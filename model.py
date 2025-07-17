import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from transformers import BertModel
from torch_geometric.nn import GCNConv, GATConv
from torchvision import models, transforms
from torch.autograd import Variable
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from dataset_processing import build_graph
import matplotlib.pyplot as plt
import numpy as np
from configs import Config
cfg = Config()

from sklearn.preprocessing import MinMaxScaler


class BertPre1(nn.Module):
    def __init__(self, cfg):
        super(BertPre1, self).__init__()
        self.bert_pret1 = BertModel.from_pretrained(cfg.bert_model)
        unfreeze_layers = ['layer.10','layer.11', 'bert.pooler', 'out.']
        # frozen bert layersq
        for name, param in self.bert_pret1.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break
        
    def forward(self, sent):
        seman_vecs = self.bert_pret1(sent)
        seman_vecs = seman_vecs.pooler_output
        return seman_vecs


class BertPre2(nn.Module):
    def __init__(self, cfg):
        super(BertPre2, self).__init__()
        self.bert_pret2 = BertModel.from_pretrained(cfg.bert_model)
        unfreeze_layers = ['layer.10','layer.11', 'bert.pooler', 'out.']
        # frozen bert layersq
        for name, param in self.bert_pret2.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break
                
    def forward(self, sent):
        seman_vecs = self.bert_pret2(sent)
        seman_vecs = seman_vecs.pooler_output
        return seman_vecs


class EdgeWeightCal1(nn.Module):
    def __init__(self, cfg):
        super(EdgeWeightCal1, self).__init__()
        self.Wl_weight = nn.Parameter(torch.rand(cfg.input_dim,cfg.input_dim)) # cfg.input_dim*cfg.input_dim
        self.Wn_weight = nn.Parameter(torch.rand(cfg.input_dim,cfg.input_dim))
        self.Ws_weight = nn.Parameter(torch.rand(cfg.input_dim,cfg.input_dim))
        # self.hs_weight = nn.Parameter(torch.rand(1,cfg.output_dim))
        
                
    def forward(self, node_feature, sent_feature):
        node_feature = node_feature.transpose(0,1)    # cfg.input_dim*N
        sent_feature = sent_feature.transpose(0,1)    # cfg.input_dim*N

        F_matrix = torch.tanh(node_feature.transpose(0,1).matmul(self.Wl_weight.matmul(sent_feature)))   # N*N
        Rt = torch.tanh(self.Wn_weight.matmul(node_feature) + self.Ws_weight.matmul(sent_feature).matmul(F_matrix))   # cfg.input_dim*N, R^T

        R = Rt.transpose(0,1) # R
        node_feature = node_feature.transpose(0,1)

        edge_weights = torch.cosine_similarity(node_feature.unsqueeze(1), R.unsqueeze(0), dim=2)
        edge_weights = torch.clamp(edge_weights, min=0.0).view(-1)
        edge_weights = edge_weights.clone().detach().requires_grad_(True).to(dtype=torch.float32, device=cfg.device)
        return edge_weights



class MyModel(nn.Module):
    def __init__(self, cfg):
        super(MyModel, self).__init__()
        self.bert_pret1 = BertPre1(cfg)
        self.bert_pret2 = BertPre2(cfg)  # sequential encoder

        # self.resnet50 = MyResNet50(cfg) # 이미지 처리 부분 제거

        self.edge_weight1 = EdgeWeightCal1(cfg)

        self.conv1 = GCNConv(cfg.input_dim, cfg.output_dim)
        self.MLP_bert = nn.Sequential(nn.Linear(cfg.input_dim, cfg.output_dim * 2),
                                      nn.Linear(cfg.output_dim * 2, cfg.output_dim))

        self.MLP_pred = nn.Sequential(nn.Linear(cfg.output_dim * 2, cfg.hidden_dim),
                                      nn.Linear(cfg.hidden_dim, cfg.final_dim))

    # forward 함수의 인자에서 이미지 ID를 받는 'ids' 제거
    def forward(self, sent, nodes_num, Flag):
        pred_list = []
        kl_list = []
        for i, sent_i in enumerate(sent):
            # 의미 있는 노드(문장)만 사용
            sent_i = sent_i[0:nodes_num[i]]

            # 벡터화
            nodes_vecs = self.bert_pret1(sent_i)
            semantic_vecs = self.bert_pret2(sent_i)  # sequential representation of news sentences

            # --- 이미지 특징 추출 및 결합 로직 전체 제거 ---
            # img_vecs = self.resnet50(ids[i])
            #
            # len_img_vecs = len(img_vecs)
            # len_cat_vecs = nodes_num[i]+len_img_vecs
            # if len_img_vecs != 0:
            #     nodes_vecs = torch.cat((nodes_vecs, img_vecs), dim=0)
            #     semantic_vecs = torch.cat((semantic_vecs, img_vecs), dim=0) # sequential representation of news content
            # ---------------------------------------------

            # 그래프 구성 (이제 노드 개수는 순수 문장 개수)
            len_cat_vecs = nodes_num[i]
            text_graph = build_graph(len_cat_vecs, nodes_vecs).to(cfg.device)
            node_fea, edge_index = text_graph.x, text_graph.edge_index

            # 엣지 가중치 추론
            edge_weight1 = self.edge_weight1(node_fea, semantic_vecs)

            # 그래프 표현 학습
            conv1_graph = self.conv1(node_fea, edge_index, edge_weight1)

            # 차원 축소
            semantic_vecs = self.MLP_bert(semantic_vecs)

            # KL 발산 계산
            if Flag == "outer":
                kldiv = torch.nn.functional.kl_div(semantic_vecs.softmax(dim=-1).log(),
                                                   conv1_graph.softmax(dim=-1), reduction="sum")
                if (kldiv > 999):  # avoid too large kl
                    kldiv = torch.tensor(999).to(cfg.device)
                kl_list.append(kldiv)

            graph_seman = torch.cat((semantic_vecs, conv1_graph), dim=1)
            pred = self.MLP_pred(graph_seman)
            pred = torch.mean(pred, dim=0)
            pred_list.append(pred)

        pred = torch.stack(pred_list, 0)
        if Flag == "outer":
            kldiv = torch.stack(kl_list, 0)
            return pred, kldiv
        else:
            return pred