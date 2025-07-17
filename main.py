from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import parameter
from configs import Config
from model import MyModel
from dataset_processing import *
from mydataset import MyDataSet
from earlystopping import EarlyStopping

from tqdm import tqdm
from transformers import BertModel, BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from torch.utils.data import DataLoader, RandomSampler, dataloader

import numpy as np
import re
import os
import torch
import time
import argparse


parser = argparse.ArgumentParser(description="code")
parser.add_argument('--dataset_name', '-dn', type=str, default=-1, help='dataset name')
parser.add_argument('--Epoch', '-epoch', type=int, default=100, help='epoch number')
parser.add_argument('--hyperpara', '-beta', type=float, default=-1, help='hypterparamter beta')
args = parser.parse_args()


# Random seed.
def set_seed(seed):
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Metircs
def metric_new(pred, label):
    p = metrics.precision_score(label, pred, zero_division=0,average='weighted')
    r = metrics.recall_score(label, pred,average='weighted')
    F1 = metrics.f1_score(label, pred,average='weighted')
    acc = metrics.accuracy_score(label, pred)
    # fpr, tpr, thresholds = metrics.roc_curve(label, pred_old.detach().numpy()[:,1], pos_label=1)
    # auc = metrics.auc(fpr, tpr)
    # if(mode=='test'):
    #     print(metrics.classification_report(label, pred, digits=4))
    # return p, r, F1, acc, auc
    return p, r, F1, acc


if __name__ == "__main__":
    cfg = Config() # Instantiate the config class

    set_seed(cfg.Seed)
    # Setting the inputs
    if args.dataset_name != -1:
        cfg.dataset_name = args.dataset_name
        cfg.dataset_path = "data/"+ cfg.dataset_name + "/" + cfg.dataset_name +"_news.tsv"
        cfg.img_path = "data/"+ cfg.dataset_name + "/imgs"
        #cfg.model_path = "data/"+ cfg.dataset_name + "/" + cfg.dataset_name +"_checkpoint.pt"
        cfg.news_list = "data/"+ cfg.dataset_name + "/" + cfg.dataset_name +"_news.npy"
        cfg.news_label = "data/"+ cfg.dataset_name + "/" + cfg.dataset_name +"_label.npy"
        cfg.news_id = "data/"+ cfg.dataset_name + "/" + cfg.dataset_name +"_id.npy"
        cfg.nodes_num_data = "data/"+ cfg.dataset_name + "/" + cfg.dataset_name +"_node.npy"

        
    if args.hyperpara != -1:
        cfg.beta = args.hyperpara
    
    if args.Epoch != 100:
        cfg.num_epoch = args.Epoch
    # for i in cfg:
    print(vars(cfg))

    print("*" * 80)  
    # Setting random seed
    torch.manual_seed(cfg.Seed)
    torch.cuda.manual_seed_all(cfg.Seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(cfg.Seed)
    print("*" * 80)

    print("*" * 80)  
    print("Constructing Model...")
    LSSDN = MyModel(cfg)
    
    print("device:",cfg.device)
    LSSDN = LSSDN.to(cfg.device)
    early_stopping = EarlyStopping()
    print("*" * 80)

    print("*" * 80)  
    print("Defining Loss Function...")
    loss_pred = torch.nn.CrossEntropyLoss()
    print("*" * 80)
    
    print("*" * 80)
    print("Taking Adam as the Ooptimizer...")

    outer_optimizer = torch.optim.Adam([
        {'params': LSSDN.bert_pret1.parameters()},
        {'params': LSSDN.bert_pret2.parameters()},
        #{'params': LSSDN.resnet50.parameters()},
        {'params': LSSDN.conv1.parameters()}, 
        {'params': LSSDN.MLP_bert.parameters()}, # MLP
        {'params': LSSDN.MLP_pred.parameters()}], cfg.lr, weight_decay=cfg.weight_decay, eps=1e-6)
    inner_optimizer = torch.optim.Adam([
        {'params': LSSDN.edge_weight1.parameters()},
        ], cfg.lr_edge, weight_decay=cfg.weight_decay, eps=1e-6)
    print("*" * 80)
    #scheduler = ReduceLROnPlateau(outer_optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    # Get data from the dataset
    news_tit_cont, news_label, news_id = get_data(cfg.dataset_path)


    # Getting token ids and  the sentences (nodes) number
    print("convert_text_to_token......")
    tokenizer = BertTokenizer.from_pretrained(cfg.bert_model) # Tokenizer
    sents_list = []
    nodes_num = []
    for tit in tqdm(news_tit_cont):
        sent, node_num = convert_text_to_token(tokenizer, tit, cfg.limit_num_sen, cfg.limit_num_words)
        sents_list.append(sent)
        nodes_num.append(node_num)
    
    # # Save the tokens for convenient
    np.save(cfg.news_list,sents_list)
    np.save(cfg.news_label,news_label)
    np.save(cfg.nodes_num_data,nodes_num)
    np.save(cfg.news_id,news_id)

    # --- 아래 필터링 코드 추가 ---
    print(f"데이터 필터링 전, 원본 데이터 개수: {len(news_label)}")

    # 유효한 레이블을 가진 인덱스만 추출
    valid_indices = [i for i, label in enumerate(news_label) if label >= 0]

    sents_list = [sents_list[i] for i in valid_indices]

    # NumPy 배열들은 기존 방식대로 필터링
    news_label = [news_label[i] for i in valid_indices]
    nodes_num = [nodes_num[i] for i in valid_indices]
    news_id = [news_id[i] for i in valid_indices]

    print(f"데이터 필터링 후, 유효 데이터 개수: {len(news_label)}")
# --- 여기까지 추가 ---

    # sents_list = np.load(cfg.news_list,allow_pickle=True)
    # news_label = np.load(cfg.news_label,allow_pickle=True)
    # news_id = np.load(cfg.news_id,allow_pickle=True)
    # nodes_num = np.load(cfg.nodes_num_data,allow_pickle=True)

    train_val_news_list, test_news_list, train_val_news_label, test_news_label, train_val_news_node, test_news_node, train_val_news_id, test_news_id \
        = train_test_split(sents_list, news_label, nodes_num, news_id, random_state=cfg.Seed, test_size=cfg.test_perc)

    train_news_list, val_news_list, train_news_label, val_news_label, train_news_node, val_news_node, train_news_id, val_news_id \
        = train_test_split(train_val_news_list, train_val_news_label, train_val_news_node, train_val_news_id, random_state=cfg.Seed, test_size=cfg.val_perc)
    
    print(f"the length of train_news_list is: {len(train_news_list)}")
    print(f"the length of val_news_list is: {len(val_news_list)}")
    print(f"the length of test_news_list is: {len(test_news_list)}")

    train_news = MyDataSet(list(zip(train_news_list,train_news_label, train_news_node, train_news_id)))
    val_news = MyDataSet(list(zip(val_news_list,val_news_label, val_news_node, val_news_id)))
    test_news = MyDataSet(list(zip(test_news_list,test_news_label, test_news_node, test_news_id)))
    

    train_sampler = RandomSampler(train_news)
    val_sampler = RandomSampler(val_news)
    test_sampler = RandomSampler(test_news)
    
    train_dataloader = DataLoader(train_news, sampler=train_sampler,batch_size=cfg.batch_size, drop_last=True)
    val_dataloader = DataLoader(val_news, sampler=val_sampler,batch_size=cfg.batch_size, drop_last=True)
    test_dataloader = DataLoader(test_news, sampler=test_sampler,batch_size=cfg.batch_size, drop_last=True)

    draw_train_loss, draw_val_loss = [], []
    min_val_loss = 10.0

    torch.set_num_threads(32)
    start_time = time.time()
    
    for epoch in range(cfg.epoch):
        print("epoch:{}".format(epoch + 1))
        LSSDN.train()
        epoch_loss_train = 0.0
        
        train_y_epoch = []
        start_time_epoch = time.time()
        
        for i, batch in enumerate(train_dataloader):
            train_news = batch[0].long().to(cfg.device)
            train_label = batch[1].long().to(cfg.device)
            train_node = batch[2].long().to(cfg.device)
            train_id = batch[3]
            train_y_epoch.append(train_label)
            
            # Inner optimization
            inner_optimizer.zero_grad()
            pred_res = LSSDN.forward(train_news, train_node, Flag="inner")


            #print("--- Debug Info ---")
            #print("Prediction shape:", pred_res.shape)
            #print("Label shape:", train_label.shape)
            #print("Label unique values:", torch.unique(train_label))
            #print("Label min:", torch.min(train_label))
            #print("Label max:", torch.max(train_label))
            #print("------------------")

            batch_loss_train = loss_pred(pred_res, train_label)
            epoch_loss_train += train_label.size()[0] * batch_loss_train.item()
            batch_loss_train.backward()
            inner_optimizer.step()

            # Outer optimization
            outer_optimizer.zero_grad()
            pred_res, kldiv = LSSDN.forward(train_news, train_node, Flag="outer")
            kldiv = torch.mean(kldiv)

            batch_loss_train = loss_pred(pred_res, train_label) + cfg.beta*kldiv
            epoch_loss_train += train_label.size()[0] * batch_loss_train.item()
            batch_loss_train.backward()
            outer_optimizer.step()

        
        epoch_loss_train /= len(train_y_epoch)
        draw_train_loss.append(epoch_loss_train)
        print(f"epoch:{epoch+1}, train_loss: {epoch_loss_train:.4f}......")

        with torch.no_grad():
            LSSDN.eval()
            epoch_loss_val = 0.0
            pred_epoch, pred_one_epoch, val_y_epoch = [], [], []

            for i, batch in enumerate(val_dataloader):
                # node_num = nodes_num[i]

                val_news = batch[0].long().to(cfg.device)
                val_label = batch[1].long().to(cfg.device)
                val_node = batch[2].long().to(cfg.device)
                val_id = batch[3]
                pred_res, _ = LSSDN.forward(val_news, val_node, Flag="outer")

                batch_loss_val = loss_pred(pred_res, val_label)
                epoch_loss_val += val_label.size()[0]*batch_loss_val.item()
                pred_one = torch.argmax(pred_res, dim=1)
                pred_one_epoch.append(pred_one)
                val_y_epoch.append(val_label)

            pred_one = torch.cat([i for i in pred_one_epoch],0).cpu()
            val_label = torch.cat([i for i in val_y_epoch],0).cpu()

            Pre, Rec, F1, Acc = metric_new(pred_one, val_label)
            epoch_loss_val /= len(val_label)
            draw_val_loss.append(epoch_loss_val)
            print(f"********epoch:{epoch+1}, val_loss:{epoch_loss_val:.4f},  Acc:{Acc:.3f}, Pre:{Pre:.3f}, Rec:{Rec:.3f}, F1:{F1:.3f}")

            print("***********************************")
    
            #scheduler.step(epoch_loss_val)
            # early stopping
            early_stopping(epoch_loss_val, LSSDN)
            if early_stopping.early_stop:
                print('Early stopping!')
                break
        print(f"Time consuming for this epoch：{(time.time()-start_time_epoch)}s")
        print("*" * 80)
    print("*" * 80)
    print(f"Time consuming：{(time.time()-start_time)/60}mins, now testing...")
    

    Model = MyModel(cfg).to(cfg.device)
    Model.load_state_dict(torch.load(cfg.model_path))
    pred_one_batch, test_y_batch = [], []

    with torch.no_grad():
        Model.eval()
        for i, batch in enumerate(test_dataloader):

            test_news = batch[0].long().to(cfg.device)
            test_label = batch[1].long().to(cfg.device)
            test_node = batch[2].long().to(cfg.device)
            test_id = batch[3]

            pred_res, _ = Model.forward(test_news, test_node, Flag="outer")
            pred_one = torch.argmax(pred_res, dim=1)
            pred_one_batch.append(pred_one)
            test_y_batch.append(test_label)

        pred_one = torch.cat([i for i in pred_one_batch],0).cpu()
        test_label = torch.cat([i for i in test_y_batch],0).cpu()

        Pre, Rec, F1, Acc = metric_new(pred_one, test_label)
        print(f"Acc:{Acc:.3f}, Pre:{Pre:.3f}, Rec:{Rec:.3f}, F1:{F1:.3f}")
