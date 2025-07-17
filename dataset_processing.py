import re
from networkx.algorithms.shortest_paths import weighted
from networkx.classes.function import is_directed
import pandas as pd

import numpy as np
import networkx as nx

import torch
from torch_geometric.data import Data



# Get the token ids of sentences
def convert_text_to_token(tokenizer, new, limit_sens, limit_words):
    result_new = []
    new = re.split(r'[.?!;]',new)
    new = new[:-1] # drop the last full stop
    nodes_num = limit_sens
    for sen in new: # get the tkoen id of every sentece

        if len(sen.split(" ")) < 3: # filter out the sentence less than two words
            pass
        else:
            tokens = tokenizer.encode(sen, max_length=limit_words, padding="max_length", truncation=True, return_tensors='pt') # token ids of one sentence's word
            result_new.append(tokens) 
            if len(result_new) >= limit_sens: 
                break
    if len(result_new) < limit_sens: # if less than setting senteces number, we padding it
        nodes_num = len(result_new)
        need_pad = limit_sens - len(result_new)
        result_new.extend([torch.Tensor([[0] * limit_words]) for i in range(need_pad)])
    result_new = torch.stack(result_new, 0)
    result_new = torch.squeeze(result_new, 1)
    return result_new, nodes_num


# Build a graph
def build_graph(node_num, nodes_feature):

    G = nx.Graph()
    edge_list = []
    for i in range(node_num):
        for j in range(node_num):
            edge_list.append((i,j))
    G.add_edges_from(edge_list)

    adj = nx.to_scipy_sparse_array(G).tocoo()
    row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
    col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)

    data = Data(x=nodes_feature, edge_index=edge_index)
    return data

# Get data from the dataset
def get_data(data_path):
    news_list = pd.read_csv(data_path, sep="\t")


    # Get the title, content, and label of news, respectively.
    news_title, news_content, news_label, news_id = [],[],[],[]
    count = 0
    # list_name = ["clean_title","content","2_way_label","id"]
    list_name = ["clean_title","content","2_way_label","id"]
    for names_index in range(len(list_name)):
        for news_attr in news_list[list_name[names_index]]:   
            if count == 0:
                news_title.append(news_attr)
            if count == 1:
                news_content.append(news_attr)
            if count == 2:
                news_label.append(news_attr)
            if count == 3:
                news_id.append(news_attr)
        count += 1

    # Merge the news title and content
    news_tit_cont = []
    print(f"the length of news_list is: {len(news_list)}")
    for i, tit in enumerate(news_title):
        news_tit_cont.append(f"{tit}. {news_content[i]}")
    print(f"the length of news_tit_cont is: {len(news_tit_cont)}")
    return news_tit_cont, news_label, news_id





