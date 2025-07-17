import torch

class Config():
    def __init__(self):
        # some hyperparameters
        self.limit_num_sen = 25 # 25 for GossipCop and PolitiFact, 30 for Snopes, and 50 for PolitFact-S
        self.limit_num_words = 13 # 13 for GossipCop and PolitiFact, 80 for Snopes, and 40 for PolitFact-S
        #self.lr = 0.00001
        self.lr = 5e-5 
        self.lr_edge = 0.10 
        self.Seed = 1998 
        self.weight_decay = 0.0005
        self.hidden_dim = 100  # dimension of MLP
        self.input_dim = 768 # dimension of d
        self.output_dim = 128 # dimension of h
        self.final_dim = 2
        self.test_perc = 0.1 # 10% of all dataset for testing
        self.val_perc =  0.11 # 11% of the left data (90%) for validating, i.e., ≈ 10% of all dataset
        self.beta = 0.1
        #self.beta = 0.1
        # training parameters   
        self.epoch = 100
        self.batch_size = 8 # 8 for GossipCop and PolitiFact, 64 for Snopes and PolitFact-S

        # set devices and pre_trained model
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.bert_model = 'bert-base-uncased' # you can download BERT model and store it in this direction

        # data path
        self.dataset_name = "politic"
        self.dataset_path = "data/"+ self.dataset_name + "/" + self.dataset_name +"_news.tsv" # news textual content
        self.img_path = "data/"+ self.dataset_name + "/imgs" # news visual content
        self.model_path = "model/"+ self.dataset_name + "/" + self.dataset_name +"_checkpoint_op.pt" # save your trained LSSDN model

        # # save token path
        # self.news_list = "data/"+ self.dataset_name + "/" + self.dataset_name +"_news.npy"
        # self.news_label = "data/"+ self.dataset_name + "/" + self.dataset_name +"_label.npy"
        # self.news_id = "data/"+ self.dataset_name + "/" + self.dataset_name +"_id.npy"
        # self.nodes_num_data = "data/"+ self.dataset_name + "/" + self.dataset_name +"_node.npy"

