import numpy as np
import pickle
import time
import datetime
import random
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from transformers import BertModel, BertTokenizer
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
# from transformers import AdamW, BertConfig
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import collections
import warnings


from torchsampler import ImbalancedDatasetSampler

warnings.filterwarnings('ignore')


class CustomDataset(Dataset):
    """自定义数据集类，支持 ImbalancedDatasetSampler"""
    
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': self.labels[idx]
        }
    
    def get_labels(self):
        """返回所有标签，ImbalancedDatasetSampler 需要这个方法"""
        if torch.is_tensor(self.labels):
            return self.labels.cpu().numpy()
        else:
            return self.labels


def load_BERT():
    # TODO change the model_path of CodeBERT to fit your environment
    # model_path = './models/roberta-base'
    model_path = '/kaggle/input/dl-test/what-makes-a-good-TODO-comment/package/todo-classifier/models/roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaModel.from_pretrained(model_path)
    if torch.cuda.is_available():
        model.cuda()
    return model, tokenizer


class Data_processor(object):

    def __init__(self, modelcard_data, label_data, batch_size, use_imbalanced_sampler=True, is_training=True):
        self.batch_size = batch_size
        self.use_imbalanced_sampler = use_imbalanced_sampler and is_training
        self.is_training = is_training
        
        print("Loading BERT model...")
        self.bert_model, self.tokenizer = load_BERT()
        print('BERT loaded')

        print("Begin encoding...")
        self.encoded_modelcard = self.bert_encode(modelcard_data)
        self.labels = np.asarray(label_data)

        print("Making data loader...")
        self.train_modelcard = self.make_data(self.encoded_modelcard)
        self.processed_dataloader = self.make_loader()
    
    def bert_encode(self, input_lst):
        encoded_input = self.tokenizer(input_lst,
                                            padding=True, truncation=True,
                                            max_length=512, return_tensors='pt')
        return encoded_input

    def make_data(self, encoded_data):
        # 获取模型所在的设备，确保数据和模型在同一设备上
        device = next(self.bert_model.parameters()).device
        input_ids, attention_masks = encoded_data['input_ids'], encoded_data['attention_mask']
        
        # Convert to Pytorch Data Types and move to same device as model
        inputs = torch.tensor(input_ids).to(device)
        masks = torch.tensor(attention_masks).to(device)
        # labels = torch.tensor(self.labels)
        labels = torch.tensor(self.labels, dtype=torch.long).to(device)
        train_data = (inputs, masks, labels)
        return train_data

    def make_loader(self):
        if self.use_imbalanced_sampler:
            # 使用自定义数据集以支持 ImbalancedDatasetSampler
            custom_dataset = CustomDataset(
                self.train_modelcard[0], 
                self.train_modelcard[1], 
                self.train_modelcard[2]
            )
            
            # 打印类别分布信息
            unique, counts = np.unique(self.labels, return_counts=True)
            print(f"类别分布: {dict(zip(unique, counts))}")
            
            # 创建采样器
            sampler = ImbalancedDatasetSampler(custom_dataset)
            
            # 创建数据加载器（使用采样器时不能使用 shuffle）
            dataloader = DataLoader(
                custom_dataset, 
                sampler=sampler,
                batch_size=self.batch_size,
                collate_fn=self._collate_fn
            )
            print("使用 ImbalancedDatasetSampler 处理数据不平衡")
        else:
            # 使用原始的 TensorDataset
            tensor_data = TensorDataset(self.train_modelcard[0],
                                        self.train_modelcard[1], self.train_modelcard[2])
            shuffle = self.is_training  # 只在训练时打乱数据
            dataloader = DataLoader(tensor_data, batch_size=self.batch_size, shuffle=shuffle)
            print("使用标准 DataLoader")
        
        return dataloader
    
    def _collate_fn(self, batch):
        """自定义 collate 函数，将字典格式的 batch 转换为元组格式"""
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_masks = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        return input_ids, attention_masks, labels


class Config(object):

    def __init__(self):
        self.model_name = 'bert'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = 2
        # TODO change the model_path of CodeBERT to fit your environment
        # self.bert_path = './models/roberta-base'
        self.bert_path = '/kaggle/input/dl-test/what-makes-a-good-TODO-comment/package/todo-classifier/models/roberta-base'
        self.tokenizer = RobertaTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.batch_size = 16
        self.num_epochs = 10
        ## bi-lstm config
        self.hidden_dim = 384
        self.n_layers = 2
        self.bidirectional = True
        self.drop_prob = 0.5
        ## CNN config
        self.filter_num = 64
        self.filter_sizes = (2,3,4)

class bert_trans_model(nn.Module):
    def __init__(self, config):
        super(bert_trans_model, self).__init__()
        self.device = config.device
        self.batch_size = config.batch_size
        self.bert = RobertaModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.transformer_layer=nn.TransformerEncoderLayer(config.hidden_size, 8)
        self.transformer_encoder=nn.TransformerEncoder(self.transformer_layer, 6)
        # 使用平均池化降维而不是展平整个序列
        self.pool = nn.AdaptiveAvgPool1d(1)
        # 在初始化时创建固定大小的fc0层
        self.fc0 = nn.Linear(config.hidden_size, 2)

    def forward(self, modelcard_input):
        modelcard_ids, modelcard_mask = modelcard_input[0], modelcard_input[1]
        modelcard_outputs = self.bert(input_ids=modelcard_ids, attention_mask=modelcard_mask)
        features = modelcard_outputs[0]
        out = self.transformer_encoder(features)        
        out = out.transpose(1, 2)
        out = self.pool(out)
        out = out.squeeze(2)        
        out = self.fc0(out)
        return out

class bilstm_model(nn.Module):
    def __init__(self, config):
        super(bilstm_model, self).__init__()
        self.device = config.device
        self.hidden_dim = config.hidden_dim
        self.n_layers = config.n_layers
        self.bidirectional = True
        self.drop_prob = config.drop_prob
        self.batch_size = config.batch_size

        self.bert = RobertaModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.lstm = nn.LSTM(config.hidden_size, self.hidden_dim, self.n_layers, bidirectional=self.bidirectional)
        self.fc0 = nn.Linear(self.hidden_dim *2, 2)

    def forward(self, modelcard_input):
        input_ids, input_mask = modelcard_input[0], modelcard_input[1]
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=input_mask)
        features = bert_outputs[0].transpose(0,1).contiguous()
        hidden = self.init_hidden(features.size(1))
        _, (h_n, c_n) = self.lstm(features, hidden)
        features = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        output = self.fc0(features)
        return output

    def init_hidden(self, batch_size):
        weight = 0
        weight = torch.tensor(weight,dtype=torch.float32)
        if torch.cuda.is_available():
            weight = weight.cuda()
        hidden = (weight.new(self.n_layers*2, batch_size, self.hidden_dim).zero_().float().to(self.device),
                   weight.new(self.n_layers*2, batch_size, self.hidden_dim).zero_().float().to(self.device)
                   )
        return hidden


class cnn_model(nn.Module):
    def __init__(self, config):
        super(cnn_model, self).__init__()
        self.device = config.device
        self.filter_num = config.filter_num
        self.n_layers = config.n_layers
        self.drop_prob = config.drop_prob
        self.batch_size = config.batch_size
        self.filter_sizes = config.filter_sizes
        self.emb_size = config.hidden_size

        self.bert = RobertaModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.convs = nn.ModuleList([nn.Conv2d(1, self.filter_num, (fsz, self.emb_size)) for fsz in self.filter_sizes])
        self.dropout = nn.Dropout(self.drop_prob)
        self.fc0 = nn.Linear(len(self.filter_sizes)*self.filter_num, 2)

    def forward(self, modelcard_input):
        modelcard_ids, modelcard_mask = modelcard_input[0], modelcard_input[1]
        modelcard_outputs = self.bert(input_ids=modelcard_ids, attention_mask=modelcard_mask)
        features = modelcard_outputs[0]
        x = features.unsqueeze(1)
        x = [F.relu(conv(x)) for conv in self.convs]
        x = [F.max_pool2d(input=x_item, kernel_size=(x_item.size(2), x_item.size(3))) for x_item in x]
        x = [x_item.view(x_item.size(0), -1) for x_item in x]
        x = torch.cat(x, 1)
        dropout_out = self.dropout(x)
        output = self.fc0(dropout_out)
        return output