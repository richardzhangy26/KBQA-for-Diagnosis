from transformers import BertTokenizer,AdamW
from torch.utils.data import Dataset, DataLoader
import csv
import torch
import math
from transformers import BertModel
from torch import nn
from torch.nn import CrossEntropyLoss
import numpy as np
# basic config
class Config:
    def __init__(self):
        self.device = 'cuda'
        self.bert_dim = 768
        self.lstm_dim = 384
        self.lstm_layers = 3
        self.dropout = 0.5
        self.tokenizer = BertTokenizer.from_pretrained('../input/huggingface-bert/bert-base-chinese')

        self.batch_size = 64
        self.max_length = 70
        self.model_path = '../input/huggingface-bert/bert-base-chinese'
        self.learning_rate = 1e-5
        
        self.train_data_path = '../input/dataiphanchortext/train_20200228.csv'
        self.dev_data_path = '../input/dataiphanchortext/dev_20200228.csv'
        self.epochs = 20

# ESIM 
class ESIM(nn.Module):
    def __init__(self, config):
        super(ESIM, self).__init__()
        self.bert = BertModel.from_pretrained(config.model_path)
        self.lstm = nn.LSTM(config.bert_dim * 4, config.lstm_dim, config.lstm_layers, dropout=config.dropout,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(config.lstm_dim * 2 * 4, 2)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, bert_inputs_01, bert_inputs_02):
        sequence_output_01 = self.bert(**bert_inputs_01)
        sequence_heatmap_01 = sequence_output_01.last_hidden_state  # batch size, seq len,hid_dim
        seq_len = sequence_heatmap_01.shape[1]
        attention_mask_01 = bert_inputs_01['attention_mask']
        attention_mask_01 = attention_mask_01.unsqueeze(2).repeat(1, 1, seq_len)

        sequence_output_02 = self.bert(**bert_inputs_02)
        sequence_heatmap_02 = sequence_output_02.last_hidden_state
        attention_mask_02 = bert_inputs_02['attention_mask']
        attention_mask_02 = attention_mask_02.unsqueeze(1).repeat(1, seq_len, 1)
        sequence_heatmap_02_T = sequence_heatmap_02.permute(0, 2, 1)

        d_k = sequence_heatmap_01.shape[-1]
        scores = torch.matmul(sequence_heatmap_01, sequence_heatmap_02_T) / math.sqrt(d_k)

        mask = attention_mask_01 * attention_mask_02
        scores = scores.masked_fill(mask == 0, -1e9)

        scores_01 = torch.nn.functional.softmax(scores, dim=-1)  # 第一个句子对第二个句子做注意力
        scores_02 = torch.nn.functional.softmax(scores, dim=1)  # 第二个句子对第一个句子做注意力

        _sequence_heatmap_01 = torch.matmul(scores_01, sequence_heatmap_02)
        _sequence_heatmap_02 = torch.matmul(scores_02.permute(0, 2, 1), sequence_heatmap_01)

        m1 = torch.cat([sequence_heatmap_01, _sequence_heatmap_01, sequence_heatmap_01 - _sequence_heatmap_01,
                        sequence_heatmap_01 * _sequence_heatmap_01], dim=-1)

        m2 = torch.cat([sequence_heatmap_02, _sequence_heatmap_02, sequence_heatmap_02 - _sequence_heatmap_02,
                        sequence_heatmap_02 * _sequence_heatmap_02], dim=-1)

        y1, _ = self.lstm(m1)  # batch size, seq len, lstm hiddim *2
        y2, _ = self.lstm(m2)
        mx1, _ = torch.max(y1, dim=1)  # batch size, hid dim * 2
        av1 = torch.mean(y1, dim=1)
        mx2, _ = torch.max(y2, dim=1)
        av2 = torch.mean(y2, dim=1)

        y = torch.cat([mx1, av1, mx2, av2], dim=-1)
        y = self.fc(y)
        y = self.dropout(y)
        return y

#dataset
class EsimDataset(Dataset):  # 继承Dataset
    def __init__(self, path):  # __init__是初始化该类的一些基础参数
        with open(path, encoding='utf8') as F:
            data = csv.reader(F)
            next(data)
            self.data = list(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        sentence_1 = self.data[index][2]
        sentence_2 = self.data[index][3]
        label = int(self.data[index][4])

        return sentence_1, sentence_2, label

# Batch 
class Batch:
    """
    根据句子构造input_ids token_type_ids, attention_mask
    """

    def __init__(self, tokenizer, max_length, device):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def __call__(self, batch_data):
        bert_inputs_01 = self.tokenizer(batch_data[0], truncation=True, return_tensors="pt",
                                        padding='max_length', max_length=self.max_length)

        bert_inputs_02 = self.tokenizer(batch_data[1], truncation=True, return_tensors="pt",
                                        padding='max_length', max_length=self.max_length)
        labels = batch_data[2]
        labels = torch.tensor(labels).to(self.device)
        # for key in bert_inputs:
        #     bert_inputs[key] = bert_inputs[key].to(self.device)
        bert_inputs_01 = bert_inputs_01.to(self.device)
        bert_inputs_02 = bert_inputs_02.to(self.device)
        return bert_inputs_01, bert_inputs_02, labels


def create_date_iter(config):
    train_data = EsimDataset(config.train_data_path)
    dev_data = EsimDataset(config.dev_data_path)

    train_iter = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    dev_iter = DataLoader(dev_data, batch_size=config.batch_size, shuffle=True)

    return train_iter,dev_iter


def load_model(config):
    device = config.device
    model = ESIM(config, )
    model.to(device)

    # prepare optimzier
    param_optimizer = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=10e-8)
    sheduler = None

    return model, optimizer, sheduler, device

def train_epoch(model, train_iter, dev_iter, optimizer, batch, loss_fct, best_acc, epoch):
    for step, i in enumerate(train_iter):
        model.train()
        bert_inputs_01, bert_inputs_02, labels = batch(i)

        out = model(bert_inputs_01, bert_inputs_02)
        loss = loss_fct(out, labels)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            acc, matrix = test(model, dev_iter, batch)
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), 'best_acc.pth')
                print('epoch:{},step:{},acc:{:.4f},train loss:{:.4f}'.format(epoch, step, acc, loss.item()))
                print(matrix)
    return best_acc


def test(model, dev_iter, batch):
    model.eval()
    matrix = np.zeros((2, 3))
    total_count = 0
    acc_count = 0
    for step, i in enumerate(dev_iter):
        bert_inputs_01, bert_inputs_02, labels = batch(i)

        out = model(bert_inputs_01, bert_inputs_02)
        total_count += out.shape[0]
        _, out = torch.max(out, dim=-1)
        acc_count += torch.sum(out == labels)
        for class_index in range(2):  # class_index=0 表示不属于一类的情况 class_index=1 表示属于1类的情况
            pred = out == class_index
            gold = labels == class_index
            tp = pred[pred == gold]
            matrix[class_index, 0] += torch.sum(tp)
            matrix[class_index, 1] += torch.sum(pred)
            matrix[class_index, 2] += torch.sum(gold)

    return acc_count / total_count, matrix


def train(model, train_iter, dev_iter, optimizer, batch, epochs):
    loss_fct = CrossEntropyLoss()
    best_acc = 0
    for epoch in range(epochs):
        best_acc = train_epoch(model, train_iter, dev_iter, optimizer, batch, loss_fct, best_acc, epoch)
    
if __name__ == '__main__':
    config = Config()
    train_iter, dev_iter = create_date_iter(config)
    batch = Batch(config.tokenizer, config.max_length, config.device)
    model, optimizer, sheduler, device = load_model(config)
    # 训练代码
    batch = Batch(config.tokenizer, config.max_length, config.device)
    train(model, train_iter, dev_iter, optimizer, batch, config.epochs)

    model.load_state_dict(torch.load('./best_acc.pth'))
    # 预测代码
    test_data = EsimDataset('../input/xinguanyiqingxiangsidutest/test.csv')

    test_iter = DataLoader(test_data, batch_size=config.batch_size, shuffle=False)
    acc, matrix = test(model, test_iter, batch) 