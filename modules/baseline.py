# the baseline model:
# text --BERT--> token-wised encodings --MultiHeadAttention--> sentence-wised encodings --classifier--> prediction --log-likelihood--> loss

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoTokenizer

Constants = {
    "PAD": 0,
    "UNK": 100,
    "BOS": 101,
    "EOS": 102,
    "PAD_WORD": '[PAD]',
    "UNK_WORD": '[UNK]',
    "BOS_WORD": '[CLS]',
    "EOS_WORD": '[SEP]',
}

class Model(nn.Module):
    r""" the baseline model
    """
    def __init__(self, args):
        super().__init__()
        self.bert = AutoModel.from_pretrained(args.pretrained_model_name_or_path)
        if args.freeze_pretrained:
            self.freeze_bert()
        self.aggregator = args.agg_class(
            input_size = self.bert.config.hidden_size
        )
        self.head = args.head_class(args, self.bert.config)
    
    def forward(self, input_ids, label, **kwargs):
        batch_size, input_len = input_ids.shape
        if "padding_mask" not in kwargs:
            padding_mask = input_ids.ne(Constants["PAD"])
        else:
            padding_mask = kwargs["padding_mask"]
            
        outputs = self.bert(input_ids)
        hidden_states = outputs.last_hidden_state
        #last_hidden_state = outputs.pooler_output
        aggregation = self.aggregator(hidden_states, padding_mask)
        loss, prediction = self.head(aggregation, label, extra_labels=kwargs["extra_labels"])
        return loss, prediction
    
    def freeze_bert(self, reverse=False):
        for name,para in self.bert.named_parameters():
            para.requires_grad = False if not reverse else True

class Aggregator(nn.Module):
    r""" a static attention layer to aggregate the information from a length-variable sequence.
    """
    def __init__(self, input_size, hidden_size=512, num_heads=16):
        super().__init__()
        self.hidden_states_to_attention_scores = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_size, num_heads, bias=False)
        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
    def forward(self, hidden_states, attention_mask):
        r"""
        hidden_states: [batch_size, seq_len, input_size]
        attention_mask: [batch_size, seq_len]
        """
        logits = self.hidden_states_to_attention_scores(hidden_states)
        # [batch_size, seq_len, num_heads]
        
        attention = F.softmax(
            logits - 1e4*(1-attention_mask.float().cuda()).unsqueeze(2).repeat(1, 1, self.num_heads),
            dim = 1).transpose(2, 1)
        # [batch_size, num_heads, seq_len]
        
        aggregation = torch.mean(torch.bmm(attention, hidden_states), dim=1)
        # [batch_size, input_size]
        
        return aggregation

class DebertaAggregator(nn.Module):
    r""" "pool" the model by simply taking the hidden state corresponding """
    def __init__(self, input_size, **kwargs):
        super().__init__()
        self.pooler = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(input_size, input_size),
            nn.GELU()
        )
    def forward(self, hidden_states, attention_mask):
        return self.pooler(hidden_states[:, 0])

class DebertaRegressionHead(nn.Module):
    """ cls -> dropout -> dense -> gelu -> dropout -> dense """
    def __init__(self, args, bert_config):
        super().__init__()
        self.MLP = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(bert_config.hidden_size, 1)
        )
    def forward(self, aggregation, label, **kwargs):
        pred = self.MLP(aggregation).squeeze(-1)
        loss = F.mse_loss(pred, label)
        return loss, pred

class BasicRegressionHead(nn.Module):
    def __init__(self, args, bert_config):
        super().__init__()
        self.MLP = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(bert_config.hidden_size, 1),
            nn.Sigmoid()
        )
    def forward(self, aggregation, label, **kwargs):
        pred = self.MLP(aggregation).squeeze(-1)
        loss = F.mse_loss(pred, label)
        return loss, pred

class ReluRegressionHead(BasicRegressionHead):
    def __init__(self, args, bert_config):
        super().__init__(args, bert_config)
        self.MLP = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(bert_config.hidden_size, int(bert_config.hidden_size**0.5)),
            nn.Tanh(),
            nn.Linear(int(bert_config.hidden_size**0.5), 1),
            nn.LeakyReLU()
        )

class TwoLayerRegressionHead(BasicRegressionHead):
    def __init__(self, args, bert_config):
        super().__init__(args, bert_config)
        self.MLP = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(bert_config.hidden_size, int(bert_config.hidden_size**0.5)),
            nn.Tanh(),
            nn.Linear(int(bert_config.hidden_size**0.5), 1),
            nn.Sigmoid()
        )

class SimCLR_MLP(BasicRegressionHead):
    def __init__(self, args, bert_config):
        super().__init__(args, bert_config)
        self.MLP = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(bert_config.hidden_size, int(bert_config.hidden_size**0.5)),
            nn.BatchNorm1d(int(bert_config.hidden_size**0.5)),
            nn.ReLU(),
            nn.Linear(int(bert_config.hidden_size**0.5), 1),
            nn.Sigmoid()
        )

def extend_with_celoss(BaseHead):
    """
    q = sigmoid(x) = 1 / (exp(-x) + 1) = 1 / x_
    ce = plog(q) + (1-p)log(1-q)
       = p(-log(x_)) + (1-p)(log(exp(-x)) - log(x_))
       = -plog(x_) + (1-p)(-x - log(x_))
       = -plog(x_) - x - log(x_) + px + plog(x_)
       = (p-1)x - log(x_)
       = (p-1)x - log(exp(-x)+1)
    """
    class CEBaseHead(BaseHead):
        def forward(self, aggregation, label, **kwargs):
            loss, pred = super(CEBaseHead, self).forward(aggregation, label)
            loss = -torch.mean(label*torch.log(pred) + \
                              (1-label)*torch.log(1-pred))
            return loss, pred
    return CEBaseHead

def extend_with_extra_regressions(BaseHead, extra_counts, extra_weights):
    class ExtraRegressionHead(BaseHead):
        def __init__(self, args, bert_config):
            super().__init__(args, bert_config)
            self.extra_weights = torch.tensor(extra_weights).float().cuda()
            self.extra_counts = extra_counts
            self.extra_MLPs = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(bert_config.hidden_size, extra_counts),
                #nn.Sigmoid()
            )
        def forward(self, aggregation, label, **kwargs):
            loss, pred = super(ExtraRegressionHead, self).forward(aggregation, label)
            if self.extra_counts == -1:
                # transform from multi-task fine-tuning to single-task fine-tuning
                return loss, pred
            loss_extra = torch.sum(F.mse_loss(self.extra_MLPs(aggregation), kwargs["extra_labels"]) * self.extra_weights)
            return loss + loss_extra, pred
    return ExtraRegressionHead
            
available_agg_classes = {agg_class.__name__:agg_class for agg_class in [Aggregator, DebertaAggregator]}

available_head_classes = {head_class.__name__:head_class for head_class in [BasicRegressionHead, TwoLayerRegressionHead, SimCLR_MLP, DebertaRegressionHead, ReluRegressionHead]}