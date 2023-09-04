import torch
import logging
from torch import nn
import torch.nn.functional as F
from model.tcanet import TCANet

class MultiModalNet(nn.Module):

    def __init__(self, emb_size, input_output_size, k_num_channels, v_num_channels, visual_emsize, seq_len, num_sub_blocks, temp_attn, nheads, en_res,
                 conv, key_size, kernel_size=2, dropout=0.3, wdrop=0.0, emb_dropout=0.1, tied_weights=False, 
                 signal_type=None, visual=True):
        super(MultiModalNet, self).__init__()
        self.k_model = TCANet(emb_size, input_output_size, k_num_channels, visual_emsize, seq_len, num_sub_blocks, temp_attn=temp_attn, nheads=nheads,
                en_res=en_res, conv=conv, dropout=dropout, emb_dropout=emb_dropout, key_size=key_size, 
                kernel_size=kernel_size, tied_weights=tied_weights, signal_type='kinematic', visual=visual)
        self.v_model = TCANet(emb_size, input_output_size, k_num_channels, visual_emsize, seq_len, num_sub_blocks, temp_attn=temp_attn, nheads=nheads,
                en_res=en_res, conv=conv, dropout=dropout, emb_dropout=emb_dropout, key_size=key_size, 
                kernel_size=kernel_size, tied_weights=tied_weights, signal_type='visual', visual=visual)
        self.one_hot_encoder = OneHotEncoderLayer(input_output_size)
        

    def forward(self, input, labels, task):
        k_input, v_input = input
        one_hot_label = self.one_hot_encoder(labels)
        k_outbatch = self.k_model(k_input)
        v_outbatch = self.v_model(v_input)
        

class OneHotEncoderLayer(nn.Module):
    def __init__(self, n_classes):
        super(OneHotEncoderLayer, self).__init__()
        self.n_classes = n_classes

    def forward(self, x):
        # Ensure the input is of type Long
        x = x.long()
        
        # Allocate a tensor for the one-hot encoded labels
        one_hot = torch.zeros(x.size(0), self.n_classes).to(x.device)
        
        # Scatter ones to the appropriate indices
        one_hot.scatter_(1, x.unsqueeze(1), 1)
        return one_hot

        

