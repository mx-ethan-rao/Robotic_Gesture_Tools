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
        self.k_model = TCANet(emb_size, input_output_size, k_num_channels, visual_emsize, seq_len, num_sub_blocks, temp_attn=False, nheads=nheads,
                en_res=en_res, conv=conv, dropout=dropout, emb_dropout=emb_dropout, key_size=key_size, 
                kernel_size=kernel_size, tied_weights=tied_weights, signal_type='kinematic', visual=visual)
        self.v_model = TCANet(emb_size, input_output_size, v_num_channels, visual_emsize, seq_len, num_sub_blocks, temp_attn=False, nheads=nheads,
                en_res=en_res, conv=conv, dropout=dropout, emb_dropout=emb_dropout, key_size=key_size, 
                kernel_size=kernel_size, tied_weights=tied_weights, signal_type='visual', visual=visual)
        self.one_hot_encoder = OneHotEncoderLayer(input_output_size)
        self.mlp = nn.Linear(k_num_channels[-1] + v_num_channels[-1], input_output_size)
        

    def forward(self, input, labels, task):

        k_input, v_input = input
        one_hot_labels = self.one_hot_encoder(labels)
        k_outbatch, k_encoded_input = self.k_model(k_input)
        v_outbatch, v_encoded_input = self.v_model(v_input)
        k_input = F.softmax(k_outbatch, dim=1)
        v_input = F.softmax(v_outbatch, dim=1)

        if task == 'train':
        # Compute the attention for each stream
            k_attention = torch.sum(k_input * one_hot_labels, dim=1, keepdim=True)
            v_attention= torch.sum(v_input * one_hot_labels, dim=1, keepdim=True)

            # element-wise product with attention
            k_encoded_input = k_encoded_input * k_attention
            v_encoded_input = v_encoded_input * v_attention
        
        input = self.mlp(torch.concat([k_encoded_input, v_encoded_input], dim=-1))
        return input.contiguous(), k_outbatch.contiguous(), v_outbatch.contiguous()
        
        
        

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

        

