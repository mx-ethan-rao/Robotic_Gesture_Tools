import torch
import logging
from torch import nn
import torch.nn.functional as F
from model.tcn_block import TemporalConvNet
# from model.pe import PositionEmbedding
# from model.optimizations import VariationalDropout, WeightDropout

from IPython import embed

logging.basicConfig( \
    level = logging.INFO, \
    format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s')

class TCANet(nn.Module):

    def __init__(self, emb_size, input_output_size, num_channels, visual_emsize, seq_len, num_sub_blocks, temp_attn, nheads, en_res,
                 conv, key_size, kernel_size=2, dropout=0.3, wdrop=0.0, emb_dropout=0.1, tied_weights=False, 
                 signal_type=None, visual=True):
        super(TCANet, self).__init__()
        self.temp_attn = temp_attn
        self.signal_type = signal_type
        self.num_levels = len(num_channels)
        # self.word_encoder = nn.Embedding(input_output_size, emb_size)
        if signal_type == 'visual':
            emb_size = visual_emsize
            self.visual_embedding = nn.Sequential(
            # First convolutional layer followed by ReLU and max-pooling
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                # Second convolutional layer followed by ReLU and max-pooling
                nn.Conv2d(32, visual_emsize, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1)
                # nn.MaxPool2d(kernel_size=2, stride=2)
            )
            # self.fc_visual = nn.Linear(64 * 56 * 56, visual_emsize)
            # self.word_encoder = nn.Embedding(256, emb_size)
        # self.position_encoder = PositionEmbedding(emb_size, seq_len)
        self.tcanet = TemporalConvNet(input_output_size, emb_size, num_channels, \
            num_sub_blocks, temp_attn, nheads, en_res, conv, key_size, kernel_size, visual=visual, dropout=dropout)
        # self.tcanet = WeightDropout(self.tcanet, self.get_conv_names(num_channels), wdrop)
        # self.drop = VariationalDropout(emb_dropout) # drop some embeded features, e.g. [16,80,600]->[16,80,421]
        self.drop = nn.Dropout(emb_dropout)
        self.decoder = nn.Linear(num_channels[-1], input_output_size)
        if tied_weights:
            if self.signal_type != 'mnist':
                pass
                # self.decoder.weight = self.word_encoder.weight
        self.emb_dropout = emb_dropout
        self.init_weights()

    def init_weights(self):
        # self.word_encoder.weight.data.normal_(0, 0.01)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def get_conv_names(self, num_channels):
        conv_names_list = []
        for level_i in range(len(num_channels)):
            conv_names_list.append(['network', level_i, 'net', 0, 'weight_v'])
            conv_names_list.append(['network', level_i, 'net', 4, 'weight_v'])
        return conv_names_list

    def forward(self, input):
        """Input ought to have dimension (N, C_in, L_in), where L_in is the seq_len; here the input is (N, L, C)"""
        # input: [batchsize, seq_len]
        # emb = self.drop(torch.cat([self.word_encoder(input), self.position_encoder(input)], dim=2))
        if self.signal_type == 'visual':

            # image embedding
            batch_size = input.size(0)
            num_time_steps = input.size(1)
            input = input.view(batch_size * num_time_steps, input.size(2), input.size(3), input.size(4)) # [N = batch_size * number_time_steps, H, W, C]
            input = input.permute(0, 3, 1, 2)  # [N, C, H, W]
            input = self.visual_embedding(input)
            input = input.contiguous().view(input.size(0), -1) # [N, 64 * 60 * 80]
            # input = self.fc_visual(input)  # [N, C = visual_emb]
            input = input.view(batch_size, num_time_steps, -1) # [N = batch_size, L = number_time_steps, C]
            
            # TCAN
            input = self.drop(input)
            if self.temp_attn:
                y, attn_weight_list = self.tcanet(input.transpose(1, 2))
                y = self.decoder(y.transpose(1, 2)[:, -1])
                return y.contiguous(), [attn_weight_list[0], attn_weight_list[self.num_levels//2], attn_weight_list[-1]]
            else:
                y = self.tcanet(input.transpose(1, 2))
                encoded_input = y.transpose(1, 2)[:, -1]
                y = self.decoder(y.transpose(1, 2)[:, -1])
                return y.contiguous(), encoded_input.contiguous()
            
            # emb = self.drop(self.word_encoder(input))
            # if self.temp_attn:
            #     y, attn_weight_list = self.tcanet(input) # input should have dimension (N, C, L)
            #     # y, attn_weight_list = self.tcanet(emb.transpose(1, 2)) # input should have dimension (N, C, L)
            #     o = self.decoder(y[:, :, -1])
            #     # return F.log_softmax(o, dim=1).contiguous(), [attn_weight_list[0], attn_weight_list[self.num_levels//2], attn_weight_list[-1]]
            #     return F.log_softmax(o, dim=1).contiguous()
            # else:
            #     y = self.tcanet(input) # input should have dimension (N, C, L)
            #     # y = self.tcanet(emb.transpose(1, 2)) # input should have dimension (N, C, L)
            #     o = self.decoder(y[:, :, -1])
            #     return F.log_softmax(o, dim=1).contiguous()

        if self.signal_type == 'kinematic':
            input = self.drop(input)
            if self.temp_attn:
                y, attn_weight_list = self.tcanet(input.transpose(1, 2))
                y = self.decoder(y.transpose(1, 2)[:, -1])
                return y.contiguous(), [attn_weight_list[0], attn_weight_list[self.num_levels//2], attn_weight_list[-1]]
            else:
                y = self.tcanet(input.transpose(1, 2))
                encoded_input = y.transpose(1, 2)[:, -1]
                y = self.decoder(y.transpose(1, 2)[:, -1])
                return y.contiguous(), encoded_input.contiguous()
            
        # emb = self.drop(self.word_encoder(input))
        # if self.temp_attn:
        #     y, attn_weight_list = self.tcanet(emb.transpose(1, 2))
        #     y = self.decoder(y.transpose(1, 2))
        #     return y.contiguous(), [attn_weight_list[0], attn_weight_list[self.num_levels//2], attn_weight_list[-1]]
        # else:
        #     y = self.tcanet(emb.transpose(1, 2))
        #     y = self.decoder(y.transpose(1, 2))
        #     return y.contiguous()

