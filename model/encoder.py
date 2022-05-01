from unicodedata import bidirectional
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            num_layers,
            dropout):

        super(Encoder,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.rnn = nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            bidirectional = True,
            batch_first  = True,
            dropout = self.dropout)
    
    def forward(self, inputs, input_lengths):
        packed_inputs = pack_padded_sequence(
            inputs,
            input_lengths,
            batch_first = True)
        
        packed_outputs, encoder_outputs = self.rnn(packed_inputs)
        outputs, _ = pad_packed_sequence(
            packed_outputs, 
            batch_first=True,
            total_length = inputs.size(1))
        return outputs, encoder_outputs
    
    def flatten_parameters(self):
        self.rnn.flatten_parameters()


def build_encoder(config):
    encoder = Encoder(
        input_size = config.input_size,
        hidden_size = config.hidden_size,
        num_layers = config.num_layers,
        dropout = config.dropout)
    return encoder
