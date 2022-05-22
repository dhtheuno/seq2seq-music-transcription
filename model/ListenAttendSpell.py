import torch
import torch.nn as nn

from model.encoder import Encoder, build_encoder
from model.decoder import Decoder, build_decoder
from utils import AttrDict

class ListenAttendSpell(nn.Module):
    def __init__(
        self,
        encoder,
        decoder):
        super(ListenAttendSpell, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputs, input_lengths, targets, target_lengths):
        encoder_outputs, _ = self.encoder(inputs, input_lengths)
        loss = self.decoder(targets, encoder_outputs, target_lengths)
        return loss
    
    @torch.no_grad()
    def recognize(self, input, input_length, batch=False, gpu=False):
        if not batch:
            encoder_outputs, _ = self.encoder(input.unsqueeze(0), input_length)
            output = self.decoder.greedy_decoding(encoder_outputs[0])
            return output
        else:
            if gpu:
                input = input.cuda()
            encoder_outputs, _  = self.encoder(input, input_length)
            output = self.decoder.greedy_batch_decode(encoder_outputs)
            return output   
    
    @classmethod
    def load_model(cls, model_states):
        encoder_config = AttrDict(model_states['encoder'])
        encoder = build_encoder(encoder_config)
        decoder_config = AttrDict(model_states['decoder'])
        decoder = build_decoder(decoder_config)
        encoder.flatten_parameters()
        model = cls(encoder, decoder)
        model.load_state_dict(model_states['model_state_dict'])
        return model 

    @staticmethod
    def get_model_state(model, epoch):
        model_states = {
            'encoder':{
                'input_size': model.encoder.input_size,
                'hidden_size': model.encoder.hidden_size,
                'num_layers': model.encoder.num_layers,
                'dropout': model.encoder.dropout},
            'decoder':{
                'vocab_size': model.decoder.vocab_size,
                'embedding_dim': model.decoder.embedding_dim,
                'decoder_hidden_size': model.decoder.decoder_hidden_size,
                'encoder_hidden_size': int(model.decoder.encoder_hidden_size/2),
                'num_layers': model.decoder.num_layers,
                'sos_id': model.decoder.sos_id,
                'eos_id': model.decoder.eos_id},
            'encoder_state_dict': model.encoder.state_dict(),
            'decoder_state_dict': model.decoder.state_dict(),
            'model_state_dict': model.state_dict(),
            'epoch': epoch}
        return model_states
