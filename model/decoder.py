import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import IGNORE_ID, pad_list

from model.attention import DotProductAttention

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_mask(lens):
    #lens = torch.tensor(lens).to(DEVICE)
    lens = lens.to(DEVICE)
    max_len = int(torch.max(lens))
    mask = (torch.arange(0, max_len).repeat(lens.size(0), 1).to(DEVICE) < \
                lens.unsqueeze(1).expand(lens.size(0), max_len)).int()
    return mask


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size, 
        embedding_dim,
        decoder_hidden_size,
        encoder_hidden_size,
        num_layers,
        sos_id,
        eos_id):

        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.decoder_hidden_size = decoder_hidden_size
        self.encoder_hidden_size = encoder_hidden_size * 2 #Since it's bidirectional
        self.num_layers = num_layers
        
        self.sos_id = sos_id
        self.eos_id = eos_id

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        self.rnn = nn.ModuleList()
        self.rnn += [nn.LSTMCell(self.embedding_dim 
                                 + self.encoder_hidden_size,
                                self.decoder_hidden_size)]

        for _ in range(1, self.num_layers):
            self.rnn += [nn.LSTMCell(self.decoder_hidden_size,
                                     self.decoder_hidden_size)]
        
        self.attention = DotProductAttention()
        
        self.linear = nn.Sequential(
            nn.Linear(self.encoder_hidden_size + self.decoder_hidden_size,
                      self.decoder_hidden_size),
            nn.Tanh(),
            nn.Linear(self.decoder_hidden_size, self.vocab_size))
        
        #self.crit = nn.CrossEntropyLoss(reduction='none')
        self.crit = nn.CrossEntropyLoss(ignore_index = -1)

    def zero_state(self, encoder_outputs, hidden_size=None):
        batch_size = encoder_outputs.size(0)
        hidden_size = self.decoder_hidden_size if hidden_size == None else hidden_size
        return encoder_outputs.new_zeros(batch_size, hidden_size)
    
    def initalize_decoder(self, encoder_outputs):
        previous_hidden = [self.zero_state(encoder_outputs)]
        previous_context = [self.zero_state(encoder_outputs)]
        for _ in range(1, self.num_layers):
            previous_hidden.append(self.zero_state(encoder_outputs))
            previous_context.append(self.zero_state(encoder_outputs))
        initial_context = self.zero_state(
            encoder_outputs,
            hidden_size = encoder_outputs.size(2))
        return previous_hidden, previous_context, initial_context


    def forward(self, inputs, encoder_outputs, target_lengths):
        # Going to use Teacher Forcing all the time
        #print(inputs.size())
        targets = [t[t != IGNORE_ID] for t in inputs]
        eos = targets[0].new([self.eos_id])
        sos = targets[0].new([self.sos_id])
        decoder_inputs = [torch.cat([sos, target], dim=0) for target in targets]
        decoder_outputs = [torch.cat([target, eos], dim=0) for target in targets]

        #decoder_inputs = pad_list(decoder_inputs, self.eos_id)
        #decoder_inputs = pad_list(decoder_inputs, self.eos_id)
        #decoder_outputs = pad_list(decoder_outputs, 0)


        decoder_inputs = nn.utils.rnn.pad_sequence(decoder_inputs, batch_first=True, padding_value=self.eos_id)
        decoder_outputs = nn.utils.rnn.pad_sequence(decoder_outputs, batch_first=True, padding_value=IGNORE_ID)
        #print(decoder_outputs.size())
        assert decoder_inputs.size() == decoder_outputs.size()
        batch_size, output_length = decoder_inputs.size(0), decoder_inputs.size(1)

        previous_hidden, previous_context, attention_context = self.initalize_decoder(encoder_outputs)
        output_distributions = list()

        embedded_output = self.embedding(decoder_inputs)
        for t in range(output_length):
            rnn_input = torch.cat((embedded_output[:, t, :], attention_context), dim=1)
            previous_hidden[0], previous_context[0] = self.rnn[0](
                rnn_input, (previous_hidden[0], previous_context[0]))
            for l in range(1, self.num_layers):
                previous_hidden[l], previous_context[l] = self.rnn[l](
                    previous_hidden[l-1], (previous_hidden[l], previous_context[l]))
            
            rnn_output = previous_hidden[-1]
            attention_context, _ = self.attention(rnn_output.unsqueeze(dim=1), 
                                                  encoder_outputs)
            attention_context = attention_context.squeeze(dim=1)
            linear_input = torch.cat((rnn_output, attention_context), dim=1)
            output_distribution = self.linear(linear_input)
            output_distributions.append(output_distribution)
        
        output_distributions = torch.stack(output_distributions, dim=1)
        #output_distributions  = output_distributions.permute(0,2,1)
        output_distributions = output_distributions.view(batch_size*output_length, self.vocab_size)
        loss = self.crit(output_distributions, decoder_outputs.view(-1))
        #loss = self.crit(output_distributions, decoder_outputs)

        #mask = generate_mask(target_lengths).to(DEVICE)

        #masked_loss = torch.sum(loss * mask.view(-1)) / torch.sum(mask)
        #return masked_loss
        return loss
    @torch.no_grad()
    def greedy_decoding(self, encoder_outputs):
        maxlen = 300
        h_list = [self.zero_state(encoder_outputs.unsqueeze(0))]
        c_list = [self.zero_state(encoder_outputs.unsqueeze(0))]
        for l in range(1, self.num_layers):
            h_list.append(self.zero_state(encoder_outputs.unsqueeze(0)))
            c_list.append(self.zero_state(encoder_outputs.unsqueeze(0)))
        att_c = self.zero_state(encoder_outputs.unsqueeze(0),
                                hidden_size=encoder_outputs.unsqueeze(0).size(2))

        y = []
        predictions = []
        y.append(self.sos_id)
        vy = encoder_outputs.new_zeros(1).long()
        for i in range(maxlen):
            vy[0] = y[0]
            embedded = self.embedding(vy)
            rnn_input = torch.cat((embedded, att_c), dim=1)
            h_list[0], c_list[0] = self.rnn[0](
                        rnn_input, (h_list[0], c_list[0]))
            for l in range(1, self.num_layers):
                h_list[l], c_list[l] = self.rnn[l](
                    h_list[l-1], (h_list[l], c_list[l]))
            rnn_output = h_list[-1]
            att_c, att_w = self.attention(rnn_output.unsqueeze(dim=1),
                                        encoder_outputs.unsqueeze(0))
            att_c = att_c.squeeze(dim=1)   

            mlp_input = torch.cat((rnn_output, att_c), dim=1)
            predicted_y_t = self.linear(mlp_input)
            local_scores = F.log_softmax(predicted_y_t, dim=1)
            prediction = torch.argmax(local_scores).item()
            if prediction == self.eos_id:
                return predictions
                break
            y[0] = prediction
            predictions.append(prediction)
        return predictions
    
    @torch.no_grad()
    def greedy_batch_decode(self, encoder_padded_outputs):
        max_len = 500
        h_list = [self.zero_state(encoder_padded_outputs)]
        c_list = [self.zero_state(encoder_padded_outputs)]
        for l in range(1, self.num_layers):
            h_list.append(self.zero_state(encoder_padded_outputs))
            c_list.append(self.zero_state(encoder_padded_outputs))
        att_c = self.zero_state(encoder_padded_outputs,
                                hidden_size=encoder_padded_outputs.size(2))
        
        batch_size = encoder_padded_outputs.size(0)
        y_all = torch.LongTensor([[self.sos_id]for _ in range(batch_size)])
        if encoder_padded_outputs.is_cuda:
            y_all = y_all.cuda()
        for i in range(max_len):
            #if encoder_padded_outputs.is_cuda:
            input_y = y_all[:, i]
            embedded = self.embedding(input_y)
            rnn_input = torch.cat((embedded, att_c), dim=1)
            
            h_list[0], c_list[0] = self.rnn[0](
                rnn_input, (h_list[0], c_list[0]))
            
            for l in range(1, self.num_layers):
                h_list[l], c_list[l] = self.rnn[l](
                    h_list[l-1], (h_list[l], c_list[l]))
            rnn_output = h_list[-1]  # below unsqueeze: (N x H) -> (N x 1 x H)
            # step 2. attention: c_i = AttentionContext(s_i,h)
            att_c, att_w = self.attention(rnn_output.unsqueeze(dim=1),
                                            encoder_padded_outputs)
            att_c = att_c.squeeze(dim=1)
        
            mlp_input = torch.cat((rnn_output, att_c), dim=1)
            predicted_y_t = self.linear(mlp_input)
            local_scores = F.log_softmax(predicted_y_t, dim=1)
            prediction = torch.argmax(local_scores, dim=1).unsqueeze(1)
            y_all = torch.cat((y_all, prediction), dim=1)
        return y_all    

def build_decoder(config):
    decoder = Decoder(
        vocab_size = config.vocab_size, 
        embedding_dim = config.embedding_dim,
        decoder_hidden_size = config.decoder_hidden_size,
        encoder_hidden_size = config.encoder_hidden_size,
        num_layers = config.num_layers,
        sos_id = config.sos_id,
        eos_id = config.eos_id)
    return decoder