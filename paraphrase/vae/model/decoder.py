import torch as t
import torch.nn as nn
import torch.nn.functional as F

from utils.functional import parameters_allocation_check


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()

        self.params = params

        self.rnn = nn.LSTM(input_size=self.params.latent_variable_size + self.params.word_embed_size,
                           hidden_size=self.params.decoder_rnn_size,
                           num_layers=self.params.decoder_num_layers,
                           batch_first=True)

        self.fc = nn.Linear(self.params.decoder_rnn_size, self.params.word_vocab_size)

    def only_decoder_beam(self, decoder_input, z, drop_prob, initial_state=None):
        assert parameters_allocation_check(self)

        #         print decoder_input.size()

        [beam_batch_size, _, _] = decoder_input.size()

        decoder_input = F.dropout(decoder_input, drop_prob)

        z = z.unsqueeze(0)

        #         print z.size()

        z = t.cat([z] * beam_batch_size, 0)

        #         print z.size()
        #         z = z.contiguous().view(1, -1)

        #         z = z.view(beam_batch_size, self.params.latent_variable_size)

        #         print z.size()

        decoder_input = t.cat([decoder_input, z], 2)

        #         print "decoder_input:",decoder_input.size()

        rnn_out, final_state = self.rnn(decoder_input, initial_state)

        #         print "rnn_out:",rnn_out.size()
        #         print "final_state_1:",final_state[0].size()
        #         print "final_state_1:",final_state[1].size()

        return rnn_out, final_state

    def forward(self, decoder_input, z, drop_prob, initial_state=None):
        assert parameters_allocation_check(self)
        [batch_size, seq_len, _] = decoder_input.size()

        decoder_input = F.dropout(decoder_input, drop_prob)

        z = t.cat([z] * seq_len, 1).view(batch_size, seq_len, self.params.latent_variable_size)
        decoder_input = t.cat([decoder_input, z], 2)

        rnn_out, final_state = self.rnn(decoder_input, initial_state)
        rnn_out = rnn_out.contiguous().view(-1, self.params.decoder_rnn_size)

        result = self.fc(rnn_out)
        result = result.view(batch_size, seq_len, self.params.word_vocab_size)

        return result, final_state
