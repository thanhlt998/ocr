import math
import torch
from torch import nn


class LanguageTransformer(nn.Module):
    def __init__(self, vocab_size,
                 input_size,
                 d_model, nhead,
                 num_encoder_layers, num_decoder_layers,
                 dim_feedforward, max_seq_length,
                 pos_dropout, trans_dropout):
        super().__init__()

        self.d_model = d_model
        self.embed_tgt = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, pos_dropout, max_seq_length)
        #        self.learned_pos_enc = LearnedPositionalEncoding(d_model, pos_dropout, max_seq_length)

        self.transformer = nn.Transformer(d_model, nhead,
                                          num_encoder_layers, num_decoder_layers,
                                          dim_feedforward, trans_dropout)

        self.fc = nn.Linear(d_model, vocab_size)
        self.i2d = nn.Linear(input_size, d_model)

        self._init_params()

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Shape:
            - src: (W, N, C)
            - tgt: (T, N)
            - src_key_padding_mask: (N, S)
            - tgt_key_padding_mask: (N, T)
            - memory_key_padding_mask: (N, S)
            - output: (N, T, E)

        """
        src = self.i2d(src)
        tgt_mask = self.gen_nopeek_mask(tgt.shape[0]).to(src.device)

        # src = self.pos_enc(src * math.sqrt(self.d_model))
        src = self.pos_enc(src)

        tgt = self.pos_enc(self.embed_tgt(tgt) * math.sqrt(self.d_model))

        output = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask)
        #        output = rearrange(output, 't n e -> n t e')
        output = output.transpose(0, 1)
        return self.fc(output)

    def gen_nopeek_mask(self, length):
        mask = (torch.triu(torch.ones(length, length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        return mask

    def forward_encoder(self, src, mask=None):
        src = self.i2d(src)
        # src = self.pos_enc(src * math.sqrt(self.d_model))
        src = self.pos_enc(src)
        memory = self.transformer.encoder(src, mask=mask,)
        return memory

    def forward_decoder(self, tgt, memory):
        tgt_mask = self.gen_nopeek_mask(tgt.shape[0]).to(tgt.device)
        tgt = self.pos_enc(self.embed_tgt(tgt) * math.sqrt(self.d_model))

        output = self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask)
        #        output = rearrange(output, 't n e -> n t e')
        output = output.transpose(0, 1)

        return self.fc(output), memory

    def expand_memory(self, memory, beam_size):
        memory = memory.repeat(1, beam_size, 1)
        return memory

    def get_memory(self, memory, i):
        memory = memory[:, [i], :]
        return memory

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]

        return self.dropout(x)

