import torch
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'blank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[blank]'] + dict_character  # dummy '[blank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]

        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        text = ''.join(text)
        text = [self.dict[char] for char in text]

        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        index = 0
        for l in length:
            t = text_index[index:index + l]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
            index += l
        return texts


class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts


class TransformerConverter:
    def __init__(
            self,
            character,
            mask_language_model=True,
            p_mask_token=0.05,
            max_seq_length=256,
    ):
        self.mask_language_model = mask_language_model
        self.p_mask_token = p_mask_token
        self.pad = '<PAD>'
        self.bos = '<BOS>'
        self.eos = '<EOS>'
        self.mask_token = '<MASK>'
        self.unk_token = '<UNK>'
        list_character = list(character)
        self.character = [self.pad, self.bos, self.eos, self.mask_token, self.unk_token] + list_character
        self.dict = {
            c: i
            for i, c in enumerate(self.character)
        }
        self.pad_idx = self.dict[self.pad]
        self.bos_idx = self.dict[self.bos]
        self.eos_idx = self.dict[self.eos]
        self.mask_token_idx = self.dict[self.mask_token]
        self.unk_idx = self.dict[self.unk_token]
        self.max_seq_length = 256

    @property
    def n_classes(self):
        return len(self.character)

    def encode(self, text, batch_max_length=None):
        length = [len(s) + 2 for s in text]
        max_length = max(length)
        batch_text = torch.zeros(len(text), max_length, dtype=torch.long) + self.pad_idx
        for i, s in enumerate(text):
            token_ids = [self.bos_idx, *[self.dict.get(c, self.unk_idx) for c in s], self.eos_idx]
            batch_text[i, :length[i]] = torch.LongTensor(token_ids)
        batch_text = batch_text.to(device)

        if self.mask_language_model:
            # mask = batch_text.new(*batch_text.size()).bernoulli_(1-self.p_mask_token).div_(1-self.p_mask_token)
            mask = torch.rand(batch_text.size(), device=batch_text.device) < self.p_mask_token
            mask = mask & (batch_text != self.pad_idx) & (batch_text != self.bos_idx) & (batch_text != self.eos_idx)
            batch_text[mask] = self.mask_token_idx

        tgt_input = batch_text[:, :-1]
        tgt_output = batch_text[:, 1:]
        tgt_padding_mask = tgt_input == self.pad_idx
        # return batch_text, tgt_mask, torch.LongTensor(length, device=device)
        return (
            {
                'tgt_input': tgt_input,
                'tgt_padding_mask': tgt_padding_mask,
                'tgt_output': tgt_output,
            },
            torch.LongTensor(length).to(device)
        )

    def _decode_one(self, token_ids,):
        first = 1 if self.bos_idx in token_ids else 0
        last = token_ids.index(self.eos_idx) if self.eos_idx in token_ids else None
        sent = ''.join([self.character[i] for i in token_ids[first:last]])
        return sent

    def decode(self, batch_ids):
        if torch.is_tensor(batch_ids):
            batch_ids = batch_ids.cpu().tolist()
        elif type(batch_ids) == np.ndarray:
            batch_ids = batch_ids.tolist()
        return [self._decode_one(token_ids) for token_ids in batch_ids]


class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res
