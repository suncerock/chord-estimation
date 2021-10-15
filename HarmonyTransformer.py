import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_positional_encoding(batch_size, time_steps, n_units, zero_pad=False, scale=False):
    N, T = batch_size, time_steps

    position_enc = np.array([[pos / np.power(1000, 2.*i/n_units) for i in range(n_units)]
                             for pos in range(T)], dtype=np.float32)

    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])

    lookup_table = torch.from_numpy(position_enc)
    outputs = lookup_table[torch.arange(T)].reshape(1, T, n_units).repeat(N, 1, 1)

    if scale:
        outputs = outputs * n_units ** 0.5
    return outputs


def get_relative_position(n_steps, n_units, max_dist=2):
    n_vectors = 2 * max_dist + 1
    center = n_vectors // 2
    pos_enc = nn.Parameter(torch.randn(n_vectors, n_units))

    n_left = [min(max_dist, i) for i in range(n_steps)]
    n_right = n_left[::-1]
    pos_enc_pad = []
    self = torch.unsqueeze(pos_enc[center], 0)
    for i, n_l, n_r in zip(range(n_steps), n_left, n_right):
        left = pos_enc[(center - n_l): center]
        right = pos_enc[(center + 1): (center + 1 + n_r)]
        temp = torch.cat([left, self, right], dim=0)

        n_left_pad = i - n_l
        n_right_pad = n_steps - i - n_r - 1
        if n_left_pad > 0:
            temp = torch.cat([temp[0].unsqueeze(dim=0).repeat(n_left_pad, 1), temp], dim=0)
        if n_right_pad > 0:
            temp = torch.cat([temp, temp[-1].unsqueeze(dim=0).repeat(n_right_pad, 1)], dim=0)

        pos_enc_pad.append(temp)

    return torch.stack(pos_enc_pad)


def chord_block_compression(hidden_states, chord_changes):
    N, L, d = hidden_states.shape
    res = []
    for i in range(N):
        smoothed_state = []
        p = 0
        for q in range(1, L):
            if chord_changes[i, q]:
                smoothed_state.append(hidden_states[i, p:q].mean(dim=0, keepdims=True).repeat(q - p, 1))
                p = q
        smoothed_state.append(hidden_states[i, p:L].mean(dim=0, keepdims=True).repeat(L - p, 1))
        res.append(torch.cat(smoothed_state, dim=0))

    return torch.stack(res)


class MultiHeadAttention(nn.Module):
    """
    Input:
        - q: torch.Tensor, (batch_size, seq_len_q, input_dim)
        - k: torch.Tensor, (batch_size, seq_len_k, input_dim)
        - v: torch.Tensor, (batch_size, seq_len_k, input_dim)
    """
    def __init__(self, input_dim, n_units=None, n_heads=8, max_dist=16,
                 relative_position=False, causal=False, self_mask=False, drop=0.):
        super(MultiHeadAttention, self).__init__()
        if n_units is None:
            n_units = input_dim
        self.n_units = n_units
        self.n_heads = n_heads
        self.head_dim = n_units // self.n_heads
        self.relative_position = relative_position
        self.max_dist = max_dist
        self.scale = self.head_dim ** 0.5
        self.causal = causal
        self.self_mask = self_mask

        self.wq = nn.Linear(input_dim, n_units)
        self.wk = nn.Linear(input_dim, n_units)
        self.wv = nn.Linear(input_dim, n_units)

        self.dropout = nn.Dropout(drop)

        self.proj = nn.Linear(n_units, n_units)
        self.ln = nn.LayerNorm(n_units)

    def forward(self, q, k, v=None):
        res = q
        N = q.size(0)
        if v is None:
            v = k

        q = self.wq(q).reshape(N, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(k).reshape(N, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(v).reshape(N, -1, self.n_heads, self.head_dim).transpose(1, 2)

        outputs = q @ k.transpose(-2, -1)
        if self.relative_position:
            pass
            # TODO: fix relative position
            # relative_position_enc_k = get_relative_position(n_steps=k.size(2), n_units=self.head_dim,
            #                                                 max_dist=self.max_dist)
            # relative_position_enc_k =
        outputs = outputs / self.scale

        if self.causal:
            diag_vals = torch.ones_like(outputs[0, 0])
            tril_mask = torch.tril(diag_vals)
            outputs = outputs * tril_mask.unsqueeze(dim=0).unsqueeze(dim=0)

        if self.self_mask:
            # TODO: Check whether self_mask is implemented correctly
            diag_vals = torch.ones_like(outputs[0, 0])
            diag_mask = torch.tril(torch.triu(diag_vals))
            outputs = outputs * diag_mask.unsqueeze(dim=0).unsqueeze(dim=0)

        outputs = torch.softmax(outputs, dim=-1)
        outputs = self.dropout(outputs)
        if self.relative_position:
            outputs = (outputs @ v).transpose(1, 2).reshape(N, -1, self.n_units)
        else:
            outputs = (outputs @ v).transpose(1, 2).reshape(N, -1, self.n_units)
        outputs = self.proj(outputs) + res
        return self.ln(outputs)


class FFN(nn.Module):
    """
    Input:
        - torch.Tensor, (batch_size, seq_len, input_dim)
    """
    def __init__(self, input_dim, hidden_dim, drop=0.):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)

        self.ln = nn.LayerNorm(input_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        h = F.relu(self.linear1(x))
        h = self.drop(self.linear2(h))
        x = self.ln(x + h)
        return x


class EncodeSegmentTime(nn.Module):
    """
    Input:
        - torch.Tensor, (batch_size, seq_len, input_dim)
    """
    def __init__(self, frequency_size=24, segment_width=21, drop=0.):
        super(EncodeSegmentTime, self).__init__()
        self.frequency_size = frequency_size
        self.segment_width = segment_width
        self.embed_dim = frequency_size * segment_width

        self.mhsa = MultiHeadAttention(input_dim=frequency_size, n_units=frequency_size, n_heads=2,
                                       max_dist=4, drop=drop, relative_position=True)
        self.ffn = FFN(frequency_size, 4 * frequency_size, drop=drop)

        self.drop = nn.Dropout(drop)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.ln = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        N = x.shape[0]
        x = x.reshape(-1, self.frequency_size, self.segment_width)
        x = x.transpose(1, 2)

        x += get_positional_encoding(x.size(0), self.segment_width, n_units=self.frequency_size) * 0.01 + 0.01
        x = self.mhsa(q=x, k=x)
        x = self.ffn(x)

        x = self.drop(x.reshape(N, -1, self.embed_dim))
        x = F.relu(self.proj(x), inplace=True)
        x = self.ln(x)

        return x


class EncodeSegmentFrequency(nn.Module):
    """
    Input:
        - torch.Tensor, (batch_size, seq_len, input_dim)
    """
    def __init__(self, frequency_size=24, segment_width=21, drop=0.):
        super(EncodeSegmentFrequency, self).__init__()
        self.frequency_size = frequency_size
        self.segment_width = segment_width
        self.embed_dim = frequency_size * segment_width

        self.mhsa = MultiHeadAttention(input_dim=segment_width, n_units=segment_width, n_heads=1,
                                       max_dist=4, drop=drop)
        self.ffn = FFN(segment_width, 4 * segment_width, drop=drop)

        self.drop = nn.Dropout(drop)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.ln = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        N = x.shape[0]
        x = x.reshape(-1, self.frequency_size, self.segment_width)

        x += get_positional_encoding(x.size(0), self.frequency_size, n_units=self.segment_width) * 0.01 + 0.01
        x = self.mhsa(q=x, k=x)
        x = self.ffn(x)

        x = self.drop(x.reshape(N, -1, self.embed_dim))
        x = F.relu(self.proj(x), inplace=True)
        x = self.ln(x)

        return x


class Encoder(nn.Module):
    """
    Input:
        - torch.Tensor, (batch_size, seq_len, input_dim)
    """
    def __init__(self, n_steps=100, frequency_size=24, segment_width=21, drop=0.):
        super(Encoder, self).__init__()
        self.n_steps = n_steps
        self.encoder_input_embedding_size = frequency_size * segment_width

        self.segment_encodings_enc = EncodeSegmentTime(frequency_size=frequency_size,
                                                       segment_width=segment_width,
                                                       drop=drop)
        self.embed_dropout = nn.Dropout(drop)

        self.layer_weight = nn.Parameter(torch.ones(2))
        self.mha1 = MultiHeadAttention(
            input_dim=self.encoder_input_embedding_size,
            n_heads=8,
            relative_position=True,
            max_dist=16,
            drop=drop
        )

        self.ffn1 = FFN(self.encoder_input_embedding_size, self.encoder_input_embedding_size * 4)

        self.mha2 = MultiHeadAttention(
            input_dim=self.encoder_input_embedding_size,
            n_heads=8,
            relative_position=True,
            max_dist=16,
            drop=drop
        )

        self.ffn2 = FFN(self.encoder_input_embedding_size, self.encoder_input_embedding_size * 4)

        self.chord_change = nn.Linear(self.encoder_input_embedding_size, 1)

    def forward(self, x):
        x = self.segment_encodings_enc(x)
        x += get_positional_encoding(x.size(0), time_steps=self.n_steps, n_units=self.encoder_input_embedding_size)
        x = self.embed_dropout(x)

        x1 = self.ffn1(self.mha1(x, x))
        x2 = self.ffn2(self.mha2(x1, x1))
        weight = torch.softmax(self.layer_weight, dim=0)
        x = weight[0] * x1 + weight[1] * x2

        chord_change_prob = torch.sigmoid(self.chord_change(x)).squeeze(dim=-1)
        return x, chord_change_prob


class Decoder(nn.Module):
    def __init__(self, n_classes=26, n_steps=100, frequency_size=24, segment_width=21, drop=0.,
                 encoder_input_embedding_size=None):
        super(Decoder, self).__init__()
        self.n_classes = n_classes
        self.n_steps = n_steps
        self.decoder_input_embedding_size = frequency_size * segment_width
        self.encoder_input_embedding_size = encoder_input_embedding_size or self.decoder_input_embedding_size
        self.segment_encodings_dec = EncodeSegmentFrequency(frequency_size=frequency_size,
                                                            segment_width=segment_width,
                                                            drop=drop)

        self.embed_dropout = nn.Dropout(drop)

        self.layer_weight = nn.Parameter(torch.ones(2))
        self.mhsa1 = MultiHeadAttention(
            input_dim=self.decoder_input_embedding_size,
            n_units=self.decoder_input_embedding_size,
            n_heads=8,
            relative_position=True,
            max_dist=16,
            self_mask=False,
            drop=drop
        )

        self.mha1 = MultiHeadAttention(
            input_dim=self.encoder_input_embedding_size,
            n_units=self.decoder_input_embedding_size,
            n_heads=8,
            relative_position=False,
            max_dist=16,
            drop=drop
        )

        self.ffn1 = FFN(self.decoder_input_embedding_size, self.decoder_input_embedding_size * 4)

        self.mhsa2 = MultiHeadAttention(
            input_dim=self.decoder_input_embedding_size,
            n_units=self.decoder_input_embedding_size,
            n_heads=8,
            relative_position=True,
            max_dist=16,
            self_mask=False,
            drop=drop
        )

        self.mha2 = MultiHeadAttention(
            input_dim=self.encoder_input_embedding_size,
            n_units=self.decoder_input_embedding_size,
            n_heads=8,
            relative_position=False,
            max_dist=16,
            drop=drop
        )

        self.ffn2 = FFN(self.decoder_input_embedding_size, self.decoder_input_embedding_size * 4)

        self.proj = nn.Linear(self.decoder_input_embedding_size, self.n_classes)

    def forward(self, x, encoder_inputs_embedded, chord_change_prob):
        x = self.segment_encodings_dec(x)
        segment_encodings_dec_blocked = chord_block_compression(x, chord_change_prob > 0.5)
        x = x + segment_encodings_dec_blocked + encoder_inputs_embedded
        x += get_positional_encoding(x.size(0), time_steps=self.n_steps, n_units=self.decoder_input_embedding_size)
        x = self.embed_dropout(x)

        x1 = self.mhsa1(x, x)
        x1 = self.mha1(x1, encoder_inputs_embedded)
        x1 = self.ffn1(x1)

        x2 = self.mhsa2(x1, x1)
        x2 = self.mha2(x2, encoder_inputs_embedded)
        x2 = self.ffn2(x2)

        weight = torch.softmax(self.layer_weight, dim=0)
        x = weight[0] * x1 + weight[1] * x2

        return self.proj(x)


class HarmonyTransformer(nn.Module):
    def __init__(self,
                 n_steps=100,
                 frequency_size=24,
                 segment_width=21,
                 drop=0.,
                 n_classes=26
                 ):
        super(HarmonyTransformer, self).__init__()
        self.encoder = Encoder(n_steps, frequency_size, segment_width, drop)
        self.decoder = Decoder(n_classes, n_steps, frequency_size, segment_width, drop)

    def forward(self, x):
        encoder_embed, chord_change_prob = self.encoder(x)
        output = self.decoder(x, encoder_embed, chord_change_prob)
        return chord_change_prob, output


if __name__ == '__main__':
    net = HarmonyTransformer()
    x = torch.randn(4, 100, 504)
    y_cc_pred, y_pred = net(x)
    print(y_cc_pred.shape, y_pred.shape)

