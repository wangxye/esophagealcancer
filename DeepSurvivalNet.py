import torch
import torch.nn as nn
from pycox.evaluation import EvalSurv
from pycox.models.loss import NLLLogistiHazardLoss
import torchtuples as tt
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def init_embedding(emb):
    """Weight initialization of embeddings (in place).
    Best practise from fastai

    Arguments:
        emb {torch.nn.Embedding} -- Embedding
    """
    w = emb.weight.data
    sc = 2 / (w.shape[1] + 1)
    w.uniform_(-sc, sc)


# Linear
class TimeDistributed(nn.Module):
    def __init__(self, input_size, output_size, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = nn.Linear(input_size, output_size, bias=True)
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # reshape input data --> (samples * timesteps, input_size)
        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


class MLP(nn.Module):
    """ Multilayer Perception """

    def __init__(self, hidden_size, output_size, dropout=0.1):
        super().__init__()
        self.dense = TimeDistributed(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output = TimeDistributed(hidden_size, output_size)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.output(x)
        return x


# Entity embedding
class EntityEmbeddings(nn.Module):
    def __init__(self, num_embeddings, embedding_dims, dropout=0.):
        super().__init__()
        if not hasattr(num_embeddings, '__iter__'):
            num_embeddings = [num_embeddings]

        if not hasattr(embedding_dims, '__iter__'):
            embedding_dims = [embedding_dims]

        if len(num_embeddings) != len(embedding_dims):
            raise ValueError("Need 'num_embeddings' and 'embedding_dims' to have the same length")

        self.embeddings = nn.ModuleList()
        for n_emb, emb_dim in zip(num_embeddings, embedding_dims):
            emb = nn.Embedding(n_emb, emb_dim)
            init_embedding(emb)
            self.embeddings.append(emb)

        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, input):
        if input.shape[1] != len(self.embeddings):
            raise RuntimeError(f"Got input of shape '{input.shape}', but need dim 1 to be {len(self.embeddings)}.")
        input = [emb(input[:, i].cuda(device=DEVICE)) for i, emb in enumerate(self.embeddings.cuda(device=DEVICE))]
        input = torch.cat(input, 1).cuda(device=DEVICE)
        if self.dropout:
            input = self.dropout(input)
        return input.cuda(device=DEVICE)


class ConvLayer(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(ConvLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_channels = hidden_size // 3
        self.hidden_size = hidden_size

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.num_channels, kernel_size=(1, hidden_size),
                               stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=self.num_channels, kernel_size=(5, hidden_size),
                               stride=1, padding=(2, 0), dilation=1, groups=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=self.num_channels + 1, kernel_size=(9, hidden_size),
                               stride=1, padding=(4, 0), dilation=1, groups=1, bias=True)

    def convlution(self, x, conv):
        # print(x.shape)
        # print(self.conv1)
        out = conv(x)  # [batch_size, hidden_dim // 3, sequence_len]
        # print(out.shape)
        activation = F.relu(out.squeeze(3))
        out = activation
        return out

    def forward(self, x):
        pooled_output = x.unsqueeze(1)
        h1 = self.convlution(pooled_output, self.conv1)
        h2 = self.convlution(pooled_output, self.conv2)
        h3 = self.convlution(pooled_output, self.conv3)

        pooled_output = torch.cat([h1, h2, h3], 1)
        pooled_output = self.dropout(pooled_output)
        pooled_output = pooled_output.permute(0, 2, 1)  # [batch_size, sequence_len, hidden_dim]
        return pooled_output


# DEVICE = torch.device("cpu")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTM(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(LSTM, self).__init__()
        self.embed_dim = hidden_size
        self.hidden_size = hidden_size // 2
        self.layer_size = 1
        self.bidirectional = True
        self.lstm = nn.LSTM(self.embed_dim,
                            self.hidden_size,
                            self.layer_size,
                            dropout=dropout,
                            bidirectional=self.bidirectional)

        if self.bidirectional:
            self.layer_size = self.layer_size * 2
        else:
            self.layer_size = self.layer_size

    def forward(self, x):
        # print(x.shape)
        x = x.permute(1, 0, 2).cuda(device=DEVICE)
        # print(x.shape)
        h_0 = Variable(torch.zeros(self.layer_size, x.size(1), self.hidden_size)).cuda(device=DEVICE)
        c_0 = Variable(torch.zeros(self.layer_size, x.size(1), self.hidden_size)).cuda(device=DEVICE)
        lstm_output, (_, _) = self.lstm(x, (h_0, c_0))
        return lstm_output


# 神经网络（生存概率预测网络
class NetAESurv(nn.Module):
    def __init__(self, in_features, encoded_features, out_features, num_embeddings, embedding_dims, dropout,
                 hidden_dim):
        super().__init__()
        # 实体嵌入: 用于将离散值映射到多维空间中，使其中具有相似函数输出的值彼此靠得更近; 用于解决独热编码变量导致的向量稀疏问题
        self.embeddings = EntityEmbeddings(num_embeddings, embedding_dims, dropout=dropout)
        input_features = in_features + sum(embedding_dims)
        #         self.attention = SelfAttentionNetwork(encoded_features, encoded_features)
        self.num_heads = encoded_features // 1
        # self.num_heads = input_features // 1

        # 多头注意力: 指导模型关注更重要的维度特征
        # self.attention = nn.MultiheadAttention(encoded_features, self.num_heads)
        # Layer Normalization: 对单个样本的所有维度特征进行归一化
        # self.layer_norm = nn.LayerNorm(encoded_features)
        # self.sigmoid = nn.Sigmoid()

        # self.attention = nn.MultiheadAttention(input_features, self.num_heads)

        self.encoder = nn.Sequential(
            # nn.Dropout(dropout),
            TimeDistributed(input_features, hidden_dim[0]),
            # nn.MultiheadAttention(encoded_features, self.num_heads),
            nn.ReLU(),
            # nn.Dropout(dropout),
            TimeDistributed(hidden_dim[0], hidden_dim[1]),  # 32 16
            nn.ReLU(),
            # nn.Dropout(dropout),
            TimeDistributed(hidden_dim[1], encoded_features),  # 8
        )  # 编码层

        # self.encoder = Encoder(src_vocab_size, src_max_len, num_layers, model_dim,
        #                        num_heads, ffn_dim, dropout)
        # self.decoder = Decoder(tgt_vocab_size, tgt_max_len, num_layers, model_dim,
        #                        num_heads, ffn_dim, dropout)

        # self.attention = nn.MultiheadAttention(encoded_features, self.num_heads)

        self.decoder = nn.Sequential(
            # nn.Dropout(dropout),
            TimeDistributed(encoded_features, hidden_dim[1]),
            nn.ReLU(),
            # nn.Dropout(dropout),
            TimeDistributed(hidden_dim[1], hidden_dim[0]),  # 16 32
            nn.ReLU(),
            # nn.Dropout(dropout),
            TimeDistributed(hidden_dim[0], input_features),
        )  # 解码层：仅用于训练阶段计算编码器MSELoss

        # self.attention = nn.MultiheadAttention(input_features, self.num_heads)

        # self.conv = ConvLayer(encoded_features)
        self.lstm = LSTM(encoded_features)

        self.surv_net = nn.Sequential(
            # nn.Dropout(dropout),
            TimeDistributed(encoded_features, hidden_dim[2]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim[2]),
            nn.Dropout(0.1),
            TimeDistributed(hidden_dim[2], hidden_dim[3]),  # 16 16
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim[3]),
            nn.Dropout(0.1),
            TimeDistributed(hidden_dim[3], out_features),  # 8
        )  # 生存预测层 MLP
        # self.mlp = MLP(out_features, out_features)

    def forward(self, input_numeric, input_categoric):
        input = torch.cat([input_numeric, self.embeddings(input_categoric)], 1)

        encoded = self.encoder(input)
        decoded = self.decoder(encoded)

        # encoded = self.encoder(input)
        # attention, _ = self.attention(encoded.unsqueeze(1), encoded.unsqueeze(1), encoded.unsqueeze(1))
        # decoded = self.decoder(attention.squeeze(1))

        # attention, _ = self.attention(input.unsqueeze(1), input.unsqueeze(1), input.unsqueeze(1))
        # encoded = self.encoder(attention.squeeze(1))
        # decoded = self.decoder(encoded)

        # encoded = self.encoder(input)
        # decoded = self.decoder(encoded)
        # attention, _ = self.attention(decoded.unsqueeze(1), decoded.unsqueeze(1), decoded.unsqueeze(1))
        # decoded = attention.squeeze(1)

        x = encoded
        # x = self.conv(x.unsqueeze(1)).squeeze(1)
        x = self.lstm(x.unsqueeze(1)).squeeze(0)
        # print(x.shape)
        phi = self.surv_net(x)
        return phi, decoded
        # return phi

    def predict(self, input_numeric, input_categoric):
        input = torch.cat([input_numeric, self.embeddings(input_categoric)], 1)
        encoded = self.encoder(input)
        x = encoded
        # x = self.conv(x.unsqueeze(1)).squeeze(1)
        x = self.lstm(x.unsqueeze(1)).squeeze(0)
        return self.surv_net(x)


# 损失函数：由LogisticHazardLoss与自动编码器MSELoss组成
class LossAELogHaz(nn.Module):
    def __init__(self, alpha, num_embeddings, embedding_dims, dropout=0.2) -> object:
        super().__init__()
        assert (alpha >= 0) and (alpha <= 1), 'Need `alpha` in [0, 1].'
        self.alpha = alpha
        self.embeddings = EntityEmbeddings(num_embeddings, embedding_dims, dropout=dropout)
        self.loss_surv = NLLLogistiHazardLoss()
        self.loss_ae = nn.MSELoss()

    # def forward(self, phi, decoded, target_loghaz, target_ae):
    #     idx_durations, events = target_loghaz
    #     loss_surv = self.loss_surv(phi, idx_durations, events)  # 生存Loss
    #     loss_ae = self.loss_ae(decoded, target_ae)  # 编码器MSELoss
    #     return self.alpha * loss_surv + (1 - self.alpha) * loss_ae
    def forward(self, phi, decoded, target_loghaz, target_ae):
        target_ae_numeric, target_ae_categoric = target_ae
        target_ae = torch.cat([target_ae_numeric, self.embeddings(target_ae_categoric)], 1)
        idx_durations, events = target_loghaz
        loss_surv = self.loss_surv(phi, idx_durations, events)  # 生存Loss
        loss_ae = self.loss_ae(decoded, target_ae)  # 编码器MSELoss
        return self.alpha * loss_surv + (1 - self.alpha) * loss_ae


class Concordance(tt.cb.MonitorMetrics):
    def __init__(self, x, durations, events, per_epoch=1, verbose=True):
        super().__init__(per_epoch)
        self.x = x
        self.durations = durations
        self.events = events
        self.verbose = verbose

    def on_epoch_end(self):
        super().on_epoch_end()
        if self.epoch % self.per_epoch == 0:
            surv = self.model.interpolate(10).predict_surv_df(self.x)
            ev = EvalSurv(surv, self.durations, self.events, censor_surv='km')
            concordance = ev.concordance_td()
            self.append_score('c-index', concordance)
            if self.verbose:
                print('c-index:', concordance)
