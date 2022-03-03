import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init
import torch.nn.functional as F



class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, input):
        mu = torch.mean(input, dim=-1, keepdim=True)
        sigma = torch.std(input, dim=-1, keepdim=True).clamp(min=self.eps)
        output = (input - mu) / sigma
        return output * self.weight.expand_as(output) + self.bias.expand_as(output)


class LSTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        for k, v in args.__dict__.items():
            self.__setattr__(k, v)

        self.num_directions = 2 if self.bidirectional else 1

        self.projection = nn.Linear(self.n_capteur, self.embed_dim)
        self.lstm = nn.LSTM(self.embed_dim,
                            self.hidden_size,
                            self.lstm_layers,
                            batch_first=True,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional)
        self.ln = LayerNorm(self.hidden_size * self.num_directions)
        self.logistic = nn.Linear(self.hidden_size * self.num_directions,
                                  self.label_size)
        self.conv1 = nn.Conv1d(self.embed_dim, self.embed_dim * 2, kernel_size=5, stride=2, padding=0)
        self.conv2 = nn.Conv1d(self.embed_dim * 2, self.embed_dim, kernel_size=5, stride=2, padding=0)

        # self._init_weights()

    def _init_weights(self, scope=1.):
        self.projection.weight.data.uniform_(-scope, scope)
        self.logistic.weight.data.uniform_(-scope, scope)
        self.logistic.bias.data.fill_(0)

    def init_hidden(self, batch):
        num_layers = self.lstm_layers * self.num_directions
        return torch.zeros(num_layers, batch, self.hidden_size).double(), \
               torch.zeros(num_layers, batch, self.hidden_size).double()

    def forward(self, input, hidden):
        encode = self.projection(input)  # [B, L, D]
        conv1_out = self.conv1(encode.transpose(1, 2))
        conv2_out = self.conv2(conv1_out).transpose(1, 2)
        lstm_out, hidden = self.lstm(conv2_out, hidden)
        # output = self.ln(lstm_out)
        return torch.sigmoid(self.logistic(hidden[0][-1])), hidden

        # def forward(self, input, hidden):
    #     # encode = self.projection(input)  # [B, L, D]
    #     lstm_out, hidden = self.lstm(input, hidden)
    #     return lstm_out, hidden

class simple(nn.Module):
    def __init__(self, hidden_size, layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(10,
                            hidden_size,
                            layers,
                            batch_first=True)
        self.logistic = nn.Linear(self.hidden_size, 2)

    def forward(self, input):

        lstm_out, hidden = self.lstm(input)
        return torch.sigmoid(self.logistic(hidden[0][-1])), hidden

        # def forward(self, input, hidden):
    #     # encode = self.projection(input)  # [B, L, D]
    #     lstm_out, hidden = self.lstm(input, hidden)
    #     return lstm_out, hidden
