import numpy as np
from tool import get_data, construct_dataset
import torch
import argparse
import time
import torch
from torch import nn
from torch.autograd import Variable
from matplotlib import pyplot as plt
import time
from tqdm import tqdm

parser = argparse.ArgumentParser(description='LSTM text classification')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate [default: 0.001]')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs for train')
parser.add_argument('--batch-size', type=int, default=32,
                    help='batch size for training [default: 16]')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda-able', action='store_true',
                    help='enables cuda')

parser.add_argument('--save', type=str, default='./LSTM_Text.pt',
                    help='path to save the final model')
parser.add_argument('--data', type=str, default='./data/corpus.pt',
                    help='location of the data corpus')

parser.add_argument('--dropout', type=float, default=0,
                    help='the probability for dropout (0 = no dropout) [default: 0.5]')
parser.add_argument('--embed-dim', type=int, default=10,
                    help='number of embedding dimension [default: 64]')
parser.add_argument('--hidden-size', type=int, default=20,
                    help='number of lstm hidden dimension [default: 128]')
parser.add_argument('--lstm-layers', type=int, default=2,
                    help='biLSTM layer numbers')
parser.add_argument('--bidirectional', action='store_true',
                    help='If True, becomes a bidirectional LSTM [default: False]')

args = parser.parse_args()
torch.manual_seed(args.seed)
use_cuda = torch.cuda.is_available() and args.cuda_able

# ##############################################################################
# Load data
###############################################################################
train_loader, valid_loader = construct_dataset(V=10, batch_size=args.batch_size)
args.n_capteur = 10
args.label_size = 2


# ##############################################################################
# Build model
# ##############################################################################
import lstm

# model = lstm.LSTM(args).double()
model = lstm.simple(48, 1).double()
if use_cuda:
    model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)
criterion = torch.nn.MSELoss()

# ##############################################################################
# Training
# ##############################################################################

train_loss = []
valid_loss = []
accuracy = []

print([1]+[1]*3)

def evaluate():
    model.eval()
    correct_e_R = correct_y_c = eval_loss = 0
    for data, label in tqdm(valid_loader, mininterval=0.2,
                desc='Evaluate Processing', leave=False):
        # hidden = model.init_hidden(data.shape[0])
        pred, hidden = model(data)

        loss = criterion(pred, label)
        eval_loss += loss.item()
        for i in range(data.shape[0]):
            if torch.abs(pred[i, 0] - label[i, 0]).item() <= 0.5/89:
                correct_y_c += 1
            if torch.abs(pred[i, 1] - label[i, 1]).item() <= 0.5/89:
                correct_e_R += 1
    print(pred)

    return eval_loss, correct_y_c/args.batch_size/len(valid_loader), correct_e_R/args.batch_size/len(valid_loader)

def train():
    model.train()
    total_loss = 0
    for data, label in tqdm(train_loader, mininterval=1,
                desc='Train Processing', leave=False):
        optimizer.zero_grad()
        # hidden = model.init_hidden(data.shape[0])
        target, hidden = model(data)
        loss = criterion(target, label)

        loss.backward()
        optimizer.step()

        total_loss += loss.data
    return total_loss

# ##############################################################################
# Save Model
# ##############################################################################
best_acc = None
total_start_time = time.time()

data, label = next(iter(train_loader))
# for i in range(data.shape[0]):
#     for j in range(data.shape[-1]):
#         plt.plot(data[i, :, j])
#     plt.show()
#     # hidden = model.init_hidden(1)
#     # pred, hidden = model(torch.randn(data.shape[1], data.shape[2]).unsqueeze(0).double(), hidden)
#     # # pred, hidden = model(data[i, :, :].unsqueeze(0), hidden)
#     # print(pred)

# projection = nn.Linear(args.n_capteur, args.embed_dim).double()
# rnn = nn.LSTM(args.embed_dim, args.hidden_size, args.lstm_layers).double()
# data = projection(data)
# print(data)
# pred, hidden = rnn(data.transpose(0, 1), hidden)
# print(pred)
#
#
# rnn = torch.nn.LSTM(10, 20, 2, batch_first=True).double()
# input = torch.randn(5, 3, 10).double()
# h0 = torch.randn(2, 3, 20).double()
# c0 = torch.randn(2, 3, 20).double()
# output, (hn, cn) = rnn(data, hidden)
# print(output)

try:
    print('-' * 90)
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        loss = train()
        train_loss.append(loss*1000.)

        print('| start of epoch {:3d} | time: {:2.2f}s | loss {:5.6f}'.format(epoch, time.time() - epoch_start_time, loss))

        epoch_start_time = time.time()
        loss, y_c_acc, e_R_acc = evaluate()
        valid_loss.append(loss*1000.)

        print('-' * 90)
        print('| end of epoch {:3d} | time: {:2.2f}s | loss {:.4f} | y_c_acc: {:.4f} | e_R_acc: {:.4f}'.format(
            epoch, time.time() - epoch_start_time, loss, y_c_acc, e_R_acc))
        print('-' * 90)

except KeyboardInterrupt:
    print("-"*90)
    print("Exiting from training early | cost time: {:5.2f}min".format((time.time() - total_start_time)/60.0))