import torch
from tool import construct_dataset
from resnet18 import ResNet18
import time
from tqdm import tqdm

use_cuda = torch.cuda.is_available()
train_loader, valid_loader = construct_dataset(V=10, batch_size=64)
model = ResNet18().double()

if use_cuda:
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
criterion = torch.nn.MSELoss()

# ##############################################################################
# Training
# ##############################################################################

train_loss = []
valid_loss = []
accuracy = []

def evaluate():
    model.eval()
    err_e_R = err_y_c = err_v = err_charge = eval_loss = 0
    for n, (data, label) in enumerate(valid_loader):
        if use_cuda:
            data, label = data.cuda(), label.cuda()
        pred = model(data.transpose(1, 2))
        eval_loss += criterion(pred, label).item()

        err_e_R = (err_e_R * n + torch.abs(pred[:, 1] - label[:, 1]).mean().item()) / (n + 1)
        err_y_c = (err_y_c * n + torch.abs(pred[:, 0] - label[:, 0]).mean().item()) / (n + 1)
        err_v = (err_v * n + torch.abs(pred[:, 2] - label[:, 2]).mean().item()) / (n + 1)
        err_charge = (err_charge * n + torch.abs(pred[:, 3] - label[:, 3]).mean().item()) / (n + 1)


    return eval_loss, err_e_R, err_y_c, err_charge, err_v

def train():
    model.train()
    total_loss = 0
    for data, label in tqdm(train_loader, mininterval=1,
                desc='Train Processing', leave=False):
        optimizer.zero_grad()
        if use_cuda:
            data, label = data.cuda(), label.cuda()
        target = model(data.transpose(1, 2))
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



try:
    print('-' * 90)
    for epoch in range(1, 1000+1):
        epoch_start_time = time.time()
        loss = train()
        train_loss.append(loss*1000.)

        print('| start of epoch {:3d} | time: {:2.2f}s | loss {:5.6f}'.format(epoch, time.time() - epoch_start_time, loss))

        epoch_start_time = time.time()
        loss, avg_error_e_R, avg_error_y_c, err_charge, err_v = evaluate()
        valid_loss.append(loss*1000.)

        print('-' * 90)
        print('| end of epoch {:3d} | time: {:2.2f}s | loss {:.4f} | y_c_err: {:.4f}  | e_R_err {:.4f}'.format(
            epoch, time.time() - epoch_start_time, loss, avg_error_y_c, avg_error_e_R)
            + '  | err_charge: {:.4f}  |  err_v: {:.4f}'.format(err_charge, err_v))
        print('-' * 90)

except KeyboardInterrupt:
    print("-"*90)
    print("Exiting from training early | cost time: {:5.2f}min".format((time.time() - total_start_time)/60.0))
