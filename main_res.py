import torch
from tool import construct_dataset
from resnet18 import ResNet18
import time
from tqdm import tqdm

use_cuda = torch.cuda.is_available()
train_loader, valid_loader = construct_dataset(V=10, batch_size=32)
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
    correct_e_R = correct_y_c = eval_loss = 0
    preds, labels = [], []
    for data, label in tqdm(valid_loader, mininterval=0.2,
                desc='Evaluate Processing', leave=False):
        if use_cuda:
            data, label = data.cuda(), label.cuda()
        pred = model(data.transpose(1, 2))
        preds.append(pred)
        labels.append(label)
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    eval_loss = criterion(preds, labels).item()
    for i in range(preds.shape[0]):
        if torch.abs(preds[i, 0] - labels[i, 0]).item() <= 1/89:
            correct_y_c += 1
        if torch.abs(preds[i, 1] - labels[i, 1]).item() <= 1/89:
            correct_e_R += 1
    err_e_R = torch.abs(preds[:, 1] - labels[:, 1]).mean().item()
    err_y_c = torch.abs(preds[:, 0] - labels[:, 0]).mean().item()
    err_charge = torch.abs(preds[:, 3] - labels[:, 3]).mean().item()
    err_v = torch.abs(preds[:, 2] - labels[:, 2]).mean().item()


    return eval_loss, correct_y_c/preds.shape[0], correct_e_R/preds.shape[0], err_e_R, err_y_c, err_charge, err_v

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
        loss, y_c_acc, e_R_acc, avg_error_e_R, avg_error_y_c, err_charge, err_v = evaluate()
        valid_loss.append(loss*1000.)

        print('-' * 90)
        print('| end of epoch {:3d} | time: {:2.2f}s | loss {:.4f} | y_c_acc: {:.4f} err: {:.4f}  | e_R_acc: {:.4f} err {:.4f}'.format(
            epoch, time.time() - epoch_start_time, loss, y_c_acc, avg_error_y_c, e_R_acc, avg_error_e_R)
            + '  | err_charge: {:.4f}  |  err_v: {:.4f}'.format(err_charge, err_v))
        print('-' * 90)

except KeyboardInterrupt:
    print("-"*90)
    print("Exiting from training early | cost time: {:5.2f}min".format((time.time() - total_start_time)/60.0))
