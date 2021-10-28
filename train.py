import torch
import torch.nn as nn

from dataloader import build_dataloader
from models.HarmonyTransformer import HarmonyTransformer


def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataloader, valid_dataloader = build_dataloader()
    net = HarmonyTransformer().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=2e-4)
    LossC = nn.CrossEntropyLoss()
    LossCt = nn.BCELoss()
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(10):
        for step, (x, y_cc, y, y_len) in enumerate(train_dataloader):
            net.train()
            optimizer.zero_grad()

            y_cc_pred, y_pred = net(x.to(device))
            loss_c = LossC(y_pred.transpose(1, 2), y.long())
            loss_ct = LossCt(y_cc_pred, y_cc.float())
            loss = 3 * loss_c + loss_ct

            loss.backward()
            optimizer.step()

            if step % 1 != 0:
                continue

            net.eval()
            with torch.no_grad():
                y_pred_label = y_pred.argmax(dim=-1)
                train_acc = (y_pred_label == y).float().mean().item() * 100

                total, correct = 0, 0
                for x_val, _, y_val, _ in valid_dataloader:
                    _, y_val_pred = net(x_val.to(device))
                    total += x_val.size(0) * x_val.size(1)
                    correct += (y_val_pred.argmax(dim=-1) == y_val).float().sum().item()
                val_acc = correct / total * 100
                print("Epoch {:2d} | step {:d}\tloss_c = {:.3f}\tloss_ct = {:.3f}\tloss = {:.3f}\t"
                      "training acc = {:.2f}\tvalidation acc = {:.2f}".
                      format(epoch + 1, step + 1, 3 * loss_c.item(), loss_ct.item(), loss.item(),
                             train_acc, val_acc))
                torch.save(net.state_dict(), './harmony/checkpoint_epoch_{}_{}.pth'.format(epoch + 1, step + 1))
        torch.save(net.state_dict(), './harmony/checkpoint_epoch_{}.pth'.format(epoch + 1))


if __name__ == '__main__':
    train()