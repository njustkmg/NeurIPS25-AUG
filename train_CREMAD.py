#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import functional as F
import os
import warnings

warnings.filterwarnings("ignore")
import json
import numpy as np
import argparse
import random
from sklearn.metrics import f1_score, average_precision_score
from data.template import config
from dataset.CREMA import CramedDataset
from model.AudioVideo import AVGBShareClassifier
from utils.utils import (
    Averager,
    deep_update_dict,
    weight_init
)

def compute_mAP(outputs, labels):
    y_true = labels.cpu().detach().numpy()
    y_pred = outputs.cpu().detach().numpy()
    AP = []
    for i in range(y_true.shape[1]):
        AP.append(average_precision_score(y_true[:, i], y_pred[:, i]))
    return np.mean(AP)

def train_audio_video(epoch, train_loader, model, optimizer, merge_alpha=0.5):
    model.train()
    # ----- RECORD LOSS AND ACC -----
    tl = Averager()
    ta = Averager()
    tv = Averager()
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    score_v = 0.0
    score_a = 0.0
    for step, (spectrogram, image, y) in enumerate(train_loader):
        image = image.float().cuda()
        y = y.cuda()
        spectrogram = spectrogram.unsqueeze(1).float().cuda()

        optimizer.zero_grad()

        o_a = model.audio_encoder(spectrogram)
        o_v = model.video_encoder(image)
        # audio
        out_a, o_fea, add_fea = model.classfier(o_a, is_a=True)
        if add_fea is None:
            loss_a = criterion(out_a, y).mean()
        else:
            kl = y*o_fea.detach().softmax(1)
            loss_a = criterion(out_a, y).mean() + criterion(o_fea, y).mean() + criterion(add_fea, y).mean() - 0.5 * criterion(add_fea, kl).mean() # criterion(add_fea, y-0.2*kl).mean() = criterion(add_fea, y).mean() - 0.2 * criterion(add_fea, kl).mean()
            # loss_a = criterion(out_a, y).mean() + criterion(o_fea, y).mean() + criterion(add_fea, y-0.2*kl).mean()
        loss_a.backward()

        optimizer.step()
        optimizer.zero_grad()

        # video
        out_v, o_fea, add_fea = model.classfier(o_v, is_a=False)
        if add_fea is None:
            loss_v = criterion(out_v, y).mean()
        else:
            kl = y*o_fea.detach().softmax(1)
            loss_v = criterion(out_v, y).mean() + criterion(o_fea, y).mean() + criterion(add_fea, y).mean() - 0.5 * criterion(add_fea, kl).mean() # criterion(add_fea, y-0.2*kl).mean() = criterion(add_fea, y).mean() - 0.2 * criterion(add_fea, kl).mean()
            # loss_v = criterion(out_v, y).mean() + criterion(o_fea, y).mean() + criterion(add_fea, y-0.2*kl).mean()
        loss_v.backward()

        optimizer.step()
        optimizer.zero_grad()

        loss = loss_a * merge_alpha + loss_v * (1 - merge_alpha)
        tl.add(loss.item())
        ta.add(loss_a.item())
        tv.add(loss_v.item())

        tmp_v = sum([F.softmax(out_v)[i][torch.argmax(y[i])] for i in range(out_v.size(0))])
        tmp_a = sum([F.softmax(out_a)[i][torch.argmax(y[i])] for i in range(out_a.size(0))])

        score_v += tmp_v
        score_a += tmp_a
        for n, p in model.named_parameters():
            if p.grad != None:
                del p.grad

        if step % cfg['print_inteval'] == 0:
            print((
                'Epoch:{epoch}, Trainnig Loss:{train_loss:.3f}, Training Loss_a:{loss_a:.3f}, Training Loss_v:{loss_v:.3f}').format(
                epoch=epoch, train_loss=loss.item(), loss_a=loss_a.item(), loss_v=loss_v.item()))

    ratio_a = score_a / score_v
    loss_ave = tl.item()
    loss_a_ave = ta.item()
    loss_v_ave = tv.item()

    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(('Epoch {epoch:d}: Average Training Loss:{loss_ave:.3f}, Average Training Loss_a:{loss_a_ave:.2f}, Average Training Loss_v:{loss_v_ave:.2f}').format(
        epoch=epoch, loss_ave=loss_ave, loss_a_ave=loss_a_ave, loss_v_ave=loss_v_ave))

    return model, ratio_a


def val(epoch, val_loader, model, merge_alpha=0.5):
    model.eval()
    pred_list = []
    pred_list_a = []
    pred_list_v = []
    label_list = []
    soft_pred = []
    soft_pred_a = []
    soft_pred_v = []
    one_hot_label = []
    with torch.no_grad():
        for step, (spectrogram, image, y) in enumerate(val_loader):
            label_list = label_list + torch.argmax(y, dim=1).tolist()
            one_hot_label = one_hot_label + y.tolist()
            image = image.cuda()
            y = y.cuda()
            spectrogram = spectrogram.unsqueeze(1).float().cuda()
            o_a, o_v = model(spectrogram, image)
            out_a, _, _ = model.classfier(o_a, is_a=True)
            out_v, _, _ = model.classfier(o_v, is_a=False)
            out = merge_alpha * out_a + (1 - merge_alpha) * out_v

            soft_pred_a = soft_pred_a + (F.softmax(out_a, dim=1)).tolist()
            soft_pred_v = soft_pred_v + (F.softmax(out_v, dim=1)).tolist()
            soft_pred = soft_pred + (F.softmax(out, dim=1)).tolist()
            pred = (F.softmax(out, dim=1)).argmax(dim=1)
            pred_a = (F.softmax(out_a, dim=1)).argmax(dim=1)
            pred_v = (F.softmax(out_v, dim=1)).argmax(dim=1)

            pred_list = pred_list + pred.tolist()
            pred_list_a = pred_list_a + pred_a.tolist()
            pred_list_v = pred_list_v + pred_v.tolist()

        f1 = f1_score(label_list, pred_list, average='macro')
        f1_a = f1_score(label_list, pred_list_a, average='macro')
        f1_v = f1_score(label_list, pred_list_v, average='macro')
        correct = sum(1 for x, y in zip(label_list, pred_list) if x == y)
        correct_a = sum(1 for x, y in zip(label_list, pred_list_a) if x == y)
        correct_v = sum(1 for x, y in zip(label_list, pred_list_v) if x == y)
        acc = correct / len(label_list)
        acc_a = correct_a / len(label_list)
        acc_v = correct_v / len(label_list)
        mAP = compute_mAP(torch.Tensor(soft_pred), torch.Tensor(one_hot_label))
        mAP_a = compute_mAP(torch.Tensor(soft_pred_a), torch.Tensor(one_hot_label))
        mAP_v = compute_mAP(torch.Tensor(soft_pred_v), torch.Tensor(one_hot_label))

    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(('Epoch {epoch:d}: f1:{f1:.4f},acc:{acc:.4f},mAP:{mAP:.4f},f1_a:{f1_a:.4f},acc_a:{acc_a:.4f},mAP_a:{mAP_a:.4f},f1_v:{f1_v:.4f},acc_v:{acc_v:.4f},mAP_v:{mAP_v:.4f}').format(epoch=epoch, f1=f1, acc=acc, mAP=mAP,
                                                                                                                                                                                            f1_a=f1_a, acc_a=acc_a, mAP_a=mAP_a,
                                                                                                                                                                                            f1_v=f1_v, acc_v=acc_v, mAP_v=mAP_v))

    return acc

if __name__ == '__main__':
    # ----- LOAD PARAM -----
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=str, default='/data/hlf/Our_CVPR25/code/data/crema.json')
    parser.add_argument('--lam', default=1.0, type=float, help='lam')
    parser.add_argument('--merge_alpha', default=0.4, type=float, help='alpha')
    args = parser.parse_args()
    cfg = config

    with open(args.config, "r") as f:
        exp_params = json.load(f)

    cfg = deep_update_dict(exp_params, cfg)

    # ----- SET SEED -----
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg['gpu_id']

    # ----- SET DATALOADER -----
    train_dataset = CramedDataset(config, mode='train')
    test_dataset = CramedDataset(config, mode='test')

    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg['train']['batch_size'], shuffle=True,
                              num_workers=cfg['train']['num_workers'], pin_memory=True)

    test_loader = DataLoader(dataset=test_dataset, batch_size=cfg['test']['batch_size'], shuffle=False,
                             num_workers=cfg['test']['num_workers'], pin_memory=True)

    # ----- MODEL -----
    model = AVGBShareClassifier(config=cfg)
    model = model.cuda()
    model.apply(weight_init)

    lr_adjust = config['train']['optimizer']['lr']

    optimizer = optim.SGD(model.parameters(), lr=lr_adjust, momentum=config['train']['optimizer']['momentum'], weight_decay=config['train']['optimizer']['wc'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, config['train']['lr_scheduler']['patience'], 0.1)
    best_acc = 0

    if cfg['train']['epoch_dict'] > 10:
        check = int(cfg['train']['epoch_dict'] / 10)
    else:
        check = 1

    for epoch in range(cfg['train']['epoch_dict']):
        print(('Epoch {epoch:d} is pending...').format(epoch=epoch))
        scheduler.step()
        model, ratio_a = train_audio_video(epoch, train_loader, model, optimizer,args.merge_alpha)
        acc = val(epoch, test_loader, model, args.merge_alpha)
        print(('ratio: {ratio_a:.3f}').format(ratio_a=ratio_a))
        if acc >= best_acc:
            best_acc = acc
            print('Find a better model and save it!')
            torch.save(model.state_dict(), ('crema_GB_best_model_{acc:.4f}.pth').format(acc=best_acc))
        if ((epoch+1) % check == 0 or epoch == 0):
            print(('ratio: {ratio_a:.3f}').format(ratio_a=ratio_a))
            if ratio_a > args.lam+0.01:
                print("add_layer_v")
                model.add_layer(is_a=False)
            elif ratio_a < args.lam-0.01:
                print("add_layer_a")

                model.add_layer(is_a=True)
