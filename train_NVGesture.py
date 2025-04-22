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
import re
from sklearn.metrics import f1_score, average_precision_score
from data.template import config
from dataset.nvGesture import NvGestureDataset
from model.RODModel import JointGBShareReinClsModel
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

def train_r_o_d(epoch, train_loader, model, optimizer, merge_alpha1=0.3, merge_alpha2=0.3):
    model.train()
    global s_r
    global s_o
    global s_d
    # ----- RECORD LOSS AND ACC -----
    tl = Averager()
    ta = Averager()
    tv = Averager()
    td = Averager()
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    score_r = 0.0
    score_o = 0.0
    score_d = 0.0
    for step,(rgb,of,depth,y) in enumerate(train_loader):

        rgb = rgb.float().cuda()
        of = of.cuda()
        depth = depth.cuda()
        y = y.cuda()

        optimizer.zero_grad()

        with torch.no_grad():
            w_ = torch.Tensor([[float(s_r)], [float(s_o)], [float(s_d)]]).cuda()
            w = model.w_network(w_.unsqueeze(0))

        o_r = model.rgb_encoder(rgb)
        o_o = model.of_encoder(of)
        o_d = model.depth_encoder(depth)
        # rgb
        out_r, o_fea, add_fea = model.classfier(o_r, w[:, 0][0], is_i='rgb')
        if add_fea is None:
            loss_r = criterion(out_r, y).mean()
        else:
            kl = y * o_fea.detach().softmax(1)
            loss_r = criterion(out_r, y).mean() + criterion(o_fea, y).mean() + criterion(add_fea, y).mean() - 0.2 * criterion(add_fea, kl).mean() # criterion(add_fea, y-0.2*kl).mean() = criterion(add_fea, y).mean() - 0.2 * criterion(add_fea, kl).mean()
        loss_r.backward()
        optimizer.step()
        optimizer.zero_grad()

        # of
        out_o, o_fea, add_fea = model.classfier(o_o, w[:, 1][0], is_i='of')
        if add_fea is None:
            loss_o = criterion(out_o, y).mean()
        else:
            kl = y * o_fea.detach().softmax(1)
            loss_o = criterion(out_o, y).mean() + criterion(o_fea, y).mean() + criterion(add_fea,y).mean() - 0.2 * criterion(add_fea, kl).mean() # criterion(add_fea,y-0.2*kl).mean() = criterion(add_fea,y).mean() - 0.2 * criterion(add_fea, kl).mean()
        loss_o.backward()
        optimizer.step()
        optimizer.zero_grad()

        # depth
        out_d, o_fea, add_fea = model.classfier(o_d, w[:, 2][0], is_i='depth')
        if add_fea is None:
            loss_d = criterion(out_d, y).mean()
        else:
            kl = y * o_fea.detach().softmax(1)
            loss_d = criterion(out_d, y).mean() + criterion(o_fea, y).mean() + criterion(add_fea,y).mean() - 0.2 * criterion(add_fea, kl).mean() # criterion(add_fea,y-0.2*kl).mean() = criterion(add_fea,y).mean() - 0.2 * criterion(add_fea, kl).mean()
        loss_d.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss = loss_r * merge_alpha1 + loss_o * merge_alpha2 + loss_d * (1 - merge_alpha1 - merge_alpha2)
        tl.add(loss.item())
        ta.add(loss_r.item())
        tv.add(loss_o.item())
        td.add(loss_d.item())
        tmp_r = sum([F.softmax(out_r)[i][torch.argmax(y[i])] for i in range(out_r.size(0))])
        tmp_o = sum([F.softmax(out_o)[i][torch.argmax(y[i])] for i in range(out_o.size(0))])
        tmp_d = sum([F.softmax(out_d)[i][torch.argmax(y[i])] for i in range(out_d.size(0))])
        score_r += tmp_r
        score_o += tmp_o
        score_d += tmp_d
        s_r = (tmp_r * (1.0 / (step + 1)) + score_r * (step / (step + 1))) / (
                    tmp_o * (1.0 / (step + 1)) + score_o * (step / (step + 1)) + tmp_d * (1.0 / (step + 1)) + score_d * (step / (step + 1)))
        s_o = (tmp_o * (1.0 / (step + 1)) + score_o * (step / (step + 1))) / (
                tmp_r * (1.0 / (step + 1)) + score_r * (step / (step + 1)) + tmp_d * (1.0 / (step + 1)) + score_d * (
                    step / (step + 1)))
        s_d = (tmp_d * (1.0 / (step + 1)) + score_d * (step / (step + 1))) / (
                tmp_r * (1.0 / (step + 1)) + score_r * (step / (step + 1)) + tmp_o * (1.0 / (step + 1)) + score_o * (
                step / (step + 1)))

        w = model.w_network(w_.unsqueeze(0))
        criterion2 = nn.MSELoss().cuda()
        if s_r > 0.5:
            r_list = [0] * 1
        else:
            r_list = [1] * 1
        if s_o > 0.5:
            o_list = [0] * 1
        else:
            o_list = [1] * 1
        if s_d > 0.5:
            d_list = [0] * 1
        else:
            d_list = [1] * 1
        w_label = torch.Tensor([r_list, o_list, d_list]).unsqueeze(0).cuda()
        loss_w = criterion2(w, w_label)
        loss_w = loss_w.mean()
        loss_w.backward()
        optimizer.step()
        optimizer.zero_grad()

        for n, p in model.named_parameters():
            if p.grad != None:
                del p.grad

        if step % cfg['print_inteval'] == 0:
            print((
                'Epoch:{epoch}, Trainnig Loss:{train_loss:.3f}, Training Loss_r:{loss_a:.3f}, Training Loss_o:{loss_v:.3f}, Training Loss_d:{loss_d:.3f}').format(
                epoch=epoch, train_loss=loss.item(), loss_a=loss_r.item(), loss_v=loss_o.item(), loss_d=loss_d))
    ratio_r = score_r / (score_o + score_d)
    ratio_o = score_o / (score_r + score_d)
    ratio_d = score_d / (score_r + score_o)

    loss_ave = tl.item()
    loss_r_ave = ta.item()
    loss_o_ave = tv.item()
    loss_d_ave = td.item()

    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(('Epoch {epoch:d}: Average Training Loss:{loss_ave:.3f}, Average Training Loss_r:{loss_a_ave:.2f}, Average Training Loss_o:{loss_v_ave:.2f}, Average Training Loss_d:{loss_d_ave:.2f}').format(
        epoch=epoch, loss_ave=loss_ave, loss_a_ave=loss_r_ave, loss_v_ave=loss_o_ave, loss_d_ave=loss_d_ave))

    return model, ratio_r, ratio_o, ratio_d


def val(epoch, val_loader, model, merge_alpha1=0.3, merge_alpha2=0.5):
    model.eval()
    global s_r
    global s_o
    global s_d
    pred_list = []
    pred_list_r = []
    pred_list_o = []
    pred_list_d = []
    label_list = []
    one_hot_label = []
    with torch.no_grad():
        for step, (rgb, of, depth, y) in enumerate(val_loader):
            label_list = label_list + torch.argmax(y, dim=1).tolist()
            one_hot_label = one_hot_label + y.tolist()
            rgb = rgb.float().cuda()
            of = of.cuda()
            depth = depth.cuda()
            w_ = torch.Tensor([[float(s_r)], [float(s_o)], [float(s_d)]]).cuda()
            w = model.w_network(w_.unsqueeze(0))
            o_r, o_o, o_d = model(rgb, of, depth)
            out_r, _, _ = model.classfier(o_r, w[:, 0][0], is_i='rgb')
            out_o, _, _ = model.classfier(o_o, w[:, 1][0], is_i='of')
            out_d, _, _ = model.classfier(o_d, w[:, 2][0], is_i='depth')
            out = out_r * merge_alpha1 + out_o * merge_alpha2 + out_d * (1 - merge_alpha1 - merge_alpha2)

            pred = (F.softmax(out, dim=1)).argmax(dim=1)
            pred_r = (F.softmax(out_r, dim=1)).argmax(dim=1)
            pred_o = (F.softmax(out_o, dim=1)).argmax(dim=1)
            pred_d = (F.softmax(out_d, dim=1)).argmax(dim=1)

            pred_list = pred_list + pred.tolist()
            pred_list_r = pred_list_r + pred_r.tolist()
            pred_list_o = pred_list_o + pred_o.tolist()
            pred_list_d = pred_list_d + pred_d.tolist()

        f1 = f1_score(label_list, pred_list, average='macro')
        f1_r = f1_score(label_list, pred_list_r, average='macro')
        f1_o = f1_score(label_list, pred_list_o, average='macro')
        f1_d = f1_score(label_list, pred_list_d, average='macro')
        correct = sum(1 for x, y in zip(label_list, pred_list) if x == y)
        correct_r = sum(1 for x, y in zip(label_list, pred_list_r) if x == y)
        correct_o = sum(1 for x, y in zip(label_list, pred_list_o) if x == y)
        correct_d = sum(1 for x, y in zip(label_list, pred_list_d) if x == y)
        acc = correct / len(label_list)
        acc_r = correct_r / len(label_list)
        acc_o = correct_o / len(label_list)
        acc_d = correct_d / len(label_list)


    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(('Epoch {epoch:d}: f1:{f1:.4f},acc:{acc:.4f},f1_r:{f1_a:.4f},acc_r:{acc_a:.4f},f1_o:{f1_v:.4f},acc_o:{acc_v:.4f},f1_d:{f1_d:.4f},acc_d:{acc_d:.4f}').format(
        epoch=epoch, f1=f1, acc=acc,
        f1_a=f1_r, acc_a=acc_r,
        f1_v=f1_o, acc_v=acc_o,
        f1_d=f1_d, acc_d=acc_d))
    return acc


if __name__ == '__main__':

    # ----- LOAD PARAM -----
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/data/nvGesture.json')
    parser.add_argument('--lam', default=0.4, type=float, help='lam')
    parser.add_argument('--merge_alpha1', default=0.33, type=float, help='2 modal fusion alpha in GS')
    parser.add_argument('--merge_alpha2', default=0.33, type=float, help='2 modal fusion alpha in GS')

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
    torch.backends.cudnn.enabled = False
    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu_id']

    # ----- SET DATALOADER -----
    train_dataset = NvGestureDataset(config, mode='train')
    test_dataset = NvGestureDataset(config, mode='test')

    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg['train']['batch_size'], shuffle=True,
                              num_workers=cfg['train']['num_workers'], pin_memory=True)

    test_loader = DataLoader(dataset=test_dataset, batch_size=cfg['test']['batch_size'], shuffle=False,
                             num_workers=cfg['test']['num_workers'], pin_memory=True)

    global s_r
    global s_o
    global s_d
    s_r = 0.0
    s_o = 0.0
    s_d = 0.0

    # ----- MODEL -----
    model = JointGBShareReinClsModel(config=cfg)
    model = model.cuda()

    lr_adjust = config['train']['optimizer']['lr']
    w_param_list = list(map(id, model.w_network.parameters()))
    w_params = filter(lambda p: id(p) in w_param_list, model.parameters())
    base_params = filter(lambda p: id(p) not in w_param_list, model.parameters())
    params = [{'params': base_params, 'lr': lr_adjust},
              {'params': w_params, 'lr': lr_adjust / 10}
              ]

    optimizer = optim.SGD(params, lr=lr_adjust,
                          momentum=config['train']['optimizer']['momentum'],
                          weight_decay=config['train']['optimizer']['wc'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, config['train']['lr_scheduler']['patience'], 0.1)
    best_acc = 0

    if cfg['train']['epoch_dict'] > 10:
        check = int(cfg['train']['epoch_dict'] / 10)
    else:
        check = 1
    for epoch in range(cfg['train']['epoch_dict']):
        print(('Epoch {epoch:d} is pending...').format(epoch=epoch))
        scheduler.step()
        model, ratio_r, ratio_o, ratio_d = train_r_o_d(epoch, train_loader, model, optimizer, args.merge_alpha1, args.merge_alpha2)
        acc = val(epoch, test_loader, model, args.merge_alpha1, args.merge_alpha2)
        print(('ratio_r: {ratio_r:.3f}, ratio_o: {ratio_o:.3f}, ratio_d: {ratio_d:.3f},').format(ratio_r=ratio_r, ratio_o=ratio_o, ratio_d=ratio_d))
        if acc >= best_acc:
            best_acc = acc
            print('Find a better model and save it!')
            torch.save(model.state_dict(), ('NVGesture_GB_best_model_{acc:.4f}.pth').format(acc=best_acc))
        if ((epoch + 1) % check == 0 or epoch == 0):
            print(('ratio_r: {ratio_r:.3f}, ratio_o: {ratio_o:.3f}, ratio_d: {ratio_d:.3f},').format(ratio_r=ratio_r,ratio_o=ratio_o,ratio_d=ratio_d))
            if ratio_r < args.lam:
                print("add_layer_r")
                model.add_layer(is_i='rgb')
            if ratio_o < args.lam:
                print("add_layer_o")
                model.add_layer(is_i='of')
            if ratio_d < args.lam:
                print("add_layer_d")
                model.add_layer(is_i='depth')