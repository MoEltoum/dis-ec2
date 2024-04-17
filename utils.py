import gc
import os
import time
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
import logging
from csv_logger import CsvLogger
import numpy as np
import wget
import pandas as pd
import torch.optim as optim
from skimage import io
from torch.autograd import Variable
import torch.nn as nn
from data_loader_cache import get_im_gt_name_dict, create_dataloaders, GOSRandomHFlip, GOSNormalize
from basics import f1_mae_torch
import torch
from torch.utils.tensorboard import SummaryWriter
import cv2
from tqdm import tqdm
from glob import glob
import pickle as pkl
from skimage.morphology import disk, skeletonize
from skimage.measure import label
import mlflow
from models.isnet import *

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define model local
Path(os.path.sep.join([os.getcwd(), "weights"])).mkdir(parents=True, exist_ok=True)
weights_path = os.path.join(os.getcwd(), 'weights')


def get_gt_encoder(train_dataloaders, train_datasets, valid_dataloaders, valid_datasets, train_dataloaders_val,
                   train_datasets_val, start_ite, gt_encoder_model, model_digit, seed, early_stop, model_save_fre,
                   batch_size_train, batch_size_valid, max_ite, max_epoch_num, valid_out_dir):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    print("define gt encoder ...")
    net = ISNetGTEncoder()
    # load the existing model gt encoder
    if (gt_encoder_model != ""):
        model_path = weights_path + "/" + gt_encoder_model
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(model_path))
            net.cuda()
        else:
            net.load_state_dict(torch.load(model_path, map_location="cpu"))
        print("gt encoder restored from the saved weights ...")
        return net

    if torch.cuda.is_available():
        net.cuda()

    print("--- define optimizer for GT Encoder---")
    optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    model_path = weights_path
    model_save_fre = model_save_fre
    max_ite = max_ite
    batch_size_train = batch_size_train
    batch_size_valid = batch_size_valid

    if (not os.path.exists(model_path)):
        os.mkdir(model_path)

    ite_num = start_ite  # count the total iteration number
    ite_num4val = 0  #
    running_loss = 0.0  # count the total loss
    running_tar_loss = 0.0  # count the target output loss
    last_f1 = [0 for x in range(len(valid_dataloaders))]

    train_num = train_datasets [0].__len__()

    net.train()

    start_last = time.time()
    gos_dataloader = train_dataloaders [0]
    epoch_num = max_epoch_num
    notgood_cnt = 0
    for epoch in range(epoch_num):  # set the epoch num as 100000

        for i, data in enumerate(gos_dataloader):

            if (ite_num >= max_ite):
                print("Training Reached the Maximal Iteration Number ", max_ite)
                exit()

            # start_read = time.time()
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            # get the inputs
            labels = data ['label']

            if (model_digit == "full"):
                labels = labels.type(torch.FloatTensor)
            else:
                labels = labels.type(torch.HalfTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                labels_v = Variable(labels.cuda(), requires_grad=False)
            else:
                labels_v = Variable(labels, requires_grad=False)

            # y zero the parameter gradients
            start_inf_loss_back = time.time()
            optimizer.zero_grad()

            ds, fs = net(labels_v)  # net(inputs_v)
            loss2, loss = net.compute_loss(ds, labels_v)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_tar_loss += loss2.item()

            # del outputs, loss
            del ds, loss2, loss
            end_inf_loss_back = time.time() - start_inf_loss_back

            print("GT Encoder Training>>>" + model_path.split('/') [
                -1] + " - [epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f, time-per-iter: %3f s, time_read: %3f" % (
                      epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val,
                      running_tar_loss / ite_num4val, time.time() - start_last,
                      time.time() - start_last - end_inf_loss_back))
            start_last = time.time()

            if ite_num % model_save_fre == 0:  # validate every 2000 iterations
                notgood_cnt += 1
                tmp_f1, tmp_mae, val_loss, tar_loss, i_val, tmp_time = valid_gt_encoder(net, train_dataloaders_val,
                                                                                        train_datasets_val, model_digit,
                                                                                        batch_size_valid, max_epoch_num,
                                                                                        valid_out_dir, epoch)

                net.train()  # resume train

                tmp_out = 0
                print("last_f1:", last_f1)
                print("tmp_f1:", tmp_f1)
                for fi in range(len(last_f1)):
                    if (tmp_f1 [fi] > last_f1 [fi]):
                        tmp_out = 1
                print("tmp_out:", tmp_out)
                if (tmp_out):
                    notgood_cnt = 0
                    last_f1 = tmp_f1
                    tmp_f1_str = [str(round(f1x, 4)) for f1x in tmp_f1]
                    tmp_mae_str = [str(round(mx, 4)) for mx in tmp_mae]
                    maxf1 = '_'.join(tmp_f1_str)
                    meanM = '_'.join(tmp_mae_str)
                    # .cpu().detach().numpy()
                    model_name = "/GTENCODER-gpu_itr_" + str(ite_num) + \
                                 "_traLoss_" + str(np.round(running_loss / ite_num4val, 4)) + \
                                 "_traTarLoss_" + str(np.round(running_tar_loss / ite_num4val, 4)) + \
                                 "_valLoss_" + str(np.round(val_loss / (i_val + 1), 4)) + \
                                 "_valTarLoss_" + str(np.round(tar_loss / (i_val + 1), 4)) + \
                                 "_maxF1_" + maxf1 + \
                                 "_mae_" + meanM + \
                                 "_time_" + str(np.round(np.mean(np.array(tmp_time)) / batch_size_valid, 6)) + ".pth"
                    torch.save(net.state_dict(), model_path + model_name)

                running_loss = 0.0
                running_tar_loss = 0.0
                ite_num4val = 0

                if (tmp_f1 [0] > 0.99):
                    print("GT encoder is well-trained and obtained...")
                    return net

                if (notgood_cnt >= early_stop):
                    print("No improvements in the last " + str(
                        notgood_cnt) + " validation periods, so training stopped !")
                    exit()

    print("Training Reaches The Maximum Epoch Number")
    return net


def valid_gt_encoder(net, valid_dataloaders, valid_datasets, model_digit, batch_size_valid, max_epoch_num,
                     valid_out_dir, epoch=0):
    net.eval()
    print("Validating...")
    epoch_num = max_epoch_num

    val_loss = 0.0
    tar_loss = 0.0

    tmp_f1 = []
    tmp_mae = []
    tmp_time = []

    start_valid = time.time()
    for k in range(len(valid_dataloaders)):

        valid_dataloader = valid_dataloaders [k]
        valid_dataset = valid_datasets [k]

        val_num = valid_dataset.__len__()
        mybins = np.arange(0, 256)
        PRE = np.zeros((val_num, len(mybins) - 1))
        REC = np.zeros((val_num, len(mybins) - 1))
        F1 = np.zeros((val_num, len(mybins) - 1))
        MAE = np.zeros((val_num))

        val_cnt = 0.0
        i_val = None

        for i_val, data_val in enumerate(valid_dataloader):

            imidx_val, labels_val, shapes_val = data_val ['imidx'], data_val ['label'], data_val ['shape']

            if model_digit == "full":
                labels_val = labels_val.type(torch.FloatTensor)
            else:
                labels_val = labels_val.type(torch.HalfTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                labels_val_v = Variable(labels_val.cuda(), requires_grad=False)
            else:
                labels_val_v = Variable(labels_val, requires_grad=False)

            t_start = time.time()
            ds_val = net(labels_val_v) [0]
            t_end = time.time() - t_start
            tmp_time.append(t_end)

            loss2_val, loss_val = net.compute_loss(ds_val, labels_val_v)

            # compute F measure
            for t in range(batch_size_valid):
                val_cnt = val_cnt + 1.0
                print("num of val: ", val_cnt)
                i_test = imidx_val [t].data.numpy()

                pred_val = ds_val [0] [t, :, :, :]  # B x 1 x H x W

                # recover the prediction spatial size to the orignal image size
                pred_val = torch.squeeze(
                    F.upsample(torch.unsqueeze(pred_val, 0), (shapes_val [t] [0], shapes_val [t] [1]), mode='bilinear'))

                ma = torch.max(pred_val)
                mi = torch.min(pred_val)
                pred_val = (pred_val - mi) / (ma - mi)  # max = 1

                gt = np.squeeze(io.imread(valid_dataset.dataset ["ori_gt_path"] [i_test]))  # max = 255
                with torch.no_grad():
                    gt = torch.tensor(gt).to(device)

                pre, rec, f1, mae = f1_mae_torch(pred_val * 255, gt, valid_dataset, i_test, mybins, valid_out_dir)

                PRE [i_test, :] = pre
                REC [i_test, :] = rec
                F1 [i_test, :] = f1
                MAE [i_test] = mae

            del ds_val, gt
            gc.collect()
            torch.cuda.empty_cache()

            # if(loss_val.data[0]>1):
            val_loss += loss_val.item()  # data[0]
            tar_loss += loss2_val.item()  # data[0]

            print("[validating: %5d/%5d] val_ls:%f, tar_ls: %f, f1: %f, mae: %f, time: %f" % (
                i_val, val_num, val_loss / (i_val + 1), tar_loss / (i_val + 1), np.amax(F1 [i_test, :]), MAE [i_test],
                t_end))

            del loss2_val, loss_val

        print('============================')
        PRE_m = np.mean(PRE, 0)
        REC_m = np.mean(REC, 0)
        f1_m = (1 + 0.3) * PRE_m * REC_m / (0.3 * PRE_m + REC_m + 1e-8)
        # print('--------------:', np.mean(f1_m))
        tmp_f1.append(np.amax(f1_m))
        tmp_mae.append(np.mean(MAE))
        print("The max F1 Score: %f" % (np.max(f1_m)))
        print("MAE: ", np.mean(MAE))

    return tmp_f1, tmp_mae, val_loss, tar_loss, i_val, tmp_time


def train(net, optimizer, train_dataloaders, train_datasets, valid_dataloaders, valid_datasets,
          train_dataloaders_val, train_datasets_val,
          interm_sup, start_ite, gt_encoder_model, model_digit, seed, early_stop, model_save_fre,
          batch_size_train, batch_size_valid, max_ite, max_epoch_num, valid_out_dir, s3_bucket):
    # train logs
    Path(os.path.sep.join([os.getcwd(), "logs"])).mkdir(parents=True, exist_ok=True)
    csvlogger = CsvLogger(filename=os.path.join(os.getcwd(), 'logs/train_logs.csv'),
                          delimiter=',',
                          level=logging.INFO,
                          fmt=f'%(asctime)s{","}%(message)s',
                          datefmt='%Y/%m/%d , %H:%M:%S',
                          header=['date', 'time', 'epoch', 'batch', 'ite', 'train_loss', 'tar', 'sec_per_iter'])

    # tensorboard logs
    Path(os.path.sep.join([os.getcwd(), "tensorboard"])).mkdir(parents=True, exist_ok=True)
    tensorboard_path = os.path.join(os.getcwd(), 'tensorboard')
    writer = SummaryWriter(tensorboard_path)

    if interm_sup:
        print("Get the gt encoder ...")
        featurenet = get_gt_encoder(train_dataloaders, train_datasets, valid_dataloaders, valid_datasets,
                                    train_dataloaders_val, train_datasets_val,
                                    start_ite, gt_encoder_model, model_digit, seed, early_stop, model_save_fre,
                                    batch_size_train, batch_size_valid, max_ite, max_epoch_num, valid_out_dir)
        # freeze the weights of gt encoder
        for param in featurenet.parameters():
            param.requires_grad = False

    model_path = weights_path
    model_save_fre = model_save_fre
    max_ite = max_ite
    batch_size_train = batch_size_train
    batch_size_valid = batch_size_valid

    if (not os.path.exists(model_path)):
        os.mkdir(model_path)

    ite_num = start_ite  # count the total iteration number
    ite_num4val = 0  #
    running_loss = 0.0  # count the toal loss
    running_tar_loss = 0.0  # count the target output loss
    last_f1 = [0 for x in range(len(valid_dataloaders))]

    train_num = train_datasets [0].__len__()

    net.train()

    start_last = time.time()
    gos_dataloader = train_dataloaders [0]
    epoch_num = max_epoch_num
    notgood_cnt = 0
    for epoch in range(epoch_num):  # set the epoch num as 100000

        for i, data in enumerate(gos_dataloader):

            if (ite_num >= max_ite):
                print("Training Reached the Maximal Iteration Number ", max_ite)

            # start_read = time.time()
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            # get the inputs
            inputs, labels = data ['image'], data ['label']

            if (model_digit == "full"):
                inputs = inputs.type(torch.FloatTensor)
                labels = labels.type(torch.FloatTensor)
            else:
                inputs = inputs.type(torch.HalfTensor)
                labels = labels.type(torch.HalfTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                            requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            # y zero the parameter gradients
            start_inf_loss_back = time.time()
            optimizer.zero_grad()

            if interm_sup:
                # forward + backward + optimize
                ds, dfs = net(inputs_v)
                _, fs = featurenet(labels_v)  ## extract the gt encodings
                loss2, loss = net.compute_loss_kl(ds, labels_v, dfs, fs, mode='MSE')
            else:
                # forward + backward + optimize
                ds, _ = net(inputs_v)
                # loss2, loss = net.compute_loss(ds, labels_v)
                loss2, loss = muti_loss_fusion(ds, labels_v)

            # writer.add_scalar("Loss/train", loss, epoch)

            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.item()
            running_tar_loss += loss2.item()

            # del outputs, loss
            del ds, loss2, loss
            end_inf_loss_back = time.time() - start_inf_loss_back

            print(">>>" + model_path.split('/') [
                -1] + " - [epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f, time-per-iter: %3f s, time_read: %3f" % (
                      epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val,
                      running_tar_loss / ite_num4val, time.time() - start_last,
                      time.time() - start_last - end_inf_loss_back))

            csvlogger.info("%3d, %5d, %d, %3f, %3f, %3f s" % (
                epoch + 1, (i + 1) * batch_size_train, ite_num, running_loss / ite_num4val,
                running_tar_loss / ite_num4val, time.time() - start_last))

            start_last = time.time()

            if ite_num % model_save_fre == 0:  # validate every 2000 iterations
                notgood_cnt += 1
                net.eval()
                tmp_f1, tmp_mae, val_loss, tar_loss, i_val, tmp_time = valid(net, valid_dataloaders, valid_datasets,
                                                                             model_digit, batch_size_valid,
                                                                             max_epoch_num, valid_out_dir, epoch)
                net.train()  # resume train

                tmp_out = 0
                print("last_f1:", last_f1)
                print("tmp_f1:", tmp_f1)
                for fi in range(len(last_f1)):
                    if (tmp_f1 [fi] > last_f1 [fi]):
                        tmp_out = 1
                print("tmp_out:", tmp_out)

                if (tmp_out):
                    notgood_cnt = 0
                    last_f1 = tmp_f1
                    tmp_f1_str = [str(round(f1x, 4)) for f1x in tmp_f1]
                    tmp_mae_str = [str(round(mx, 4)) for mx in tmp_mae]
                    maxf1 = '_'.join(tmp_f1_str)
                    meanM = '_'.join(tmp_mae_str)
                    # .cpu().detach().numpy()
                    model_name = "/gpu_itr_" + str(ite_num) + \
                                 "_traLoss_" + str(np.round(running_loss / ite_num4val, 4)) + \
                                 "_traTarLoss_" + str(np.round(running_tar_loss / ite_num4val, 4)) + \
                                 "_valLoss_" + str(np.round(val_loss / (i_val + 1), 4)) + \
                                 "_valTarLoss_" + str(np.round(tar_loss / (i_val + 1), 4)) + \
                                 "_maxF1_" + maxf1 + \
                                 "_mae_" + meanM + \
                                 "_time_" + str(np.round(np.mean(np.array(tmp_time)) / batch_size_valid, 6)) + ".pth"
                    torch.save(net.state_dict(), model_path + model_name)

                    writer.add_scalars('train',
                                       {'Loss': np.round(running_loss / ite_num4val, 4),
                                        'TarLoss': np.round(running_tar_loss / ite_num4val, 4)}, ite_num)
                    writer.add_scalars('valid',
                                       {'Loss': np.round(val_loss / (i_val + 1), 4),
                                        'TarLoss': np.round(tar_loss / (i_val + 1), 4)}, ite_num)
                    writer.add_scalars('F1_score', {'F1': round(tmp_f1 [0], 4)}, ite_num)
                    writer.add_scalars('MAE', {'mae': round(tmp_mae [0], 4)}, ite_num)
                # mlflow
                mlflow.log_metric("train_loss", np.round(running_loss / ite_num4val, 4), step=ite_num)
                mlflow.log_metric("val_loss", np.round(val_loss / (i_val + 1), 4), step=ite_num)
                mlflow.log_metric("F1_score", round(tmp_f1 [0], 4), step=ite_num)
                mlflow.log_metric("mae", round(tmp_mae [0], 4), step=ite_num)
                upload_folder_to_s3("/opt/ml/code/mlruns", s3_bucket, "mlruns")

                running_loss = 0.0
                running_tar_loss = 0.0
                ite_num4val = 0

                if (notgood_cnt >= early_stop):
                    print("No improvements in the last " + str(
                        notgood_cnt) + " validation periods, so training stopped !")
                    logging.info("No improvements in the last " + str(
                        notgood_cnt) + " validation periods, so training stopped !")
                    return

    print("Training Reaches The Maximum Epoch Number")


def valid(net, valid_dataloaders, valid_datasets, model_digit, batch_size_valid, max_epoch_num, valid_out_dir, epoch=0):
    net.eval()
    print("Validating...")
    epoch_num = max_epoch_num

    val_loss = 0.0
    tar_loss = 0.0
    val_cnt = 0.0

    tmp_f1 = []
    tmp_mae = []
    tmp_time = []

    start_valid = time.time()

    for k in range(len(valid_dataloaders)):

        valid_dataloader = valid_dataloaders [k]
        valid_dataset = valid_datasets [k]

        val_num = valid_dataset.__len__()
        mybins = np.arange(0, 256)
        PRE = np.zeros((val_num, len(mybins) - 1))
        REC = np.zeros((val_num, len(mybins) - 1))
        F1 = np.zeros((val_num, len(mybins) - 1))
        MAE = np.zeros((val_num))

        for i_val, data_val in enumerate(valid_dataloader):
            val_cnt = val_cnt + 1.0
            imidx_val, inputs_val, labels_val, shapes_val = data_val ['imidx'], data_val ['image'], data_val ['label'], \
                data_val ['shape']

            if model_digit == "full":
                inputs_val = inputs_val.type(torch.FloatTensor)
                labels_val = labels_val.type(torch.FloatTensor)
            else:
                inputs_val = inputs_val.type(torch.HalfTensor)
                labels_val = labels_val.type(torch.HalfTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_val_v, labels_val_v = Variable(inputs_val.cuda(), requires_grad=False), Variable(
                    labels_val.cuda(), requires_grad=False)
            else:
                inputs_val_v, labels_val_v = Variable(inputs_val, requires_grad=False), Variable(labels_val,
                                                                                                 requires_grad=False)

            t_start = time.time()
            ds_val = net(inputs_val_v) [0]
            t_end = time.time() - t_start
            tmp_time.append(t_end)

            # loss2_val, loss_val = net.compute_loss(ds_val, labels_val_v)
            loss2_val, loss_val = muti_loss_fusion(ds_val, labels_val_v)

            # compute F measure
            for t in range(batch_size_valid):
                i_test = imidx_val [t].data.numpy()

                pred_val = ds_val [0] [t, :, :, :]  # B x 1 x H x W

                # recover the prediction spatial size to the orignal image size
                pred_val = torch.squeeze(
                    F.upsample(torch.unsqueeze(pred_val, 0), (shapes_val [t] [0], shapes_val [t] [1]), mode='bilinear'))

                # pred_val = normPRED(pred_val)
                ma = torch.max(pred_val)
                mi = torch.min(pred_val)
                pred_val = (pred_val - mi) / (ma - mi)  # max = 1

                # valid_dataset.dataset ["ori_gt_path"] = list_files('/opt/ml/code/data/val/gt')
                if len(valid_dataset.dataset ["ori_gt_path"]) != 0:
                    gt = np.squeeze(io.imread(valid_dataset.dataset ["ori_gt_path"] [i_test]))  # max = 255
                else:
                    gt = np.zeros((shapes_val [t] [0], shapes_val [t] [1]))
                with torch.no_grad():
                    gt = torch.tensor(gt).to(device)

                pre, rec, f1, mae = f1_mae_torch(pred_val * 255, gt, valid_dataset, i_test, mybins, valid_out_dir)

                PRE [i_test, :] = pre
                REC [i_test, :] = rec
                F1 [i_test, :] = f1
                MAE [i_test] = mae

                del ds_val, gt
                gc.collect()
                torch.cuda.empty_cache()

            # if(loss_val.data[0]>1):
            val_loss += loss_val.item()  # data[0]
            tar_loss += loss2_val.item()  # data[0]

            print("[validating: %5d/%5d] val_ls:%f, tar_ls: %f, f1: %f, mae: %f, time: %f" % (
                i_val, val_num, val_loss / (i_val + 1), tar_loss / (i_val + 1), np.amax(F1 [i_test, :]), MAE [i_test],
                t_end))

            del loss2_val, loss_val

        print('============================')
        PRE_m = np.mean(PRE, 0)
        REC_m = np.mean(REC, 0)
        f1_m = (1 + 0.3) * PRE_m * REC_m / (0.3 * PRE_m + REC_m + 1e-8)

        tmp_f1.append(np.amax(f1_m))
        tmp_mae.append(np.mean(MAE))

    return tmp_f1, tmp_mae, val_loss, tar_loss, i_val, tmp_time


def muti_loss_fusion(preds, target):
    bce_loss = nn.BCELoss(size_average=True)
    loss0 = 0.0
    loss = 0.0

    for i in range(0, len(preds)):
        # print("i: ", i, preds[i].shape)
        if (preds [i].shape [2] != target.shape [2] or preds [i].shape [3] != target.shape [3]):
            # tmp_target = _upsample_like(target,preds[i])
            tmp_target = F.interpolate(target, size=preds [i].size() [2:], mode='bilinear', align_corners=True)
            loss = loss + bce_loss(preds [i], tmp_target)
        else:
            loss = loss + bce_loss(preds [i], target)
        if (i == 0):
            loss0 = loss
    return loss0, loss


def create_inputs(train_dir, val_dir):
    dataset_train = {"name": 'TRAIN-DATA',
                     "im_dir": os.path.join(train_dir, 'im'),
                     "gt_dir": os.path.join(train_dir, 'gt'),
                     "im_ext": ".jpg",
                     "gt_ext": ".png",
                     "cache_dir": os.path.join(train_dir, 'cache')}

    dataset_val = {"name": 'VAL-DATA',
                   "im_dir": os.path.join(val_dir, 'im'),
                   "gt_dir": os.path.join(val_dir, 'gt'),
                   "im_ext": ".jpg",
                   "gt_ext": ".png",
                   "cache_dir": os.path.join(val_dir, 'cache')}

    train_datasets = [dataset_train]
    valid_datasets = [dataset_val]

    return train_datasets, valid_datasets


def upload_file_to_s3(file_path, bucket_name, s3_key):
    s3 = boto3.client('s3')
    s3.upload_file(file_path, bucket_name, s3_key)


def upload_folder_to_s3(local_folder_path, s3_bucket_name, s3_folder_path):
    s3 = boto3.client('s3')

    for root, dirs, files in os.walk(local_folder_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_file_path, local_folder_path)
            # Convert backslashes to forward slashes
            s3_key = os.path.join(s3_folder_path, relative_path).replace("\\", "/")
            s3.upload_file(local_file_path, s3_bucket_name, s3_key)


def download_folder_from_s3(bucket_name, s3_folder, local_folder):
    s3_client = boto3.client('s3')

    # Create the local folder if it doesn't exist
    os.makedirs(local_folder, exist_ok=True)

    # List objects in the S3 folder recursively
    paginator = s3_client.get_paginator('list_objects_v2')
    operation_parameters = {'Bucket': bucket_name, 'Prefix': s3_folder}
    page_iterator = paginator.paginate(**operation_parameters)

    for page in page_iterator:
        if 'Contents' in page:
            for obj in page ['Contents']:
                # Construct the local file path
                local_file_path = os.path.join(local_folder, os.path.relpath(obj ['Key'], s3_folder))

                # Create any necessary local directories
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                # Download the object
                s3_client.download_file(bucket_name, obj ['Key'], local_file_path)


def download_mlruns(s3_bucket, s3_prefix, local_dir):
    # Initialize S3 client
    s3 = boto3.client('s3')

    # Create local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    # Function to download S3 objects recursively
    def download_objects(prefix, local_prefix):
        try:
            response = s3.list_objects_v2(Bucket=s3_bucket, Prefix=prefix, Delimiter='/')
        except ClientError as e:
            if e.response ['Error'] ['Code'] == 'NoSuchKey':
                print(f"The S3 folder '{prefix}' does not exist.")
                return

        for obj in response.get('Contents', []):
            key = obj ['Key']
            local_file_path = os.path.join(local_prefix, os.path.basename(key))

            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            s3.download_file(s3_bucket, key, local_file_path)
            print(f'Downloaded {key} to {local_file_path}')

        for subdir in response.get('CommonPrefixes', []):
            subdir_prefix = subdir ['Prefix']
            subdir_name = os.path.basename(os.path.normpath(subdir_prefix))
            subdir_path = os.path.join(local_prefix, subdir_name)
            os.makedirs(subdir_path, exist_ok=True)
            download_objects(subdir_prefix, subdir_path)

    # Start downloading objects
    download_objects(s3_prefix, local_dir)


def filter_bdy_cond(bdy_, mask, cond):
    cond = cv2.dilate(cond.astype(np.uint8), disk(1))
    labels = label(mask)  # find the connected regions
    lbls = np.unique(labels)  # the indices of the connected regions
    indep = np.ones(lbls.shape [0])  # the label of each connected regions
    indep [0] = 0  # 0 indicate the background region

    boundaries = []
    h, w = cond.shape [0:2]
    ind_map = np.zeros((h, w))
    indep_cnt = 0

    for i in range(0, len(bdy_)):
        tmp_bdies = []
        tmp_bdy = []
        for j in range(0, bdy_ [i].shape [0]):
            r, c = bdy_ [i] [j, 0, 1], bdy_ [i] [j, 0, 0]

            if (np.sum(cond [r, c]) == 0 or ind_map [r, c] != 0):
                if (len(tmp_bdy) > 0):
                    tmp_bdies.append(tmp_bdy)
                    tmp_bdy = []
                continue
            tmp_bdy.append([c, r])
            ind_map [r, c] = ind_map [r, c] + 1
            indep [labels [r, c]] = 0  # indicates part of the boundary of this region needs human correction
        if (len(tmp_bdy) > 0):
            tmp_bdies.append(tmp_bdy)

        # check if the first and the last boundaries are connected
        # if yes, invert the first boundary and attach it after the last boundary
        if (len(tmp_bdies) > 1):
            first_x, first_y = tmp_bdies [0] [0]
            last_x, last_y = tmp_bdies [-1] [-1]
            if ((abs(first_x - last_x) == 1 and first_y == last_y) or
                    (first_x == last_x and abs(first_y - last_y) == 1) or
                    (abs(first_x - last_x) == 1 and abs(first_y - last_y) == 1)
            ):
                tmp_bdies [-1].extend(tmp_bdies [0] [::-1])
                del tmp_bdies [0]

        for k in range(0, len(tmp_bdies)):
            tmp_bdies [k] = np.array(tmp_bdies [k]) [:, np.newaxis, :]
        if (len(tmp_bdies) > 0):
            boundaries.extend(tmp_bdies)

    return boundaries, np.sum(indep)


# this function approximate each boundary by DP algorithm
# https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm
def approximate_RDP(boundaries, epsilon=1.0):
    boundaries_ = []
    boundaries_len_ = []
    pixel_cnt_ = 0

    # polygon approximate of each boundary
    for i in range(0, len(boundaries)):
        boundaries_.append(cv2.approxPolyDP(boundaries [i], epsilon, False))

    # count the control points number of each boundary and the total control points number of all the boundaries
    for i in range(0, len(boundaries_)):
        boundaries_len_.append(len(boundaries_ [i]))
        pixel_cnt_ = pixel_cnt_ + len(boundaries_ [i])

    return boundaries_, boundaries_len_, pixel_cnt_


def relax_HCE(gt, rs, gt_ske, relax=5, epsilon=2.0):
    # print("max(gt_ske): ", np.amax(gt_ske))
    # gt_ske = gt_ske>128
    # print("max(gt_ske): ", np.amax(gt_ske))

    # Binarize gt
    if (len(gt.shape) > 2):
        gt = gt [:, :, 0]

    epsilon_gt = 128  # (np.amin(gt)+np.amax(gt))/2.0
    gt = (gt > epsilon_gt).astype(np.uint8)

    # Binarize rs
    if (len(rs.shape) > 2):
        rs = rs [:, :, 0]
    epsilon_rs = 128  # (np.amin(rs)+np.amax(rs))/2.0
    rs = (rs > epsilon_rs).astype(np.uint8)

    Union = np.logical_or(gt, rs)
    TP = np.logical_and(gt, rs)
    FP = rs - TP
    FN = gt - TP

    # relax the Union of gt and rs
    Union_erode = Union.copy()
    Union_erode = cv2.erode(Union_erode.astype(np.uint8), disk(1), iterations=relax)

    # --- get the relaxed False Positive regions for computing the human efforts in correcting them ---
    FP_ = np.logical_and(FP, Union_erode)  # get the relaxed FP
    for i in range(0, relax):
        FP_ = cv2.dilate(FP_.astype(np.uint8), disk(1))
        FP_ = np.logical_and(FP_, 1 - np.logical_or(TP, FN))
    FP_ = np.logical_and(FP, FP_)

    # --- get the relaxed False Negative regions for computing the human efforts in correcting them ---
    FN_ = np.logical_and(FN, Union_erode)  # preserve the structural components of FN
    ## recover the FN, where pixels are not close to the TP borders
    for i in range(0, relax):
        FN_ = cv2.dilate(FN_.astype(np.uint8), disk(1))
        FN_ = np.logical_and(FN_, 1 - np.logical_or(TP, FP))
    FN_ = np.logical_and(FN, FN_)
    FN_ = np.logical_or(FN_,
                        np.logical_xor(gt_ske, np.logical_and(TP, gt_ske)))  # preserve the structural components of FN

    ## 2. =============Find exact polygon control points and independent regions==============
    ## find contours from FP_
    ctrs_FP, hier_FP = cv2.findContours(FP_.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    ## find control points and independent regions for human correction
    bdies_FP, indep_cnt_FP = filter_bdy_cond(ctrs_FP, FP_, np.logical_or(TP, FN_))
    ## find contours from FN_
    ctrs_FN, hier_FN = cv2.findContours(FN_.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    ## find control points and independent regions for human correction
    bdies_FN, indep_cnt_FN = filter_bdy_cond(ctrs_FN, FN_, 1 - np.logical_or(np.logical_or(TP, FP_), FN_))

    poly_FP, poly_FP_len, poly_FP_point_cnt = approximate_RDP(bdies_FP, epsilon=epsilon)
    poly_FN, poly_FN_len, poly_FN_point_cnt = approximate_RDP(bdies_FN, epsilon=epsilon)

    return poly_FP_point_cnt, indep_cnt_FP, poly_FN_point_cnt, indep_cnt_FN


def compute_hce(pred_root, gt_root, gt_ske_root):
    gt_name_list = glob(pred_root + '/*.png')
    gt_name_list = sorted([x.split('/') [-1] for x in gt_name_list])

    hces = []
    for gt_name in tqdm(gt_name_list, total=len(gt_name_list)):
        gt_path = os.path.join(gt_root, gt_name)
        pred_path = os.path.join(pred_root, gt_name)

        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        ske_path = os.path.join(gt_ske_root, gt_name)
        if os.path.exists(ske_path):
            ske = cv2.imread(ske_path, cv2.IMREAD_GRAYSCALE)
            ske = ske > 128
        else:
            ske = skeletonize(gt > 128)

        FP_points, FP_indep, FN_points, FN_indep = relax_HCE(gt, pred, ske)
        print(gt_path.split('/') [-1], FP_points, FP_indep, FN_points, FN_indep)
        hces.append([FP_points, FP_indep, FN_points, FN_indep, FP_points + FP_indep + FN_points + FN_indep])

    hce_metric = {'names': gt_name_list,
                  'hces': hces}

    file_metric = open(pred_root + '/hce_metric.pkl', 'wb')
    pkl.dump(hce_metric, file_metric)
    # file_metrics.write(cmn_metrics)
    file_metric.close()

    return np.mean(np.array(hces) [:, -1])
