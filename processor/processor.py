import logging
import numpy as np
import os
import time
import torch
import torch.nn as nn
import cv2
from utils.meter import AverageMeter
from utils.metrics import R1_mAP, R1_mAP_Pseudo
import json
import datetime
try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info('start training')

    if device:
        model.to(device)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        else:
            if cfg.SOLVER.FP16:
                model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()

        model.train()
        for n_iter, (img, vid) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)

            score, feat = model(img, target)
            loss = loss_fn(score, feat, target)

            if cfg.SOLVER.FP16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                optimizer_center.step()

            acc = (score.max(1)[1] == target).float().mean()
            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler.get_lr()[0]))
        scheduler.step()
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))


def do_inference(cfg,
                 model,
                 val_loader_green,
                val_loader_normal,
                 num_query_green,
                 num_query_normal):
    device = "cuda"
    logger = logging.getLogger("reid_baseline.test")
    logger.info("Enter inferencing")

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    val_loader = [val_loader_green, val_loader_normal]
    for index, loader in enumerate(val_loader):
        if index == 0:
            subfix = '1'
            reranking_parameter = [14, 4, 0.4]
            evaluator = R1_mAP(num_query_green, max_rank=200, feat_norm=cfg.TEST.FEAT_NORM,
                               reranking=cfg.TEST.RE_RANKING)
        else:
            subfix = '2'
            reranking_parameter = [10, 3, 0.6]
            evaluator = R1_mAP(num_query_normal, max_rank=200, feat_norm=cfg.TEST.FEAT_NORM,
                               reranking=cfg.TEST.RE_RANKING)

        evaluator.reset()
        DISTMAT_PATH = os.path.join(cfg.OUTPUT_DIR, "distmat_{}.npy".format(subfix))
        QUERY_PATH = os.path.join(cfg.OUTPUT_DIR, "query_path_{}.npy".format(subfix))
        GALLERY_PATH = os.path.join(cfg.OUTPUT_DIR, "gallery_path_{}.npy".format(subfix))

        for n_iter, (img, pid, camid, imgpath) in enumerate(loader):
            with torch.no_grad():
                img = img.to(device)
                if cfg.TEST.FLIP_FEATS == 'on':
                    feat = torch.FloatTensor(img.size(0), 2048).zero_().cuda()
                    for i in range(2):
                        if i == 1:
                            inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()
                            img = img.index_select(3, inv_idx)
                        f = model(img)
                        feat = feat + f
                else:
                    feat = model(img)

                evaluator.update((feat, imgpath))

        data, distmat, img_name_q, img_name_g = evaluator.compute(reranking_parameter)
        np.save(DISTMAT_PATH, distmat)
        np.save(QUERY_PATH, img_name_q)
        np.save(GALLERY_PATH, img_name_g)

        if index == 0:
            data_1 = data

    data_all = {**data_1, **data}
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    with open(os.path.join(cfg.OUTPUT_DIR, 'result_{}.json'.format(nowTime)), 'w',encoding='utf-8') as fp:
        json.dump(data_all, fp)


def do_inference_Pseudo(cfg,
                 model,
                val_loader,
                num_query
                 ):
    device = "cuda"

    evaluator = R1_mAP_Pseudo(num_query, max_rank=200, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    reranking_parameter = [14, 4, 0.4]

    model.eval()
    for n_iter, (img, pid, camid, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            if cfg.TEST.FLIP_FEATS == 'on':
                feat = torch.FloatTensor(img.size(0), 2048).zero_().cuda()
                for i in range(2):
                    if i == 1:
                        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()
                        img = img.index_select(3, inv_idx)
                    f = model(img)
                    feat = feat + f
            else:
                feat = model(img)

            evaluator.update((feat, imgpath))

    distmat, img_name_q, img_name_g = evaluator.compute(reranking_parameter)

    return distmat, img_name_q, img_name_g