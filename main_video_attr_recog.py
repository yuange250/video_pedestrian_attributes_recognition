from __future__ import print_function, absolute_import
import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np
import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler
from tqdm import tqdm

import data_manager
from video_loader import VideoDataset
import transforms as T
import models

from utils import AverageMeter, Logger, save_checkpoint, softmax
from samplers import RandomIdentitySampler

parser = argparse.ArgumentParser(description='Train video model with cross entropy loss')
# Datasets
parser.add_argument('-d', '--dataset', type=str, default='mars',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=224,
                    help="height of an image (default: 224)")
parser.add_argument('--width', type=int, default=112,
                    help="width of an image (default: 112)")
parser.add_argument('--seq-len', type=int, default=6, help="number of images to sample in a tracklet")
parser.add_argument('--max-traclets-len', type=int, default=250, help="number of parts to sample in a tracklet")
parser.add_argument('--gpu-devices', default='0, 1', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--num-instances', type=int, default=4,
                    help="number of instances per identity")
parser.add_argument('--test-batch', default=1, type=int, help="has to be 1")

parser.add_argument('--train-batch', default=64, type=int,
                    help="train batch size")
# decor
parser.add_argument('--sample-margin', type=int, default=4, help="margin for img sample")

# Optimization options

parser.add_argument('--max-epoch', default=1000, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")

parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                    help="initial learning rate, use 0.0001 for rnn, use 0.0003 for pooling and attention")
parser.add_argument('--stepsize', default=100, type=int,
                    help="stepsize to decay learning rate (>0 means this is enabled)")
parser.add_argument('--gamma', default=0.3, type=float,
                    help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")
parser.add_argument('--margin', type=float, default=0.3, help="margin for triplet loss")

# Architecture
parser.add_argument('-a', '--arch', type=str, default='attr_resnet50tp_baseline', help="attr_resnet503d, attr_resnet50tp, attr_resnet50tp_baseline")
parser.add_argument('-model-type', '--model_type', type=str, default='ta', help="tp(temporal pooling), ta(temporal attention), rnn(rnn attention)")

parser.add_argument('--pool', type=str, default='avg', choices=['avg', 'max'])
parser.add_argument('--attr_lens', type=list, default=[[5, 6, 2, 2, 2, 2, 2, 2, 2],[9, 10, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]], help="labels len") # mars attr lens
parser.add_argument('--attr_loss', type=str, default="cropy", help="attributes loss")
parser.add_argument('--htri-only', action='store_true', default=False,
                    help="if this is True, only htri loss is used in training")
# Miscs
parser.add_argument('--print-freq', type=int, default=80, help="print frequency")
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--evaluate', default=False, action='store_true', help="evaluation only")
parser.add_argument('--eval-step', type=int, default=2,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--save-dir', type=str, default='/data/chenzy/models/mars')
parser.add_argument('--use-cpu', action='store_true', help="use cpu")
args = parser.parse_args()
tqdm_enable = False
def attr_main():
    runId = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args.save_dir = os.path.join(args.save_dir, runId)
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    print(args.save_dir)
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    if not args.evaluate:
        sys.stdout = Logger('./log_train_' + runId + '.txt')
    else:
        sys.stdout = Logger('./log_test_' + runId + '.txt')
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(args.dataset))

    dataset = data_manager.init_dataset(name=args.dataset, min_seq_len=args.seq_len, attr=True)
    args.attr_lens = dataset.attr_lens
    args.columns = dataset.columns
    print("Initializing model: {}".format(args.arch))
    # if args.arch == "resnet50ta_attr" or args.arch == "resnet50ta_attr_newarch":
    if args.arch == 'attr_resnet503d':
        model = models.init_model(name=args.arch, attr_lens=args.attr_lens, model_type=args.model_type, num_classes=dataset.num_train_pids, sample_width=args.width,
                                  sample_height=args.height, sample_duration=args.seq_len)
        torch.backends.cudnn.benchmark = False
    else:
        model = models.init_model(name=args.arch, attr_lens=args.attr_lens, model_type=args.model_type)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    if args.dataset == "duke":
        transform_train = T.Compose([
            T.Random2DTranslation(args.height, args.width),
            # T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        transform_test = T.Compose([
            T.Resize((args.height, args.width)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif args.dataset == "mars":
        transform_train = T.Compose([
            T.Random2DTranslation(args.height, args.width),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        transform_test = T.Compose([
            T.Resize((args.height, args.width)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    pin_memory = True if use_gpu else False

    trainloader = DataLoader(
        VideoDataset(dataset.train, seq_len=args.seq_len, sample='random', transform=transform_train, attr=True,
                     attr_loss=args.attr_loss, attr_lens=args.attr_lens),
        sampler=RandomIdentitySampler(dataset.train, num_instances=args.num_instances),
        batch_size=args.train_batch, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,
    )

    queryloader = VideoDataset(dataset.query + dataset.gallery, seq_len=args.seq_len, sample='dense', transform=transform_test, attr=True,
                     attr_loss=args.attr_loss, attr_lens=args.attr_lens)

    start_epoch = args.start_epoch

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    start_time = time.time()
    if args.arch == 'resnet503d':
        torch.backends.cudnn.benchmark = False

    # print("Run attribute pre-training")
    if args.attr_loss == "cropy":
        criterion = nn.CrossEntropyLoss()
    elif args.attr_loss == "mse":
        criterion = nn.MSELoss()

    if args.evaluate:
        print("Evaluate only")
        model_root = "/data/chenzy/models/mars/2019-02-26_21-02-13"
        model_paths = []
        for m in os.listdir(model_root):
            if m.endswith("pth"):
                model_paths.append(m)

        model_paths = sorted(model_paths, key=lambda a:float(a.split("_")[1]), reverse=True)
        # model_paths = ['rank1_2.8755379380596713_checkpoint_ep500.pth']
        for m in model_paths:
            model_path = os.path.join(model_root, m)
            print(model_path)

            old_weights = torch.load(model_path)
            new_weights = model.module.state_dict()
            for k in new_weights:
                if k in old_weights:
                    new_weights[k] = old_weights[k]
            model.module.load_state_dict(new_weights)
            avr_acc = attr_test(model, criterion, queryloader, use_gpu)
            # break
        # test(model, queryloader, galleryloader, args.pool, use_gpu)
        return
    if use_gpu:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=args.lr,
                                     weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                     weight_decay=args.weight_decay)
    # avr_acc = attr_test(model, criterion, queryloader, use_gpu)
    if args.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

    best_avr = 0
    no_rise = 0
    for epoch in range(start_epoch, args.max_epoch):
        print("==> Epoch {}/{}".format(epoch + 1, args.max_epoch))
        attr_train(model, criterion, optimizer, trainloader, use_gpu)

        if args.stepsize > 0: scheduler.step()

        if args.eval_step > 0 and ((epoch + 1) % (args.eval_step) == 0 or (epoch + 1) == args.max_epoch):
            avr_acc = attr_test(model, criterion, queryloader, use_gpu)
            print("avr", avr_acc)
            if avr_acc > best_avr:
                no_rise = 0
                print("==> Test")
                best_avr = avr_acc
                if use_gpu:
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()
                torch.save(state_dict, osp.join(args.save_dir, "avr_" + str(avr_acc) +  '_checkpoint_ep' + str(epoch + 1) + '.pth'))
            else:
                no_rise += 1
                print("no_rise:", no_rise)
                if no_rise > 20:
                    break
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

def attr_train(model, criterion, optimizer, trainloader, use_gpu):
    model.train()
    losses = AverageMeter()
    attr_lens = args.attr_lens[0] + args.attr_lens[1]

    for batch_idx, (imgs, pids, _, attrs) in enumerate(trainloader):
        # torch.cuda.empty_cache()
        if use_gpu:
            imgs = imgs.cuda()
            if args.attr_loss == "mse":
                attrs = [a.cuda() for a in attrs]
            else:
                attrs = [a.view(-1).cuda() for a in attrs]
        outputs = model(imgs)
        loss = criterion(outputs[0], attrs[0]) / attr_lens[0]
        for i in range(1, len(outputs)):
            loss += criterion(outputs[i], attrs[i]) / attr_lens[i]
        losses.update(loss.item(), pids.size(0))
        # if loss.item() > backward_thresold:
        # loss = loss / (len(args.attr_lens[0]) + len(args.attr_lens[1]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % args.print_freq == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(batch_idx + 1, len(trainloader), losses.val, losses.avg))
    print("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(batch_idx + 1, len(trainloader), losses.val, losses.avg))
    return losses.avg
def attr_test(model, criterion, testloader, use_gpu):
    columns = args.columns
    # accs = np.array([0 for _ in range(len(args.attr_lens))])
    accs = np.array([0 for _ in range(len(args.attr_lens[0]) + len(args.attr_lens[1]))])
    num = 0
    y_preds = [[] for _ in range(len(args.attr_lens[0]) + len(args.attr_lens[1]))]
    y_trues = [[] for _ in range(len(args.attr_lens[0]) + len(args.attr_lens[1]))]


    with torch.no_grad():
        model.eval()
        losses = AverageMeter()
        for batch_idx, (imgs, pids, _, attrs, img_path) in enumerate(tqdm(testloader)):
            # if batch_idx > 100:
            #     break
            if use_gpu:
                if args.attr_loss == "mse":
                    attrs = [a.cuda() for a in attrs]
                else:
                    attrs = [a.view(-1).cuda() for a in attrs]

            if len(attrs) >= len(args.attr_lens[0]) + len(args.attr_lens[1]):
                num += 1
                outputs = model(imgs)
                outputs = [torch.mean(out, 0).view(1, -1) for out in outputs]
                loss = criterion(outputs[0], attrs[0])
                for i in range(1, len(outputs)):
                    loss += criterion(outputs[i], attrs[i])
                # losses.update(loss.item(), pids.size(0))
                losses.update(loss.item(), 1)
                for i in range(len(outputs)):
                    outs = outputs[i].cpu().numpy()
                    # outs = torch.mean(outs, 0)
                    if args.attr_loss == "mse":
                        accs[i] += np.sum(np.argmax(outs, 1) == np.argmax(attrs[i].cpu().numpy(), 1))
                    elif args.attr_loss == "cropy":
                        y_preds[i].append(np.argmax(outs, 1))
                        y_trues[i].append(attrs[i].cpu().numpy())
                        accs[i] += np.sum(np.argmax(outs, 1) == attrs[i].cpu().numpy())


        print("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(batch_idx + 1, len(testloader), losses.val, losses.avg))
        accs = accs / num
        avr = np.mean(accs)
        f1_scores_macros = [f1_score(y_pred=y_preds[i], y_true=y_trues[i], average='macro') for i in [0, 1] + list(
            range(len(args.attr_lens[0]), len(args.attr_lens[0]) + len(args.attr_lens[1])))]
        colum_str = "|".join(["%15s" % c for c in columns])
        acc_str = "|".join(["%15f" % acc for acc in accs])
        f1_scores_macros_str = "|".join(["%15f" % f for f in f1_scores_macros])
        print(colum_str)
        print(acc_str)
        print(f1_scores_macros_str)
        origin_avr = np.mean(accs[[0, 1] + list(range(len(args.attr_lens[0]), len(args.attr_lens[0]) + len(args.attr_lens[1])))])
        origin_avr_without_motion_angle = np.mean(accs[len(args.attr_lens[0]):])
        f1_scores_macro = np.mean(f1_scores_macros)
        f1_scores_macro_without_motion_angle = np.mean(f1_scores_macros[2:])
        f1_scores_micro = np.mean([f1_score(y_pred=y_preds[i], y_true=y_trues[i], average='micro') for i in [0, 1] + list(range(len(args.attr_lens[0]), len(args.attr_lens[0]) + len(args.attr_lens[1])))])

        print("avr acc", origin_avr)
        print("f1_score_macro", f1_scores_macro)
        print("f1_score_micro", f1_scores_micro)

        print("avr acc id-related", origin_avr_without_motion_angle)
        print("f1_score_macro id-related", f1_scores_macro_without_motion_angle)

        print("avr", avr)
        return origin_avr + f1_scores_macro

if __name__ == '__main__':
    attr_main()
    # paper_fig_plot()




