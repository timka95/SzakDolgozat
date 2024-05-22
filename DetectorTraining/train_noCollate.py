import argparse
import os
import shutil
import time

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import cv2

import dataloader_SimpliestDetector
# import PyramidNet_modified as PYRM
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

#import stage1_Loader
import matplotlib.pyplot as plt
from tensorboard_logger import configure, log_value

from lcnn.models.HT import hough_transform
from lcnn.models.houghtransform_RAL_TIMKA import ht
from lcnn.models.multitask_learner import MultitaskHead
from Class_losses.MSE_LOSS import MseLoss

import cv2

from lcnn.config import C, M
import pynvml

### GPU Memory ###
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(1)
meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
used_mem_in_GB = meminfo.used / 1024 ** 3
print(f"BEFORE Everything:  {used_mem_in_GB} GB")
pynvml.nvmlShutdown()
### GPU Memory ###


print("///////////////// TRAIN GOOD //////////////////")

import pickle
from datetime import datetime
import torch.nn as nn
import numpy as np

from torch import autograd

from torch.autograd import Variable

from Class_losses.losses_new import MultiLoss_new
# from Class_losses.sampler_new import HichemSampler
# from Class_losses.sampler_new import NghSampler2
# from Class_losses.triplet_loss_new import  TripletLoss

from Class_losses.MSE_loss_new import MSELoss_new

import scipy.io as sio
import os

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--net_type', default='PyramidNet', type=str,
                    help='networktype: resnet, resnext, densenet, pyamidnet, and so on')

parser.add_argument('--print-freq', '-p', default=1, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')

# parser.add_argument('--resume', default="/home/ntimea/l2d2/IN_OUT_DATA/INPUT_NETWEIGHT/checkpoint_0829_SMALL.pth.tar", type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

#parser.add_argument('--resume', default="/home/ntimea/l2d2/TRAIN_Planned_0828/runs_TIMKA_Planned/Omni_Stage1_detection_512_wd1e_minus_6_basic/checkpoint_90.pth.tar", type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
# parser.add_argument('--resume', default='/home/hichem/MainProjects/R2D2_RAL_network_style/Pycharm_projects/training_128_detection_RAL/runs/HT_128_detect_mse_rel_b20/checkpoint_97.pth.tar', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

parser.add_argument('--resume', default="/project/ntimea/l2d2/TRAIN_Github_DOCU_1022/WORKED_May11/gyak/TEST1/checkpoint_100.pth.tar", type=str, metavar='PATH', help='path to latest checkpoint (default: none)')


parser.add_argument('--epochs', default=101, type=int, metavar='N', help='number of total epochs to run')
# parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.025, type=float, metavar='LR', help='initial learning rate')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-6, type=float, metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument('-b', '--batch_size', default=1, type=int, metavar='N', help='mini-batch size (default: 1)')

parser.add_argument('--imagessave', default='/project/ntimea/l2d2/TRAIN_Github_DOCU_1022/WORKED_May11/IMAGES/TEST2/', type=str,
                    help='name of experiment')

parser.add_argument('--expname', default='M2_EVAL_SOLD_05_M2', type=str,
                    help='name of experiment')




# TIMKAAA
device_name = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
if torch.cuda.is_available():
    device_name = "cuda"
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(0)
    print("Let's use", torch.cuda.device_count(), "GPU(s)!")
else:
    print("CUDA is not available")
device = torch.device(device_name)

# TIMKAA


parser.set_defaults(bottleneck=False)
parser.set_defaults(verbose=True)

best_err1 = 100
best_err5 = 100
global step
step = 0
global filetowrite


def main():
    global args, best_err1, best_err5
    global filetowrite
    global step
    ####################################Dimension of the input image--> hough #######################################################################
    global Img_dimension
    Img_dimension = 128
    Img_dimension1 = 128
    theta_res = 3
    ####################################Dimension of the input image--> hough #######################################################################

    args = parser.parse_args()
    # if args.tensorboard:
    configure("/project/ntimea/l2d2/TRAIN_Github_DOCU_1022/WORKED_May11/gyak/%s" % (args.expname))
    namefile = "/project/ntimea/l2d2/TRAIN_Github_DOCU_1022/WORKED_May11/gyak/%s/consol.txt" % (args.expname)
    filetowrite = open(namefile, 'w')

    # normalize = transforms.Normalize(mean=[0.54975, 0.60135, 0.63516], std=[0.34359,0.34472,0.34718])
    normalize = transforms.Normalize(mean=[0.492967568115862], std=[0.272086182765434])
    # mean_image =0.492967568115862
    # std_image = 0.272086182765434

    train_loader = dataloader_SimpliestDetector.DataLoaderMatrix()
    # transform_train = transforms.Compose([transforms.ToTensor(), normalize, ])
    #
    # train_loader = torch.utils.data.DataLoader(
    #     # stage1_Loader.LyftLoader('/home/hichem/OmniProject/data/equirectangular_512_training_berm_1/',img_count =0,  train=True, transform=transform_train,bsize= args.batch_size),
    #     # batch_size=1, shuffle=True, num_workers=4, pin_memory=False)
    #
    #     stage1_Loader.LyftLoader('/home/ntimea/l2d2/DATASETS/Lyft_Bin/training/', img_count=0, train=True,
    #                              transform=transform_train, bsize=args.batch_size),
    #     # stage1_Loader.LyftLoader('/home/ntimea/l2d2/equirectangular_512_training_berm_1/',img_count =0,  train=True, transform=transform_train,bsize= args.batch_size),
    #     batch_size=1, shuffle=True, num_workers=4, pin_memory=False)

    print("=> creating model")

    if os.path.isfile(C.io.vote_index):
        print('load vote_index ... ')
        vote_index = sio.loadmat(C.io.vote_index)['vote_index']
    else:
        print('compute vote_index ... ')
        vote_index = hough_transform(rows=128, cols=128, theta_res=3, rho_res=1)
        sio.savemat(C.io.vote_index, {'vote_index': vote_index})
    vote_index = torch.from_numpy(vote_index).float().contiguous().to(device)
    print('vote_index loaded', vote_index.shape)

    model = ht(
        head=lambda c_in, c_out: MultitaskHead(c_in, c_out),
        vote_index=vote_index,
        batches_size=args.batch_size,

        depth=M.depth,
        num_stacks=M.num_stacks,
        num_blocks=M.num_blocks,
        num_classes=sum(sum(M.head_size, [])),
    )

    def weight_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(m.bias)

    

    model = model.to('cuda')
    model = torch.nn.DataParallel(model).cuda()
    model.apply(weight_init)

    # print(model)
    print('the number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    # define loss function (criterion) and optimizer

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)

    optimizer_hough = torch.optim.SGD(model.parameters(), args.lr,
                                      momentum=args.momentum,
                                      weight_decay=args.weight_decay, nesterov=True)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch'] + 1
            step = checkpoint['step']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            optimizer_hough.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return 0

    cudnn.benchmark = True

    default_loss = """MultiLoss_new(1, MSELoss_new(), )"""
    # default_loss = """MultiLoss_new(1, MSELoss_new())"""

    MultiLoss = eval(default_loss)

    print("\n>> Creating loss functions")

    # torch.autograd.set_detect_anomaly(True)

    # for param in model.parameters():
    #    print(param.dtype)

    for epoch in range(args.start_epoch, args.epochs + 1):

        def calculate_gamma(epoch):
            gamma = 0.25
            return gamma
        # if args.distributed:
        # train_sampler.set_epoch(epoch)
        print()
        print("here 2 adjust_learning_rate(optimizer, epoch)")
        print()
        # train for one epoch

        #scheduler = StepLR(optimizer, step_size=1, gamma=0.999)

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, factor=0.1, patience=10, verbose=True
        # )

        adjust_learning_rate(optimizer, epoch)
        train(train_loader, model, optimizer, optimizer_hough, epoch, MultiLoss, args.batch_size)
        save_checkpoint2({
            'epoch': epoch,
            'step': step,
            'arch': args.net_type,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'optimizer_hough': optimizer_hough.state_dict(),
        }, epoch)
        print("here 3)")

    filetowrite.close()
    # evaluate(val_loader, model)
    # print ('Best accuracy (top-1 and 5 error):', best_err1, best_err5)
    print('end')


def saveimage(tensor, imagename):
    imagepath = args.imagessave

    ii = np.transpose(tensor[0, :, :, :].cpu().detach().numpy(), axes=[1, 2, 0])
    normalized = (ii - np.min(ii)) / (np.max(ii) - np.min(ii))
    print("MIN", np.min(ii))
    print("MAX", np.max(ii))


    cv2.imwrite(imagepath + str(imagename) + '.png',  normalized*255)



def train(train_loader, model, optimizer, optimizer_hough, epoch, MultiLoss, batches_size):
    global step
    global filetowrite
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    h_losses = AverageMeter()
    mse_losses = AverageMeter()
    h_mse_losses = AverageMeter()
    triplet_losses = AverageMeter()
    # switch to train mode
    model.train()
    # torch.set_grad_enabled(True)
    lossdata = 0
    mse_lossdata = 0

    h_lossdata = 0
    h_mse_lossdata = 0
    triplet_lossdata = 0

    end = time.time()
    #current_LR_hough = get_learning_rate(optimizer_hough)[0]
    batchnumber = train_loader.batchnumber()


    losses = AverageMeter()
    print("All Batch:", batchnumber)
    FirstVisul = True

    for CurrentBatch in range(batchnumber):
        # measure data loading time

        # TIMKA smaller data
        # if i >= 10:  # Change this value to the desired number of iterations
        # break

        # TIMKA smaller data

        data_time.update(time.time() - end)

        optimizer.zero_grad()

        data = train_loader.detectorarray(CurrentBatch)
        allimage = data["images"]
        batch_loss = 0.0  # Initialize loss for the current batch

        current_LR = get_learning_rate(optimizer)[0]

        for image_num in range(len(allimage)):
            
            imagename = data["imagenames"][image_num]
            input1 = data["images"][image_num]["image"].cuda()
            target_lines = data["images"][image_num]["GT"].cuda()
            #hough_target_lines = data["images"][image_num]["Hough_GT"].cuda()     

            

            input1 = input1.reshape(1, 1, 512, 512)
            target_lines = target_lines.reshape(1, 1, 512, 512)
            #hough_target_lines = hough_target_lines.reshape(1, 1, 728, 240)

            line_detected, hough_line_detected = model.forward(input1)
            
            #line_detected_target, hough_line_detected_target = model.forward(target_lines)

            ############## VISUL ##############
            if True:
                PathSave =  "/project/ntimea/l2d2/TRAIN_Github_DOCU_1022/WORKED_May11/IMAGES/M2_EVAL_SOLD_05_M2/"
                if not os.path.exists(PathSave):
                    os.makedirs(PathSave)

                PathSave =  "/project/ntimea/l2d2/TRAIN_Github_DOCU_1022/WORKED_May11/IMAGES/M2_EVAL_SOLD_05_M2/input_img/"
                if not os.path.exists(PathSave):
                    os.makedirs(PathSave)
                input_img = input1.cpu().detach().numpy()
                input_img = np.squeeze(input_img)
                cv2.imwrite(PathSave + imagename + '.png', input_img)
                
                PathSave =  "/project/ntimea/l2d2/TRAIN_Github_DOCU_1022/WORKED_May11/IMAGES/M2_EVAL_SOLD_05_M2/GT/"
                if not os.path.exists(PathSave):
                    os.makedirs(PathSave)
                visulGT = target_lines.cpu().detach().numpy()

                visulGT = (visulGT - visulGT.min()) / (visulGT.max() - visulGT.min())
                visulGT = np.squeeze(visulGT)
                #cv2.imwrite(PathSave + imagename + '.png', visulGT)

            if True:
                PathSave = "/project/ntimea/l2d2/TRAIN_Github_DOCU_1022/WORKED_May11/IMAGES/M2_EVAL_SOLD_05_M2/"
                PathSave_orig = PathSave + str(epoch)

                PathSave_orig = PathSave_orig + "/"

                PathSave_line_detected = PathSave_orig + "line_detected/"
                PathSave_norm = PathSave_orig + "line_detected_norm/"
                PathSave_hough = PathSave_orig + "line_detected_hough/"

                if not os.path.exists(PathSave_orig):
                    os.makedirs(PathSave_orig)
                    os.makedirs(PathSave_line_detected)
                    os.makedirs(PathSave_norm)
                    os.makedirs(PathSave_hough)


                visulimg = line_detected.cpu().detach().numpy()
                visu_hough = hough_line_detected.cpu().detach().numpy()
                
                visulimg_norm = (visulimg - visulimg.min()) / (visulimg.max() - visulimg.min())
                visulimg_norm = np.squeeze(visulimg_norm)

                visulimg = np.squeeze(visulimg)
                visu_hough = np.squeeze(visu_hough)

                

                cv2.imwrite(PathSave_line_detected + imagename + '.png', visulimg * 255)
                cv2.imwrite(PathSave_hough + imagename + '.png', visu_hough * 255)
                cv2.imwrite(PathSave_norm + imagename + '.png', visulimg_norm * 255)
                
                endvisul = 0


                ############## VISUL ##############
            
            


            #saveimage(line_detected,imagename)

            inputs = {
                "lines": line_detected,
                "hough_lines": hough_line_detected,
                "lines_gt": target_lines,
                #"hough_gt": hough_target_lines,
                "batches_size": batches_size,
            }

    #         allvars = dict(inputs)
    #         #lines_loss, details, h_loss, h_details = MultiLoss.forward_all(**allvars)
    #         Mse_loss_class = MseLoss()
    #         lines_loss = Mse_loss_class.mseloss(line_detected, target_lines, imagename)
            
    #         batch_loss +=  lines_loss


    #     batch_loss.backward()
    #     if 'allbatchloss' in locals():
    #         allbatchloss = allbatchloss + batch_loss
    #     else:
    #         allbatchloss = batch_loss

    #     optimizer.step()
    #     #scheduler.step()

    #     losses.update(batch_loss.data.detach().cpu().numpy(), 1)

    #     batch_time.update(time.time() - end)
    #     end = time.time()

    #     if CurrentBatch % args.print_freq == 0 and args.verbose == True:
    #         print('Epoch: [{0}/{1}][{2}/{3}]\t'
    #               'LR: {LR:.6f}\t'
    #               'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
    #               'Loss {loss.val:.7f} ({loss.avg:.7f})\t'
    #         .format(
    #             epoch, args.epochs, CurrentBatch, batchnumber, LR=current_LR, batch_time=batch_time,
    #             loss=losses))
    #         print('Epoch: [{0}/{1}][{2}/{3}]\t'
    #               'LR: {LR:.6f}\t'
    #               'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
    #               'Loss {loss.val:.7f} ({loss.avg:.7f})\t'
    #         .format(
    #             epoch, args.epochs, CurrentBatch, batchnumber, LR=current_LR, batch_time=batch_time,
    #             loss=losses), file=filetowrite)
    # batch_loss_average = allbatchloss /batchnumber
    # log_value('train_loss', batch_loss_average, epoch)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    directory = "gyak/%s/" % (args.expname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'gyak/%s/' % (args.expname) + 'model_best.pth.tar')


def save_checkpoint2(state, epoch_s, filename='checkpoint_'):
    directory = "/project/ntimea/l2d2/TRAIN_Github_DOCU_1022/WORKED_May11/gyak/%s/" % (args.expname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename1 = directory + filename + str(epoch_s) + '.pth.tar'
    torch.save(state, filename1)
    filename2 = directory + filename + '.pth.tar'
    torch.save(state, filename2)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Adjusts the learning rate based on the epoch"""
    # Calculate the learning rate according to your schedule
    lr = args.lr

    if lr < 0.0001:
        lr = 0.0001

    # Set the learning rate for all parameter groups
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr
    log_value('learning_rate', lr, epoch)


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res


if __name__ == '__main__':
    main()
