# Import necessary libraries
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn.functional as F
from dataset import get_loader
import math
from Models.USOD_Net import ImageDepthNet
import os
import pytorch_iou
import pytorch_ssim

# Define loss functions
criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy with Logits
ssim_loss = pytorch_ssim.SSIM(window_size=7, size_average=True)  # SSIM loss
iou_loss = pytorch_iou.IOU(size_average=True)  # IOU loss

# Function to save loss values to a file
def save_loss(save_dir, whole_iter_num, epoch_total_loss, epoch_loss, epoch):
    # Open file in append mode
    fh = open(save_dir, 'a')
    # Convert losses to string for writing
    epoch_total_loss = str(epoch_total_loss)
    epoch_loss = str(epoch_loss)
    # Write epoch and loss information
    fh.write('until_' + str(epoch) + '_run_iter_num' + str(whole_iter_num) + '\n')
    fh.write(str(epoch) + '_epoch_total_loss' + epoch_total_loss + '\n')
    fh.write(str(epoch) + '_epoch_loss' + epoch_loss + '\n')
    fh.write('\n')
    fh.close()

# Function to adjust the learning rate of the optimizer
def adjust_learning_rate(optimizer, decay_rate=.1):
    update_lr_group = optimizer.param_groups
    for param_group in update_lr_group:
        print('before lr: ', param_group['lr'])  # Print current learning rate
        param_group['lr'] = param_group['lr'] * decay_rate  # Update learning rate
        print('after lr: ', param_group['lr'])  # Print updated learning rate
    return optimizer

# Function to save the current learning rate to a file
def save_lr(save_dir, optimizer):
    update_lr_group = optimizer.param_groups[0]
    fh = open(save_dir, 'a')
    fh.write('encode:update:lr' + str(update_lr_group['lr']) + '\n')
    fh.write('decode:update:lr' + str(update_lr_group['lr']) + '\n')
    fh.write('\n')
    fh.close()

# Main training function that spawns processes for distributed training
def train_net(num_gpus, args):
    mp.spawn(main, nprocs=num_gpus, args=(num_gpus, args))

# Function to compute BCE and SSIM loss
def bce_ssim_loss(pred, target):
    bce_out = criterion(pred, target)  # Compute BCE loss
    ssim_out = 1 - ssim_loss(pred, target)  # Compute SSIM loss
    loss = bce_out + ssim_out  # Total loss
    return loss

# Function to compute BCE and IOU loss
def bce_iou_loss(pred, target):
    bce_out = criterion(pred, target)  # Compute BCE loss
    iou_out = iou_loss(pred, target)  # Compute IOU loss
    loss = bce_out + iou_out  # Total loss
    return loss

# Function to compute Dice loss
def dice_loss(score, target):
    target = target.float()  # Convert target to float
    smooth = 1e-5  # Smoothing factor to avoid division by zero
    intersect = torch.sum(score * target)  # Intersection
    y_sum = torch.sum(target * target)  # Sum of target
    z_sum = torch.sum(score * score)  # Sum of score
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)  # Dice loss calculation
    loss = 1 - loss  # Return the final loss
    return loss

# Main function for training the network
def main(local_rank, num_gpus, args):
    cudnn.benchmark = True  # Enable benchmark mode for faster training
    dist.init_process_group(backend='nccl', init_method=args.init_method, world_size=num_gpus, rank=local_rank)  # Initialize distributed training
    torch.cuda.set_device(local_rank)  # Set the device for the current process
    net = ImageDepthNet(args)  # Initialize the network
    net.train()  # Set the network to training mode
    net.cuda()  # Move the network to GPU
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)  # Convert to synchronized batch normalization
    net = torch.nn.parallel.DistributedDataParallel(  # Wrap the network for distributed training
        net,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True)

    # Separate parameters for different learning rates
    base_params = [params for name, params in net.named_parameters() if ("backbone" in name)]
    other_params = [params for name, params in net.named_parameters() if ("backbone" not in name)]

    # Initialize optimizer with different learning rates for different parameter groups
    optimizer = optim.Adam([{'params': base_params, 'lr': args.lr * 0.1},
                            {'params': other_params, 'lr': args.lr}])
    train_dataset = get_loader(args.trainset, args.data_root, args.img_size, mode='train')  # Load training dataset

    # Set up distributed sampler and data loader
    sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=num_gpus,
        rank=local_rank,
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=6,
                                               pin_memory=True,
                                               sampler=sampler,
                                               drop_last=True,
                                               )

    print('''
        Starting training:
            Train steps: {}
            Batch size: {}
            Learning rate: {}
            Training size: {}
        '''.format(args.train_steps, args.batch_size, args.lr, len(train_loader.dataset)))

    N_train = len(train_loader) * args.batch_size  # Total number of training samples

    # Create directory to save model if it doesn't exist
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)

    whole_iter_num = 0  # Initialize iteration number
    iter_num = math.ceil(len(train_loader.dataset) / args.batch_size)  # Calculate number of iterations per epoch
    for epoch in range(args.epochs):

        print('Starting epoch {}/{}.'.format(epoch + 1, args.epochs))
        print('epoch:{0}-------lr:{1}'.format(epoch + 1, args.lr))

        epoch_total_loss = 0  # Initialize total loss for the epoch
        epoch_loss = 0  # Initialize loss for the epoch

        for i, data_batch in enumerate(train_loader):
            if (i + 1) > iter_num: break  # Break if the number of iterations exceeds

            # Unpack data batch
            images, depths, label_224, label_14, label_28, label_56, label_112, \
            contour_224, contour_14, contour_28, contour_56, contour_112 = data_batch
            
            # Move data to GPU
            images, depths, label_224, contour_224 = Variable(images.cuda(local_rank, non_blocking=True)), \
                                                     Variable(depths.cuda(local_rank, non_blocking=True)), \
                                                     Variable(label_224.cuda(local_rank, non_blocking=True)), \
                                                     Variable(contour_224.cuda(local_rank, non_blocking=True))

            label_14, label_28, label_56, label_112 = Variable(label_14.cuda()), Variable(label_28.cuda()), \
                                                      Variable(label_56.cuda()), Variable(label_112.cuda())

            contour_14, contour_28, contour_56, contour_112 = Variable(contour_14.cuda()), \
                                                              Variable(contour_28.cuda()), \
                                                              Variable(contour_56.cuda()), Variable(contour_112.cuda())

            outputs_saliency = net(images, depths)  # Forward pass through the network

            # Unpack outputs
            d1, d2, d3, d4, d5, db, ud2, ud3, ud4, ud5, udb = outputs_saliency

            # Compute various losses
            bce_loss1 = criterion(d1, label_224)
            bce_loss2 = criterion(d2, label_112)
            bce_loss3 = criterion(d3, label_56)
            bce_loss4 = criterion(d4, label_28)
            bce_loss5 = criterion(d5, label_14)
            bce_loss6 = criterion(db, label_14)

            iou_loss1 = bce_iou_loss(d1,  label_224)
            iou_loss2 = bce_iou_loss(ud2, label_224)
            iou_loss3 = bce_iou_loss(ud3, label_224)
            iou_loss4 = bce_iou_loss(ud4, label_224)
            iou_loss5 = bce_iou_loss(ud5, label_224)
            iou_loss6 = bce_iou_loss(udb, label_224)

            c_loss1 = bce_ssim_loss(d1,  contour_224)
            c_loss2 = bce_ssim_loss(ud2, label_224)
            c_loss3 = bce_ssim_loss(ud3, label_224)
            c_loss4 = bce_ssim_loss(ud4, label_224)
            c_loss5 = bce_ssim_loss(ud5, label_224)
            c_loss6 = bce_ssim_loss(udb, label_224)

            d_loss1 = dice_loss(d1,   label_224)
            d_loss2 = dice_loss(ud2,  label_224)
            d_loss3 = dice_loss(ud3,  label_224)
            d_loss4 = dice_loss(ud4,  label_224)
            d_loss5 = dice_loss(ud5,  label_224)
            d_loss6 = dice_loss(udb,  label_224)

            # Calculate total losses
            BCE_total_loss = bce_loss1 + bce_loss2 + bce_loss3 + bce_loss4 + bce_loss5 + bce_loss6
            IoU_total_loss = iou_loss1 + iou_loss2 + iou_loss3 + iou_loss4 + iou_loss5 + iou_loss6
            Edge_total_loss = c_loss1 + c_loss2 + c_loss3 + c_loss4 + c_loss5 + c_loss6
            Dice_total_loss = d_loss1 + d_loss2 + d_loss3 + d_loss4 + d_loss5 + d_loss6
            total_loss = Edge_total_loss + BCE_total_loss + IoU_total_loss + Dice_total_loss

            epoch_total_loss += total_loss.cpu().data.item()  # Accumulate total loss for the epoch
            epoch_loss += bce_loss1.cpu().data.item()  # Accumulate BCE loss for the epoch

            # Print loss information
            print(
                'whole_iter_num: {0} --- {1:.4f} --- total_loss: {2:.6f} --- bce loss: {3:.6f} --- e loss: {4:.6f}'.format(
                    (whole_iter_num + 1), (i + 1) * args.batch_size / N_train, total_loss.item(), bce_loss1.item(), c_loss1.item()
                    ))

            optimizer.zero_grad()  # Zero the gradients

            total_loss.backward()  # Backpropagation

            optimizer.step()  # Update the weights
            whole_iter_num += 1  # Increment iteration number

            # Save model if training steps reached
            if (local_rank == 0) and (whole_iter_num == args.train_steps):
                torch.save(net.state_dict(),
                           args.save_model_dir + 'UVST.pth')

            # Return if training steps reached
            if whole_iter_num == args.train_steps:
                return 0

            # Adjust learning rate at specified steps
            if whole_iter_num == args.stepvalue1 or whole_iter_num == args.stepvalue2:
                optimizer = adjust_learning_rate(optimizer, decay_rate=args.lr_decay_gamma)
                save_dir = './loss.txt'
                save_lr(save_dir, optimizer)
                print('have updated lr!!')

        print('Epoch finished ! Loss: {}'.format(epoch_total_loss / iter_num))  # Print epoch loss
        save_lossdir = './loss.txt'
        save_loss(save_lossdir, whole_iter_num, epoch_total_loss / iter_num, epoch_loss / iter_num, epoch + 1)  # Save loss information

