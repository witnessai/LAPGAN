# -*- coding: utf-8 -*-
# @Author: morjio

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2
from torchvision.utils import save_image
from torch.autograd import Variable

#morjio import
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import time
current_time = time.strftime('%Y-%m-%d %H%M%S')
if not os.path.exists('./result/%s' % current_time):
    os.makedirs('./result/%s' % current_time)

from lapgan import LAPGAN, gen_noise


from logger import Logger
logger = Logger('./result/%s/logs' % current_time)



def load_dataset(batch_size=256, download=False):
    """
    The output of torchvision datasets are PILImage images of range [0, 1].
    Transform them to Tensors of normalized range [-1, 1]
    """
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                          download=download,
                                          transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                         download=download,
                                         transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=True, num_workers=2)

    return trainloader, testloader


def train_LAPGAN(LapGan_model, n_level, D_criterions, G_criterions,
                 D_optimizers, G_optimizers, trainloader, n_epoch,
                 batch_size, noise_dim, n_update_dis=1, n_update_gen=1,
                 use_gpu=True, print_every=10, update_max=None):
    """train LAPGAN and print out the losses for Ds and Gs"""
    for G in LapGan_model.Gen_models:
        G.train()
    for D in LapGan_model.Dis_models:
        D.train()
    one = torch.Tensor([1])
    mone = one * -1
    if torch.cuda.is_available():
        one = one.cuda()
        mone = mone.cuda()
    
    iter_G = 5
        
    for epoch in range(n_epoch):

        D_running_losses = [0.0 for i in range(n_level)]
        G_running_losses = [0.0 for i in range(n_level)]


        total_data = len(trainloader)
        for ind, data in enumerate(trainloader, 0):
            # get the inputs from true distribution
            true_inputs, lab = data
            down_imgs = true_inputs.numpy()
            n_minibatch, n_channel, _, _ = down_imgs.shape

            for l in range(n_level):
                # calculate input images for models at the particular level
                if l == (n_level - 1):
                    condi_inputs = None
                    true_inputs = Variable(torch.Tensor(down_imgs))
                    if use_gpu:
                        true_inputs = true_inputs.cuda()
                else:
                    new_down_imgs = []
                    up_imgs = []
                    residual_imgs = []

                    # compute a Laplacian Pyramid
                    for i in range(n_minibatch):
                        down_img = []
                        up_img = []
                        residual_img = []

                        for j in range(n_channel):
                            previous = down_imgs[i, j, :]
                            down_img.append(cv2.pyrDown(previous))
                            up_img.append(cv2.pyrUp(down_img[-1]))
                            residual_img.append(previous - up_img[-1])

                        new_down_imgs.append(down_img)
                        up_imgs.append(up_img)
                        residual_imgs.append(residual_img)

                    down_imgs = np.array(new_down_imgs)
                    up_imgs = np.array(up_imgs)
                    residual_imgs = np.array(residual_imgs)

                    condi_inputs = Variable(torch.Tensor(up_imgs))
                    true_inputs = Variable(torch.Tensor(residual_imgs))
                    if use_gpu:
                        condi_inputs = condi_inputs.cuda()
                        true_inputs = true_inputs.cuda()

                # get inputs for discriminators from generators and real data
                
                
                #update D
                for p in LapGan_model.Dis_models[l].parameters():
                    p.requires_grad = True
                    p.data.clamp_(-0.01, 0.01)                
                #train with real
                D_optimizers[l].zero_grad()
                errD_real = LapGan_model.Dis_models[l](true_inputs, condi_inputs)
                errD_real.backward(one)
                
                #train with fake
                if l == 0: noise_dim = 32*32
                elif l == 1: noise_dim = 16*16
                else: noise_dim = 100
                noise = Variable(gen_noise(batch_size, noise_dim), volatile=False)
                if use_gpu:
                    noise = noise.cuda()
                fake_inputs = LapGan_model.Gen_models[l](noise, condi_inputs)
                errD_fake = LapGan_model.Dis_models[l](fake_inputs, condi_inputs)
                errD_fake.backward(mone)
                D_loss = errD_real - errD_fake
                D_optimizers[l].step()
                
                info = {'D_loss': D_loss.data[0]}
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, epoch*total_data+ind)
                
                #update G

                if ind % iter_G == 0:
                    for p in LapGan_model.Dis_models[l].parameters():
                        p.requires_grad = False
                    G_optimizers[l].zero_grad()
                    noise = Variable(gen_noise(batch_size, noise_dim))
                    if use_gpu:
                        noise = noise.cuda()
                    fake_inputs = LapGan_model.Gen_models[l](noise, condi_inputs)
                    G_loss = LapGan_model.Dis_models[l](fake_inputs, condi_inputs)
                    G_loss.backward(one)
                    G_optimizers[l].step()
                    
                    info = {'G_loss': G_loss.data[0]}
                    for tag, value in info.items():
                        logger.scalar_summary(tag, value, epoch*total_data+ind)
                
                
               
                
                
                
                
                
                D_running_losses[l] += D_loss.data[0]
                G_running_losses[l] += G_loss.data[0]
                if ind % print_every == (print_every - 1):
                    print('[%d, %5d, %d] D loss: %.3f ; G loss: %.3f' %
                          (epoch+1, ind+1, l+1,
                           D_running_losses[l] / print_every,
                           G_running_losses[l] / print_every))
                    D_running_losses[l] = 0.0
                    G_running_losses[l] = 0.0

            if update_max and ind > update_max:
                samples = LapGan_model.generate(32)
                samples = torch.Tensor(samples)
                save_image(samples, './result/%s/epoch%d.png'% (current_time, epoch), normalize=True)
                break

            

    print('Finished Training')


def run_LAPGAN(n_level=3, n_epoch=1, batch_size=256, use_gpu=True,
               dis_lrs=None, gen_lrs=None, noise_dim=100, n_update_dis=1, 
               n_update_gen=1, n_channel=3, n_sample=32,update_max=None):
    # loading data
    trainloader, testloader = load_dataset(batch_size=batch_size)

    # initialize models
    LapGan_model = LAPGAN(n_level, use_gpu, n_channel)

    # assign loss function and optimizer (Adam) to D and G
    D_criterions = []
    G_criterions = []

    D_optimizers = []
    G_optimizers = []

    if not dis_lrs:
        #dis_lrs = [0.0002, 0.0003, 0.001]
        #dis_lrs = [0.02 for i in range(3)]
        dis_lrs = [0.0001, 0.0001, 0.0001]

    if not gen_lrs:
        #gen_lrs = [0.001, 0.005, 0.001]
        #gen_lrs = [0.02 for i in range(3)]
        gen_lrs = [0.0003, 0.0005, 0.003]

    for l in range(n_level):
        #D_criterions.append(nn.BCELoss())
        D_optim = optim.Adam(LapGan_model.Dis_models[l].parameters(),
                             lr=dis_lrs[l], betas=(0.5, 0.999))
        D_optimizers.append(D_optim)
  
        #G_criterions.append(nn.BCELoss())
        G_optim = optim.Adam(LapGan_model.Gen_models[l].parameters(),
                             lr=gen_lrs[l], betas=(0.5, 0.999))
        G_optimizers.append(G_optim)

    train_LAPGAN(LapGan_model, n_level, D_criterions, G_criterions,
                 D_optimizers, G_optimizers, trainloader, n_epoch,
                 batch_size, noise_dim, n_update_dis, n_update_gen,
                 update_max=update_max)
    
    samples = LapGan_model.generate(n_sample)
    samples = torch.Tensor(samples)
    save_image(samples, './result/%s/epoch%d.png'% (current_time, n_epoch), normalize=True)
    return samples.numpy()


if __name__ == '__main__':
    run_LAPGAN(n_epoch=50, update_max=180)
