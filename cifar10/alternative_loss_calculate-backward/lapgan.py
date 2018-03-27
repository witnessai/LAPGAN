# -*- coding: utf-8 -*-
# @Author: morjio

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

from torch.autograd import Variable


class Discriminator_zero(nn.Module):

    def __init__(self):
        super(Discriminator_zero, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=5)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=5, stride=2)
        self.sz = ((32-5+1)-5)//2+1
        self.fc1 = nn.Linear(128*self.sz*self.sz, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.drop1 = nn.Dropout(0.5)

    def forward(self, x, condi_x=None):
        #print(x.size(), condi_x.size())
        ## how to add condition information?
        ## answer: condition information(low-pass) plus residual image(high-pass)
        x = x + condi_x
        #print(x.size())
        x = self.bn1(F.leaky_relu(self.conv1(x)))
        x = self.bn2(F.leaky_relu(self.conv2(x)))
        #print(x.size())
        x = x.view(-1, 128*self.sz*self.sz)
        #x = self.drop1(x)
        x = F.sigmoid(self.fc1(x))
        # wgan format:
        # x = self.fc1(x)
        # x = x.mean(0).view(1)
        return x



class Discriminator_one(nn.Module):
    def __init__(self):
        super(Discriminator_one, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=2)
        self.sz = ((16-5+1)-5)//2+1
        self.fc1 = nn.Linear(64*self.sz*self.sz, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.drop1 = nn.Dropout(0.5)
        

    def forward(self, x, condi_x=None):
        x = x + condi_x
        x = self.bn1(F.leaky_relu(self.conv1(x)))
        x = self.bn2(F.leaky_relu(self.conv2(x)))
        x = x.view(-1, 64*self.sz*self.sz)
        #x = self.drop1(x)
        x = F.sigmoid(self.fc1(x))
        # wgan format
        # x = self.fc1(x)
        # x = x.mean(0).view(1)

        return x



class Discriminator_two(nn.Module):
    def __init__(self):
        super(Discriminator_two, self).__init__()
        self.fc1 = nn.Linear(3*8*8, 600)
        self.fc2 = nn.Linear(600, 600)
        self.fc3 = nn.Linear(600, 1)
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)

    def forward(self, x, condi_x=None):
        x = x.view(-1, 3*8*8)
        #x = self.drop1(F.leaky_relu(self.fc1(x)))
        #x = self.drop2(F.leaky_relu(self.fc2(x)))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        # wgan format
        # x = self.fc3(x)
        # x = x.mean(0).view(1)

        return x


class Generator_zero(nn.Module):

    def __init__(self):
        super(Generator_zero, self).__init__()
        self.conv1 = nn.ConvTranspose2d(3+1, 128, kernel_size=3, padding=1)
        self.conv2 = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.ConvTranspose2d(128, 3, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)

    def forward(self, x, condi_x=None):
        #condi_x is low-pass image, and x is noise
        #print(x.size())
        x = x.view(-1, 1, 32, 32)
        #print(condi_x.size(), x.size())
        x = torch.cat((condi_x, x), 1)
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.conv3(x)
        #print(x.size())
        return x



class Generator_one(nn.Module):

    def __init__(self):
        super(Generator_one, self).__init__()
        #set padding=2 to constrain the output's size is 28 * 28
        self.conv1 = nn.ConvTranspose2d(3+1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.ConvTranspose2d(64, 3, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x, condi_x=None):
        #print(x.size())
        x = x.view(-1, 1, 16, 16)
        x = torch.cat((condi_x, x), 1)
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.conv3(x)
        #x = x.view(-1, 14, 14)

        return x


class Generator_two(nn.Module):

    def __init__(self):
        super(Generator_two, self).__init__()
        self.fc1 = nn.Linear(100, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, 8*8*3)

    def forward(self, x, condi_x=None):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, 3, 8, 8)

        return x


class LAPGAN(object):

    def __init__(self, n_level, use_gpu=False, n_channel=3):
        self.n_level = n_level
        self.n_channel = n_channel
        self.use_gpu = use_gpu
        self.Dis_models = []
        self.Gen_models = []

        #D zero and D one both inputs contain condition information
        D_model0 = Discriminator_zero()
        if use_gpu: D_model0 = D_model0.cuda()
        self.Dis_models.append(D_model0)

        D_model1 = Discriminator_one()
        if use_gpu: D_model1 = D_model1.cuda()
        self.Dis_models.append(D_model1)
        
        #D two inputs have no condition information
        D_model2 = Discriminator_two()
        if use_gpu: D_model2 = D_model2.cuda()
        self.Dis_models.append(D_model2)


        #G zero and G one both inputs contain condition information
        G_model0 = Generator_zero()
        if use_gpu: G_model0 = G_model0.cuda()
        self.Gen_models.append(G_model0)

        G_model1 = Generator_one()
        if use_gpu: G_model1 = G_model1.cuda()
        self.Gen_models.append(G_model1)

        #G two inputs have no condition information
        G_model2 = Generator_two()
        if use_gpu: G_model2 = G_model2.cuda()
        self.Gen_models.append(G_model2)

        print(self.Gen_models)
        print(self.Dis_models)

    def generate(self, batchsize, get_level=None, generator=False):
        """Generate images from LAPGAN generators"""
        for G in self.Gen_models:
            G.eval()
            
        self.outputs = []
        self.generator_outputs = []
        for level in range(self.n_level):
            Gen_model = self.Gen_models[self.n_level - level - 1]

            # generate noise
            if level == 0: self.noise_dim = 100
            elif level == 1: self.noise_dim = 16*16
            else: self.noise_dim = 32*32
            noise = Variable(gen_noise(batchsize, self.noise_dim))
            if self.use_gpu:
                noise = noise.cuda()

            x = []
            if level == 0:
                # directly generate images
                output_imgs = Gen_model.forward(noise)
                if self.use_gpu:
                    output_imgs = output_imgs.cpu()
                output_imgs = output_imgs.data.numpy()
                x.append(output_imgs)
                self.generator_outputs.append(output_imgs)
            else:
                # upsize
                input_imgs = np.array([[cv2.pyrUp(output_imgs[i, j, :])
                                      for j in range(self.n_channel)]
                                      for i in range(batchsize)])
                condi_imgs = Variable(torch.Tensor(input_imgs))
                if self.use_gpu:
                    condi_imgs = condi_imgs.cuda()

                # generate images with extra information
                residual_imgs = Gen_model.forward(noise, condi_imgs)
                if self.use_gpu:
                    residual_imgs = residual_imgs.cpu()
                output_imgs = residual_imgs.data.numpy() + input_imgs
                self.generator_outputs.append(residual_imgs.data.numpy())
                x.append(output_imgs)

            self.outputs.append(x[-1])

        if get_level is None:
            get_level = -1

        x = self.outputs[0]
        t = np.zeros(batchsize * self.n_channel * 32 * 32).reshape(batchsize, self.n_channel, 32, 32)
        t[:, :, :x.shape[2], :x.shape[3]] = x
        result_imgs = t
        x = self.outputs[1]
        t = np.zeros(batchsize * self.n_channel * 32 * 32).reshape(batchsize, self.n_channel, 32, 32)
        t[:, :, :x.shape[2], :x.shape[3]] = x
        result_imgs = np.concatenate([result_imgs, t], axis=0)
        x = self.outputs[2]
        t = np.zeros(batchsize * self.n_channel * 32 * 32).reshape(batchsize, self.n_channel, 32, 32)
        t[:, :, :x.shape[2], :x.shape[3]] = x
        result_imgs = np.concatenate([result_imgs, t], axis=0)
        #result_imgs = torch.from_numpy(result_imgs)
        #torch.clamp(result_imgs, min=-1, max=1)
        #result_imgs = result_imgs.numpy()
        #result_imgs = (result_imgs+1)/2 
        #result_imgs = 1 - result_imgs
        return result_imgs


def gen_noise(n_instance, n_dim):
    #return torch.Tensor(np.random.uniform(low=-1.0, high=1.0, size=(n_instance, n_dim)))
    return torch.Tensor(np.random.normal(loc=0, scale=0.1, size=(n_instance, n_dim)))
