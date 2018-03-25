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
        self.conv1 = nn.Conv2d(1, 128, kernel_size=5)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=5, stride=2)
        self.sz = ((28-5+1)-5)//2+1
        self.fc1 = nn.Linear(128*self.sz*self.sz, 1)

    def forward(self, x, condi_x=None):
        #print(x.size(), condi_x.size())
        ## how to add condition information?
        ## answer: condition information(low-pass) plus residual image(high-pass)
        x = x + condi_x
        #print(x.size())
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 128*self.sz*self.sz)
        x = F.dropout(x)
        x = F.sigmoid(self.fc1(x))
        #what's the difference between F.sigmoid and nn.Sigmoid?
        return x



class Discriminator_one(nn.Module):
    def __init__(self):
        super(Discriminator_one, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=2)
        self.sz = ((14-5+1)-5)//2+1
        self.fc1 = nn.Linear(64*self.sz*self.sz, 1)

    def forward(self, x, condi_x=None):
        x = x + condi_x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 64*self.sz*self.sz)
        x = F.dropout(x)
        x = F.sigmoid(self.fc1(x))

        return x



class Discriminator_two(nn.Module):
    def __init__(self):
        super(Discriminator_two, self).__init__()
        self.fc1 = nn.Linear(7*7, 600)
        self.fc2 = nn.Linear(600, 600)
        self.fc3 = nn.Linear(600, 1)

    def forward(self, x, condi_x=None):
        x = x.view(-1, 7*7)
        x = F.dropout(F.relu(self.fc1(x)))
        x = F.dropout(F.relu(self.fc2(x)))
        x = F.sigmoid(self.fc3(x))

        return x


class Generator_zero(nn.Module):

    def __init__(self):
        super(Generator_zero, self).__init__()
        self.conv1 = nn.ConvTranspose2d(1+1, 128, kernel_size=7, padding=3)
        self.conv2 = nn.ConvTranspose2d(128, 128, kernel_size=7, padding=3)
        self.conv3 = nn.ConvTranspose2d(128, 1, kernel_size=5, padding=2)

    def forward(self, x, condi_x=None):
        #condi_x is low-pass image, and x is noise
        x = x.view(-1, 1, 28, 28)
        #print(condi_x.size(), x.size())
        x = torch.cat((condi_x, x), 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        #print(x.size())
        return x



class Generator_one(nn.Module):

    def __init__(self):
        super(Generator_one, self).__init__()
        #set padding=2 to constrain the output's size is 28 * 28
        self.conv1 = nn.ConvTranspose2d(1+1, 64, kernel_size=5, padding=2)
        self.conv2 = nn.ConvTranspose2d(64, 64, kernel_size=5, padding=2)
        self.conv3 = nn.ConvTranspose2d(64, 1, kernel_size=5, padding=2)

    def forward(self, x, condi_x=None):
        #print(x.size())
        x = x.view(-1, 1, 14, 14)
        x = torch.cat((condi_x, x), 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        #x = x.view(-1, 14, 14)

        return x


class Generator_two(nn.Module):

    def __init__(self):
        super(Generator_two, self).__init__()
        self.fc1 = nn.Linear(100, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, 7*7)

    def forward(self, x, condi_x=None):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, 1, 7, 7)

        return x


class LAPGAN(object):

    def __init__(self, n_level, use_gpu=False, n_channel=1):
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
        self.outputs = []
        self.generator_outputs = []
        for level in range(self.n_level):
            Gen_model = self.Gen_models[self.n_level - level - 1]

            # generate noise
            if level == 0: self.noise_dim = 100
            elif level == 1: self.noise_dim = 14*14
            else: self.noise_dim = 28*28
            noise = Variable(gen_noise(batchsize, self.noise_dim))
            if self.use_gpu:
                noise = noise.cuda()

            if level == 0:
                # directly generate images
                output_imgs = Gen_model.forward(noise)
                if self.use_gpu:
                    output_imgs = output_imgs.cpu()
                output_imgs = output_imgs.data.numpy()
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

            self.outputs.append(output_imgs)

        if get_level is None:
            get_level = -1

        if generator:
            result_imgs = self.generator_outputs[get_level]
        else:
            result_imgs = self.outputs[get_level]

        return result_imgs


def gen_noise(n_instance, n_dim):
    return torch.Tensor(np.random.uniform(low=-1.0, high=1.0, size=(n_instance, n_dim)))
