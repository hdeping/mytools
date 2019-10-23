# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class baseConv1d(nn.Module):
    def __init__(self,input_chanel,output_chanel,kernel_size,stride,padding):
        super(baseConv1d,self).__init__()
        # architeture of the base conv1d
        self.conv = nn.Conv1d(input_chanel,output_chanel,kernel_size=kernel_size,stride=stride,padding=padding)
        self.bn   = nn.BatchNorm1d(output_chanel)
    def forward(self,x):
        # conv 
        #print(x.size())
        x = self.conv(x)
        # batchnorm 
        x = self.bn(x)
        # 1d avg pool 
        x = F.max_pool1d(x,kernel_size=2)
        # relu output
        x = F.relu(x)
        return x
class baseLinear(nn.Module):
    def __init__(self,input_chanel,output_chanel,relu=True):
        super(baseLinear,self).__init__()

        self.fc = nn.Linear(input_chanel,output_chanel)
        self.relu = relu
    def forward(self,x):
        x = self.fc(x)
        if self.relu :
            x = F.relu(x)
        else:
            x = F.tanh(x)
        return x

class LanNet(nn.Module):
    def __init__(self, input_dim=48):
        super(LanNet, self).__init__()
        self.input_dim = input_dim
        
        self.layer = nn.Sequential()
        self.layer.add_module('fc1', baseLinear(self.input_dim,1024))
        self.layer.add_module('fc2', baseLinear(1024,1024))
        self.layer.add_module('fc3', baseLinear(1024,40))
        self.layer.add_module('fc4', baseLinear(40,1024))
        self.layer.add_module('fc5', baseLinear(1024,1024))
        self.layer.add_module('fc6', baseLinear(1024,self.input_dim,relu=False))

    def forward(self, x):
        batch_size, fea_frames, fea_dim = x.size()
        #print(x.size)
        # reshape the input
        x = x.contiguous().view(batch_size*fea_frames,-1)
        out_target = self.layer(x)

        return out_target

class conv_VAE(nn.Module):
    def __init__(self,intermediate_size,hidden_size):
        super(conv_VAE, self).__init__()

        self.intermediate_size  = intermediate_size
        self.hidden_size  = hidden_size

        # Encoder
        self.conv1 = nn.Conv1d(1, 10, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(10, 20, kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv1d(20, 40, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(40, 40, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 16 * 32, args.intermediate_size)

        # Latent space
        self.fc21 = nn.Linear(self.intermediate_size, self.hidden_size)
        self.fc22 = nn.Linear(self.intermediate_size, self.hidden_size)

        # Decoder
        self.fc3 = nn.Linear(self.hidden_size, self.intermediate_size)
        self.fc4 = nn.Linear(self.intermediate_size, 8192)
        self.deconv1 = nn.ConvTranspose1d(40, 40, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose1d(40, 20, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose1d(20, 10, kernel_size=2, stride=2, padding=0)
        self.conv5 = nn.Conv2d(10, 1, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        out = out.view(out.size(0), -1)
        h1 = self.relu(self.fc1(out))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        out = self.relu(self.fc4(h3))
        # import pdb; pdb.set_trace()
        out = out.view(out.size(0), 32, 16, 16)
        out = self.relu(self.deconv1(out))
        out = self.relu(self.deconv2(out))
        out = self.relu(self.deconv3(out))
        out = self.sigmoid(self.conv5(out))
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = conv_VAE(500,40)

if args.cuda:
    model.cuda()
optimizer = optim.RMSprop(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x.view(-1, 32 * 32 * 3),
                                 x.view(-1, 32 * 32 * 3), size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
