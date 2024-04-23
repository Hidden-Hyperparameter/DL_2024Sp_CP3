# ### uncomment this cell if you're using Google colab
# from google.colab import drive
# drive.mount('/content/drive')

# ### change GOOGLE_DRIVE_PATH to the path of your CP3 folder
# GOOGLE_DRIVE_PATH = '/content/drive/MyDrive/Colab Notebooks/DL23SP/CP3'
# %cd $GOOGLE_DRIVE_PATH

# import sys
# sys.path.append(GOOGLE_DRIVE_PATH)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rcParams


# figure size in inches optional
rcParams['figure.figsize'] = 11, 8

# read images
img_A = mpimg.imread('./flow/recovered.png')
img_B = mpimg.imread('./flow/corrupted.png')

# display images
fig, ax = plt.subplots(1, 2)
ax[0].imshow(img_A)
ax[1].imshow(img_B)
from utils import hello
hello()
from collections import deque
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from utils import save_model, load_model, corruption, train_set, val_set

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda')
class BatchNormFlow(nn.Module):
    """ An implementation of a batch normalization layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs, momentum=0.0, eps=1e-5):
        super(BatchNormFlow, self).__init__()

        self.log_gamma = nn.Parameter(torch.zeros(num_inputs))
        self.beta = nn.Parameter(torch.zeros(num_inputs))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('running_mean', torch.zeros(num_inputs))
        self.register_buffer('running_var', torch.ones(num_inputs))

    def forward(self, inputs, mode='direct', **kwargs):
        if mode == 'direct':
            if self.training:
                self.batch_mean = inputs.mean(0)
                self.batch_var = (
                    inputs - self.batch_mean).pow(2).mean(0) + self.eps

                self.running_mean.mul_(self.momentum)
                self.running_var.mul_(self.momentum)

                self.running_mean.add_(self.batch_mean.data *
                                       (1 - self.momentum))
                self.running_var.add_(self.batch_var.data *
                                      (1 - self.momentum))

                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - mean) / var.sqrt()
            y = torch.exp(self.log_gamma) * x_hat + self.beta
            return y, (self.log_gamma - 0.5 * torch.log(var)).sum(-1, keepdim=True)
        else:
            if self.training:
                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - self.beta) / torch.exp(self.log_gamma)

            y = x_hat * var.sqrt() + mean

            return y, (- self.log_gamma + 0.5 * torch.log(var)).sum(-1, keepdim=True)
class Shuffle(nn.Module):
    """ An implementation of a shuffling layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs):
        super(Shuffle, self).__init__()
        self.perm = np.random.permutation(num_inputs)
        self.inv_perm = np.argsort(self.perm)

    def forward(self, inputs:torch.Tensor, mode='direct'):
        n = self.perm.shape[0]
        batch_size = inputs.size(0)
        if mode == 'direct':
            # return z = f(x) and logdet, z has the same shape with x, logdet has the shape (batch size, 1)
            ##############################################################################
            #                  TODO: You need to complete the code here                  #
            ##############################################################################
            # YOUR CODE HERE
            return inputs[:,self.perm[:]],torch.zeros([batch_size,1]).to(device)
            # raise NotImplementedError()
            ##############################################################################
            #                              END OF YOUR CODE                              #
            ##############################################################################
        else:
            # return x = f^-1(z) and logdet, x has the same shape with z, logdet has the shape (batch size, 1)
            ##############################################################################
            #                  TODO: You need to complete the code here                  #
            ##############################################################################
            # YOUR CODE HERE
            return inputs[:,self.inv_perm[:]],torch.zeros([batch_size,1]).to(device)
            # raise NotImplementedError()
            ##############################################################################
            #                              END OF YOUR CODE                              #
            ##############################################################################
class CouplingLayer(nn.Module):
    """ An implementation of a coupling layer
    from RealNVP (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self,
                 num_inputs,
                 num_hidden,
                 mask,
                 s_act=nn.Tanh(),
                 t_act=nn.ReLU()):
        super(CouplingLayer, self).__init__()

        self.num_inputs = num_inputs
        self.mask = mask

        # define your scale_net and translate_net
        ##############################################################################
        #                  TODO: You need to complete the code here                  #
        ##############################################################################
        # YOUR CODE HERE
        self.num_hidden = num_hidden
        self.s_act = s_act
        self.t_act = t_act
        # scale net: from num_input to num_hidden
        # print(self.num_inputs) # 784
        # print(self.num_hidden) # 512
        # print('mask shape',self.mask.shape) # [784]
        self.scale_net = nn.Sequential(
            nn.Unflatten(1,(1,28,28)),    
            nn.Conv2d(1,64,kernel_size=(4,4)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # self.s_act,
            nn.Conv2d(64,16,kernel_size=(4,4),stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # self.s_act,
            nn.Conv2d(16,8,kernel_size=(4,4)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            # self.s_act,
            nn.Flatten(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,self.num_inputs),
            self.s_act
        )
        # translate net: from num_input to num_hidden
        self.translate_net = nn.Sequential(
            nn.Unflatten(1,(1,28,28)),    
            nn.Conv2d(1,64,kernel_size=(4,4)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # self.t_act,
            nn.Conv2d(64,16,kernel_size=(4,4),stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # self.t_act,
            nn.Conv2d(16,8,kernel_size=(4,4)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            # self.t_act,
            nn.Flatten(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,self.num_inputs),
            nn.Tanh()
            # self.t_act
        )
        # raise NotImplementedError()
        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################

    def forward(self, x, mode='direct'):
        mask = self.mask

        masked_inputs = x * mask
        log_s = self.scale_net(masked_inputs)*(1-mask)
        s = torch.exp(log_s)
        t = self.translate_net(masked_inputs)

        if mode == 'direct':
            # complete code here;  x - > z
            # return z = f(x) and logdet, z has the same shape with x, logdet has the shape (batch size, 1)
            ##############################################################################
            #                  TODO: You need to complete the code here                  #
            ##############################################################################
            # YOUR CODE HERE
            # x: image z: latent; logdet: det(dz/dx)
            # print('x.shape',x.shape)    # torch.Size([128, 784])
            # print('s.shape',s.shape)    # torch.Size([128, 784])
            # print('t.shape',t.shape)    # torch.Size([128, 784])
            z = masked_inputs + (1-mask)*(s*x+t)
            logdet = torch.sum(log_s,dim=1,keepdim=True)
            # assert z.shape==x.shape
            # assert list(logdet.shape) == [x.shape[0],1],(logdet.shape,[x.shape[0],1])
            return z,logdet
            # raise NotImplementedError()
            ##############################################################################
            #                              END OF YOUR CODE                              #
            ##############################################################################
        else:
            # complete code here; z - > x
            # return x = f^-1(z) and logdet, x has the same shape with z, logdet has the shape (batch size, 1)
            ##############################################################################
            #                  TODO: You need to complete the code here                  #
            ##############################################################################
            # YOUR CODE HERE
            z = masked_inputs + (1-mask)*(1/s)*((1-mask)*x-t)
            logdet = -torch.sum(log_s,dim=1,keepdim=True)
            # assert z.shape==x.shape
            # assert logdet.shape == [x.shape[0],1]
            return z,logdet
            ##############################################################################
            #                              END OF YOUR CODE                              #
            ##############################################################################
class FlowSequential(nn.Sequential):
    """ A sequential container for flows.
    In addition to a forward pass it implements a backward pass and
    computes log jacobians.
    """

    def __init__(self, device, *args):
        super().__init__(*args)
        self.prior = torch.distributions.Normal(
            torch.tensor([0.0]).to(device),
            torch.tensor([1.0]).to(device)
        )

    def _pre_process(self, x):
        """
        Args:
            x (torch.Tensor): Input image.
        Returns:
            y (torch.Tensor): logits of `x`.
        See Also:
            - Modeling logits: https://arxiv.org/abs/1605.08803, Section 4.1
        """
        return x.log() - (1. - x).log()

    def forward(self, inputs, mode='direct', logdets=None, **kwargs):
        """ Performs a forward or backward pass for flow modules.
        Args:
            inputs: a tuple of inputs and logdets
            mode: to run direct computation or inverse
        """
        self.num_inputs = inputs.size(-1)

        if logdets is None:
            logdets = torch.zeros(inputs.size(0), 1, device=inputs.device)

        assert mode in ['direct', 'inverse']
        if mode == 'direct':
            if kwargs.get("pre_process", True):
                inputs = self._pre_process(inputs)
            for module in self._modules.values():
                inputs, logdet = module(inputs, mode)
                logdets += logdet
        else:
            for module in reversed(self._modules.values()):
                inputs, logdet = module(inputs, mode, **kwargs)
                logdets += logdet

        return inputs, logdets

    def log_probs(self, inputs, pre_process=True):
        u, log_jacob = self(inputs, pre_process=pre_process)
        # return the log probability with shape (batch size, 1)
        ##############################################################################
        #                  TODO: You need to complete the code here                  #
        ##############################################################################
        # YOUR CODE HERE
        # inputs = real data
        # u = latent
        # print(self.prior.log_prob(u).sum(dim=1))
        # print(log_jacob)
        return self.prior.log_prob(u).sum(dim=1,keepdim=True) + log_jacob
        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################

    @torch.no_grad()
    def sample_images(self, n_samples=100, save=True, save_dir='./flow'):
        self.eval()
        samples, _ = self.forward(self.prior.sample([n_samples, 28 * 28]).squeeze(-1), mode='inverse')
        imgs = torch.sigmoid(samples).view(n_samples, 1, 28, 28)
        if save:
            os.makedirs(save_dir, exist_ok=True)
            torchvision.utils.save_image(imgs, os.path.join(save_dir, 'sample.png'), nrow=int(np.sqrt(n_samples)))
        return imgs
def inpainting(model, inputs, mask, device, save=True, save_dir='./flow/inpainting'):
    num_samples = inputs.shape[0]
#     torchvision.utils.save_image(inputs[:100], os.path.join(save_dir, 'corrupted.png'), nrow=10)
    inputs = inputs.view(num_samples, -1).to(device)
    mask = mask.view(num_samples, -1).to(device)
    inputs = inputs.log() - (1. - inputs).log()

    ep = torch.randn(inputs.size()).to(device)
    os.makedirs(os.path.join(save_dir, 'recover_process'), exist_ok=True)
    for i in range(1000):

        alpha = 0.2
        inputs.requires_grad_()
        log_probs = model.log_probs(inputs, pre_process=False)
        dx = torch.autograd.grad([log_probs.sum()], [inputs])[0]
        dx = torch.clip(dx, -10, 10)

        with torch.no_grad():
            inputs = inputs * mask + (1 - mask) * (inputs + alpha * dx).clip(-10, 10)

        imgs = torch.sigmoid(inputs.view(num_samples, 1, 28, 28))

        if i % 10 == 0 and save:
            torchvision.utils.save_image(imgs[:100], os.path.join(
                save_dir, 'recover_process/recovered_iter{:03d}.png'.format(i + 1)), nrow=10)
    if save:
        torchvision.utils.save_image(imgs[:100], os.path.join(
        save_dir, 'recovered.png'.format(i + 1)), nrow=10)
    return imgs
def evaluate(epoch, flow_model, loader, device):
    flow_model.eval()
    val_loss = c_mse = r_mse = 0
    n_batches = 0

    pbar = tqdm(total=len(loader.dataset))
    pbar.set_description('Eval')
    for batch_idx, (data, _) in enumerate(loader):
        bs = data.shape[0]
        n_batches += data.shape[0]
        data = data.to(device)

        # compute validation loss
        with torch.no_grad():
            # sum up batch loss
            val_loss += -flow_model.log_probs(data.reshape(bs, -1)).sum().item()

        # inpainting
        imgs = data
        c_imgs, mask = corruption(imgs, type_='flow')
        r_imgs = inpainting(flow_model, c_imgs, mask, device,
                   save=(batch_idx == 0), save_dir=f"./flow/{epoch + 1}/")
        r_mse += ((imgs.view(bs, -1) - r_imgs.view(bs, -1))**2).sum().item()
        c_mse += ((imgs.view(bs, -1) - c_imgs.view(bs, -1))**2).sum().item()

        pbar.update(data.size(0))
        pbar.set_description('Val, Log likelihood in nats: {:.6f}, Corruption MSE: {:.6f}, Recovered MSE: {:.6f}'.format(
            -val_loss / n_batches, c_mse / n_batches, r_mse / n_batches))

    pbar.close()
    return val_loss / n_batches, c_mse / n_batches, r_mse / n_batches
def train(n_epochs, flow_model, train_loader, val_loader, optimizer, device=torch.device('cuda'), save_interval=1):
    flow_model.to(device)
    best_val_loss = np.inf

    for epoch in range(n_epochs):
        train_loss = 0
        n_batches = 0
        pbar = tqdm(total=len(train_loader.dataset))
        for i, (x, _) in enumerate(train_loader):
            # compute loss
            n_batches += x.shape[0]
            flow_model.train()
            x = x.to(device)
            loss = -flow_model.log_probs(x.reshape(x.shape[0], -1))
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

            train_loss += loss.sum().item()

            pbar.update(x.size(0))
            pbar.set_description('Train Epoch {}, Log likelihood in nats: {:.6f}'.format(epoch + 1,
                                                                                         -train_loss / n_batches))
        pbar.close()

        if (epoch + 1) % save_interval == 0:
            os.makedirs(f'./flow/{epoch + 1}', exist_ok=True)
            flow_model.eval()
            save_model(f'./flow/{epoch + 1}/flow.pth', flow_model, optimizer)

            val_loss, c_mse, r_mse = evaluate(epoch, flow_model, val_loader, device)

            # sample and save images
            flow_model.sample_images(
                save=True, save_dir=f"./flow/{epoch + 1}/")

            # feel free to change the following metric to MSE for model selection!
            if val_loss < best_val_loss:
                print(
                    f'Current validation loss: {best_val_loss} -> {val_loss}')
                best_val_loss = val_loss
                save_model('./flow/flow_best.pth', flow_model)
num_inputs = 28 * 28
num_hidden = 512
num_blocks = 8

modules = []
masks = []

mask = torch.arange(0, num_inputs) % 2
mask = mask.to(device).float()
masks.extend([mask, 1 - mask])
mask2 = torch.zeros_like(mask)
mask2[: num_inputs//2] = 1
masks.extend([mask2, 1 - mask2])

for i in range(num_blocks):
    modules += [
        CouplingLayer(
            num_inputs, num_hidden, masks[i % len(masks)]),
        BatchNormFlow(num_inputs, 0.1),
        Shuffle(num_inputs)
    ]

model = FlowSequential(device, *modules)

for module in model.modules():
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight)
        # nn.init.xavier_normal_(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias.data.fill_(0)

model.to(device)
val_loader = DataLoader(val_set, batch_size=256, pin_memory=True,  shuffle=True, num_workers=1)
# from torch.utils.data import Subset
# # feel free to change training hyper-parameters!
# train_loader = DataLoader(train_set, batch_size=128, pin_memory=True,
#                           drop_last=False, shuffle=True, num_workers=8)

# optimizer = torch.optim.Adam(model.parameters(), lr=6e-5, weight_decay=1e-4)

# val_subset = Subset(val_set, range(2000))

# val_loader = DataLoader(val_subset, batch_size=500, pin_memory=True,  shuffle=True, num_workers=1)
# train(300, model, train_loader, val_loader, optimizer, device, save_interval=10)
# assert(False)
model.load_state_dict(load_model('./flow/flow_04230846.pth')[0])
val_loader = DataLoader(val_set, batch_size=256, pin_memory=True,  shuffle=True, num_workers=1)
val_loss, c_mse, r_mse = evaluate(int(1e4), model, val_loader, device)
print("[Evaluation] val_loss: {}, c_mse: {}, r_mse: {}".format(val_loss, c_mse, r_mse), flush=True)
