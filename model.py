import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from projection1 import backward_projection, forward_projection


class PrimalNet(nn.Module):
    def __init__(self):
        super(PrimalNet, self).__init__()

        layers = [
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1)
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, img, img_back):
        x = torch.cat((img, img_back), dim=0).unsqueeze(0)
        x = img + self.block(x).squeeze(0)
        return x

class DualNet(nn.Module):
    def __init__(self):
        super(DualNet,self).__init__()

        layers = [
            nn.Linear(3, 64),
            nn.PReLU(),
            nn.Linear(64, 64),
            nn.PReLU(),
            nn.Linear(64, 16),
            nn.PReLU(),
            nn.Linear(16, 1),    
        ]
        self.block = nn.Sequential(*layers)
    
    def forward(self, lst, img_fwd, lst_fixed):
        x = torch.cat((lst, img_fwd, lst_fixed), dim=1)   
        x = self.block(x)
        x = lst + x
        return x

class LMNet(nn.Module):
    def __init__(self, primal_net=PrimalNet, dual_net=DualNet, n_iter=5):
        super(LMNet,self).__init__()

        self.primal_net = primal_net
        self.dual_net = dual_net
        self.n_iter = n_iter

        self.image_dim = (1, 128, 128)

        self.primal_nets = nn.ModuleList()
        self.dual_nets = nn.ModuleList()

        for _ in range(n_iter):
            self.primal_nets.append(primal_net())
            self.dual_nets.append(dual_net())
    
    def forward(self, sysG, lst_events):
        nevents = sysG.shape[0]
        lst = torch.zeros((nevents, 1), device=sysG.device)
        img = torch.zeros(self.image_dim, device=sysG.device)



        for i in range(self.n_iter):
            img_fwd = forward_projection(sysG, img).unsqueeze(1)
            lst = self.dual_nets[i](lst, img_fwd, lst_events)
            img_back = backward_projection(sysG, lst).view(*self.image_dim)
            img = self.primal_nets[i](img, img_back)

        return img.clamp(min=0)
    

def SSIM(image, label):
    image = image.reshape([-1, 128, 128])
    num, w, h = image.size()
    image = image.reshape(-1, w * h)
    label = label.reshape(-1,  w * h)
    num = w * h

    K1 = 0.01
    K2 = 0.03
    L = 1.0
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    C3 = C2 / 2

    mu_x, sigma_x = torch.mean(image, 1), torch.std(image, 1)
    mu_y, sigma_y = torch.mean(label, 1), torch.std(label, 1)
    mu_x_v = mu_x.reshape(-1, 1).repeat(1, 1, w * h)
    mu_y_v = mu_y.reshape(-1, 1).repeat(1, 1, w * h)

    sigma_xy = torch.sum((image - mu_x_v) * (label - mu_y_v), 2) / num
    lxy = (2 * mu_x * mu_y + C1) / (mu_x ** 2 + mu_y ** 2 + C1)
    cxy = (2 * sigma_x * sigma_y + C2) / (sigma_x ** 2 + sigma_y ** 2 + C2)
    sxy = (sigma_xy + C3) / (sigma_x * sigma_y + C3)

    ssim = lxy * cxy * sxy
    final_ssim = ssim
    final_ssim = torch.mean(final_ssim)

    return final_ssim

def psnr(img1, img2):
    mse = torch.mean(torch.pow(img1 - img2, 2))
    if mse == 0:
        return 100
    PIXEL_MAX1 = img1.max()
    PIXEL_MAX2 = img2.max()
    PIXEL_MAX = max(PIXEL_MAX1, PIXEL_MAX2)
    return torch.tensor(10 * math.log10(PIXEL_MAX * PIXEL_MAX / mse))
