import torch.nn.functional as F
from my_tools import *
from utils import A, At, shift, shift_back
import torch
import torchvision

class GAP_net(nn.Module):

    def __init__(self, out_ch=28):
        super(GAP_net, self).__init__()
        middle_channel = 28
        self.unet1 = Unet(28, 28)
        self.unet2 = Unet(28, 28)
        self.unet3 = Unet(28, 28)
        self.unet4 = Unet_2(middle_channel, middle_channel)
        self.unet5 = Unet_2(middle_channel, middle_channel)
        self.unet6 = Unet_2(middle_channel, middle_channel)
        self.unet7 = Unet(28, 28)
        self.unet8 = Unet(28, 28)
        self.unet9 = Unet(28, 28)
        
        # unet-arc define
        self.maxpool_x = nn.MaxPool2d(2,2)
        self.maxpool_phi = nn.MaxPool2d(2,2)
        self.maxpool_y = nn.MaxPool2d(2,2)
        
        self.downsample_x = nn.Sequential(
            nn.Conv2d(28, middle_channel, kernel_size=2, stride=2),
            # nn.ReLU()
        )
        self.downsample_phi = nn.Sequential(
            nn.Conv2d(28, middle_channel, kernel_size=2, stride=2),
            nn.ReLU() 
        )
        self.downsample_y = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.upsample1_x = nn.Sequential(
            nn.ConvTranspose2d(middle_channel, 28, kernel_size=2, stride=2),
            # nn.ReLU()
        )
        # phi
        self.upsample1_phi = nn.Sequential(
            nn.ConvTranspose2d(middle_channel, 28, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.relu = nn.ReLU()

    # 纯烂
    def forward(self, y, Phi, Phi_s):
        x_list = []
        x = At(y,Phi)
        input_x = x
        y1 = y
        ## down
        ### stage 1
        yb = A(x,Phi)
        x = x + At(torch.div(y1-yb,Phi_s),Phi)  # c=28
        x = shift_back(x)
        x = self.unet1(x)
        x = shift(x)
        
        # ### stage 2
        yb = A(x,Phi)
        x = x + At(torch.div(y1-yb,Phi_s),Phi)
        x = shift_back(x)
        x = self.unet2(x)
        x = shift(x)
        
        ## stage 3
        yb = A(x,Phi)
        x = x + At(torch.div(y1-yb,Phi_s),Phi)
        x = shift_back(x)
        x = self.unet3(x)
        x = shift(x)
        
        x = self.downsample_x(x)
        Phi = self.downsample_phi(Phi)
        # TODO：交换在dim0之后增加一个维度为1
        y_tmp = torch.unsqueeze(y1, 1)
        y2 = self.downsample_y(y_tmp)
        # y2的dim1维度去掉
        y2 = torch.squeeze(y2, 1)
        Phi_s = torch.sum(Phi, dim=1)
        Phi_s[Phi_s==0] = 1
        
        ### stage 4
        yb = A(x,Phi)
        x = x + At(torch.div(y2-yb,Phi_s),Phi)
        x = shift_back(x)
        x = self.unet4(x)
        x = shift(x)
        
        ## bottom
        ### stage 5
        yb = A(x,Phi)
        x = x + At(torch.div(y2-yb,Phi_s),Phi)
        x = shift_back(x)
        x = self.unet5(x)        
        x = shift(x)
        
        ## up
        ### stage 6
        yb = A(x,Phi)
        x = x + At(torch.div(y2-yb,Phi_s),Phi)
        x = shift_back(x)
        x = self.unet6(x)        
        
        x = self.upsample1_x(x)           # 128-64
        Phi = self.upsample1_phi(Phi)
        # print(x)
        # y = self.upsample1_y(y)
        Phi_s = torch.sum(Phi, dim=1)
        Phi_s[Phi_s==0] = 1
        
        ### stage 7
        yb = A(x,Phi)
        x = x + At(torch.div(y1-yb,Phi_s),Phi)
        x = shift_back(x)
        x = self.unet7(x)      
        x = shift(x)
        
        ### stage 8
        yb = A(x,Phi)
        x = x + At(torch.div(y1-yb,Phi_s),Phi)
        x = shift_back(x)
        x = self.unet8(x)
        x = shift(x)
        
        
        
        ### stage 9
        yb = A(x,Phi)
        x = x + At(torch.div(y1-yb, Phi_s),Phi)
        x = shift_back(x)
        x = self.unet9(x)
        
        x_list.append(x[:,:,:,0:256])
        output_list = x_list
        return output_list