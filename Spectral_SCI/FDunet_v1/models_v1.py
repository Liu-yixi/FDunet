import torch.nn.functional as F
from my_tools import *
from utils import A, At, shift, shift_back
import torch
import torchvision

class GAP_net(nn.Module):

    def __init__(self, out_ch=28):
        super(GAP_net, self).__init__()
                
        self.unet1 = Unet(28, 28)
        self.unet2 = Unet_2(28, 28)
        self.unet3 = Unet_3(28, 28)
        self.unet4 = Unet_2(28, 28)
        self.unet5 = Unet(28, 28)

        # unet-arc define
        self.maxpool_x = nn.MaxPool2d(2,2)
        self.maxpool_phi = nn.MaxPool2d(2,2)
        self.maxpool_y = nn.MaxPool2d(2,2)
        
        self.upsample2_x = nn.Sequential(
            nn.ConvTranspose2d(28, 28, kernel_size=2, stride=2, output_padding=(0,1)),
            #nn.Conv2d(64, 64, (1,2), padding=(0,1)),
            # nn.ReLU()
        )
        self.upsample1_x = nn.Sequential(
            nn.ConvTranspose2d(28, 28, kernel_size=2, stride=2),
            # nn.ReLU()
        )
        
        self.upsample2_phi = nn.Sequential(
            nn.ConvTranspose2d(28, 28, kernel_size=2, stride=2, output_padding=(0,1)),
            #nn.Conv2d(64, 64, (1,2), padding=(0,1)),
            nn.ReLU()
        )
        self.upsample1_phi = nn.Sequential(
            nn.ConvTranspose2d(28, 28, kernel_size=2, stride=2),
            nn.ReLU()
        )
        
        


    def forward(self, y, Phi, Phi_s):
        x_list = []
        x = At(y,Phi)
        y1 = y
        ## down
        ### stage 1
        yb = A(x,Phi)
        x = x + At(torch.div(y1-yb,Phi_s),Phi)  # c=28
        x = shift_back(x)
        x = self.unet1(x)
                            
        x = self.maxpool_x(x)
        Phi = self.maxpool_phi(Phi)
        y2 = self.maxpool_y(y1)
        Phi_s = torch.sum(Phi, dim=1)
        Phi_s[Phi_s==0] = 1
        x = shift(x)
        
        ### stage 2
        yb = A(x,Phi)
        x = x + At(torch.div(y2-yb,Phi_s),Phi)
        x = shift_back(x)
        x = self.unet2(x)
        x = self.maxpool_x(x)
        Phi = self.maxpool_phi(Phi)
        y3 = self.maxpool_y(y2)
        Phi_s = torch.sum(Phi, dim=1)
        Phi_s[Phi_s==0] = 1
        x = shift(x)
        
        ## stage 3
        yb = A(x,Phi)
        x = x + At(torch.div(y3-yb,Phi_s),Phi)
        x = shift_back(x)
        x = self.unet3(x)
        x = self.upsample2_x(x)           # 128-64
        Phi = self.upsample2_phi(Phi)
        # y = self.upsample2_y(y)
        Phi_s = torch.sum(Phi, dim=1)
        Phi_s[Phi_s==0] = 1

        x = shift(x)
        ## up 
        ### stage 4
        yb = A(x,Phi)
        x = x + At(torch.div(y2-yb,Phi_s),Phi)
        x = shift_back(x)
        x = self.unet4(x)
        
        x = self.upsample1_x(x)           # 128-64
        Phi = self.upsample1_phi(Phi)
        # y = self.upsample1_y(y)
        Phi_s = torch.sum(Phi, dim=1)
        Phi_s[Phi_s==0] = 1
        x = shift(x)
        
        ### stage 5
        yb = A(x,Phi)
        x = x + At(torch.div(y1-yb,Phi_s),Phi)
        x = shift_back(x)
        x = self.unet5(x)
        x = shift(x)
        x_list.append(x[:,:,:,0:256])
        
        

        output_list = x_list
        return output_list