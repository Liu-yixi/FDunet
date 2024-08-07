""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU()
    )   
'''
class double_conv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(double_conv, self).__init__()
        self.d_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.d_conv(x)
        return x

class Unet(nn.Module):

    def __init__(self,in_ch, out_ch):
        super(Unet, self).__init__()
                
        self.dconv_down1 = double_conv(in_ch, 32)
        self.dconv_down2 = double_conv(32, 64)
        self.dconv_down3 = double_conv(64, 128)       

        self.maxpool = nn.MaxPool2d(2,2)
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, output_padding=(0,1)),
            #nn.Conv2d(64, 64, (1,2), padding=(0,1)),
            nn.ReLU()
        )
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.dconv_up2 = double_conv(64 + 64, 64)
        self.dconv_up1 = double_conv(32 + 32, 32)
        
        self.conv_last = nn.Conv2d(32, out_ch, 1)
        self.afn_last = nn.Tanh()
        
        
        
    def forward(self, x):
        inputs = x
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)

        x = self.upsample2(conv3)  
        x = torch.cat([x, conv2], dim=1)
        
        x = self.dconv_up2(x)
        x = self.upsample1(x)        
        x = torch.cat([x, conv1], dim=1)       

        x = self.dconv_up1(x)  
        
        x = self.conv_last(x)
        x = self.afn_last(x)
        out = x + inputs
        
        return out
    
class Unet_2(nn.Module):

    def __init__(self,in_ch, out_ch):
        super(Unet_2, self).__init__()
                
        self.dconv_down1 = double_conv(in_ch, 32)
        self.dconv_down2 = double_conv(32, 64)
        self.dconv_down3 = double_conv(64, 128)       

        self.maxpool = nn.MaxPool2d(2,2)
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, output_padding=(0,1)),
            #nn.Conv2d(64, 64, (1,2), padding=(0,1)),
            nn.ReLU()
        )
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, output_padding=(0,1)),
            nn.ReLU()
        )
        self.dconv_up2 = double_conv(64 + 64, 64)
        self.dconv_up1 = double_conv(32 + 32, 32)
        
        self.conv_last = nn.Conv2d(32, out_ch, 1)
        self.afn_last = nn.Tanh()
        
        
        
    def forward(self, x):
        inputs = x
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.upsample2(conv3)  
        x = torch.cat([x, conv2], dim=1)
        
        x = self.dconv_up2(x)
        x = self.upsample1(x)        
        x = torch.cat([x, conv1], dim=1)       

        x = self.dconv_up1(x)  
        
        x = self.conv_last(x)
        x = self.afn_last(x)
        out = x + inputs
        
        return out

class Unet_3(nn.Module):

    def __init__(self,in_ch, out_ch):
        super(Unet_3, self).__init__()
                
        self.dconv_down1 = double_conv(in_ch, 32)
        self.dconv_down2 = double_conv(32, 64)
        self.dconv_down3 = double_conv(64, 128)       

        self.maxpool = nn.MaxPool2d(2,2)
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            #nn.Conv2d(64, 64, (1,2), padding=(0,1)),
            nn.ReLU()
        )
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, output_padding=(0,1)),
            nn.ReLU()
        )
        self.dconv_up2 = double_conv(64 + 64, 64)
        self.dconv_up1 = double_conv(32 + 32, 32)
        
        self.conv_last = nn.Conv2d(32, out_ch, 1)
        self.afn_last = nn.Tanh()
        
        
        
    def forward(self, x):
        inputs = x
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.upsample2(conv3)  
        x = torch.cat([x, conv2], dim=1)
        
        x = self.dconv_up2(x)
        x = self.upsample1(x)        
        x = torch.cat([x, conv1], dim=1)       

        x = self.dconv_up1(x)  
        
        x = self.conv_last(x)
        x = self.afn_last(x)
        out = x + inputs
        
        return out

class Unet_4(nn.Module):
    # no output padding
    def __init__(self,in_ch, out_ch):
        super(Unet_4, self).__init__()
                
        self.dconv_down1 = double_conv(in_ch, 32)
        self.dconv_down2 = double_conv(32, 64)
        self.dconv_down3 = double_conv(64, 128)       

        self.maxpool = nn.MaxPool2d(2,2)
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            #nn.Conv2d(64, 64, (1,2), padding=(0,1)),
            nn.ReLU()
        )
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.dconv_up2 = double_conv(64 + 64, 64)
        self.dconv_up1 = double_conv(32 + 32, 32)
        
        self.conv_last = nn.Conv2d(32, out_ch, 1)
        self.afn_last = nn.Tanh()
        
        
        
    def forward(self, x):
        inputs = x
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.upsample2(conv3)  
        x = torch.cat([x, conv2], dim=1)
        
        x = self.dconv_up2(x)
        x = self.upsample1(x)        
        x = torch.cat([x, conv1], dim=1)       

        x = self.dconv_up1(x)  
        
        x = self.conv_last(x)
        x = self.afn_last(x)
        out = x + inputs
        
        return out
    
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class FGD(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(28, 28, kernel_size = 1, stride = 1),
            nn.GELU(),
            nn.Conv2d(28, 28, kernel_size = 3, stride = 1, dilation = 2, padding = 1),
            nn.GELU(),
            nn.Conv2d(28, 28, kernel_size = 1, stride = 1, dilation = 2, padding = 1),
            )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(28, 28, kernel_size = 1, stride = 1),
            nn.GELU(),
            nn.Conv2d(28, 28, kernel_size = 3, stride = 1, dilation = 2, padding = 1),
            nn.GELU(),
            nn.Conv2d(28, 28, kernel_size = 1, stride = 1, dilation = 2, padding = 1),
            )
        
    
    def forward(self, x):
        x_in = x
        x = self.block1(x)
        x = self.block2(x)
        return x + x_in

if __name__ == "__main__":
    x = torch.randn(1, 28, 256, 155)
    print(x.shape)
    model = FGD()
    print(model(x).shape)