import torch.nn.functional as F
from my_tools import *
from utils import A, At, shift, shift_back
import torch
import torchvision

class AttentionPooling(nn.Module):
    def __init__(self, in_channels, out_channels, k):
        super(AttentionPooling, self).__init__()
        self.k = k  # 池化窗口大小
        self.fc = nn.Linear(in_channels, out_channels)  # 注意力权重的生成
        self.out_channels = out_channels

    def forward(self, x):
        # x: (batch_size, in_channels, H, W)
        batch_size, in_channels, H, W = x.size()
        # 确保特征图的高度和宽度可以被k整除
        # assert H % self.k == 0 and W % self.k == 0, "The height and width of the feature map must be divisible by k."

        # 调整特征图的维度以便于应用注意力机制
        x_reshaped = x.view(batch_size, in_channels, H // self.k, self.k, W // self.k,  self.k)

        # 通过线性层生成注意力权重
        attention = self.fc(x_reshaped.contiguous().view(-1, in_channels))
        attention = attention.view(batch_size, self.out_channels, H // self.k, self.k, W // self.k,  self.k)
        attention = F.softmax(attention, dim=-1)  # 对每个kxk窗口内的权重求softmax

        # 应用注意力权重并进行池化
        pooled_features = torch.sum(x_reshaped * attention, dim=-1)  # 对每个kxk窗口内的元素进行加权求和
        pooled_features = torch.sum(pooled_features, dim=-2)  # 对每个kxk窗口内的元素进行加权求和
        # 调整输出的维度
        pooled_features = pooled_features.view(batch_size, self.out_channels, H // self.k, W // self.k)

        return pooled_features
    
class GAP_net(nn.Module):

    def __init__(self, out_ch=28):
        super(GAP_net, self).__init__()
        middle_channel = 28
        self.unet1 = Unet(28, 28)
        self.unet2 = Unet(28, 28)
        self.unet3 = Unet_2(28, 28)
        self.unet4 = Unet_2(middle_channel, middle_channel)
        self.unet5 = Unet_2(middle_channel, middle_channel)
        self.unet6 = Unet_2(middle_channel, middle_channel)
        self.unet7 = Unet_2(28, 28)
        self.unet8 = Unet(28, 28)
        self.unet9 = Unet(28, 28)
        
        # unet-arc define
        self.pool_x = AttentionPooling(28, 28, 2)
        self.pool_phi = AttentionPooling(28, 28, 2)
        self.pool_y = AttentionPooling(1, 1, 2)
        
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
        
        x = self.pool_x(x)
        Phi = self.pool_phi(Phi)
        # TODO：交换在dim0之后增加一个维度为1
        y_tmp = torch.unsqueeze(y1, 1)
        y2 = self.pool_y(y_tmp)
        y2 = torch.squeeze(y2, 1)
        Phi_s = torch.sum(Phi, dim=1)
        Phi_s[Phi_s==0] = 1
        
        ## stage 3
        yb = A(x,Phi)
        x = x + At(torch.div(y2-yb,Phi_s),Phi)
        x = shift_back(x)
        x = self.unet3(x)
        x = shift(x)
        

        
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
        
        
        ### stage 7
        yb = A(x,Phi)
        x = x + At(torch.div(y2-yb,Phi_s),Phi)
        x = shift_back(x)
        x = self.unet7(x)      
        x = shift(x)
        
        x = self.upsample1_x(x)           # 128-64
        Phi = self.upsample1_phi(Phi)
        # print(x)
        # y = self.upsample1_y(y)
        Phi_s = torch.sum(Phi, dim=1)
        Phi_s[Phi_s==0] = 1
        
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