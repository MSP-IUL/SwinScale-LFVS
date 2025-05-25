import math
import torch
import torch.nn.functional as F
from torch import nn
import custom_layers
import blocks
import math
import torch
import torch.nn.functional as F
from torch import nn
import custom_layers
import blocks

class MSCS(nn.Module):
    def __init__(self):
        super(MSCS, self).__init__()
        self.conv3d_3 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.conv3d_5 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(5,5,5), stride=(1,1,1), padding=(2,2,2))
        self.conv3d_7 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(7,7,7), stride=(1,1,1), padding=(3,3,3))
        
        self.confuse = nn.Conv3d(in_channels=192, out_channels=64, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))

    def forward (self, x_init):
        x1 = self.conv3d_3(x_init)
        x2 = self.conv3d_5(x_init)
        x3 = self.conv3d_7(x_init)
        Initial_featfuse = torch.cat([x1, x2, x3], dim=1)
        
        fuse = self.confuse(Initial_featfuse)
        return fuse
class FuseBlock3D(nn.Module):
    def __init__(self, kernel_size, i):
        super(FuseBlock3D, self).__init__()
       
        padding = tuple(k // 2 for k in kernel_size)  # Dynamically compute padding for 3D
        self.conv1 = nn.Conv3d(128, 128, kernel_size=kernel_size, stride=(1, 1, 1), padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(128, 128, kernel_size=kernel_size, stride=(1, 1, 1), padding=padding)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = x + out  # Residual connection
        return out

def make_FuseBlock3D(layer_num, kernel_size):
    layers = []
    for i in range(layer_num):
        layers.append(FuseBlock3D(kernel_size, i))
    return nn.Sequential(*layers)


class TRNS(nn.Module):
    def __init__(self, cfg) -> None:
        super(TRNS, self).__init__()
        channel = 64
        factor = cfg.factor
        angular_in = cfg.angin
        angular_out = cfg.angout
        self.factor = factor
        self.angRes = angular_in
        self.angRes_out = angular_out

        # Architecture
        self.conv3d_initial = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(1,1,1), stride=(1,1,1))
        
        self.down_stage1 = blocks.Stage(
            in_channels=64,
            dim=128,
            depth=4,
            num_heads=4,
            window_size=[4,16,16],
            qkv_bias=True,
            qk_scale=None,
            mul_attn_ratio=0.75,
            mlp_ratio=2.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            reshape="down"
        )

        self.down_stage2 = blocks.Stage(
            in_channels=128,
            dim=256,
            depth=4,
            num_heads=8,
            window_size=[4,16,16],
            qkv_bias=True,
            qk_scale=None,
            mul_attn_ratio=0.75,
            mlp_ratio=2.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            reshape="down"
        )

        self.BottleNeck = blocks.Stage(
            in_channels=256,
            dim=256,
            depth=4,
            num_heads=8,
            window_size=[4,16,16],
            qkv_bias=True,
            qk_scale=None,
            mul_attn_ratio=0.75,
            mlp_ratio=2.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            reshape="none"
        )

        self.up_stage1 = blocks.Stage(
            in_channels=256,
            dim=128,
            depth=4,
            num_heads=8,
            window_size=[4,16,16],
            qkv_bias=True,
            qk_scale=None,
            mul_attn_ratio=0.75,
            mlp_ratio=2.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            reshape="up"
        )

        self.up_stage2 = blocks.Stage(
            in_channels=128,
            dim=64,
            depth=4,
            num_heads=4,
            window_size=[4,16,16],
            qkv_bias=True,
            qk_scale=None,
            mul_attn_ratio=0.75,
            mlp_ratio=2.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            reshape="up"
        )

          # Reduce channels
        self.conv3x3 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        #self.conv1x1 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))
        self.MSCS = MSCS()
        self.vsfus = make_FuseBlock3D(2, (3, 3, 3))
        self.conv_vs1 = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.conv_vs2 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))  # Output 64 channels

        self.Angular_UpSample = Upsample(channel, angular_in, factor)
        self.Out = nn.Conv2d(64, 1, 1, 1, 0, bias=False)
        self.Resup = Interpolation(angular_in, factor)

    def forward(self, x):
        x_initial = LFsplit(x, self.angRes)
        b, n, c, h, w = x_initial.shape

        Bicubic_up = self.Resup(x_initial)
        x_mean = x_initial.mean([1,3,4], keepdim=True)
        x_initial = x_initial - x_mean
        x_initial = x_initial.transpose(1,2)
        
        x = self.conv3d_initial(x_initial)
        
        x1 = self.down_stage1(x)  # 1, 128, 4, 32, 32
      
        x2 = self.down_stage2(x1)  # 1, 256, 4, 16, 16
      
        
        b = self.BottleNeck(x2)  # 1, 256, 4, 16, 16
        
        BX2= b+x2 
        #print('BX2', BX2.shape)
        y2 = self.up_stage1(BX2)  # Use the output of BottleNeck
       

        # Ensure the spatial dimensions of x2 and y2 match
        x1y2=x1+y2
        #print('skip connection x1y2:', x1y2.shape)
        y1 = self.up_stage2(x1y2)  # Concatenate y2 and x2 (output of down_stage2)
       
         
        # Apply 1x1 convolution to reduce channels from 128 to 64
        swin = self.conv3x3(y1)
       
        #out = self.conv1x1(y1)
        scale = self.MSCS(x_initial)
        featsfuse = torch.cat([swin, scale],dim=1)  # Addition after matching channels
        vsfus=self.vsfus(featsfuse) 
       
        vs1=self.conv_vs1(vsfus) 
        LFVS = self.conv_vs2(vs1)  
        b, c, d, h, w = LFVS.shape 
        x = LFVS.view(b, -1, c, h, w)
       
        HAR = self.Angular_UpSample(x)
        out = self.Out(HAR.contiguous().view(b * self.angRes_out * self.angRes_out, -1, h, w))
        out = FormOutput(out.view(b, -1, 1, h, w)) + FormOutput(Bicubic_up)
       
        return out



class Upsample(nn.Module):
    def __init__(self, channel, angular_in, factor):
        super(Upsample, self).__init__()
        self.an = angular_in
        self.an_out = angular_in * factor
        self.angconv = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=angular_in, stride=angular_in, padding=0),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.upsp = nn.Sequential(
            nn.Conv2d(channel, channel * factor * factor, kernel_size=1, padding=0),
            nn.PixelShuffle(factor),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        b, n, c, h, w = x.shape
        x = x.contiguous().view(b, n, c, h * w)
        x = torch.transpose(x, 1, 3)
        x = x.contiguous().view(b * h * w, c, self.an, self.an)
        up_in = self.angconv(x)

        out = self.upsp(up_in)

        out = out.view(b, h * w, -1, self.an_out * self.an_out)
        out = torch.transpose(out, 1, 3)
        out = out.contiguous().view(b, self.an_out * self.an_out, -1, h, w)
        return out


class Interpolation(nn.Module):
    def __init__(self, angular_in, factor):
        super(Interpolation, self).__init__()
        self.an = angular_in
        self.an_out = factor
        self.factor = factor

    def forward(self, x_mv):
        b, n, c, h, w = x_mv.shape
        x = x_mv.contiguous().view(b, n, c, h * w)
        x = torch.transpose(x, 1, 3)
        x = x.contiguous().view(b * h * w, c, self.an, self.an)

        out = F.interpolate(x, size=(self.factor, self.factor), mode='bicubic', align_corners=False)

        out = out.view(b, h * w, c, self.an_out * self.an_out)
        out = torch.transpose(out, 1, 3)
        out = out.contiguous().view(b, self.an_out * self.an_out, c, h, w)
        return out

def LFsplit(data, angRes):
    b, _, H, W = data.shape
    h = int(H / angRes)
    w = int(W / angRes)
    data_sv = []
    for u in range(angRes):
        for v in range(angRes):
            data_sv.append(data[:, :, u * h:(u + 1) * h, v * w:(v + 1) * w])

    data_st = torch.stack(data_sv, dim=1)
    return data_st


def FormOutput(x_sv):
    b, n, c, h, w = x_sv.shape
    angRes = int(math.sqrt(n))
    out = []
    kk = 0
    for u in range(angRes):
        buffer = []
        for v in range(angRes):
            buffer.append(x_sv[:, kk, :, :, :])
            kk += 1
        buffer = torch.cat(buffer, 3)
        out.append(buffer)
    out = torch.cat(out, 2)

    return out

