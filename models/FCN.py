import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class conv1d_act_norm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, 
            bias=False, stride = 1, dilation = 1, padding = 'valid', act=None, 
            using_bn=False):
        super().__init__()

        self.conv_1d = nn.Conv1d(
                in_channels, out_channels, kernel_size,
                # padding=(kernel_size//2), bias=bias, stride = stride, dilation = dilation)
                padding=padding, 
                bias=bias, stride = stride, 
                dilation = dilation,
                )

        self.act = act
        self.using_bn = using_bn
        self.batch_norm = nn.BatchNorm1d(out_channels)

        # self.kernel_size = kernel_size
        self.stride = stride
        # self.dilation = dilation

    def forward(self, x):
        ori_len = x.shape[-1]

        x = self.conv_1d(x)
        if self.stride >= 2 and ori_len%self.stride == 0:
            new_len = x.shape[-1]
            target_len = ori_len//self.stride
            pad_w = abs(target_len-new_len)
            if pad_w != 0:
                x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2])

        if self.using_bn is True:
            x = self.batch_norm(x)
        
        x = self.act(x)
        return x

class trans_conv1d_act_norm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, 
            bias=False, stride = 1, dilation = 1, padding='valid', act=None, 
            using_bn=False):
        super().__init__()

        # print('padding',padding)
        self.conv_1d = nn.ConvTranspose1d(
                in_channels, out_channels, kernel_size,
                # padding=(kernel_size//2), bias=bias, stride = stride, dilation = dilation)
                # padding=padding, 
                bias=bias, stride = stride, 
                dilation = dilation,
                )

        self.act = act
        self.using_bn = using_bn
        self.batch_norm = nn.BatchNorm1d(out_channels)

        self.stride = stride
        self.kernel_size = kernel_size
        self.dilation = dilation

    def forward(self, x):
        ori_len = x.shape[-1]

        x = self.conv_1d(x)

        new_len = x.shape[-1]
        pad_w = abs(ori_len*self.stride - new_len)

        # print(ori_len)
        # print(new_len)
        # print(pad_w)
        if pad_w>0:
            # x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2])
            x = x[:,:,pad_w//2:new_len-(pad_w-pad_w // 2)]

        # print(x.shape[-1])

        if self.using_bn is True:
            x = self.batch_norm(x)
        x = self.act(x)
        return x

class context_contrast(nn.Module):
    def __init__(self, in_channels,out_channels,act=None,using_bn=False):
        super(context_contrast, self).__init__()

        self.local_path = nn.Sequential(conv1d_act_norm(in_channels, out_channels, 5, bias=False, 
                                            stride = 1, dilation = 10, act = act, using_bn=using_bn),
                                        conv1d_act_norm(out_channels, out_channels, 5, bias=False, 
                                            stride = 1, dilation = 10, act = act,using_bn=using_bn),
        )

        self.global_path = nn.Sequential(conv1d_act_norm(in_channels, out_channels, 5, bias=False, 
                                        stride = 1, dilation = 1, act = act, using_bn=using_bn),
                                        conv1d_act_norm(out_channels, out_channels, 5, bias=False, 
                                        stride = 1, dilation = 1, act = act, using_bn=using_bn),
        )

    def forward(self, x):
        x1 = self.local_path(x)
        x2 = self.global_path(x)
        x3 = x1-x2
        x_out = torch.cat((x1,x2,x3),1) # (bs, length, channel*3)
        return x_out

class DownSample(nn.Module):
    def __init__(self, in_channels,out_channels,downsample_rate=0.5):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=downsample_rate),
                                  nn.Conv1d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x

class SkipUpSample(nn.Module):
    def __init__(self, in_channels,s_factor,upsample_rate=2):
        super().__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=upsample_rate),
                                nn.Conv1d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x

class decode_stage(nn.Module):
    def __init__(self, in_channels,in_channels_skip,kernel_size=3,upsample_rate=2,act=None,using_bn=None):
        super().__init__()

        self.upsample = nn.Sequential(nn.Upsample(scale_factor=upsample_rate),
                        nn.Conv1d(in_channels, in_channels, 1, stride=1, padding='same', bias=False))
        
        self.conv1 = conv1d_act_norm(in_channels+in_channels_skip, in_channels_skip, kernel_size=kernel_size, 
                                        act=act,using_bn=using_bn)
        self.conv2 = conv1d_act_norm(in_channels_skip, in_channels_skip, kernel_size=kernel_size, 
                                        act=act,using_bn=using_bn)
        self.conv3 = conv1d_act_norm(in_channels_skip, in_channels_skip, kernel_size=kernel_size, 
                                        act=act,using_bn=using_bn)

    def forward(self, x, y):
        x = self.upsample(x)
        x_out = torch.cat((x,y),1) # (bs, length, channel*3)
        x_out = self.conv1(x_out)

        x_out = x_out+y
        x_out = self.conv2(x_out)

        x_out = x_out+y
        x_out = self.conv3(x_out)
        
        x_out = x_out+y
        return x_out




class Model(nn.Module):
    def __init__(self, in_c=1,out_c=1):
        super().__init__()

        # act=nn.PReLU()
        act=nn.ELU()

        self.encode_cnn1 = conv1d_act_norm(in_c, 40, kernel_size=16, stride=2, act=act,using_bn=True)
        self.encode_cnn2 = conv1d_act_norm(40, 20, kernel_size=16, stride=2, act=act,using_bn=True)
        self.encode_cnn3 = conv1d_act_norm(20, 20, kernel_size=16, stride=2, act=act,using_bn=True)
        self.encode_cnn4 = conv1d_act_norm(20, 20, kernel_size=16, stride=2, act=act,using_bn=True)
        self.encode_cnn5 = conv1d_act_norm(20, 40, kernel_size=16, stride=2, act=act,using_bn=True)
        self.encode_cnn6 = conv1d_act_norm(40, 1, kernel_size=16, stride=1, padding='same', act=act,using_bn=True)


        self.decode_cnn6 = trans_conv1d_act_norm(1, 1, kernel_size=1, stride=1, act=act,using_bn=True)
        self.decode_cnn5 = trans_conv1d_act_norm(1, 40, kernel_size=16, stride=2, act=act,using_bn=True)
        self.decode_cnn4 = trans_conv1d_act_norm(40, 20, kernel_size=16, stride=2, act=act,using_bn=True)
        self.decode_cnn3 = trans_conv1d_act_norm(20, 20, kernel_size=16, stride=2, act=act,using_bn=True)
        self.decode_cnn2 = trans_conv1d_act_norm(20, 20, kernel_size=16, stride=2, act=act,using_bn=True)
        self.decode_cnn1 = trans_conv1d_act_norm(20, 40, kernel_size=16, stride=2, act=act,using_bn=True)


        self.reg = nn.Sequential(
                nn.Conv1d(40, out_c, 1, padding='same', bias=False, stride = 1),
                # nn.Tanh()
        )


    def forward(self, x):
        # X (N,C,L)
        x = self.encode_cnn1(x)
        # print('encode_cnn1',x.shape)
        x = self.encode_cnn2(x)
        # print('encode_cnn2',x.shape)
        x = self.encode_cnn3(x)
        # print('encode_cnn3',x.shape)
        x = self.encode_cnn4(x)
        # print('encode_cnn4',x.shape)
        x = self.encode_cnn5(x)
        # print('encode_cnn5',x.shape)
        x = self.encode_cnn6(x)
        # print('encode_cnn6',x.shape)

        x = self.decode_cnn6(x)
        # print('decode_cnn6',x.shape)
        x = self.decode_cnn5(x)
        # print('decode_cnn5',x.shape)
        x = self.decode_cnn4(x)
        # print('decode_cnn4',x.shape)
        x = self.decode_cnn3(x)
        # print('decode_cnn3',x.shape)
        x = self.decode_cnn2(x)
        # print('decode_cnn2',x.shape)
        x = self.decode_cnn1(x)

        fc_out = self.reg(x)
        # print('fc_out',fc_out.shape)
        return fc_out



if __name__ == "__main__":

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    x = torch.rand(2,1,7168).to(device)

    # act=nn.PReLU()
    # bias=False

    model2 = Model(in_c=1)
    # model2.cuda()
    model2.to(device)

    print('x',x.size())
    y = model2(x)
    print(y.size())

