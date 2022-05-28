import torch
import torch.nn as nn
import torch.nn.functional as F


# class conv1d_act_norm(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, 
#             bias=False, stride = 1, dilation = 1, padding = 'same', act=None, 
#             using_bn=False):
#         super().__init__()

#         self.conv_1d = nn.Conv1d(
#                 in_channels, out_channels, kernel_size,
#                 # padding=(kernel_size//2), bias=bias, stride = stride, dilation = dilation)
#                 padding=padding, bias=bias, stride = stride, 
#                 dilation = dilation,
#                 )

#         self.act = act
#         self.using_bn = using_bn
#         self.batch_norm = nn.BatchNorm1d(out_channels)
        
#     def forward(self, x):
#         x = self.conv_1d(x)
#         if self.using_bn is True:
#             x = self.batch_norm(x)
#         x = self.act(x)
#         return x


class conv1d_act_norm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, 
            bias=False, stride = 1, dilation = 1, padding = 'same', act=None, 
            using_bn=False):
        super().__init__()

        if stride == 2: padding='valid'
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
            bias=False, stride = 1, dilation = 1, padding='same', act=None, 
            using_bn=True):
        super().__init__()

        # if stride == 2: padding='vaild'
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

class decode_stage_v1(nn.Module):
    def __init__(self, in_channels,in_channels_skip,upsample_rate=2,act=None,kernel_size=3,using_bn=None):
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


class decode_stage(nn.Module):
    def __init__(self, in_channels,out_channel,act=None):
        super().__init__()

        self.conv1_upsample = trans_conv1d_act_norm(in_channels, in_channels, kernel_size=8, stride=2, act=act)
        self.conv2 = trans_conv1d_act_norm(in_channels+out_channel, out_channel, kernel_size=8, stride=1, act=act)
        self.conv3 = trans_conv1d_act_norm(out_channel, out_channel, kernel_size=8, stride=1, act=act)

    def forward(self, x, y):

        # print('x',x.shape)
        x = self.conv1_upsample(x)
        # print('x',x.shape)
        # print('y',y.shape)
        x_out = torch.cat((x,y),1)

        # x_out = x_out+y
        x_out = self.conv2(x_out)

        # x_out = x_out+y
        x_out = self.conv3(x_out)
        
        # x_out = x_out+y
        return x_out


class Model(nn.Module):
    def __init__(self, in_c=1,out_c=1,act=nn.PReLU(),using_bn=True,denoise_mode=False):
        super().__init__()

        self.encode_stage1 = nn.Sequential(
                conv1d_act_norm(in_c, 4, kernel_size=9, act=act,using_bn=using_bn),
                conv1d_act_norm(4, 4, kernel_size=9, act=act,using_bn=using_bn),
                # nn.MaxPool1d(3, stride=2)
                )

        self.encode_stage2 = nn.Sequential(
                conv1d_act_norm(4, 4, kernel_size=9, act=act,stride=2, using_bn=using_bn),
                conv1d_act_norm(4, 8, kernel_size=9, act=act,using_bn=using_bn),
                conv1d_act_norm(8, 8, kernel_size=9, act=act,using_bn=using_bn),
                )

        self.encode_stage3 = nn.Sequential(
                conv1d_act_norm(8, 8, kernel_size=9, act=act,stride=2, using_bn=using_bn),
                conv1d_act_norm(8, 16, kernel_size=9, act=act,using_bn=using_bn),
                conv1d_act_norm(16, 16, kernel_size=9, act=act,using_bn=using_bn),
                )

        self.encode_stage4 = nn.Sequential(
                conv1d_act_norm(16, 16, kernel_size=9, act=act,stride=2, using_bn=using_bn),
                conv1d_act_norm(16, 32, kernel_size=9, act=act,using_bn=using_bn),
                conv1d_act_norm(32, 32, kernel_size=9, act=act,using_bn=using_bn),
                )

        self.encode_stage5 = nn.Sequential(
                conv1d_act_norm(32, 32, kernel_size=9, act=act,stride=2, using_bn=using_bn),
                conv1d_act_norm(32, 64, kernel_size=9, act=act,using_bn=using_bn),
                conv1d_act_norm(64, 64, kernel_size=9, act=act,using_bn=using_bn),
                )


        self.decode_stage4 = decode_stage(64,32,act=act)
        self.decode_stage3 = decode_stage(32,16,act=act)
        self.decode_stage2 = decode_stage(16,8,act=act)
        self.decode_stage1 = decode_stage(8,4,act=act)

        self.reg = nn.Conv1d(4, out_c, 1, padding='same', bias=True, stride = 1)
        # self.denoise_mode = denoise_mode


    def forward(self, x,verbose=False):
        x1 = self.encode_stage1(x)
        if verbose==True: print('x1',x1.shape)
        x2 = self.encode_stage2(x1)
        if verbose==True: print('x2',x2.shape)
        x3 = self.encode_stage3(x2)
        if verbose==True: print('x3',x3.shape)
        x4 = self.encode_stage4(x3)
        if verbose==True: print('x4',x4.shape)
        x5 = self.encode_stage5(x4)
        if verbose==True: print('x5',x5.shape)

        out4 = self.decode_stage4(x5,x4)
        if verbose==True: print('out4',out4.shape)
        out3 = self.decode_stage3(out4,x3)
        if verbose==True: print('out3',out3.shape)
        out2 = self.decode_stage2(out3,x2)
        if verbose==True: print('out2',out2.shape)
        out1 = self.decode_stage1(out2,x1)
        if verbose==True: print('out1',out1.shape)
        fc_out = self.reg(out1)
        if verbose==True: print('fc_out',fc_out.shape)
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
    y = model2(x ,verbose=True)
    print(y.size()) #2 1 5000

