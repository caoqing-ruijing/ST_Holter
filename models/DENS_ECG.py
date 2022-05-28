import torch
import torch.nn as nn
import torch.nn.functional as F


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

        self.cnn1 = conv1d_act_norm(in_c, 32, kernel_size=3, act=act,using_bn=False)
        self.cnn2 = conv1d_act_norm(32, 64, kernel_size=3, act=act,using_bn=False)
        self.cnn3 = conv1d_act_norm(64, 128, kernel_size=3, act=act,using_bn=False)

        self.LSTM1 = nn.LSTM(128,250,1,bidirectional=True)
        self.LSTM2 = nn.LSTM(500,125,1,bidirectional=True)

        # self.reg = nn.Conv1d(4, out_c, 1, padding='same', bias=True, stride = 1)
        # self.denoise_mode = denoise_mode
        self.reg = nn.Linear(250, out_c)

    def forward(self, x,verbose=False):
        
        x = self.cnn1(x)
        # print('cnn1',x.shape)
        x = self.cnn2(x)
        # print('cnn2',x.shape)
        x = self.cnn3(x)
        # print('cnn3',x.shape) # bs,128,L

        x = torch.transpose(x, 1, 2) # bs,128,L -> # bs,L,128        
        # print('transpose',x.shape) # bs,128,L

        x = self.LSTM1(x)[0]
        # print('LSTM1',x.shape)

        x = self.LSTM2(x)[0]
        # print('LSTM2',x.shape)

        fc_out = self.reg(x)
        fc_out = torch.transpose(fc_out, 1, 2) # bs,L,C -> # bs,C,L        
        return fc_out
        


if __name__ == "__main__":

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    x = torch.rand(2,1,7168).to(device)

    # act=nn.PReLU()
    # bias=False

    model2 = Model(in_c=1,out_c=3)
    # model2.cuda()
    model2.to(device)

    print('x',x.size())
    y = model2(x ,verbose=True)
    print(y.size()) #2 1 5000

    predict=y
    num = predict.shape[0]
    print('predict',predict)
    print('num',num)
    # pre = torch.sigmoid(predict)
    # .view(num, -1)
    pre = torch.sigmoid(predict).reshape(num, -1)

    print('pre',pre.size()) #2 1 5000
