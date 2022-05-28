import torch
import torch.nn as nn
import torch.nn.functional as F


class conv1d_act_norm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, 
            bias=False, stride = 1, dilation = 1, padding = 'same', act=None, 
            using_bn=False):
        super().__init__()

        self.conv_1d = nn.Conv1d(
                in_channels, out_channels, kernel_size,
                # padding=(kernel_size//2), bias=bias, stride = stride, dilation = dilation)
                padding=padding, bias=bias, stride = stride, 
                dilation = dilation,
                )

        self.act = act
        self.using_bn = using_bn
        self.batch_norm = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        x = self.conv_1d(x)
        if self.using_bn is True:
            x = self.batch_norm(x)
        x = self.act(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class DR_Block(nn.Module):
    def __init__(self, in_channels, kernel_size=3, 
            # bias=False, stride = 1, dilation = 1, padding = 'same', 
            act=None, 
            using_bn=False):
        super().__init__()

        self.conv_1d_1 = nn.Conv1d(in_channels, in_channels*4, kernel_size,padding='same')
        self.conv_1d_2 = nn.Conv1d(in_channels*4, in_channels, kernel_size,padding='same')

        self.act = act
        self.SELayer = SELayer(in_channels)

    def forward(self, x):
        input = x
        x = self.conv_1d_1(x)
        x = self.act(x)
        
        x = self.conv_1d_2(x)
        x = self.SELayer(x)

        output = x+input
        output = self.act(output)
        return output


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


class Model(nn.Module):
    def __init__(self, in_c=1,out_c=1,act=nn.PReLU(),using_bn=True,denoise_mode=False):
        super().__init__()

        self.path1 = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding='same', bias=True, stride = 1),
            DR_Block(32,kernel_size=3,act=act),
            DR_Block(32,kernel_size=3,act=act),
            DR_Block(32,kernel_size=3,act=act),
        )
        self.path1_conv1 = nn.Conv1d(32, out_c, 3, padding='same', bias=True, stride=1)

        self.path2 = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding='same', bias=True, stride = 1),
            DR_Block(32,kernel_size=5,act=act),
            DR_Block(32,kernel_size=5,act=act),
            DR_Block(32,kernel_size=5,act=act),
        )
        self.path2_conv1 = nn.Conv1d(32, out_c, 3, padding='same', bias=True, stride=1)

        self.path3 = nn.Sequential(
            nn.Conv1d(1, 32, 9, padding='same', bias=True, stride = 1),
            DR_Block(32,kernel_size=9,act=act),
            DR_Block(32,kernel_size=9,act=act),
            DR_Block(32,kernel_size=9,act=act),
        )
        self.path3_conv1 = nn.Conv1d(32, out_c, 3, padding='same', bias=True, stride=1)

        # self.reg = nn.Conv1d(12, out_c, 1, padding='same', bias=True, stride = 1)
        # self.denoise_mode = denoise_mode


    def forward(self, x):

        x1 = self.path1(x)
        # print('x1',x1.shape)
        x1 = self.path1_conv1(x1)
        # print('x1',x1.shape)

        x2 = self.path2(x)
        # print('x2',x2.shape)
        x2 = self.path2_conv1(x2)
        # print('x2',x2.shape)
        
        x3 = self.path3(x)
        # print('x3',x3.shape)
        x3 = self.path3_conv1(x3)
        # print('x3',x3.shape)

        fc_out = x1+x2+x3

        # fc_out = self.path3_conv1(fc_out)
        # print('fc_out',fc_out.shape)
        return fc_out



def load_pertrain(model,model_path='./'):
    pertrain_model = torch.load(model_path)
    pertrain_dict = pertrain_model.state_dict()

    model_dict = model.state_dict()

    pertrained_dict = {}
    for k,v in pertrain_dict.items():
        if k in model_dict:
            if pertrain_dict[k].size() == model_dict[k].size():
                pertrained_dict[k]=v
            else:
                print('{} filter'.format(k))
        else:
            print('{} filter'.format(k))
    model_dict.update(pertrained_dict)
    model.load_state_dict(model_dict)
    return model


if __name__ == "__main__":

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    x = torch.rand(2,1,7168).to(device)

    # act=nn.PReLU()
    # bias=False



    model2 = Model(in_c=1,out_c=3)
    # model2.cuda()
    model2.to(device)

    print('x',x.size())
    y = model2(x)
    print(y.size()) #2 1 5000

