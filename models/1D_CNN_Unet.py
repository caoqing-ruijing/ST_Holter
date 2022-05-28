# swin transformer block is borrowed https://github.com/microsoft/Swin-Transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])

        # coords_h = torch.arange(1)
        # coords_w = torch.arange(window_size[0])

        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        # print('coords',coords)
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        # print('coords_flatten',coords_flatten)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        # print('relative_coords',relative_coords)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        # print('relative_coords',relative_coords)

        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        # print('relative_coords',relative_coords)
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        # print('relative_position_index',relative_position_index)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        # print('relative_position_bias',relative_position_bias.shape)
        relative_position_bias = relative_position_bias.view(
                    self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        # print('relative_position_bias',relative_position_bias.shape)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


def window_reverse_1d(windows, window_size, L):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, L, C)
    """
    B = int(windows.shape[0] / (L / window_size))
    x = windows.view(B, L // window_size, window_size, -1)
    # print('x window_reverse_1d',x.shape)
    x = x.permute(0, 1, 2, 3).contiguous().view(B, L, -1)
    # print('x',x.shape)
    return x


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            # dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            dim, window_size=(1,self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # print('shift_size',self.shift_size)
        if self.shift_size > 0:
            # print('shift_size',shift_size)
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            # print('h_slices',h_slices)
            # print('w_slices',w_slices)/
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # print('window_size',self.window_size)
            # print('img_mask',img_mask.shape)
            img_mask = img_mask.view(1,W,1)
            # mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            # print('img_mask',img_mask,img_mask.shape,self.window_size)
            # assert 1>2
            mask_windows = window_partition_1d(img_mask, self.window_size)  # nW, window_size, window_size, 1
            # print('mask_windows',mask_windows.shape)
            # mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            mask_windows = mask_windows.view(-1, 1 * self.window_size)
            # print('mask_windows',mask_windows.shape)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            # print('attn_mask',attn_mask.shape) #no batch size
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        # H, W = self.input_resolution
        W = self.input_resolution[0]
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"
        # print('Swin input',x.shape)

        shortcut = x
        x = self.norm1(x)
        # x = x.view(B, H, W, C)
        # x = x.view(B, 1, W, C) #
        # print('x view',x.shape)

        # cyclic shift
        if self.shift_size > 0:
            # print('x',x.shape)
            # shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_x = torch.roll(x, shifts=(-self.shift_size), dims=(1))
            # print('shifted_x',shifted_x.shape)
            # assert 1>2
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition_1d(shifted_x, self.window_size)  # nW*B, window_size, C
        # print('x_windows',x_windows.shape) # 500,10,24

        # x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        # print('x_windows',x_windows.shape)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        # print('attn_windows',attn_windows.shape)

        # merge windows
        # attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        # shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        shifted_x = window_reverse_1d(attn_windows, self.window_size, L)  # B H' W' C
        # print('shifted_x window_reverse_1d',shifted_x.shape)

        # reverse cyclic shift
        if self.shift_size > 0:
            # print('self.shift_size',self.shift_size)
            # x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            x = torch.roll(shifted_x, shifts=(self.shift_size), dims=(1))
        else:
            x = shifted_x
        # x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=5000, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        # img_size = to_2tuple(img_size)
        # patch_size = to_2tuple(patch_size)

        img_size = [img_size]
        patch_size = [patch_size]

        patches_resolution = [img_size[0] // patch_size[0],]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # B, C, H, W = x.shape
        B, C, L = x.shape
        # FIXME look at relaxing size constraints
        x = self.proj(x).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    # def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        # self.input_resolution = input_resolution
        self.dim = dim
        # self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        # self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(2 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)
    def forward(self, x):
        """
        x: B, H*W, C
        """
        # H, W = self.input_resolution

        # L = self.input_resolution
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"
        assert L % 2 == 0, f"x size ({L}) are not even."

        # x = x.view(B, H, W, C)
        x0 = x[:, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, :]  # B H/2 W/2 C
        # x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        # x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        # print(' PatchMerging x0',x0.shape)
        # print(' PatchMerging x1',x1.shape)
        x = torch.cat([x0, x1], -1)  # B H/2 W/2 4*C
        # print('x PatchMerging x cat',x.shape)
        x = x.view(B, -1, 2 * C)  # B H/2*W/2 4*C
        # print('PatchMerging x',x.shape)

        # x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        # x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        # x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        # x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        # x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        # x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)
        # print('PatchMerging reduction',x.shape)
        return x

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, downsample_scale=2,
                 upsample_mode='down',use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.upsample_mode = upsample_mode

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
                for i in range(depth)
                ]
            )

        # patch merging layer
        if downsample is not None:
            # self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)

            # self.downsample = downsample(dim, norm_layer=norm_layer)

            if upsample_mode == 'down':
                self.downsample = nn.Sequential(nn.Upsample(scale_factor=0.5),
                                nn.Conv1d(int(dim), dim*2, 1, stride=1, padding=0, bias=False)
                                )

                # self.downsample = nn.Upsample(scale_factor=0.5)
                # self.Conv1d = nn.Conv1d(dim, dim*2, 1, stride=1, padding=0, bias=False)

            else:
                self.downsample = nn.Sequential(nn.Upsample(scale_factor=2),
                                nn.Conv1d(dim, int(dim/2), 1, stride=1, padding=0, bias=False)                                )

                # self.downsample = nn.Upsample(scale_factor=2)
                # self.Conv1d = nn.Conv1d(dim, int(dim/2), 1, stride=1, padding=0, bias=False)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        # print('x',x.shape)
        if self.downsample is not None:
            # x = self.downsample(x)
            if self.upsample_mode in ['up','down']:
                x = x.transpose(1, 2)
                # print('self.input_resolution',self.input_resolution,'dim',self.dim)
                # print('x -> transpose',x.shape)
                x = self.downsample(x)
                # print('x -> downsample',x.shape)
                # x = self.Conv1d(x)
                # print('x -> Conv1d',x.shape)
                x = x.transpose(1, 2)
                # print('x -> transpose',x.shape)
            else:
                x = self.downsample(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_partition_1d(x, window_size):
    """
    Args:
        x: (B, L, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, C)
    """
    B, L, C = x.shape # [2, 5000, 1]
    # print('x',x.shape,'window_size',window_size)
    x = x.view(B, L // window_size, window_size, C) # # B, nW, window_size, C [2, 20, 250, 1]
    # print(x.shape)
    windows = x.permute(0, 1, 2, 3).contiguous().view(-1, window_size, C)
    # windows = x.view(-1, window_size, C) # nW*B, window_size, window_size, C [40, 250, 1]
    return windows

class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, upsample_mode='down',use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            # self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
            if upsample_mode == 'down':
                self.upsample = nn.Sequential(nn.Upsample(scale_factor=0.5),
                                nn.Conv1d(input_resolution, input_resolution, 1, stride=1, padding=0, bias=False))
            else:
                self.upsample = nn.Sequential(nn.Upsample(scale_factor=2),
                                nn.Conv1d(input_resolution, input_resolution, 1, stride=1, padding=0, bias=False))

        else:
            self.upsample = None

            
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x

class PatchExpand_orginl(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B,-1,C//4)
        x= self.norm(x)
        return x

class PatchExpand(nn.Module):
    # def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        # self.input_resolution = input_resolution
        self.dim = dim
        # self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        # self.expand = nn.Linear(dim, dim, bias=False) if dim_scale==2 else nn.Identity()
        self.expand = nn.Linear(dim, dim, bias=False)
        self.norm = norm_layer(dim // dim_scale)
        self.dim_scale = dim_scale
        # self.norm = norm_layer(dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        x = rearrange(x, 'b w (p1 c)-> b (w p1) c', p1=self.dim_scale, c=C//self.dim_scale)
        x= self.norm(x)
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


class Model(nn.Module):
    def __init__(self, in_c=1,out_c=1,
                img_size = 3584,
                patch_size=2,
                embed_dim = 24,
                norm_layer=nn.LayerNorm,
                window_size=56,
                patch_norm=True,
                drop_rate=0,
                act=nn.PReLU(),
                # using_bn=True,
                denoise_mode=False,
                mlp_ratio=4,
                drop_path_rate=0,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                ):
        super().__init__()

        self.denoise_mode = denoise_mode
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_c, embed_dim=embed_dim,
            norm_layer=nn.LayerNorm)

        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # print('patches_resolution',patches_resolution)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        i_layer = 0

        self.num_layers = len(depths)
        # num_heads=[3, 6, 12, 24]
        qkv_bias=True
        qk_scale=None
        attn_drop_rate=0.
        drop_path_rate=0.1
        self.mlp_ratio = mlp_ratio

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(1,patches_resolution[0] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               upsample_mode='down',
                               )
            self.layers.append(layer)

        self.norm = norm_layer(int(embed_dim * 2 ** i_layer))

        # build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2*int(embed_dim*2**(self.num_layers-1-i_layer)),
            int(embed_dim*2**(self.num_layers-1-i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer ==0 :
                layer_up = PatchExpand(int(embed_dim * 2 ** (self.num_layers-1-i_layer)),  #192
                                    dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer(
                        dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer)),
                        input_resolution=(1,patches_resolution[0] // (2 ** (self.num_layers-1-i_layer))),
                        depth=depths[(self.num_layers-1-i_layer)],
                        num_heads=num_heads[(self.num_layers-1-i_layer)],
                        window_size=window_size,
                        mlp_ratio=self.mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:(self.num_layers-1-i_layer)]):sum(depths[:(self.num_layers-1-i_layer) + 1])],
                        norm_layer=norm_layer,
                        downsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                        upsample_mode='up',
                    )

            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.final_swin_layer = BasicLayer(
                        dim=int(embed_dim * 2),
                        input_resolution=(1,patches_resolution[0] // (2 ** (self.num_layers-1-i_layer))),
                        depth=depths[(self.num_layers-1-i_layer)],
                        num_heads=num_heads[(self.num_layers-1-i_layer)],
                        window_size=window_size,
                        mlp_ratio=self.mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:(self.num_layers-1-i_layer)]):sum(depths[:(self.num_layers-1-i_layer) + 1])],
                        norm_layer=norm_layer,
                        # downsample=PatchExpand,
                    )
        self.layer_up = PatchExpand(int(embed_dim * 2),dim_scale=patch_size)

        if patch_size > 1:
            self.norm_up= norm_layer(embed_dim // int(patch_size*0.5))
            self.reg = nn.Conv1d(embed_dim // int(patch_size*0.5), out_c, 1, padding='same', bias=True, stride = 1)
        elif patch_size == 1:
            self.norm_up= norm_layer(embed_dim*2)
            self.reg = nn.Conv1d(embed_dim*2, out_c, 1, padding='same', bias=True, stride = 1)
        

    def forward(self, input):
        x_stem = self.patch_embed(input)

        x = self.pos_drop(x_stem)

        x_downsample = []
        for i,layer in enumerate(self.layers):
            x_downsample.append(x)
            x = layer(x)

        x = self.norm(x)  # B L C
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x,x_downsample[len(x_downsample)-1-inx]],-1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

        x = torch.cat([x,x_stem],-1)
        x = self.final_swin_layer(x)
        if x.shape[1] != input.shape[1]:
            x = self.layer_up(x)

        x = self.norm_up(x)  # B L C
        x = x.transpose(1, 2)  #B,C,H,W 

        fc_out = self.reg(x)

        if self.denoise_mode is True:
            return fc_out+input
        else:
            return fc_out


if __name__ == "__main__":

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    input_length = 7168 #3584 7168
    x = torch.rand(2,1,input_length).to(device)

    # act=nn.PReLU()
    # bias=False

    model2 = Model(
                    in_c=1,
                    # img_size=3584,
                    patch_size=1,
                    img_size=input_length,
                    embed_dim=24, #24 48
                    window_size=112, # 112 224 448
                    depths=[2, 2, 6, 2],
                    num_heads=[3, 6, 12, 24],
                    # depths=[2, 2],
                    # num_heads=[3, 6],
                    # depths=[2, 6, 2],
                    # num_heads=[3, 12, 24],
                    )
    # model2.cuda()
    model2.to(device)

    print('x',x.size())
    y = model2(x)
    print(y.size()) # 2 1 5000
    
