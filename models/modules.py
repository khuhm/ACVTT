from torch import nn
import torch
import importlib
import torch.nn.functional as F


class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, act='ReLU'):
        super(SingleConv, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_block=1, down_sample=False, up_sample=False, down_op=None, up_op=None, do_residual=False, act_after_res=True):
        super(ConvBlock, self).__init__()
        self.up_sample = up_sample
        self.down_sample = down_sample
        self.do_residual = do_residual
        self.act_after_res = act_after_res

        if up_sample:
            if up_op == 'transp_conv':
                self.upsampling_op = nn.ConvTranspose2d(in_channels, in_channels, kernel_size, 2, 1, output_padding=1)
            else:
                self.upsampling_op = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.blocks = nn.Sequential()
        if down_sample:
            if down_op == 'stride_conv':
                # self.downsample_op = nn.Conv2d(in_channels, in_channels, kernel_size, stride=2, padding=1)
                self.blocks.append(nn.Conv2d(in_channels, in_channels, kernel_size, stride=2, padding=1))
            else:
                # self.downsample_op = nn.AvgPool2d(2)
                self.blocks.append(nn.AvgPool2d(2))
        for i in range(num_block):
            if i > 0:
                in_channels_ = out_channels
            elif up_sample:
                in_channels_ = in_channels + out_channels
            else:
                in_channels_ = in_channels
            self.blocks.append(SingleConv(in_channels_, out_channels, kernel_size))

        if do_residual:
            if self.up_sample:
                self.skip = nn.Conv2d(in_channels + out_channels, out_channels, 1, bias=False)
            else:
                self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if act_after_res:
                self.nonlin = nn.ReLU(inplace=True)

    def forward(self, x, y=None):
        if self.up_sample:
            x = self.upsampling_op(x)
            x = torch.cat([x, y], dim=1)

        if self.do_residual:
            if self.down_sample:
                x = self.blocks[0](x)
            residual = self.skip(x)
            x = self.blocks[1:](x)
            x = x + residual
            if self.act_after_res:
                x = self.nonlin(x)
        else:
            x = self.blocks(x)

        return x


class TransferBlock(nn.Module):
    def __init__(self, args, dim, block_size, win_block_size):
        super(TransferBlock, self).__init__()

        self.do_norm = args.do_norm
        self.do_proj_qkv = args.do_proj_qkv
        self.do_proj_after_attn = args.do_proj_after_attn
        self.apply_attention = args.apply_attention
        self.block_size = block_size
        self.alpha_transf = args.alpha_transf
        self.zero_conv = args.zero_conv
        self.norm_type_for_attn = args.norm_type_for_attn
        self.do_proj_after_norm = args.do_proj_after_norm
        self.proj_after_norm_qk_only = args.proj_after_norm_qk_only
        self.num_block_proj_in = args.num_block_proj_in
        self.kernel_size_proj_in = args.kernel_size_proj_in
        self.num_block_proj_out = args.num_block_proj_out
        self.kernel_size_proj_out = args.kernel_size_proj_out
        self.do_proj_after_unfold = args.do_proj_after_unfold
        self.output_attention = args.output_attention
        self.output_residual = args.output_residual
        self.norm_order = args.norm_order
        self.norm_dim = args.norm_dim
        self.attn_softmax = args.attn_softmax
        self.do_proj_out_unfold = args.do_proj_out_unfold
        self.reduce_proj_qk_by_block = args.reduce_proj_qk_by_block
        self.reduce_proj_v_by_block = args.reduce_proj_v_by_block

        self.NL_in_NL = args.NL_in_NL
        self.win_do_proj_qkv = args.win_do_proj_qkv
        self.win_block_size = win_block_size
        self.win_do_norm = args.win_do_norm
        self.win_do_proj_out = args.win_do_proj_out

        if args.do_proj_qkv:
            self.proj_qkv = nn.ModuleList()
            for i in range(3):
                if (i == 2) and self.proj_after_norm_qk_only:
                    continue
                layers = nn.Sequential()
                for j in range(args.num_block_proj_in):
                    layers.append(ConvBlock(dim, dim, args.kernel_size_proj_in))
                    pass
                layers.append(nn.Conv2d(dim, dim, 1))
                self.proj_qkv.append(layers)

        block_dim = dim * self.block_size ** 2
        norm_dim = block_dim

        if args.do_proj_after_unfold:
            self.linear_unfold = nn.ModuleList()
            for i in range(3):
                if (i == 2) and self.proj_after_norm_qk_only:
                    continue
                if i == 2:
                    if args.reduce_proj_v_by_block:
                        proj_dim = block_dim // self.block_size
                    else:
                        proj_dim = block_dim
                elif args.reduce_proj_qk_dim:
                    if args.reduce_proj_qk_by_block:
                        proj_dim = block_dim // self.block_size
                    else:
                        proj_dim = args.proj_dim
                    norm_dim = proj_dim
                else:
                    proj_dim = block_dim
                layers = nn.Sequential()
                layers.append(nn.Linear(block_dim, proj_dim))
                self.linear_unfold.append(layers)
            pass

        if args.do_norm:
            if args.norm_type_for_attn == 'l2norm':
                pass
            elif args.norm_type_for_attn == 'sep_layernorm':
                self.norm_q = nn.LayerNorm(norm_dim, elementwise_affine=args.layernorm_affine,
                                           bias=args.layernorm_bias)
                self.norm_k = nn.LayerNorm(norm_dim, elementwise_affine=args.layernorm_affine,
                                           bias=args.layernorm_bias)
            else:
                self.norm = nn.LayerNorm(norm_dim, elementwise_affine=args.layernorm_affine, bias=args.layernorm_bias)

        if args.do_proj_after_norm:
            self.linear_qkv = nn.ModuleList()
            for i in range(3):
                if (i == 2) and self.proj_after_norm_qk_only:
                    continue
                layers = nn.Sequential()
                layers.append(nn.Linear(block_dim, block_dim))
                self.linear_qkv.append(layers)
            pass

        if args.do_proj_out_unfold:
            self.proj_out_unfold = nn.Sequential()
            for i in range(args.num_block_proj_out):
                if (i == 0) and args.reduce_proj_v_by_block:
                    dim_in = block_dim // self.block_size
                else:
                    dim_in = block_dim
                self.proj_out_unfold.append(nn.Linear(dim_in, block_dim, args.kernel_size_proj_out))
                self.proj_out_unfold.append(nn.ReLU(inplace=True))
                pass

            if len(self.proj_out_unfold) == 0:
                if args.reduce_proj_v_by_block:
                    dim_in = block_dim // self.block_size
                else:
                    dim_in = block_dim
            else:
                dim_in = block_dim
            self.proj_out_unfold.append(nn.Linear(dim_in, block_dim))
            pass

        if args.do_proj_after_attn:
            self.proj_out = nn.Sequential()
            for i in range(args.num_block_proj_out):
                self.proj_out.append(ConvBlock(dim, dim, args.kernel_size_proj_out))
                pass
            self.proj_out.append(nn.Conv2d(dim, dim, 1))
            if args.zero_conv:
                zero_module(self.proj_out)

        if args.NL_in_NL:

            if args.win_do_proj_qkv:
                self.win_proj_qkv = nn.ModuleList()
                for i in range(3):
                    if (i == 2) and self.proj_after_norm_qk_only:
                        continue
                    layers = nn.Sequential()
                    for j in range(args.num_block_proj_in):
                        layers.append(ConvBlock(dim, dim, args.kernel_size_proj_in))
                        pass
                    layers.append(nn.Conv2d(dim, dim, 1))
                    self.win_proj_qkv.append(layers)

            block_dim = dim * self.win_block_size ** 2
            norm_dim = block_dim

            if args.win_do_norm:
                if args.norm_type_for_attn == 'l2norm':
                    pass
                elif args.norm_type_for_attn == 'sep_layernorm':
                    self.win_norm_q = nn.LayerNorm(norm_dim, elementwise_affine=args.layernorm_affine,
                                               bias=args.layernorm_bias)
                    self.win_norm_k = nn.LayerNorm(norm_dim, elementwise_affine=args.layernorm_affine,
                                               bias=args.layernorm_bias)
                else:
                    self.win_norm = nn.LayerNorm(norm_dim, elementwise_affine=args.layernorm_affine,
                                             bias=args.layernorm_bias)

            if args.win_do_proj_out:
                self.win_proj_out = nn.Sequential()
                for i in range(args.num_block_proj_out):
                    self.win_proj_out.append(ConvBlock(dim, dim, args.kernel_size_proj_out))
                    pass
                self.win_proj_out.append(nn.Conv2d(dim, dim, 1))
                if args.zero_conv:
                    zero_module(self.win_proj_out)

        pass

    def forward(self, x, y=None):
        h, w = x.shape[-2:]
        residual = x
        # unfold
        if self.do_proj_qkv:
            proj_q = self.proj_qkv[0](x)
            proj_k = self.proj_qkv[1](y)
            if self.proj_after_norm_qk_only:
                proj_v = y
            else:
                proj_v = self.proj_qkv[2](y)
        else:
            proj_q = x
            proj_k = y
            proj_v = y

        q = F.unfold(proj_q, self.block_size, padding=0, stride=self.block_size)
        q = q.permute(0, -1, -2)

        k = F.unfold(proj_k, self.block_size, padding=0, stride=self.block_size)
        k = k.permute(0, -1, -2)

        v = F.unfold(proj_v, self.block_size, padding=0, stride=self.block_size)
        v = v.permute(0, -1, -2)

        if self.do_proj_after_unfold:
            q = self.linear_unfold[0](q)
            k = self.linear_unfold[1](k)
            if not self.proj_after_norm_qk_only:
                v = self.linear_unfold[2](v)

        if self.do_norm:
            if self.norm_type_for_attn == 'l2norm':
                q = F.normalize(q, p=self.norm_order, dim=self.norm_dim)
                k = F.normalize(k, p=self.norm_order, dim=self.norm_dim)
                denom = 1
            elif self.norm_type_for_attn == 'sep_layernorm':
                q = self.norm_q(q)
                k = self.norm_k(k)
                denom = (q.shape[-1] ** -0.5)
            else:
                q = self.norm(q)
                k = self.norm(k)
                denom = (q.shape[-1] ** -0.5)

        if self.do_proj_after_norm:
            q = self.linear_qkv[0](q)
            k = self.linear_qkv[1](k)
            if not self.proj_after_norm_qk_only:
                v = self.linear_qkv[2](v)


        if self.apply_attention:
            sim = torch.matmul(q * denom, k.transpose(-2, -1))
            if self.attn_softmax:
                attn = F.softmax(sim, dim=-1)
            else:
                attn = sim
            # ori = torch.mean(sim, dim=(-1), keepdim=True)
            # relevance = F.softmax(ori, dim=0)
            relevance = torch.sum(attn * sim, dim=(-1), keepdim=True)
            relevance = F.softmax(relevance, dim=0)
            # relevance_aggr = torch.sum(relevance, dim=(-1, -2))
            out = torch.matmul(attn, v)
            out = out * relevance
        else:
            out = v

        # output projection
        if self.do_proj_out_unfold:
            out = self.proj_out_unfold(out)

        out = F.fold(out.permute(0, -1, -2), (h, w), self.block_size, stride=self.block_size)
        out = torch.sum(out, dim=0, keepdim=True)

        # sliding window non-local block
        if self.NL_in_NL:
            out = self.sliding_window_non_local(proj_q, out)

        if self.do_proj_after_attn:
            out = self.proj_out(out)
        x = residual + out * self.alpha_transf

        out_dict = {'out': x}

        if self.output_residual:
            out_dict['out'] = residual

        if self.output_attention:
            out_dict['attn'] = attn

        return out_dict


    def sliding_window_non_local(self, x, y):
        h, w = x.shape[-2:]

        # project
        if self.win_do_proj_qkv:
            proj_q = self.win_proj_qkv[0](x)
            proj_k = self.win_proj_qkv[1](y)
            if self.proj_after_norm_qk_only:
                proj_v = y
            else:
                proj_v = self.win_proj_qkv[2](y)
        else:
            proj_q = x
            proj_k = y
            proj_v = y

        # unfold
        q = F.unfold(proj_q, self.block_size, padding=0, stride=self.block_size)
        q = q.permute(-1, 1, 0)
        q = F.fold(q, (self.block_size, self.block_size), self.block_size)
        q = F.unfold(q, self.win_block_size, padding=0, stride=self.win_block_size)
        q = q.permute(0, -1, -2)

        k = F.unfold(proj_k, self.block_size, padding=0, stride=self.block_size)
        k = k.permute(-1, 1, 0)
        k = F.fold(k, (self.block_size, self.block_size), self.block_size)
        k = F.unfold(k, self.win_block_size, padding=0, stride=self.win_block_size)
        k = k.permute(0, -1, -2)

        v = F.unfold(proj_v, self.block_size, padding=0, stride=self.block_size)
        v = v.permute(-1, 1, 0)
        v = F.fold(v, (self.block_size, self.block_size), self.block_size)
        v = F.unfold(v, self.win_block_size, padding=0, stride=self.win_block_size)
        v = v.permute(0, -1, -2)

        # normalize
        if self.do_norm:
            if self.norm_type_for_attn == 'l2norm':
                q = F.normalize(q, p=self.norm_order, dim=self.norm_dim)
                k = F.normalize(k, p=self.norm_order, dim=self.norm_dim)
                denom = 1
            elif self.norm_type_for_attn == 'sep_layernorm':
                q = self.win_norm_q(q)
                k = self.win_norm_k(k)
                denom = (q.shape[-1] ** -0.5)
            else:
                q = self.win_norm(q)
                k = self.win_norm(k)
                denom = (q.shape[-1] ** -0.5)

        # attention
        if self.apply_attention:
            sim = torch.matmul(q * denom, k.transpose(-2, -1))
            if self.attn_softmax:
                attn = F.softmax(sim, dim=-1)
            else:
                attn = sim
            out = torch.matmul(attn, v)
        else:
            out = v

        # sum
        out = F.fold(out.permute(0, -1, -2), (self.block_size, self.block_size), self.win_block_size, stride=self.win_block_size)
        out = F.unfold(out, self.block_size, padding=0, stride=self.block_size)
        out = F.fold(out.permute(-1, 1, 0), (h, w), self.block_size, stride=self.block_size)
        # out = torch.sum(out, dim=0, keepdim=True)

        if self.win_do_proj_out:
            out = self.win_proj_out(out)

        return out

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

