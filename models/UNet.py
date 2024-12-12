import torch
from torch import nn
from models.modules import ConvBlock
from tqdm.auto import tqdm
import math


class UNet(nn.Module):
    def __init__(self, args):
        super(UNet, self).__init__()
        self.num_level = args.num_level
        self.num_channel = args.num_channel
        self.num_block = args.num_block
        self.kernel_size = args.kernel_size
        self.scale = args.scale
        self.patch_size = args.patch_size
        self.slide_overlap = args.slide_overlap
        self.long_skip = args.long_skip
        self.long_skip_feat = args.long_skip_feat
        self.do_fusion = args.do_fusion
        self.output_lr_up = args.output_lr_up
        self.act_first = args.act_first
        self.cat_feat = args.cat_feat
        self.use_resenc = args.use_resenc
        self.use_resdec = args.use_resdec
        self.act_after_res = args.act_after_res

        # first conv
        if args.do_fusion and not args.cat_feat:
            in_channels = 2
        else:
            in_channels = 1

        if args.cat_feat:
            out_channels = self.num_channel // 2
        else:
            out_channels = self.num_channel

        if args.act_first:
            self.first_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, self.kernel_size, padding=1), nn.ReLU(inplace=True))
        else:
            self.first_conv = nn.Conv2d(in_channels, out_channels, self.kernel_size, padding=1)
        # encoder layers
        self.encoder = nn.ModuleList()
        for i in range(self.num_level):
            if i == 0:
                in_channel = self.num_channel
                down_sample = False
            else:
                in_channel = self.num_channel * (2 ** (i - 1))
                down_sample = True
            out_channel = self.num_channel * (2 ** i)
            self.encoder.append(
                ConvBlock(in_channel, out_channel, self.kernel_size, num_block=self.num_block, down_sample=down_sample, down_op=args.down_op, do_residual=self.use_resenc, act_after_res=self.act_after_res))

        # decoder layers
        self.decoder = nn.ModuleList()
        for i in reversed(range(self.num_level - 1)):
            in_channel = self.num_channel * (2 ** (i + 1))
            out_channel = self.num_channel * (2 ** (i))
            up_sample = True
            self.decoder.append(
                ConvBlock(in_channel, out_channel, self.kernel_size, num_block=self.num_block, up_sample=up_sample, up_op=args.up_op, do_residual=self.use_resdec, act_after_res=self.act_after_res))

        # last conv
        self.last_conv = nn.Conv2d(self.num_channel, 1, self.kernel_size, padding=1)
        pass

    def forward(self, data, pre_upsampled=False):

        # lr upsampling
        if pre_upsampled:
            lr_up = data
        else:
            lr_up = nn.functional.interpolate(data['slice_lr'], data['slice_hr'].shape[-2:], mode='bilinear',
                                              align_corners=True)

        # if output upsampled lr
        if self.output_lr_up:
            if self.do_fusion:
                lr_up = torch.mean(lr_up, dim=1, keepdim=True)
            return {'out': lr_up}

        # first conv
        if self.cat_feat:
            # concat at feature level
            lr_up_perm = lr_up.permute(1, 0, 2, 3)
            x = self.first_conv(lr_up_perm)
            h, w = x.shape[-2:]
            x = x.view(1, -1, h, w)
        else:
            x = self.first_conv(lr_up)

        init_feat = x
        # pass encoder
        enc_list = []
        for i, module in enumerate(self.encoder):
            x = module(x)
            enc_list.append(x)

        enc_list.pop()
        # pass decoder
        for i, module in enumerate(self.decoder):
            x = module(x, enc_list.pop())

        if self.long_skip_feat:
            x = x + init_feat
        # last conv
        x = self.last_conv(x)

        if self.do_fusion:
            lr_up = torch.mean(lr_up, dim=1, keepdim=True)

        if self.long_skip:
            x = x + lr_up
        out_dict = {'lr_up': lr_up,
                    'out': x}
        return out_dict

    def test_volume(self, data):
        # lr upsampling
        lr_up = nn.functional.interpolate(data['img_lr'], data['img_hr'].shape[-3:], mode='trilinear',
                                          align_corners=True)

        # padding to be size of 2^x
        lr_up, padding = self.padding_by_level(lr_up)

        # output tensor dict
        out_dict = {'out_cor': torch.zeros_like(lr_up),
                    'out_sag': torch.zeros_like(lr_up)}

        # cor slices
        progress_bar = tqdm(range(lr_up.shape[-2]), desc='cor iter', leave=False)
        for i in progress_bar:
            out_dict['out_cor'][..., i, :] = self.forward(lr_up[..., i, :], pre_upsampled=True)['out']

        # sag slices
        progress_bar = tqdm(range(lr_up.shape[-1]), desc='sag iter', leave=False)
        for i in progress_bar:
            out_dict['out_sag'][..., i] = self.forward(lr_up[..., i], pre_upsampled=True)['out']

        # average
        out_dict['out_avg'] = 0.5 * (out_dict['out_cor'] + out_dict['out_sag'])

        # un-pad images
        self.unpad_images(out_dict, padding)

        return out_dict

    def test_slicewise(self, data):
        # lr upsampling
        lr_up = nn.functional.interpolate(data['img_lr'], data['img_hr'].shape[-3:], mode='trilinear',
                                          align_corners=True)

        # padding to be size of 2^x
        lr_up, padding = self.padding_by_level(lr_up)

        # output tensor dict
        _, _, d, h, w = lr_up.shape
        out_dict = {'out_avg': torch.zeros((1, 1, d, h, w), device='cuda')}

        # axial slices
        progress_bar = tqdm(range(lr_up.shape[-3]), desc='axial iter', leave=False)
        for i in progress_bar:
            out_dict['out_avg'][..., i, :, :] = self.forward(lr_up[..., i, :, :], pre_upsampled=True)['out']

        # un-pad images
        self.unpad_images(out_dict, padding)

        return out_dict

    def sliding_window_inference(self, data):
        # lr upsampling
        lr_up = nn.functional.interpolate(data['img_lr'], data['img_hr'].shape[-3:], mode='trilinear',
                                          align_corners=True)

        # padding to be larger than patch size
        lr_up, padding = self.padding_by_patch_size(lr_up)

        # allocate memory output tensor dict
        out_dict = {'out_cor': torch.zeros_like(lr_up),
                    'out_sag': torch.zeros_like(lr_up)}

        # get scan interval
        slicer_list = self.get_slicer(lr_up)

        # create importance map (currently only 'constant' mode supported)
        importance_map = torch.ones_like(lr_up)

        # cor slices
        count_map = torch.zeros_like(lr_up)
        progress_bar = tqdm(range(lr_up.shape[-2]), desc='cor iter', leave=False)
        for i in progress_bar:
            for slicer in slicer_list['cor']:
                out_dict['out_cor'][..., slicer[0], i, slicer[1]] += \
                    self.forward(lr_up[..., slicer[0], i, slicer[1]], pre_upsampled=True)['out']
                count_map[..., slicer[0], i, slicer[1]] += importance_map[..., slicer[0], i, slicer[1]]
                pass
        out_dict['out_cor'] = out_dict['out_cor'] / count_map

        # sag slices
        count_map = torch.zeros_like(lr_up)
        progress_bar = tqdm(range(lr_up.shape[-1]), desc='sag iter', leave=False)
        for i in progress_bar:
            for slicer in slicer_list['sag']:
                out_dict['out_sag'][..., slicer[0], slicer[1], i] += \
                    self.forward(lr_up[..., slicer[0], slicer[1], i], pre_upsampled=True)['out']
                count_map[..., slicer[0], slicer[1], i] += importance_map[..., slicer[0], slicer[1], i]
                pass
        out_dict['out_sag'] = out_dict['out_sag'] / count_map

        # average
        out_dict['out_avg'] = 0.5 * (out_dict['out_cor'] + out_dict['out_sag'])

        # un-pad images
        self.unpad_images(out_dict, padding)
        return out_dict

    def padding_by_level(self, img):
        _, _, d, h, w = img.shape
        mult = pow(2, self.num_level - 1)
        pad_d = math.ceil(d / mult) * mult - d
        pad_h = math.ceil(h / mult) * mult - h
        pad_w = math.ceil(w / mult) * mult - w
        img = nn.functional.pad(img, (0, pad_w, 0, pad_h, 0, pad_d), mode='reflect')
        padding = (pad_d, pad_h, pad_w)
        return img, padding

    def padding_by_patch_size(self, img):
        _, _, d, h, w = img.shape
        pad_d = max(0, (self.patch_size - d))
        pad_h = max(0, (self.patch_size - h))
        pad_w = max(0, (self.patch_size - w))

        img = nn.functional.pad(img, (0, pad_w, 0, pad_h, 0, pad_d), mode='reflect')
        padding = (pad_d, pad_h, pad_w)
        return img, padding

    def get_slicer(self, img):
        _, _, d, h, w = img.shape
        slicer_list = {'cor': [],
                       'sag': []}
        for key, value in slicer_list.items():
            if key == 'cor':
                t = w
            else:
                t = h

            for z in range(0, d - self.slide_overlap, self.patch_size - self.slide_overlap):
                z = min(z, d - self.patch_size)
                for yx in range(0, t - self.slide_overlap, self.patch_size - self.slide_overlap):
                    yx = min(yx, t - self.patch_size)
                    slicer_list[key].append((slice(z, z + self.patch_size), slice(yx, yx + self.patch_size)))
        return slicer_list

    def unpad_images(self, data, padding):
        (pad_d, pad_h, pad_w) = padding
        for key, value in data.items():
            (_, _, d, h, w) = data[key].shape
            data[key] = data[key][..., :d - pad_d, :h - pad_h, :w - pad_w]
        pass
