from torch.utils.data import Dataset
import os
import nibabel as nib
import numpy as np
from time import time
import random
import json

class KiTS23(Dataset):
    def __init__(self, args, mode=str()):
        self.data_dir = args.data_dir
        self.mode = mode
        self.scale = args.scale
        self.patch_size = args.patch_size
        self.mirror_axis = args.mirror_axis
        self.test_all_data = args.test_all_data
        self.fusion_data_dir = args.fusion_data_dir
        self.do_fusion = args.do_fusion
        self.use_ref_axial = args.use_ref_axial
        self.num_ref_slices = args.num_ref_slices
        self.use_hr_ref = args.use_hr_ref
        self.roll_hr_ref = args.roll_hr_ref
        self.roll_xy = args.roll_xy
        self.roll_xy_start = args.roll_xy_start
        self.roll_xy_step = args.roll_xy_step

        self.random_roll = args.random_roll
        self.identical_shift = args.identical_shift
        self.append_orig_hr_ref = args.append_orig_hr_ref
        self.get_uniform_train_ref = args.get_uniform_train_ref
        self.uniform_train_ref_in_patch = args.uniform_train_ref_in_patch

        self.split_file = args.split_file
        self.num_level = args.num_level

        # load case_ids
        self.case_ids = self.load_case_ids()

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        # start_time = time()
        # load case images
        case_data = self.load_case_data(idx)

        # get train/val/test data
        data_dict = {}
        if self.test_all_data:
            data_dict = self.get_test_data(case_data)
        elif self.mode in ['train', 'val']:
            data_dict = self.get_train_data(case_data)
        elif self.mode == 'test':
            data_dict = self.get_test_data(case_data)

        # print(f'b data load - {time() - start_time}')
        data_dict['case'] = case_data['case']
        return data_dict

    def load_case_ids(self):
        with open(self.split_file, 'r') as file:
            splits_dct = json.load(file)

        if self.mode in ['val', 'test']:
            case_ids = splits_dct['test']
        elif self.mode == 'train':
            case_ids = splits_dct['train']
        else:
            case_ids = []
        return case_ids

    def load_case_data(self, idx):
        case = self.case_ids[idx]
        # case = 'CT00000040.npy'
        case_name = case.split('.')[0]

        img_hr = np.load(os.path.join(self.data_dir, case, 'imaging_1mm.npy'), mmap_mode='r')

        if self.do_fusion:
            img_lr = np.load(os.path.join(self.fusion_data_dir, self.mode, f'{case.split('.')[0]}.npy'), mmap_mode='r')
            img_hr = img_hr[:img_lr.shape[1]]
        else:
            img_lr = None

        # img_hr = img_hr.astype(np.float32)
        # img_lr = img_lr.astype(np.float32)

        data = {'case': case_name,
                'img_hr': img_hr,
                'img_lr': img_lr}
        return data

    def get_train_data(self, data):
        # random mirror augmentation
        self.mirror_aug(data)

        # randomly crop cor/sag slice
        if self.do_fusion:
            slices = self.get_axial_slice(data)
        else:
            if random.random() < 0.5:
                slices = self.get_cor_slice(data)
            else:
                slices = self.get_sag_slice(data)

        # sample reference axial slice
        if self.use_ref_axial:
            if self.use_hr_ref:
                slices['slice_ref'] = slices['slice_hr']
            else:
                ref_axial_slices = self.get_ref_axial_slices(data, slices['z_s'], slices)
                slices['slice_ref'] = ref_axial_slices

        # roll image
        if self.roll_hr_ref:
            slice_ref = []
            for i in range(self.num_ref_slices):
                if self.random_roll:
                    shift_x = random.randrange(self.roll_xy_start[0], self.roll_xy[0] + 1, self.roll_xy_step[0])
                    shift_y = random.randrange(self.roll_xy_start[1], self.roll_xy[-1] + 1, self.roll_xy_step[1])
                    if self.identical_shift:
                        shift_y = shift_x
                    rolled_slice = np.roll(slices['slice_ref'], (shift_x, shift_y), axis=(1, 2))
                    # slices['slice_ref'] = np.roll(slices['slice_ref'], (shift_x, shift_y), axis=(1, 2))
                else:
                    rolled_slice = np.roll(slices['slice_ref'], self.roll_xy, axis=(1, 2))
                    # slices['slice_ref'] = np.roll(slices['slice_ref'], self.roll_xy, axis=(1, 2))
                slice_ref.append(rolled_slice)

            if self.append_orig_hr_ref:
                slice_ref.append(slices['slice_hr'])

            slice_ref = np.vstack(slice_ref)
            slices['slice_ref'] = slice_ref
        slices.pop('z_s', None)
        return slices

    def get_test_data(self, data):
        # append dummy gray channel
        # image size should be multiplies of 16
        d = data['img_hr'].shape[0]
        last_idx = d - (d - 1) % self.scale
        data['img_hr'] = data['img_hr'][:last_idx]


        if self.do_fusion:
            data['img_lr'] = data['img_lr'].copy()
        else:
            data['img_lr'] = data['img_hr'][::self.scale]
            data['img_lr'] = np.expand_dims(data['img_lr'], 0).copy()

        data['img_hr'] = np.expand_dims(data['img_hr'], 0).copy()

        return data

    def get_ref_axial_slices(self, data, z_s, slices=None):
        d, h, w = data['img_lr'].shape

        slice_ref = []
        for i in range(self.num_ref_slices):
            if self.get_uniform_train_ref:
                if self.uniform_train_ref_in_patch:
                    lr_patch_size = (self.patch_size - 1) // 5 + 1
                    interval = lr_patch_size // (self.num_ref_slices + 1)
                    z = z_s + (i + 1) * interval
                    pass
                else:
                    interval = d // (self.num_ref_slices + 1)
                    z = (i + 1) * interval
            else:
                z = random.randrange(0, d)

            y_s = random.randint(0, h - self.patch_size)
            x_s = random.randint(0, w - self.patch_size)

            y_e = y_s + self.patch_size
            x_e = x_s + self.patch_size

            slice_ax = data['img_lr'][z, y_s:y_e, x_s:x_e].copy()
            slice_ref.append(slice_ax)

        if self.append_orig_hr_ref:
                slice_ref.append(slices['slice_hr'].squeeze())
        slice_ref = np.array(slice_ref)
        return slice_ref

    def get_axial_slice(self, data):
        # padding
        data['img_hr'] = self.axial_padding(data['img_hr'], self.patch_size)
        data['img_lr'] = self.batch_axial_padding(data['img_lr'], self.patch_size)

        _, d, h, w = data['img_lr'].shape
        z = random.randrange(0, d)
        y_s = random.randint(0, h - self.patch_size)
        x_s = random.randint(0, w - self.patch_size)

        y_e = y_s + self.patch_size
        x_e = x_s + self.patch_size

        slice_lr = data['img_lr'][:, z, y_s:y_e, x_s:x_e].copy()
        slice_hr = data['img_hr'][z, y_s:y_e, x_s:x_e].copy()

        # append dummy gray channel
        slice_hr = np.expand_dims(slice_hr, 0)

        slice_dict = {'slice_hr': slice_hr,
                      'slice_lr': slice_lr}
        return slice_dict

    def get_cor_slice(self, data):
        # padding
        data['img_hr'] = self.cube_padding(data['img_hr'], self.patch_size)

        d, h, w = data['img_hr'].shape
        z_s = random.randint(0, d - self.patch_size)
        y = random.randrange(0, h)
        x_s = random.randint(0, w - self.patch_size)

        z_e = z_s + self.patch_size
        x_e = x_s + self.patch_size

        slice_hr = data['img_hr'][z_s:z_e, y, x_s:x_e].copy()
        slice_lr = slice_hr[::self.scale]

        # append dummy gray channel
        slice_lr = np.expand_dims(slice_lr, 0)
        slice_hr = np.expand_dims(slice_hr, 0)

        slice_dict = {'slice_hr': slice_hr,
                      'slice_lr': slice_lr,
                      'z_s': z_s}
        return slice_dict

    def get_sag_slice(self, data):
        # padding
        data['img_hr'] = self.cube_padding(data['img_hr'], self.patch_size)

        d, h, w = data['img_hr'].shape
        z_s = random.randint(0, d - self.patch_size)
        y_s = random.randint(0, h - self.patch_size)
        x = random.randrange(0, w)

        z_e = z_s + self.patch_size
        y_e = y_s + self.patch_size

        slice_hr = data['img_hr'][z_s:z_e, y_s:y_e, x].copy()
        slice_lr = slice_hr[::self.scale]

        # append dummy gray channel
        slice_lr = np.expand_dims(slice_lr, 0)
        slice_hr = np.expand_dims(slice_hr, 0)

        slice_dict = {'slice_hr': slice_hr,
                      'slice_lr': slice_lr,
                      'z_s': z_s}
        return slice_dict

    def slice_padding(self, img, slice_patch_size, mode='reflect'):
        z, _, _ = img.shape
        z_pad = max(0, (slice_patch_size - z))
        if z_pad > 0:
            img = np.pad(img, [(0, z_pad), (0, 0), (0, 0)], mode=mode)
        return img

    def cube_padding(self, img, slice_patch_size, mode='reflect'):
        z, h, w = img.shape
        z_pad = max(0, (slice_patch_size - z))
        h_pad = max(0, (slice_patch_size - h))
        w_pad = max(0, (slice_patch_size - w))
        if z_pad + h_pad + w_pad > 0:
            img = np.pad(img, [(0, z_pad), (0, h_pad), (0, w_pad)], mode=mode)
        return img

    def axial_padding(self, img, slice_patch_size, mode='reflect'):
        z, h, w = img.shape[-3:]
        z_pad = 0
        h_pad = max(0, (slice_patch_size - h))
        w_pad = max(0, (slice_patch_size - w))
        if z_pad + h_pad + w_pad > 0:
            img = np.pad(img, [(0, z_pad), (0, h_pad), (0, w_pad)], mode=mode)
        return img
    
    def batch_axial_padding(self, img, slice_patch_size, mode='reflect'):
        z, h, w = img.shape[-3:]
        z_pad = 0
        h_pad = max(0, (slice_patch_size - h))
        w_pad = max(0, (slice_patch_size - w))
        if z_pad + h_pad + w_pad > 0:
            img = np.pad(img, [(0, 0), (0, z_pad), (0, h_pad), (0, w_pad)], mode=mode)
        return img
    
    def mirror_aug(self, data):
        if 'x' in self.mirror_axis:
            if random.random() < 0.5:
                for key, value in data.items():
                    if isinstance(value, np.ndarray):
                        data[key] = value[:, :, ::-1].copy()
        if 'y' in self.mirror_axis:
            if random.random() < 0.5:
                for key, value in data.items():
                    if isinstance(value, np.ndarray):
                        data[key] = value[:, ::-1].copy()
        if 'z' in self.mirror_axis:
            if random.random() < 0.5:
                for key, value in data.items():
                    if isinstance(value, np.ndarray):
                        data[key] = value[::-1].copy()
        pass
