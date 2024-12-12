import os
import numpy as np
import nibabel as nib
import skimage.transform as skitr
from PIL import Image
import matplotlib.pyplot as plt
import scipy.ndimage
import torch.nn.functional as F
import torch
import shutil
import json

mean_intensity = 103.13614654541016
std_intensity = 73.3431396484375
lower_bound =  -58.0
upper_bound = 302.0
HU_MIN = -1024
HU_MAX = 3071

def down_sample_image():
    data_dir = '/home/user/workspace/kits23/data_handler'
    # data_dir = "/home/user/data_handler/230720_kits23_test_images"
    case_list = sorted(os.listdir(data_dir))
    for case in case_list:
        if case != 'case_00000':
            continue
        case_dir = os.path.join(data_dir, case)
        img_path = os.path.join(case_dir, 'imaging.nii.gz')
        # img_path = os.path.join(data_dir, case)
        img = nib.load(img_path)
        affine = img.affine
        spacing = [-affine[0, 2], -affine[1, 1], -affine[2, 0]]
        img_data = img.get_fdata()
        img_data = img_data[::10]
        affine_down = affine
        affine_down[2, 0] = 10 * affine[2, 0]
        nib.save(nib.Nifti1Image(img_data, affine=affine_down),
                 os.path.join(data_dir, case, f'imaging_down_5mm.nii.gz'))


def down_sample_and_move():
    data_dir = '/home/user/workspace/kits23/dataset'
    # data_dir = "/home/user/dataset/230720_kits23_test_images"
    case_list = sorted(os.listdir(data_dir))
    out_dir = '/home/user/dataset/kits23_slice'
    save_nii = False
    save_npy = True
    os.makedirs(os.path.join(out_dir, 'thumbnail'), exist_ok=True)
    for case in case_list:
        # if case != 'case_00638.nii.gz':
        #     continue
        print(case)

        img_path = os.path.join(data_dir, case, 'imaging.nii.gz')
        # img_path = os.path.join(data_dir, case)
        # case = case[:10]
        img = nib.load(img_path)
        affine = img.affine
        spacing = [-affine[0, 2], -affine[1, 1], -affine[2, 0]]

        if spacing[2] > 1.5:
            continue

        img_data = img.get_fdata()
        scale = spacing[2] / 1.0
        os.makedirs(os.path.join(out_dir, case), exist_ok=True)

        # exceptional case KiTS23. Weird long CT
        if case == 'case_00638':
            img_data = img_data[:750]

        img_data = np.clip(img_data, a_min=-1024, a_max=None)
        cor_proj = np.mean(img_data, axis=(0, 2))
        sag_proj = np.mean(img_data, axis=(0, 1))

        thr_cor = -500
        thr_sag = -600
        cor_nonzero = np.argwhere(cor_proj > thr_cor).squeeze()
        sag_nonzero = np.argwhere(sag_proj > thr_sag).squeeze()

        img_data = img_data[:, cor_nonzero[0]:cor_nonzero[-1] + 1, sag_nonzero[0]:sag_nonzero[-1]+1]

        if spacing[2] == 0.5:
            upsampled_img = img_data[::2]
            # last_slice = upsampled_img.shape[0] - (upsampled_img.shape[0] - 1) % 5
            # upsampled_img = upsampled_img[:last_slice]
            downsampled_img = upsampled_img[::5]

            if save_nii:
                affine_down = affine.copy()
                affine_down[2, 0] = affine[2, 0] / scale
                nib.save(nib.Nifti1Image(upsampled_img.astype(np.float32), affine=affine_down),
                         os.path.join(out_dir, case, f'imaging_1mm.nii.gz'))

                affine_down = affine.copy()
                affine_down[2, 0] = affine[2, 0] / scale * 5
                nib.save(nib.Nifti1Image(downsampled_img.astype(np.float32), affine=affine_down),
                         os.path.join(out_dir, case, f'imaging_5mm.nii.gz'))

            if save_npy:
                upsampled_img = np.clip(upsampled_img, HU_MIN, HU_MAX)
                upsampled_img = (upsampled_img - HU_MIN) / (HU_MAX - HU_MIN)

                thumbnail_img = upsampled_img[upsampled_img.shape[0] // 2]
                img_pil = Image.fromarray(np.uint8(255 * thumbnail_img))
                img_pil.save(os.path.join(out_dir, 'thumbnail', f'{case}.jpg'))

                np.save(os.path.join(out_dir, case, f'imaging_1mm.npy'), upsampled_img.astype(np.float16))
            continue

        if spacing[2] == 1.0:
            # last_slice = img_data.shape[0] - (img_data.shape[0] - 1) % 5
            # img_data = img_data[:last_slice]
            downsampled_img = img_data[::5]

            if save_nii:
                affine_down = affine.copy()
                affine_down[2, 0] = affine[2, 0] / scale
                nib.save(nib.Nifti1Image(img_data.astype(np.float32), affine=affine_down),
                         os.path.join(out_dir, case, f'imaging_1mm.nii.gz'))

                affine_down = affine.copy()
                affine_down[2, 0] = affine[2, 0] / scale * 5
                nib.save(nib.Nifti1Image(downsampled_img.astype(np.float32), affine=affine_down),
                         os.path.join(out_dir, case, f'imaging_5mm.nii.gz'))

            if save_npy:
                img_data = np.clip(img_data, HU_MIN, HU_MAX)
                img_data = (img_data - HU_MIN) / (HU_MAX - HU_MIN)

                thumbnail_img = img_data[img_data.shape[0] // 2]
                img_pil = Image.fromarray(np.uint8(255 * thumbnail_img))
                img_pil.save(os.path.join(out_dir, 'thumbnail', f'{case}.jpg'))

                np.save(os.path.join(out_dir, case, f'imaging_1mm.npy'), img_data.astype(np.float16))
            continue

        upsampled_img = []
        for i in range(img_data.shape[1]):
            sag_slice = img_data[:, i, :]
            upsampled_sag_slice = skitr.rescale(sag_slice, (scale, 1), order=3, preserve_range=True)
            upsampled_img.append(upsampled_sag_slice)
        upsampled_img = np.stack(upsampled_img, axis=1)
        # zoomed_img = scipy.ndimage.zoom(img_data, (scale, 1, 1), order=3, prefilter=False)

        # last_slice = upsampled_img.shape[0] - (upsampled_img.shape[0] - 1) % 5
        # upsampled_img = upsampled_img[:last_slice]
        downsampled_img = upsampled_img[::5]

        if save_nii:
            affine_down = affine.copy()
            affine_down[2, 0] = affine[2, 0] / scale
            upsampled_img = np.round(upsampled_img)
            nib.save(nib.Nifti1Image(upsampled_img.astype(np.float32), affine=affine_down),
                     os.path.join(out_dir, case, f'imaging_1mm.nii.gz'))

            affine_down = affine.copy()
            affine_down[2, 0] = affine[2, 0] / scale * 5
            downsampled_img = np.round(downsampled_img)
            nib.save(nib.Nifti1Image(downsampled_img.astype(np.float32), affine=affine_down),
                     os.path.join(out_dir, case, f'imaging_5mm.nii.gz'))

        if save_npy:
            upsampled_img = np.clip(upsampled_img, HU_MIN, HU_MAX)
            upsampled_img = (upsampled_img - HU_MIN) / (HU_MAX - HU_MIN)

            thumbnail_img = upsampled_img[upsampled_img.shape[0] // 2]
            img_pil = Image.fromarray(np.uint8(255 * thumbnail_img))
            img_pil.save(os.path.join(out_dir,  'thumbnail', f'{case}.jpg'))

            np.save(os.path.join(out_dir, case, f'imaging_1mm.npy'), upsampled_img.astype(np.float16))


def analyze_background_range():
    data_dir = '/home/user/workspace/kits23/data_handler'
    # data_dir = "/home/user/data_handler/230720_kits23_test_images"
    case_list = sorted(os.listdir(data_dir))
    out_dir = '/home/user/dataset/kits23_slice'
    save_nii = False
    for case in case_list:
        # if case != 'case_00001':
        #     continue
        print(case)
        case_dir = os.path.join(data_dir, case)
        img_path = os.path.join(case_dir, 'imaging.nii.gz')
        # img_path = os.path.join(data_dir, case)
        # case = case[:10]
        img = nib.load(img_path)
        affine = img.affine
        spacing = [-affine[0, 2], -affine[1, 1], -affine[2, 0]]
        img_data = img.get_fdata()
        img_data = np.clip(img_data, a_min=-1024, a_max=None)
        mean_cor_slices = np.expand_dims(np.mean(img_data, axis=(0, 2)), axis=1)
        plt.close()
        _ = plt.hist(mean_cor_slices, bins=[-1024, -900, -800, -700, -600, -500, -400, -300, -200, -100, 0])  # arguments are passed to np.histogram
        plt.title("Histogram with 'auto' bins")
        plt.show()
        os.makedirs(os.path.join('plot_hist'), exist_ok=True)
        plt.savefig(os.path.join('plot_hist', f'cor_{case}'))

        '''
        for i in range(512):
            img_cor = img_data[:, i, :]
            img_clip = np.clip(img_cor, lower_bound, upper_bound)
            img_norm = (img_clip - lower_bound) / (upper_bound - lower_bound)

            img_pil = Image.fromarray(np.uint8(255 * img_norm))
            os.makedirs(os.path.join('images', 'background_test', case, 'cor'), exist_ok=True)
            img_pil.save(os.path.join('images', 'background_test', case, 'cor', f'{i:03d}.jpg'))
        '''
        mean_sag_slices = np.expand_dims(np.mean(img_data, axis=(0, 1)), axis=1)
        plt.close()
        _ = plt.hist(mean_sag_slices, bins=[-1024, -900, -800, -700, -600, -500, -400, -300, -200, -100, 0])  # arguments are passed to np.histogram
        plt.title("Histogram with 'auto' bins")
        plt.show()
        plt.savefig(os.path.join('plot_hist', f'sag_{case}'))
        '''
        for i in range(512):
            img_sag = img_data[:, :, i]
            img_clip = np.clip(img_sag, lower_bound, upper_bound)
            img_norm = (img_clip - lower_bound) / (upper_bound - lower_bound)

            img_pil = Image.fromarray(np.uint8(255 * img_norm))
            os.makedirs(os.path.join('images', 'background_test', case, 'sag'), exist_ok=True)
            img_pil.save(os.path.join('images', 'background_test', case, 'sag', f'{i:03d}.jpg'))
        '''
        a = 2


def move_test_cases_to_train():
    # data_dir = '/home/user/workspace/kits23/data_handler'
    data_dir = "/home/user/dataset/230720_kits23_test_images"
    case_list = sorted(os.listdir(data_dir))
    out_dir = '/home/user/workspace/kits23/data_handler'
    save_nii = False
    for case in case_list:
        # if case != 'case_00001':
        #     continue
        print(case)
        img_path = os.path.join(data_dir, case)
        os.makedirs(os.path.join(out_dir, case[:10]), exist_ok=True)
        dst_path = os.path.join(out_dir, case[:10], 'imaging.nii.gz')
        shutil.copy(img_path, dst_path)
    pass


def move_cases_to_rplhr():
    data_dir = '/home/user/workspace/kits23/dataset'
    # data_dir = "/home/user/dataset/230720_kits23_test_images"
    case_list = sorted(os.listdir(data_dir))
    out_dir = '/home/user/dataset/kits23_rplhr'
    split_file = '/home/user/workspace/CT-slice-interpolation/case_ids/splits.json'

    with open(split_file, 'r') as file:
        splits_dct = json.load(file)

    train_cases = splits_dct['train']
    val_cases = splits_dct['test']
    test_cases = splits_dct['test']

    save_nii = False
    for case in train_cases:
        # if case != 'case_00001':
        #     continue
        print(case)
        img_path = os.path.join(data_dir, case, 'imaging.nii.gz')
        os.makedirs(os.path.join(out_dir, 'train', '1mm'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'train', '5mm'), exist_ok=True)
        dst_path = os.path.join(out_dir, 'train', '1mm', f'{case}.nii.gz')
        dst_path_5mm = os.path.join(out_dir, 'train', '5mm', f'{case}.nii.gz')

        img = nib.load(img_path)
        affine = img.affine
        spacing = [-affine[0, 2], -affine[1, 1], -affine[2, 0]]

        if spacing[2] > 1.5:
            continue

        img_data = img.get_fdata()
        scale = spacing[2] / 1.0

        if spacing[2] == 0.5:
            upsampled_img = img_data[::2]
            last_slice = upsampled_img.shape[0] - (upsampled_img.shape[0] - 1) % 5
            upsampled_img = upsampled_img[:last_slice]
            pass
        elif spacing[2] == 1.0:
            last_slice = img_data.shape[0] - (img_data.shape[0] - 1) % 5
            upsampled_img = img_data[:last_slice]
            pass
        elif spacing[2] <= 1.5:
            upsampled_img = []
            for i in range(img_data.shape[1]):
                sag_slice = img_data[:, i, :]
                upsampled_sag_slice = skitr.rescale(sag_slice, (scale, 1), order=3, preserve_range=True)
                upsampled_img.append(upsampled_sag_slice)
            upsampled_img = np.stack(upsampled_img, axis=1)
            last_slice = upsampled_img.shape[0] - (upsampled_img.shape[0] - 1) % 5
            upsampled_img = upsampled_img[:last_slice]
            pass
        else:
            pass

        downsampled_img = upsampled_img[::5].copy()

        upsampled_img = np.clip(upsampled_img, HU_MIN, HU_MAX)
        upsampled_img = (upsampled_img - HU_MIN) / (HU_MAX - HU_MIN)

        downsampled_img = np.clip(downsampled_img, HU_MIN, HU_MAX)
        downsampled_img = (downsampled_img - HU_MIN) / (HU_MAX - HU_MIN)

        affine_down = affine.copy()
        affine_down[2, 0] = affine[2, 0] / scale
        affine_down[:3, :3] = np.flip(affine_down[:3, :3], axis=1)
        upsampled_img = np.transpose(upsampled_img)
        nib.save(nib.Nifti1Image(upsampled_img.astype(np.float32), affine=affine_down),
                 dst_path)

        affine_down = affine.copy()
        affine_down[2, 0] = affine[2, 0] / scale * 5
        affine_down[:3, :3] = np.flip(affine_down[:3, :3], axis=1)
        downsampled_img = np.transpose(downsampled_img)
        nib.save(nib.Nifti1Image(downsampled_img.astype(np.float32), affine=affine_down),
                 dst_path_5mm)

    for case in val_cases:
        # if case != 'case_00001':
        #     continue
        print(case)
        img_path = os.path.join(data_dir, case, 'imaging.nii.gz')
        os.makedirs(os.path.join(out_dir, 'val', '1mm'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'val', '5mm'), exist_ok=True)
        dst_path = os.path.join(out_dir, 'val', '1mm', f'{case}.nii.gz')
        dst_path_5mm = os.path.join(out_dir, 'val', '5mm', f'{case}.nii.gz')

        img = nib.load(img_path)
        affine = img.affine
        spacing = [-affine[0, 2], -affine[1, 1], -affine[2, 0]]

        if spacing[2] > 1.5:
            continue

        img_data = img.get_fdata()
        scale = spacing[2] / 1.0

        if spacing[2] == 0.5:
            upsampled_img = img_data[::2]
            last_slice = upsampled_img.shape[0] - (upsampled_img.shape[0] - 1) % 5
            upsampled_img = upsampled_img[:last_slice]
            pass
        elif spacing[2] == 1.0:
            last_slice = img_data.shape[0] - (img_data.shape[0] - 1) % 5
            upsampled_img = img_data[:last_slice]
            pass
        elif spacing[2] <= 1.5:
            upsampled_img = []
            for i in range(img_data.shape[1]):
                sag_slice = img_data[:, i, :]
                upsampled_sag_slice = skitr.rescale(sag_slice, (scale, 1), order=3, preserve_range=True)
                upsampled_img.append(upsampled_sag_slice)
            upsampled_img = np.stack(upsampled_img, axis=1)
            last_slice = upsampled_img.shape[0] - (upsampled_img.shape[0] - 1) % 5
            upsampled_img = upsampled_img[:last_slice]
            pass
        else:
            pass

        downsampled_img = upsampled_img[::5].copy()

        upsampled_img = np.clip(upsampled_img, HU_MIN, HU_MAX)
        upsampled_img = (upsampled_img - HU_MIN) / (HU_MAX - HU_MIN)

        downsampled_img = np.clip(downsampled_img, HU_MIN, HU_MAX)
        downsampled_img = (downsampled_img - HU_MIN) / (HU_MAX - HU_MIN)

        affine_down = affine.copy()
        affine_down[2, 0] = affine[2, 0] / scale
        affine_down[:3, :3] = np.flip(affine_down[:3, :3], axis=1)
        upsampled_img = np.transpose(upsampled_img)
        nib.save(nib.Nifti1Image(upsampled_img.astype(np.float32), affine=affine_down),
                 dst_path)

        affine_down = affine.copy()
        affine_down[2, 0] = affine[2, 0] / scale * 5
        affine_down[:3, :3] = np.flip(affine_down[:3, :3], axis=1)
        downsampled_img = np.transpose(downsampled_img)
        nib.save(nib.Nifti1Image(downsampled_img.astype(np.float32), affine=affine_down),
                 dst_path_5mm)

    for case in test_cases:
        # if case != 'case_00001':
        #     continue
        print(case)
        img_path = os.path.join(data_dir, case, 'imaging.nii.gz')
        os.makedirs(os.path.join(out_dir, 'test', '1mm'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'test', '5mm'), exist_ok=True)
        dst_path = os.path.join(out_dir, 'test', '1mm', f'{case}.nii.gz')
        dst_path_5mm = os.path.join(out_dir, 'test', '5mm', f'{case}.nii.gz')

        img = nib.load(img_path)
        affine = img.affine
        spacing = [-affine[0, 2], -affine[1, 1], -affine[2, 0]]

        if spacing[2] > 1.5:
            continue

        img_data = img.get_fdata()
        scale = spacing[2] / 1.0

        if spacing[2] == 0.5:
            upsampled_img = img_data[::2]
            last_slice = upsampled_img.shape[0] - (upsampled_img.shape[0] - 1) % 5
            upsampled_img = upsampled_img[:last_slice]
            pass
        elif spacing[2] == 1.0:
            last_slice = img_data.shape[0] - (img_data.shape[0] - 1) % 5
            upsampled_img = img_data[:last_slice]
            pass
        elif spacing[2] <= 1.5:
            upsampled_img = []
            for i in range(img_data.shape[1]):
                sag_slice = img_data[:, i, :]
                upsampled_sag_slice = skitr.rescale(sag_slice, (scale, 1), order=3, preserve_range=True)
                upsampled_img.append(upsampled_sag_slice)
            upsampled_img = np.stack(upsampled_img, axis=1)
            last_slice = upsampled_img.shape[0] - (upsampled_img.shape[0] - 1) % 5
            upsampled_img = upsampled_img[:last_slice]
            pass
        else:
            pass

        downsampled_img = upsampled_img[::5].copy()

        upsampled_img = np.clip(upsampled_img, HU_MIN, HU_MAX)
        upsampled_img = (upsampled_img - HU_MIN) / (HU_MAX - HU_MIN)

        downsampled_img = np.clip(downsampled_img, HU_MIN, HU_MAX)
        downsampled_img = (downsampled_img - HU_MIN) / (HU_MAX - HU_MIN)

        affine_down = affine.copy()
        affine_down[2, 0] = affine[2, 0] / scale
        affine_down[:3, :3] = np.flip(affine_down[:3, :3], axis=1)
        upsampled_img = np.transpose(upsampled_img)
        nib.save(nib.Nifti1Image(upsampled_img.astype(np.float32), affine=affine_down),
                 dst_path)

        affine_down = affine.copy()
        affine_down[2, 0] = affine[2, 0] / scale * 5
        affine_down[:3, :3] = np.flip(affine_down[:3, :3], axis=1)
        downsampled_img = np.transpose(downsampled_img)
        nib.save(nib.Nifti1Image(downsampled_img.astype(np.float32), affine=affine_down),
                 dst_path_5mm)
    pass


if __name__ == '__main__':
    # down_sample_and_move()
    move_cases_to_rplhr()
    print('done')


