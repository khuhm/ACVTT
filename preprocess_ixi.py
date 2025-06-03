import os
import numpy as np
import nibabel as nib
import skimage.transform as skitr
from PIL import Image
import argparse

HU_MIN = 0
HU_MAX = 4095  

def down_sample_and_move_ixi(input_dir, output_dir, scale_factor=5, save_nii=False, save_npy=True, save_thumb=True):
    os.makedirs(os.path.join(output_dir, 'thumbnail'), exist_ok=True)
    case_list = sorted(os.listdir(input_dir))

    for case_file in case_list:
        if '._' in case_file or not case_file.endswith('.nii.gz'):
            continue
        case = case_file.replace('.nii.gz', '', 1)
        print(f"Processing {case}")
        img_path = os.path.join(input_dir, case_file)
        img = nib.load(img_path)
        affine = img.affine
        spacing = [affine[0, 0], affine[1, 1], affine[2, 2]]
        img_data = img.get_fdata()
        img_data = np.transpose(img_data, (2, 0, 1))  # [z, y, x]

        scale = spacing[2] / 1.0
        os.makedirs(os.path.join(output_dir, case), exist_ok=True)

        if spacing[2] == 0.5:
            upsampled_img = img_data[::2]
        elif spacing[2] == 1.0:
            upsampled_img = img_data
        else:
            upsampled_img = []
            for i in range(img_data.shape[1]):
                sag_slice = img_data[:, i, :]
                rescaled = skitr.rescale(sag_slice, (scale, 1), order=3, preserve_range=True)
                upsampled_img.append(rescaled)
            upsampled_img = np.stack(upsampled_img, axis=1)

        last = upsampled_img.shape[0] - (upsampled_img.shape[0] - 1) % scale_factor
        upsampled_img = upsampled_img[:last]
        downsampled_img = upsampled_img[::scale_factor]

        if save_nii:
            affine_up = affine.copy()
            affine_up[2, 2] = affine[2, 2] / scale
            nib.save(nib.Nifti1Image(upsampled_img.astype(np.float32), affine_up),
                     os.path.join(output_dir, case, 'imaging_1mm.nii.gz'))
            affine_down = affine_up.copy()
            affine_down[2, 2] *= scale_factor
            nib.save(nib.Nifti1Image(downsampled_img.astype(np.float32), affine_down),
                     os.path.join(output_dir, case, f'imaging_{scale_factor}mm.nii.gz'))

        upsampled_img = np.clip(upsampled_img, HU_MIN, HU_MAX)
        upsampled_img = (upsampled_img - HU_MIN) / (HU_MAX - HU_MIN)

        if save_npy:
            np.save(os.path.join(output_dir, case, 'imaging_1mm.npy'), upsampled_img.astype(np.float16))
        if save_thumb:
            thumbnail_img = upsampled_img[upsampled_img.shape[0] // 2]
            img_pil = Image.fromarray(np.uint8(255 * thumbnail_img[::-1, ::-1]))
            img_pil.save(os.path.join(output_dir, 'thumbnail', f'{case}.jpg'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--scale_factor', type=int, default=5)
    parser.add_argument('--save_nii', action='store_true')
    parser.add_argument('--save_npy', action='store_true')
    parser.add_argument('--save_thumb', action='store_true')
    args = parser.parse_args()

    down_sample_and_move_ixi(args.input_dir, args.output_dir,
                             scale_factor=args.scale_factor,
                             save_nii=args.save_nii,
                             save_npy=args.save_npy,
                             save_thumb=args.save_thumb)
