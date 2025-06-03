import os
import argparse
import numpy as np
import nibabel as nib
from PIL import Image
from tqdm import tqdm
import skimage.transform as skitr

HU_MIN = -1024
HU_MAX = 3071

def clip_and_normalize(img):
    img = np.clip(img, HU_MIN, HU_MAX)
    return (img - HU_MIN) / (HU_MAX - HU_MIN)

def save_thumbnail(img, out_path):
    mid_slice = img[img.shape[0] // 2]
    img_pil = Image.fromarray(np.uint8(255 * mid_slice))
    img_pil.save(out_path)

def resample_z_axis(img, src_spacing, target_spacing=1.0):
    scale = src_spacing / target_spacing
    resampled = []
    for i in range(img.shape[1]):
        slice_ = img[:, i, :]
        resized = skitr.rescale(slice_, (scale, 1), order=3, preserve_range=True)
        resampled.append(resized)
    return np.stack(resampled, axis=1)

def down_sample_and_move(input_dir, output_dir, scale_factor=5, save_npy=True, save_nii=False):
    os.makedirs(os.path.join(output_dir, 'thumbnail'), exist_ok=True)
    case_list = sorted(os.listdir(input_dir))

    for case in tqdm(case_list):
        case_path = os.path.join(input_dir, case)
        img_path = os.path.join(case_path, 'imaging.nii.gz')
        if not os.path.exists(img_path):
            continue

        try:
            img = nib.load(img_path)
            affine = img.affine
            spacing = [-affine[0, 2], -affine[1, 1], -affine[2, 0]]
            if spacing[2] > 1.5:
                continue

            img_data = img.get_fdata()

            # Crop using coronal and sagittal projections
            cor_proj = np.mean(img_data, axis=(0, 2))
            sag_proj = np.mean(img_data, axis=(0, 1))
            cor_mask = np.argwhere(cor_proj > -500).squeeze()
            sag_mask = np.argwhere(sag_proj > -600).squeeze()
            img_data = img_data[:, cor_mask[0]:cor_mask[-1]+1, sag_mask[0]:sag_mask[-1]+1]

            # Resample
            if spacing[2] == 0.5:
                upsampled = img_data[::2]
            elif spacing[2] == 1.0:
                upsampled = img_data
            else:
                upsampled = resample_z_axis(img_data, spacing[2])

            # Adjust slice count to multiple of scale_factor
            last = upsampled.shape[0] - (upsampled.shape[0] - 1) % scale_factor
            upsampled = upsampled[:last]
            downsampled = upsampled[::scale_factor]

            case_out = os.path.join(output_dir, case)
            os.makedirs(case_out, exist_ok=True)

            if save_npy:
                up_norm = clip_and_normalize(upsampled)
                np.save(os.path.join(case_out, f'imaging_1mm.npy'), up_norm.astype(np.float16))
                save_thumbnail(up_norm, os.path.join(output_dir, 'thumbnail', f'{case}.jpg'))

            if save_nii:
                affine_1mm = affine.copy()
                affine_1mm[2, 0] = affine[2, 0] / (spacing[2] / 1.0)
                nib.save(nib.Nifti1Image(upsampled.astype(np.float32), affine_1mm),
                         os.path.join(case_out, f'imaging_1mm.nii.gz'))

                affine_down = affine_1mm.copy()
                affine_down[2, 0] *= scale_factor
                nib.save(nib.Nifti1Image(downsampled.astype(np.float32), affine_down),
                         os.path.join(case_out, f'imaging_{scale_factor}mm.nii.gz'))

        except Exception as e:
            print(f"[!] Failed to process {case}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Downsample and move KiTS23 CT data with cropping and normalization.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing case folders with imaging.nii.gz')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output files')
    parser.add_argument('--scale_factor', type=int, default=5, help='Downsampling factor (e.g., 2, 3, 4, 5)')
    parser.add_argument('--save_npy', action='store_true', help='Save normalized .npy volume (default: False)')
    parser.add_argument('--save_nii', action='store_true', help='Save intermediate .nii.gz volumes (default: False)')
    args = parser.parse_args()

    down_sample_and_move(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        scale_factor=args.scale_factor,
        save_npy=args.save_npy,
        save_nii=args.save_nii
    )

    print("All cases processed.")
