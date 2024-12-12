# CT-slice-interpolation
Anisotropic Cross-View Texture Transfer with Multi-Reference Non-Local Attention for CT Slice Interpolation.

# Instructions
Belows are instructions for data preprocessing, model training, and inference.


## Data preprocessing
1. RPLHR-CT dataset
```bash
python others/rplhr_nii_to_npy.py
```
2. MSD dataset
```bash
python preprocess_msd.py
```
3. KiTS23 dataset
```bash
python downsample_image.py
```

## Training
Training scripts for each dataset. Change --num_ref_slices to the number of sampled axial slices for reference.
First train transfer part with L_trans, and then train with L_fuse
1. RPLHR-CT dataset
```bash
python scripts/train.py --dataset RPLHRCT --model ACVTT --num_ref_slices 3
python scripts/train.py --dataset RPLHRCT --do_fusion True long_skip True
```
2. MSD dataset
```bash
python scripts/train.py --dataset MSD --model ACVTT --num_ref_slices 3
python scripts/train.py --dataset MSD --do_fusion True long_skip True
```
3. KiTS23 dataset
```bash
python scripts/train.py --dataset KiTS23 --model ACVTT --num_ref_slices 3
python scripts/train.py --dataset KiTS23 --do_fusion True long_skip True
```

## Inference
Get inference results for test data for models 

1. RPLHR-CT dataset
```bash
python scripts/train.py --test_only True --dataset RPLHRCT --model ACVTT --num_ref_slices 3
python scripts/train.py --test_only True --dataset RPLHRCT --do_fusion True long_skip True
```
2. MSD dataset
```bash
python scripts/train.py --test_only True --dataset MSD --model ACVTT --num_ref_slices 3
python scripts/train.py --test_only True --dataset MSD --do_fusion True long_skip True
```
3. KiTS23 dataset
```bash
python scripts/train.py --test_only True --dataset KiTS23 --model ACVTT --num_ref_slices 3
python scripts/train.py --test_only True --dataset KiTS23 --do_fusion True long_skip True
```
