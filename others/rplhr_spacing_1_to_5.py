import os
import nibabel as nib
import numpy as np

data_dir = "/home/user/dataset/RPLHR-CT"
out_dir = "/home/user/dataset/RPLHR-CT-5mm"
mode_list = sorted(os.listdir(data_dir))
for mode in mode_list:
    thickness_list = sorted(os.listdir(os.path.join(data_dir, mode)))
    for thickness in thickness_list:
        if thickness not in ['5mm']:
            continue
        case_list = sorted(os.listdir(os.path.join(data_dir, mode, thickness)))
        for case in case_list:
            print(case)
            img_path = os.path.join(data_dir, mode, thickness, case)
            img = nib.load(img_path)
            img_data = img.get_fdata()
            affine = img.affine
            spacing = [affine[0, 0], affine[1, 1], affine[2, 2]]
            affine_down = affine.copy()
            affine_down[2, 2] = 5.
            os.makedirs(os.path.join(out_dir, mode, thickness), exist_ok=True)
            nib.save(nib.Nifti1Image(img_data.astype(np.float32), affine=affine_down),
                     os.path.join(out_dir, mode, thickness, case))


print('fond')