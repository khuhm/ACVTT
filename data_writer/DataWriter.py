from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
import torch
import os
import json

class DataWriter:
    def __init__(self, args=None):
        self.result_dir = args.result_dir
        self.run_name = args.run_name
        self.writer = SummaryWriter(os.path.join(self.result_dir, self.run_name))
        pass

    def plot(self, tag, data, epoch):
        for key, value in data.items():
            for idx, metric in enumerate(value):
                self.writer.add_scalar(f'{tag}/{key}_{idx}', metric, epoch)
        pass

    def visualize(self, image, file_name='img_hr', folder_name='../images', run_name='exp', case='000', data_range=(-1024, 2048), clip_range=(-40, 300)):
        case = case[0]
        HU_MIN, HU_MAX = data_range
        lower_bound, upper_bound = clip_range
        os.makedirs(os.path.join(folder_name, run_name, f'{case}', ), exist_ok=True)
        img_denorm = image * (HU_MAX - HU_MIN) + HU_MIN
        img_clip = torch.clamp(img_denorm, lower_bound, upper_bound)
        img_norm = (img_clip - lower_bound) / (upper_bound - lower_bound)
        save_image(img_norm, os.path.join(folder_name, run_name, f'{case}', f'{file_name}.jpg'), normalize=False)
        pass

    def save_metrics(self, data, epoch):
        with open(os.path.join(self.result_dir, self.run_name, f'metrics_{epoch}.json'),'w') as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False, indent=4)


