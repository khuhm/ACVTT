import importlib
import warnings
warnings.filterwarnings("ignore", message="torch.distributed.reduce_op is deprecated")

from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch
import os


class DataHandler:
    def __init__(self, args=None):
        self.result_dir = args.result_dir
        self.run_name = args.run_name
        self.show_normalize = args.show_normalize

        # load corresponding dataset class
        dataset_class = importlib.import_module(f'data_handler.{args.dataset}')
        dataset_instance = getattr(dataset_class, args.dataset)

        # dataset for train/val/test
        train_dataset = dataset_instance(args, mode='train')
        self.train_data_loader = DataLoader(train_dataset, shuffle=True)

        val_dataset = dataset_instance(args, mode='val')
        self.val_data_loader = DataLoader(val_dataset)

        test_dataset = dataset_instance(args, mode='test')
        self.test_data_loader = DataLoader(test_dataset)

    def show_data(self, data):
        case = data['case'][0]
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                if key == 'slice_lr':
                    continue
                os.makedirs(os.path.join(self.result_dir, self.run_name, 'images', f'{case}'), exist_ok=True)
                save_image(value, os.path.join(self.result_dir, self.run_name, 'images', f'{case}', f'{key}.jpg'), normalize=self.show_normalize)
        pass