import importlib
import torch
import os
import nibabel as nib
import numpy as np
import skimage
from utils.ssim import ssim_conv
from torchvision.utils import save_image
import copy

class Model:
    def __init__(self, args=None):
        # info.
        self.result_dir = args.result_dir
        self.run_name = args.run_name
        self.do_fusion = args.do_fusion
        self.save_test_image = args.save_test_image
        self.save_test_npy = args.save_test_npy
        self.model_name = args.model_name
        self.eval_metric = args.eval_metric
        self.not_eval_border = args.not_eval_border
        self.sliding_test = args.sliding_test
        self.lr = args.lr
        self.print_curr_metric = args.print_curr_metric
        self.pretrained = args.pretrained
        self.pretrained_model = args.pretrained_model
        self.freeze_pretrained = args.freeze_pretrained
        self.show_train_data = args.show_train_data
        self.show_normalize = args.show_normalize
        self.unfreeze_decoder = args.unfreeze_decoder
        self.init_ref_enc = args.init_ref_enc
        self.test_only = args.test_only
        self.not_eval_inter_hr = args.not_eval_inter_hr

        # model
        model_class = importlib.import_module(f'models.{args.model}')
        model_instance = getattr(model_class, args.model)
        self.network = model_instance(args)
        self.network.cuda()

        # loss
        loss_class = importlib.import_module(f'torch.nn')
        loss_instance = getattr(loss_class, args.loss_func)
        self.loss = loss_instance()

        # optimizer
        optim_class = importlib.import_module(f'torch.optim')
        optim_instance = getattr(optim_class, args.optimizer)
        self.optimizer = optim_instance(self.network.parameters(), lr=args.lr)

        # loss, metric dict
        self.loss_dict = {}
        self.metric_dict = {}
        pass

    def init_train_mode(self):
        self.network.train()
        self.loss_dict['count'] = 0
        self.loss_dict['avg_loss'] = 0
        pass

    def init_val_mode(self):
        self.network.eval()
        self.loss_dict['count'] = 0
        self.loss_dict['avg_loss'] = 0
        pass

    def init_test_mode(self):
        self.network.eval()
        self.loss_dict['count'] = 0
        self.loss_dict['avg_loss'] = 0
        pass

    def one_train_loop(self, data, mode='train'):
        # data to gpu
        self.data_to_gpu(data)

        # zero_grad
        self.optimizer.zero_grad()

        # forward
        output = self.network(data)

        # loss backward
        loss = self.loss(output['out'], data['slice_hr'])
        loss.backward()

        # optimize
        self.optimizer.step()

        # if save train data
        if self.show_train_data:
            output['case'] = data['case']
            self.show_data(output)

        # update avg loss
        self.update_avg_loss(loss)
        pass

    def one_val_loop(self, data):
        # data to gpu
        self.data_to_gpu(data)

        # forward
        with torch.no_grad():
            output = self.network(data)
            loss = self.loss(output['out'], data['slice_hr'])

        # if save train data
        if self.show_train_data:
            output['case'] = data['case']
            self.show_data(output)

        # update avg loss
        self.update_avg_loss(loss)
        pass

    def one_test_loop(self, data, tag='None'):
        # data to gpu
        self.data_to_gpu(data)

        # forward
        with torch.no_grad():
            if self.sliding_test:
                output = self.network.sliding_window_inference(data)
            elif self.do_fusion:
                output = self.network.test_slicewise(data)
            else:
                output = self.network.test_volume(data)

        # data to numpy
        self.data_to_numpy(data)
        self.data_to_numpy(output)

        # save result image
        if self.save_test_image:
            self.save_image(output['out_avg'], data['case'][0])

        if self.save_test_npy:
            self.save_npy(output, data['case'][0], tag)

        # compute metrics
        if self.eval_metric:
            metrics = self.compute_metrics(output, data['img_hr'])
            self.append_metrics(metrics)
            if self.print_curr_metric:
                self.print_metrics(metrics)
        pass

    def save(self, epoch):
        # save model
        torch.save({
            'epoch': epoch,
            'net_state': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
        }, os.path.join(self.result_dir, self.run_name, f'model_{epoch}.pt'))
        pass

    def load(self, model_name):
        if self.pretrained:
            checkpoint = torch.load(os.path.join(self.result_dir, self.pretrained_model, model_name))
            start_epoch = 0
        else:
            checkpoint = torch.load(os.path.join(self.result_dir, self.run_name, model_name))
            if self.test_only:
                start_epoch = checkpoint['epoch']
            else:
                start_epoch = checkpoint['epoch'] + 1

        if self.pretrained:
            model_dict = self.network.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['net_state'].items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.network.load_state_dict(model_dict)
            if self.freeze_pretrained:
                for name, param in self.network.named_parameters():
                    if name in pretrained_dict.keys():
                        if name.split('.')[0] in ['first_conv', 'encoder']:
                            param.requires_grad = False
                        else:
                            if self.unfreeze_decoder:
                                pass
                            else:
                                param.requires_grad = False
            if self.init_ref_enc:
                base_state_dict = self.network.encoder.state_dict()
                self.network.ref_encoder.load_state_dict(base_state_dict)
                base_conv_state_dict = self.network.first_conv.state_dict()
                self.network.ref_first_conv.load_state_dict(base_conv_state_dict)
                pass
        else:
            self.network.load_state_dict(checkpoint['net_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        return start_epoch

    def get_avg_loss(self):
        avg_loss_dict = {'loss': [self.loss_dict['avg_loss']]}
        return avg_loss_dict

    def get_metrics(self):
        return self.metric_dict

    def get_avg_metrics(self):
        avg_metrics = {}
        for key, value in self.metric_dict.items():
            if key not in avg_metrics:
                avg_metrics[key] = []
            for idx, metric in enumerate(value):
                avg_metrics[key].append(np.average(metric))
        return avg_metrics

    def data_to_gpu(self, data):
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = data[key].to(torch.device('cuda'), torch.float32)
                if key == 'slice_ref':
                    data[key] = data[key].permute(1, 0, 2, 3)
        return data

    def data_to_numpy(self, data):
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = data[key].detach().cpu().numpy()
        return data

    def update_avg_loss(self, loss):
        loss = loss.item()
        curr_sum = self.loss_dict['avg_loss'] * self.loss_dict['count']
        avg_loss = (curr_sum + loss) / (self.loss_dict['count'] + 1)
        self.loss_dict['count'] += 1
        self.loss_dict['avg_loss'] = avg_loss
        pass

    def save_image(self, img, case=None, tag=''):
        affine = np.eye(4)
        affine[0, 0] = -1
        affine[1, 1] = -1
        img = np.transpose(img.squeeze())
        img = np.clip(img, 0, 1)
        model_name = self.model_name.split('.')[0]
        os.makedirs(os.path.join(self.result_dir, self.run_name, model_name), exist_ok=True)
        nib.save(nib.Nifti1Image(img.astype(np.float32), affine=affine),
                 os.path.join(self.result_dir, self.run_name, model_name, f'{case}.nii.gz'))
        pass

    def save_npy(self, img, case=None, tag=''):
        img_cat = np.concatenate([img['out_cor'], img['out_sag']]).squeeze()
        model_name = self.model_name.split('.')[0]
        os.makedirs(os.path.join(self.result_dir, self.run_name, model_name, tag), exist_ok=True)
        np.save(os.path.join(self.result_dir, self.run_name, model_name, tag, case), img_cat.astype(np.float16))
        pass

    def show_data(self, data):
        case = data['case'][0]
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                os.makedirs(os.path.join(self.result_dir, self.run_name, 'images', f'{case}'), exist_ok=True)
                if key == 'attn':
                    for i in range(value.shape[1]):
                        save_image(value[:, [i]], os.path.join(self.result_dir, self.run_name, 'images', f'{case}', f'{key}_{i}.jpg'), normalize=self.show_normalize)
                else:
                    save_image(value, os.path.join(self.result_dir, self.run_name, 'images', f'{case}', f'{key}.jpg'), normalize=self.show_normalize)
        pass

    def compute_metrics(self, prediction, gt):
        gt = gt.squeeze()
        metrics = {}
        for key, value in prediction.items():
            value = value.squeeze()
            gt_data = gt.copy()
            # if not evaluate border
            if self.not_eval_border:
                value = value[5:-5]
                gt_data = gt_data[5:-5]

            if self.not_eval_inter_hr:
                inter_indices = np.array(range(gt_data.shape[0])) % 5 > 0
                value = value[inter_indices]
                gt_data = gt_data[inter_indices]
                pass

            # clamp
            value = np.clip(value, 0, 1)
            gt_data = np.clip(gt_data, 0, 1)

            # compute metrics
            psnr = skimage.metrics.peak_signal_noise_ratio(value, gt_data, data_range=1.)
            ssim = skimage.metrics.structural_similarity(value, gt_data, data_range=1.)
            metrics[key] = [psnr, ssim]
        return metrics

    def append_metrics(self, metrics):
        for key, value in metrics.items():
            if key not in self.metric_dict:
                self.metric_dict[key] = [[] for _ in range(len(value))]
            for idx, metric in enumerate(value):
                self.metric_dict[key][idx].append(metric)
        pass


    def print_metrics(self, metrics):
        print_dict = {}
        for key, value in metrics.items():
            value_rounded = np.round(value, 4)
            print_dict[key] = value_rounded
        
        print(print_dict)
        pass
