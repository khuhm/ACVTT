import argparse
import sys
sys.path.append('.')
from trainer.Trainer import Trainer
import getpass
# import torch
# torch.backends.cudnn.benchmark = True

user = getpass.getuser()

# arguments
parser = argparse.ArgumentParser(description='CT slice interpolation')
parser.add_argument('--data_dir', type=str, default=f"/home/{user}/dataset/RPLHR-CT-npy", help='dataset directory ')
parser.add_argument('--dataset', type=str, default="RPLHRCT", help=f'dataset class file name in "data_handler"')
parser.add_argument('--result_dir', type=str, default=f"/home/{user}/workspace/slice_interp_results", help=f'result directory')
parser.add_argument('--run_name', type=str, default="acvtt_ref_ran_3_transf_8_fine", help=f'run name')
parser.add_argument('--split_file', type=str, default=f"/home/{user}/workspace/CT-slice-interpolation/case_ids/splits.json", help=f'result directory')
parser.add_argument('--do_fusion', type=bool, default=False, help=f'if do fusion of cor/sag results')
parser.add_argument('--fusion_data_dir', type=str, default=f"/home/{user}/workspace/slice_interp_results/kits_acvtt_ran_2/model_11000", help='fusion dataset directory ')

parser.add_argument('--model', type=str, default="UNet", help=f'model class file name in "models"')
parser.add_argument('--loss_func', type=str, default='L1Loss', help=f'loss function')
parser.add_argument('--optimizer', type=str, default='Adam', help=f'optimizer')

parser.add_argument('--num_level', type=int, default=4, help=f'number of levels of encoder')
parser.add_argument('--num_channel', type=int, default=64, help=f'number of base channels')
parser.add_argument('--num_block', type=int, default=2, help=f'number of blocks per level')
parser.add_argument('--kernel_size', type=int, default=3, help=f'conv kernel size')
parser.add_argument('--mirror_axis', type=str, default='', help=f'augmentation mirror axis (None: '')')
parser.add_argument('--long_skip', type=bool, default=True, help=f'if use long skip connection')
parser.add_argument('--long_skip_feat', type=bool, default=False, help=f'if use long skip connection of feature')
parser.add_argument('--output_lr_up', type=bool, default=False, help=f'if output upsampled lr images')

parser.add_argument('--act_first', type=bool, default=False, help=f'if include activation in first layer')
parser.add_argument('--cat_feat', type=bool, default=False, help=f'if concat at feature levels')
parser.add_argument('--down_op', type=str, default=None, help=f'downsampling operation')
parser.add_argument('--up_op', type=str, default=None, help=f'upsampling operation')
parser.add_argument('--use_resenc', type=str, default=False, help=f'if use residual encoder')
parser.add_argument('--use_resdec', type=str, default=False, help=f'if use residual decoder')
parser.add_argument('--act_after_res', type=str, default=False, help=f'if use activation after residual')

parser.add_argument('--use_ref_axial', type=bool, default=True, help=f'if use reference axial slices')
parser.add_argument('--num_ref_slices', type=int, default=3, help=f'number of reference slices')
parser.add_argument('--num_ref_enc_level', type=int, default=4, help=f'number of reference encoder level')
parser.add_argument('--num_ref_channel', type=int, default=64, help=f'number of ref encoder channels')
parser.add_argument('--num_block_ref', type=int, default=2, help=f'number of blocks per level for ref')
parser.add_argument('--down_op_ref', type=str, default=None, help=f'downsampling operation')
parser.add_argument('--use_resenc_ref', type=str, default=False, help=f'if use residual encoder')
parser.add_argument('--act_after_res_ref', type=str, default=False, help=f'if use activation after residual')
parser.add_argument('--transfer_level', type=int, nargs='+', default=[0, 1, 2, 3], help=f'transfer feature levels')
parser.add_argument('--block_size_per_level', type=int, nargs='+', default=[8, 4, 2, 1], help=f'block size at each level')

parser.add_argument('--do_norm', type=bool, default=True, help=f'if do projection to qkv')
parser.add_argument('--norm_type_for_attn', type=str, default='sep_layernorm', help=f'normalization type for attention')
parser.add_argument('--layernorm_affine', type=bool, default=True, help=f'layernorm affine')
parser.add_argument('--layernorm_bias', type=bool, default=True, help=f'layernorm bias')
parser.add_argument('--norm_order', type=int, default='1', help=f'normalization type for attention')
parser.add_argument('--norm_dim', type=int, default='-2', help=f'normalization type for attention')

parser.add_argument('--do_proj_qkv', type=bool, default=True, help=f'if do projection to qkv')
parser.add_argument('--num_block_proj_in', type=int, default=0, help=f'num block of proj in')
parser.add_argument('--kernel_size_proj_in', type=int, default=1, help=f'num of kernel size of proj in')

parser.add_argument('--do_proj_after_unfold', type=bool, default=False, help=f'if po_proj_after_unfold')
parser.add_argument('--reduce_proj_qk_dim', type=bool, default=False, help=f'if reduce proj qk dim ')
parser.add_argument('--reduce_proj_qk_by_block', type=bool, default=False, help=f'if reduce proj dim by block size')
parser.add_argument('--reduce_proj_v_by_block', type=bool, default=False, help=f'if reduce proj dim by block size')

parser.add_argument('--do_proj_after_norm', type=bool, default=False, help=f'if project after norm')
parser.add_argument('--proj_after_norm_qk_only', type=bool, default=False, help=f'if project after norm qk only')

parser.add_argument('--do_proj_out_unfold', type=bool, default=False, help=f'if do projection output unfold')

parser.add_argument('--do_proj_after_attn', type=bool, default=True, help=f'if do proj after attention')
parser.add_argument('--num_block_proj_out', type=int, default=0, help=f'num block of proj after attention')
parser.add_argument('--kernel_size_proj_out', type=int, default=1, help=f'num of kernel size of proj out')

parser.add_argument('--output_attention', type=bool, default=True, help=f'nif output_attention')
parser.add_argument('--attn_softmax', type=bool, default=True, help=f'if attention softmax')
parser.add_argument('--show_normalize', type=bool, default=False, help=f'nif output_attention')
parser.add_argument('--output_residual', type=bool, default=False, help=f'if output residual')


parser.add_argument('--get_uniform_train_ref', type=bool, default=False, help=f'if get uniform train reference slices')
parser.add_argument('--uniform_train_ref_in_patch', type=bool, default=False, help=f'if  uniform train reference slices in patch')
parser.add_argument('--get_random_test_ref', type=bool, default=False, help=f'if get random test reference slices')
parser.add_argument('--get_uniform_test_ref', type=bool, default=True, help=f'if get uniformly sampled test ref slices')
parser.add_argument('--alpha_trainable', type=bool, default=False, help=f'alpha in transfer block')
parser.add_argument('--alpha_transf', type=float, default=1., help=f'alpha in transfer block')

parser.add_argument('--shared_ref_enc', type=bool, default=True, help=f'if share ref encoder')
parser.add_argument('--init_ref_enc', type=bool, default=False, help=f'if share ref encoder')

parser.add_argument('--apply_attention', type=bool, default=True, help=f'if apply attention')
parser.add_argument('--zero_conv', type=bool, default=False, help=f'if initialized to zero')
parser.add_argument('--pre_transfer', type=bool, default=False, help=f'if pre transfer')

parser.add_argument('--roll_hr_ref', type=bool, default=False, help=f'if roll hr reference')
parser.add_argument('--roll_xy', type=int, nargs='+', default=[255, 255], help=f'roll x y shift')
parser.add_argument('--random_roll', type=bool, default=False, help=f'random roll x y')
parser.add_argument('--roll_xy_start', type=int, nargs='+', default=[0, 0], help=f'roll x y start')
parser.add_argument('--roll_xy_step', type=int, nargs='+', default=[1, 1], help=f'roll x y step')
parser.add_argument('--identical_shift', type=bool, default=False, help=f'roll x y step')
parser.add_argument('--append_orig_hr_ref', type=bool, default=False, help=f'append original hr reference')

parser.add_argument('--NL_in_NL', type=bool, default=False, help=f'if do non local in non local')
parser.add_argument('--win_do_proj_qkv', type=bool, default=False, help=f'if do window project qkv')
parser.add_argument('--win_block_size_per_level', type=int, nargs='+', default=[2, 2, 2, 2], help=f'win_block_size_per_level')
parser.add_argument('--win_do_norm', type=bool,  default=False, help=f'win_block_size_per_level')
parser.add_argument('--win_do_proj_out', type=bool,  default=False, help=f'win_block_size_per_level')

parser.add_argument('--lr', type=float, default=1e-4, help=f'learning rate')
parser.add_argument('--last_epoch', type=int, default=20000, help=f'last epoch')
parser.add_argument('--scale', type=int, default=5, help=f'scale factor')
parser.add_argument('--patch_size', type=int, default=256, help=f'patch size')

parser.add_argument('--do_val', type=bool, default=True, help=f'if do validation')
parser.add_argument('--do_test', type=bool, default=True, help=f'if do test')
parser.add_argument('--save_freq', type=int, default=1000, help=f'model save frequency')
parser.add_argument('--val_freq', type=int, default=10, help=f'validation frequency')
parser.add_argument('--test_freq', type=int, default=1000, help=f'test frequency')
parser.add_argument('--save_test_image', type=bool, default=False, help=f'if save test result images')
parser.add_argument('--save_test_npy', type=bool, default=False, help=f'if save test result in npy')
parser.add_argument('--show_train_data', type=bool, default=False, help=f'if show(save) train data')
parser.add_argument('--print_curr_metric', type=bool, default=False, help=f'if print metrics for current case')
parser.add_argument('--eval_metric', type=bool, default=True, help=f'if evaluate metric')
parser.add_argument('--plot_metric', type=bool, default=False, help=f'if plot metric')
parser.add_argument('--not_eval_border', type=bool, default=True, help=f'if not evaluate border')
parser.add_argument('--not_eval_inter_hr', type=bool, default=False, help=f'if not evaluate border')

parser.add_argument('--load_model', type=bool, default=False, help=f'if load model')
parser.add_argument('--pretrained', type=bool, default=False, help=f'if model pretrained')
parser.add_argument('--pretrained_model', type=str, default='base_nCh64_ep11000_rfn_skip', help=f'pretrained model run name')
parser.add_argument('--model_name', type=str, default='model_0.pt', help=f'model name')
parser.add_argument('--freeze_pretrained', type=bool, default=False, help=f'if freeze model pretrained')
parser.add_argument('--unfreeze_decoder', type=bool, default=False, help=f'if unfreeze decoder')

parser.add_argument('--test_only', type=bool, default=False, help=f'if do test only')
parser.add_argument('--sliding_test', type=bool, default=False, help=f'if do sliding window inference')
parser.add_argument('--slide_overlap', type=int, default=0, help=f'slide window overlap')

parser.add_argument('--test_all_data', type=bool, default=False, help=f'if do test all data (train/val/test)')
parser.add_argument('--test_split', type=str, nargs='+', default=['train', 'test'], help=f'if do test all data (train/val/test)')

parser.add_argument('--input_gt', type=bool, default=False, help=f'if input gt instead of lr')
parser.add_argument('--use_hr_ref', type=bool, default=False, help=f'if use hr as reference')

args = parser.parse_args()
print(args)

# initialize a Trainer
trainer = Trainer(args)

# run train (intermediate validation)
if args.test_only:
    if args.test_all_data:
        trainer.run_test_all()
    else:
        trainer.run_testing()
else:
    trainer.run_training()

print('done')
