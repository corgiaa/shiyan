import os
import sys

# ----------------------------------------------------------------------
# 路径修复：强制将项目根目录添加到系统路径的最前面
# ----------------------------------------------------------------------
# 获取当前脚本所在的目录 (F:\danzi10\4.5w\daima\TransUNet-main)
base_dir = os.path.dirname(os.path.abspath(__file__))
# 将项目根目录添加到 sys.path 的最前面，确保 models 模块可以被找到
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)
# ----------------------------------------------------------------------

import argparse
import logging
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from importlib import import_module
from tqdm import tqdm
from scipy.ndimage import zoom

from utils import test_single_volume
from datasets.dataset_synapse import Synapse_dataset

# ----------------------------------------------------------------------
# 1. HARDCODED CONFIGURATION BLOCK
# ----------------------------------------------------------------------
# 定义您要加载的模型和配置
HARDCODED_CONFIG = {
    'enabled': True,  # 设置为 True 启用硬编码
    'snapshot_full_path': r"F:\danzi10\4.5w\daima\model\TU_Kvasir-SEG224\TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224\epoch_149.pth",
    'vit_name': 'R50-ViT-B_16',
    'max_epochs': 150,
    'dataset': 'Synapse',  # 保持 Synapse 只是为了使用其配置结构
    'num_classes': 2,  # Kvasir-SEG 假设为 2 类 (背景+息肉)
    'volume_path': '../data/Synapse/test_vol_h5',  # 确保测试数据路径正确
}
# ----------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='../data/Synapse/test_vol_h5', help='root dir for validation volume data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum iteration number to train')
parser.add_argument('--max_epochs', type=int, default=30, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action='store_true', help='whether to save results to nii')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=1e-2, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_name', type=str, default='ViT-B_16', help='select one vit model')
parser.add_argument('--vit_weights', type=str, default='imagenet', help='vit weights')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size')
parser.add_argument('--skip_layers', type=int, default=3, help='skip_layers')
parser.add_argument('--skip_channels', type=int, default=256, help='skip_channels')
parser.add_argument('--snapshot_path', type=str, default='../model', help='snapshot path')
parser.add_argument('--snapshot_full_path', type=str, default=None,
                    help='full path to the snapshot file')  # 增加这个参数用于硬编码

args = parser.parse_args()


def inference(args):
    # ----------------------------------------------------------------------
    # 2. APPLY HARDCODED CONFIGURATION
    # ----------------------------------------------------------------------
    if HARDCODED_CONFIG['enabled']:
        print("--- WARNING: Using Hardcoded Model Configuration ---")
        args.snapshot_full_path = HARDCODED_CONFIG['snapshot_full_path']
        args.vit_name = HARDCODED_CONFIG['vit_name']
        args.max_epochs = HARDCODED_CONFIG['max_epochs']
        args.dataset = HARDCODED_CONFIG['dataset']
        args.num_classes = HARDCODED_CONFIG['num_classes']
        args.volume_path = HARDCODED_CONFIG['volume_path']
        # 确保日志名称反映硬编码的设置
        args.exp = 'TU_' + args.dataset + str(args.img_size)
    # ----------------------------------------------------------------------

    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': args.volume_path,
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
            'z_spacing': 1,
        },
    }

    if args.dataset not in dataset_config:
        raise ValueError(f"Dataset {args.dataset} not configured.")

    db_config = dataset_config[args.dataset]

    # 覆盖 num_classes，以匹配硬编码值 (如果硬编码被启用)
    if not HARDCODED_CONFIG['enabled'] or args.dataset == 'Synapse':
        args.num_classes = db_config['num_classes']

    args.volume_path = db_config['volume_path']
    args.list_dir = db_config['list_dir']
    args.z_spacing = db_config['z_spacing']

    # ----------------------------------------------------------------------
    # 3. PATH GENERATION LOGIC
    # ----------------------------------------------------------------------
    # 无论是否使用硬编码，都优先使用 snapshot_full_path
    if args.snapshot_full_path:
        snapshot_path = args.snapshot_full_path
        # 提取模型名称用于日志
        model_name = os.path.basename(os.path.dirname(snapshot_path))
        args.exp = model_name
    else:
        # 原始的路径生成逻辑 (如果未硬编码)
        snapshot_path = os.path.join(args.snapshot_path, args.exp,
                                     args.vit_name + '_skip' + str(args.skip_layers) + '_epo' + str(
                                         args.max_epochs) + '_bs' + str(args.batch_size) + '_' + str(args.img_size))
        snapshot_path = os.path.join(snapshot_path, 'epoch_%d.pth' % (args.max_epochs - 1))

    # 检查快照文件是否存在
    if not os.path.exists(snapshot_path):
        logging.error(f"Error: Snapshot file not found at: {snapshot_path}")
        return

    # setup model
    # 这里的 import_module 现在应该能找到 models 目录了
    net = import_module('models.' + args.vit_name)
    config_vit = net.CONFIGS[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.skip_layers
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)

    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size),
                                   int(args.img_size / args.vit_patches_size))

    net = net.ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    # Load weights
    net.load_state_dict(torch.load(snapshot_path))
    print(f"Loaded model from {snapshot_path}")

    # Set up logging and output directory
    log_folder = './test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + args.exp + ".txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(f"Model loaded from {snapshot_path}")

    # Run inference
    net.eval()
    metric_list = 0.0
    db_test = db_config['Dataset'](base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    testloader = torch.utils.data.DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        metric_i = test_single_volume(sampled_batch, net, classes=args.num_classes,
                                      patch_size=[args.img_size, args.img_size],
                                      test_save_path=log_folder, is_savenii=args.is_savenii)
        metric_list += np.array(metric_i)

    metric_list = metric_list / len(db_test)

    # Output results
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i - 1][0], metric_list[i - 1][1]))

    performance = np.mean(metric_list[:, 0])
    mean_hd95 = np.mean(metric_list[:, 1])
    logging.info('Testing performance in best epoch: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return "Testing Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    inference(args)
