# File: train.py

import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_synapse
import sys

# 设置日志级别
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

parser = argparse.ArgumentParser()
# 默认使用 Kvasir-SEG
parser.add_argument('--dataset', type=str,
                    default='Kvasir-SEG', help='experiment_name (e.g., Kvasir-SEG)')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu  ')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()

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

    dataset_name = args.dataset

    # ====================================================================
    # 路径修正点：现在直接从项目根目录进入 data/data
    # 路径: F:\danzi10\4.5w\daima\TransUNet-main\data\data
    BASE_DATA_DIR_RELATIVE = os.path.join('data', 'data')
    # ====================================================================

    # --- 数据集配置字典 ---
    # 所有息肉数据集 (Polyp) 都是 2 类 (背景+前景)
    dataset_config = {
        'Synapse': {  # 保留 Synapse 以防万一
            'root_path': os.path.join('..', 'data', 'Synapse', 'train_npz'),
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
        },
        'Kvasir-SEG': {
            'root_path': os.path.join(BASE_DATA_DIR_RELATIVE, 'Kvasir-SEG'),
            'list_dir': './lists/lists_Kvasir-SEG',
            'num_classes': 2,
        },
        'CVC-ClinicDB': {
            'root_path': os.path.join(BASE_DATA_DIR_RELATIVE, 'CVC-ClinicDB'),
            'list_dir': './lists/lists_CVC-ClinicDB',
            'num_classes': 2,
        },
        'CVC-ColonDB': {
            'root_path': os.path.join(BASE_DATA_DIR_RELATIVE, 'CVC-ColonDB'),
            'list_dir': './lists/lists_CVC-ColonDB',
            'num_classes': 2,
        },
        'CVC-300': {
            'root_path': os.path.join(BASE_DATA_DIR_RELATIVE, 'CVC-300'),
            'list_dir': './lists/lists_CVC-300',
            'num_classes': 2,
        },
        'ETIS-LaribPolyp DB': {
            'root_path': os.path.join(BASE_DATA_DIR_RELATIVE, 'ETIS-LaribPolyp DB'),
            'list_dir': './lists/lists_ETIS-LaribPolyp DB',
            'num_classes': 2,
        },
    }

    if dataset_name not in dataset_config:
        print(f"Error: Dataset '{dataset_name}' not configured.")
        print(f"Available datasets: {list(dataset_config.keys())}")
        sys.exit(1)

    # 动态加载配置
    config = dataset_config[dataset_name]
    args.num_classes = config['num_classes']
    args.root_path = config['root_path']
    args.list_dir = config['list_dir']

    # 路径增强：转换为绝对路径，确保数据加载器能够准确找到文件
    args.root_path = os.path.abspath(args.root_path)
    args.list_dir = os.path.abspath(args.list_dir)

    print(f"--- Dataset Configuration ---")
    print(f"Dataset: {dataset_name}")
    print(f"Root Path: {args.root_path}")
    print(f"List Dir: {args.list_dir}")
    print(f"Num Classes: {args.num_classes}")
    print(f"-----------------------------")

    # 检查数据根目录是否存在
    if not os.path.isdir(args.root_path):
        logging.error(f"Data root path not found: {args.root_path}")
        logging.error("Please ensure the data is correctly placed in the 'data/data' directory relative to the script.")
        sys.exit(1)

    args.is_pretrain = True
    # 替换空格以确保路径有效
    exp_name = dataset_name.replace(' ', '_')
    args.exp = 'TU_' + exp_name + str(args.img_size)

    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(
        args.vit_patches_size) if args.vit_patches_size != 16 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.max_iterations)[
                                          0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.img_size)
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    config_vit = CONFIGS_ViT_seg[args.vit_name]

    # 路径修正 FIX: 解决预训练模型权重路径问题
    if config_vit.pretrained_path.startswith('../'):
        config_vit.pretrained_path = config_vit.pretrained_path[3:]
        logging.warning(f"Fixed pretrained path to: {config_vit.pretrained_path}")

    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size),
                                   int(args.img_size / args.vit_patches_size))

    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    # 确保预训练权重路径存在
    if os.path.exists(config_vit.pretrained_path):
        net.load_from(weights=np.load(config_vit.pretrained_path))
    else:
        logging.warning(
            f"Pretrained weights not found at {config_vit.pretrained_path}. Starting training from scratch.")

    trainer = {'Synapse': trainer_synapse,
               'Kvasir-SEG': trainer_synapse,
               'CVC-ClinicDB': trainer_synapse,
               'CVC-ColonDB': trainer_synapse,
               'CVC-300': trainer_synapse,
               'ETIS-LaribPolyp DB': trainer_synapse,
               }

    if dataset_name in trainer:
        trainer[dataset_name](args, net, snapshot_path)
    else:
        raise NotImplementedError(f"Trainer for {dataset_name} is not defined.")
