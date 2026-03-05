# File: F:\danzi10\4.5w\daima\TransUNet-main\main.py

import os
import sys
import argparse
import logging
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import matplotlib.pyplot as plt
import csv
import glob
from importlib import import_module
from tqdm import tqdm
from scipy.ndimage import zoom
from scipy.spatial.distance import directed_hausdorff
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
# from trainer import trainer_synapse # 原始 trainer_synapse 已被 modified_trainer_synapse 替代
# 假设 datasets.dataset_synapse 已经是最新的版本
from datasets.dataset_synapse import Synapse_dataset, RandomGenerator

# 尝试导入 torchinfo 用于模型摘要
try:
    from torchinfo import summary

    HAS_TORCHINFO = True
except ImportError:
    HAS_TORCHINFO = False
    logging.warning("torchinfo library not found. Cannot print detailed model summary.")

# ----------------------------------------------------------------------
# 路径修复：强制将项目根目录添加到系统路径的最前面
# ----------------------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)


# ----------------------------------------------------------------------
# --- 评价指标计算函数 (保持不变) ---
# ----------------------------------------------------------------------

def calculate_hausdorff_distance(pred_mask, true_mask):
    """计算Hausdorff距离"""
    try:
        # 获取边界点
        pred_points = np.argwhere(pred_mask > 0)
        true_points = np.argwhere(true_mask > 0)

        if len(pred_points) == 0 and len(true_points) == 0:
            return 0.0
        if len(pred_points) == 0 or len(true_points) == 0:
            # 如果缺少边界点，返回一个较大的值，表示分割失败或不完整
            return 100.0

        # 计算双向Hausdorff距离
        hd1 = directed_hausdorff(pred_points, true_points)[0]
        hd2 = directed_hausdorff(true_points, pred_points)[0]

        return max(hd1, hd2)
    except Exception as e:
        # logging.error(f"Error in calculate_hausdorff_distance: {e}")
        return 100.0


def calculate_hd95(pred_mask, true_mask):
    """计算95%Hausdorff距离"""
    try:
        pred_points = np.argwhere(pred_mask > 0)
        true_points = np.argwhere(true_mask > 0)

        if len(pred_points) == 0 and len(true_points) == 0:
            return 0.0
        if len(pred_points) == 0 or len(true_points) == 0:
            return 100.0

        # 计算从预测到真值的距离
        distances_pred_to_true = []
        if len(pred_points) > 0 and len(true_points) > 0:
            for pred_point in pred_points:
                min_dist = np.min(np.sqrt(np.sum((true_points - pred_point) ** 2, axis=1)))
                distances_pred_to_true.append(min_dist)

        # 计算从真值到预测的距离
        distances_true_to_pred = []
        if len(pred_points) > 0 and len(true_points) > 0:
            for true_point in true_points:
                min_dist = np.min(np.sqrt(np.sum((pred_points - true_point) ** 2, axis=1)))
                distances_true_to_pred.append(min_dist)

        all_distances = distances_pred_to_true + distances_true_to_pred
        if not all_distances: # 如果没有有效的距离，返回最大值
            return 100.0

        # 返回95%分位数
        return np.percentile(all_distances, 95)
    except Exception as e:
        # logging.error(f"Error in calculate_hd95: {e}")
        return 100.0


def calculate_assd(pred_mask, true_mask):
    """计算平均对称表面距离 (Average Symmetric Surface Distance)"""
    try:
        pred_points = np.argwhere(pred_mask > 0)
        true_points = np.argwhere(true_mask > 0)

        if len(pred_points) == 0 and len(true_points) == 0:
            return 0.0
        if len(pred_points) == 0 or len(true_points) == 0:
            return 100.0

        # 计算从预测到真值的距离
        distances_pred_to_true = []
        if len(pred_points) > 0 and len(true_points) > 0:
            for pred_point in pred_points:
                min_dist = np.min(np.sqrt(np.sum((true_points - pred_point) ** 2, axis=1)))
                distances_pred_to_true.append(min_dist)

        # 计算从真值到预测的距离
        distances_true_to_pred = []
        if len(pred_points) > 0 and len(true_points) > 0:
            for true_point in true_points:
                min_dist = np.min(np.sqrt(np.sum((pred_points - true_point) ** 2, axis=1)))
                distances_true_to_pred.append(min_dist)

        all_distances = distances_pred_to_true + distances_true_to_pred
        return np.mean(all_distances) if len(all_distances) > 0 else 100.0
    except Exception as e:
        # logging.error(f"Error in calculate_assd: {e}")
        return 100.0


def calculate_comprehensive_metrics(pred_mask, true_mask):
    """计算全面的评价指标"""
    try:
        pred_flat = pred_mask.flatten()
        true_flat = true_mask.flatten()

        # 基本统计
        tp = np.sum((pred_flat == 1) & (true_flat == 1))
        fp = np.sum((pred_flat == 1) & (true_flat == 0))
        fn = np.sum((pred_flat == 0) & (true_flat == 1))
        tn = np.sum((pred_flat == 0) & (true_flat == 0))

        # 防止除零
        epsilon = 1e-8

        # 1. Dice系数
        dice = 2 * tp / (2 * tp + fp + fn + epsilon)

        # 2. IoU (Jaccard指数)
        iou = tp / (tp + fp + fn + epsilon)

        # 3. 精确率 (Precision)
        precision = tp / (tp + fp + epsilon)

        # 4. 召回率/敏感性 (Recall/Sensitivity)
        recall = tp / (tp + fn + epsilon)

        # 5. 特异性 (Specificity)
        specificity = tn / (tn + fp + epsilon)

        # 6. F1分数
        f1 = 2 * precision * recall / (precision + recall + epsilon)

        # 7. 准确率 (Accuracy)
        accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)

        # 8. 平衡准确率 (Balanced Accuracy)
        balanced_acc = (recall + specificity) / 2

        # 9. Matthews相关系数 (MCC)
        mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if mcc_denominator == 0:
            mcc = 0.0
        else:
            mcc = (tp * tn - fp * fn) / mcc_denominator

        # 10. Hausdorff距离
        hd = calculate_hausdorff_distance(pred_mask, true_mask)

        # 11. 95% Hausdorff距离
        hd95 = calculate_hd95(pred_mask, true_mask)

        # 12. 平均对称表面距离
        assd = calculate_assd(pred_mask, true_mask)

        # 13. 相对体积误差 (Relative Volume Error)
        pred_volume = np.sum(pred_mask)
        true_volume = np.sum(true_mask)
        if true_volume > 0:
            rve = abs(pred_volume - true_volume) / true_volume
        else:
            rve = 1.0 if pred_volume > 0 else 0.0 # 如果真值体积为0，预测体积大于0，则RVE为1.0

        return {
            'dice': dice,
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1': f1,
            'accuracy': accuracy,
            'balanced_acc': balanced_acc,
            'mcc': mcc,
            'hd': hd,
            'hd95': hd95,
            'assd': assd,
            'rve': rve
        }

    except Exception as e:
        # print(f"Error calculating metrics: {e}")
        # 返回默认值
        return {
            'dice': 0.0, 'iou': 0.0, 'precision': 0.0, 'recall': 0.0,
            'specificity': 0.0, 'f1': 0.0, 'accuracy': 0.0, 'balanced_acc': 0.0,
            'mcc': 0.0, 'hd': 100.0, 'hd95': 100.0, 'assd': 100.0, 'rve': 1.0
        }


# ----------------------------------------------------------------------
# --- 可视化函数 (保持不变) ---
# ----------------------------------------------------------------------

def plot_segmentation_results(image, gt_mask, pred_mask, case_name, save_path):
    """
    绘制单张图像的输入、真值、预测和叠加结果 (4图)
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # 1. 输入图像
    # Assuming image is 0-1 float array here for consistency with the loader's output
    axes[0].imshow(image)
    axes[0].set_title("1. Input Image")
    axes[0].axis('off')

    # 2. 真值掩码
    axes[1].imshow(gt_mask, cmap='gray')
    axes[1].set_title("2. Ground Truth")
    axes[1].axis('off')

    # 3. 预测掩码
    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title("3. Prediction")
    axes[2].axis('off')

    # 4. 叠加图 (使用红色表示预测，绿色表示真值)
    # 创建一个叠加图像
    overlay = np.zeros_like(image)
    # Ensure overlay has 3 channels for RGB if image is grayscale
    if overlay.ndim == 2:
        overlay = np.stack([overlay, overlay, overlay], axis=-1)

    overlay[..., 0] = pred_mask * 255  # Red for Prediction
    overlay[..., 1] = gt_mask * 255  # Green for Ground Truth

    # 将原图转换为灰度图以进行叠加
    if image.ndim == 3:
        if image.shape[2] == 3:
            # Simple grayscale conversion for visualization if it's RGB
            gray_image = np.mean(image, axis=2)
        elif image.shape[2] == 1:
            gray_image = image[..., 0]
        else:
            # Handle non-standard channel count by just taking the first channel
            gray_image = image[..., 0]
    else:  # 2D image
        gray_image = image

    axes[3].imshow(gray_image, cmap='gray')
    # Rescale overlay to match image dimensions if necessary (e.g., if image was 3 channel)
    # The overlay is already created with the same H, W as image_resized.
    # If image_resized is 3-channel, overlay is 3-channel. If image_resized is 2D, overlay is 3-channel.
    # So, no need to recreate overlay based on gray_image shape.
    axes[3].imshow(overlay, alpha=0.5)
    axes[3].set_title("4. Overlay (R: Pred, G: GT)")
    axes[3].axis('off')

    plt.suptitle(f"Segmentation Results for Case: {case_name}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Ensure case name is file-system safe
    safe_case_name = case_name.replace('/', '_').replace('\\', '_') # Also handle backslashes
    plt.savefig(os.path.join(save_path, f"{safe_case_name}_segmentation.png"))
    plt.close(fig)


def plot_metrics(history_metrics, current_metrics_list, epoch, save_path, is_final=False):
    """
    绘制历史趋势图 (5, 6, 7) 和当前周期指标分布图 (8, 9, 10)
    """

    # --- 历史趋势图 (Plots 5, 6, 7) ---
    if history_metrics:
        epochs = [m['epoch'] for m in history_metrics]
        dice_scores = [m['dice'] for m in history_metrics]
        iou_scores = [m['iou'] for m in history_metrics]
        hd95_scores = [m['hd95'] for m in history_metrics]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 5. Dice vs Epoch
        axes[0].plot(epochs, dice_scores, marker='o', linestyle='-', color='blue')
        axes[0].set_title('5. Dice Score vs. Epoch')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Dice Score')
        axes[0].set_ylim(0, 1.0)
        axes[0].grid(True)

        # 6. IoU vs Epoch
        axes[1].plot(epochs, iou_scores, marker='o', linestyle='-', color='green')
        axes[1].set_title('6. IoU vs. Epoch')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('IoU Score')
        axes[1].set_ylim(0, 1.0)
        axes[1].grid(True)

        # 7. HD95 vs Epoch
        axes[2].plot(epochs, hd95_scores, marker='o', linestyle='-', color='red')
        axes[2].set_title('7. HD95 vs. Epoch (Lower is Better)')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('HD95 Distance')
        axes[2].grid(True)

        plt.suptitle("Training History Trend", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(save_path, "05_06_07_history_trends.png"))
        plt.close(fig)

    # --- 当前周期指标分布图 (Plots 8, 9, 10) ---
    if current_metrics_list:

        # 提取当前周期的指标值 (假设只有 Class 1, since num_classes=2 for polyp)
        # Filter out empty or invalid entries
        valid_metrics = [m[0] for m in current_metrics_list if m and len(m) > 0 and 'dice' in m[0]]

        if not valid_metrics:
            logging.warning(f"No valid metrics to plot for epoch {epoch} distributions.")
            return

        dice_values = [m['dice'] for m in valid_metrics]
        hd95_values = [m['hd95'] for m in valid_metrics]
        precision_values = [m['precision'] for m in valid_metrics]
        recall_values = [m['recall'] for m in valid_metrics]


        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 8. Dice Score Distribution (Histogram)
        axes[0].hist(dice_values, bins=15, range=(0, 1), color='skyblue', edgecolor='black')
        axes[0].set_title(f'8. Dice Score Distribution (E{epoch})')
        axes[0].set_xlabel('Dice Score')
        axes[0].set_ylabel('Frequency')
        if dice_values:
            axes[0].axvline(np.mean(dice_values), color='red', linestyle='dashed', linewidth=1,
                            label=f'Mean: {np.mean(dice_values):.4f}')
        axes[0].legend()

        # 9. HD95 Score Distribution (Histogram)
        # Filter out extremely high values for better visualization, e.g., max 50
        hd95_plot_values = [min(v, 50) for v in hd95_values]
        axes[1].hist(hd95_plot_values, bins=15, color='lightcoral', edgecolor='black')
        axes[1].set_title(f'9. HD95 Distribution (E{epoch}) (Capped at 50)')
        axes[1].set_xlabel('HD95 Distance')
        axes[1].set_ylabel('Frequency')
        if hd95_values:
            axes[1].axvline(np.mean(hd95_values), color='blue', linestyle='dashed', linewidth=1,
                            label=f'Mean: {np.mean(hd95_values):.4f}')
        axes[1].legend()

        # 10. Precision vs Recall Scatter Plot
        axes[2].scatter(precision_values, recall_values, color='purple', alpha=0.6)
        axes[2].set_title(f'10. Precision vs Recall (E{epoch})')
        axes[2].set_xlabel('Precision')
        axes[2].set_ylabel('Recall')
        axes[2].set_xlim(0, 1.05)
        axes[2].set_ylim(0, 1.05)
        axes[2].grid(True, linestyle='--')

        plt.suptitle(f"Epoch {epoch} Metric Distributions", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # 保存到当前测试周期的日志文件夹
        if is_final:
            plot_name = "08_09_10_final_metrics_distributions.png"
        else:
            plot_name = f"08_09_10_epoch_{epoch}_metrics_distributions.png"

        plt.savefig(os.path.join(save_path, plot_name))
        plt.close(fig)


def custom_test_single_volume(image_batch, label_batch, net, classes, patch_size, test_save_path=None, case=None,
                              z_spacing=1, visualize_sample=False):
    """
    自定义的测试单个样本函数，返回全面的评价指标
    """
    try:
        # 确保网络在评估模式
        net.eval()

        # 处理输入数据
        if torch.is_tensor(image_batch):
            image = image_batch
        else:
            image = image_batch['image'] if isinstance(image_batch, dict) else image_batch

        if torch.is_tensor(label_batch):
            label = label_batch
        else:
            label = label_batch['label'] if isinstance(label_batch, dict) else label_batch

        # 转换为numpy并确保在CPU上
        if torch.is_tensor(image):
            image_np = image.cpu().detach().numpy()
        if torch.is_tensor(label):
            label_np = label.cpu().detach().numpy()

        # 处理图像维度
        # Input tensor format: (1, C, H, W)
        if len(image_np.shape) == 4:  # (1, C, H, W)
            image_np = image_np[0]
        if len(image_np.shape) == 3 and image_np.shape[0] == 3:  # (C, H, W)
            image_np = np.transpose(image_np, (1, 2, 0))  # (H, W, C)

        # 处理标签维度
        if len(label_np.shape) == 4:  # (1, 1, H, W)
            label_np = label_np[0, 0]
        elif len(label_np.shape) == 3:  # (1, H, W) 或 (C, H, W)
            if label_np.shape[0] == 1:
                label_np = label_np[0]
            else:
                label_np = label_np[0]

        # 确保图像是3通道的 (如果输入是灰度图，需要复制通道)
        if image_np.ndim == 2:
            image_np = np.stack([image_np, image_np, image_np], axis=-1)
        elif image_np.ndim == 3 and image_np.shape[2] == 1:
            image_np = np.repeat(image_np, 3, axis=2)

        # 图像此时应该是 (H, W, C) 且归一化到 [0, 1]

        # 调整大小到网络输入尺寸
        h, w = image_np.shape[:2]
        target_h, target_w = patch_size[0], patch_size[1]

        if h != target_h or w != target_w:
            # 图像缩放 (插值1)
            image_resized = zoom(image_np, (target_h / h, target_w / w, 1), order=1)
            # 标签缩放 (最近邻插值0)
            label_resized = zoom(label_np.astype(np.float32), (target_h / h, target_w / w), order=0)
            label_resized = label_resized.astype(np.int32)
        else:
            image_resized = image_np
            label_resized = label_np

        # 转换为tensor格式 (1, C, H, W)
        image_tensor = torch.from_numpy(image_resized).float()
        if len(image_tensor.shape) == 3:  # (H, W, C)
            image_tensor = image_tensor.permute(2, 0, 1)  # (C, H, W)
        image_tensor = image_tensor.unsqueeze(0).cuda()  # (1, C, H, W)

        # 网络推理
        with torch.no_grad():
            outputs = net(image_tensor)
            # 假设输出是 (1, C, H, W)
            prediction = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            prediction = prediction.squeeze(0).cpu().numpy()

        # 如果预测结果和标签大小不匹配，调整预测结果
        if prediction.shape != label_resized.shape:
            pred_h, pred_w = prediction.shape
            label_h, label_w = label_resized.shape
            if pred_h != label_h or pred_w != label_w:
                prediction = zoom(prediction.astype(np.float32),
                                  (label_h / pred_h, label_w / pred_w), order=0)
                prediction = prediction.astype(np.int32)

        # 获取案例名称 (如果 case 在调用时没有传入，则尝试从 sampled_batch 中获取)
        if case is None:
            if isinstance(image_batch, dict) and 'case_name' in image_batch:
                case = image_batch['case_name'][0]
            else:
                case = f"Unknown_Case_{random.randint(0, 9999)}"

        # 计算每个类别的详细指标
        class_metrics = []
        # 从类别 1 开始计算 (0 是背景)
        for class_idx in range(1, classes):
            pred_class = (prediction == class_idx).astype(np.uint8)
            label_class = (label_resized == class_idx).astype(np.uint8)

            metrics = calculate_comprehensive_metrics(pred_class, label_class)
            class_metrics.append(metrics)

            # --- 可视化：仅对第一个类别进行可视化 (息肉分割通常只有前景一个类别) ---
            if visualize_sample and class_idx == 1:
                plot_segmentation_results(image_resized, label_class, pred_class, case, test_save_path)

        return class_metrics

    except Exception as e:
        logging.error(f"Error in custom_test_single_volume for case {case}: {str(e)}")
        import traceback
        traceback.print_exc()
        # 返回默认值
        default_metrics = {
            'dice': 0.0, 'iou': 0.0, 'precision': 0.0, 'recall': 0.0,
            'specificity': 0.0, 'f1': 0.0, 'accuracy': 0.0, 'balanced_acc': 0.0,
            'mcc': 0.0, 'hd': 100.0, 'hd95': 100.0, 'assd': 100.0, 'rve': 1.0
        }
        return [default_metrics for _ in range(1, classes)]


# 设置日志级别
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

parser = argparse.ArgumentParser()
# 默认使用 Kvasir-SEG，但在 main 中会被覆盖为组合数据集名称
parser.add_argument('--dataset', type=str,
                    default='Kvasir-SEG', help='experiment_name (e.g., Kvasir-SEG)')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.02,
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
parser.add_argument('--test_interval', type=int,
                    default=10, help='test every N epochs')
parser.add_argument('--is_savenii', action='store_true', help='whether to save results to nii')
parser.add_argument('--num_visualize', type=int,
                    default=5, help='Number of samples to visualize during testing')

args = parser.parse_args()


def worker_init_fn(worker_id):
    """全局定义的worker初始化函数，避免多进程问题"""
    random.seed(args.seed + worker_id)


# ----------------------------------------------------------------------
# --- 新增：单个数据集分割函数 (替代 create_combined_split) ---
# ----------------------------------------------------------------------

def create_single_dataset_split(dataset_folder_name, dataset_full_path, dataset_list_dir, split_ratio=0.8):
    """
    为单个数据集创建训练和测试列表：
    训练: 80%
    测试: 20%
    """
    import glob
    import random

    random.seed(args.seed)

    train_cases = []
    test_cases = []

    logging.info(f"Creating {split_ratio*100:.0f}/{(1-split_ratio)*100:.0f} split for dataset: {dataset_folder_name}")

    # 路径结构: dataset_full_path / images / *
    images_dir = os.path.join(dataset_full_path, 'images')

    if not os.path.exists(images_dir):
        logging.error(f"Skipping {dataset_folder_name}: Image directory not found at {images_dir}")
        return [], []

    # 获取所有图像名称 (不带扩展名)
    image_files_png = glob.glob(os.path.join(images_dir, '*.png'))
    image_files_jpg = glob.glob(os.path.join(images_dir, '*.jpg'))
    image_files_jpeg = glob.glob(os.path.join(images_dir, '*.jpeg'))
    image_files_tif = glob.glob(os.path.join(images_dir, '*.tif'))
    image_files_tiff = glob.glob(os.path.join(images_dir, '*.tiff'))

    image_files = image_files_png + image_files_jpg + image_files_jpeg + image_files_tif + image_files_tiff

    image_names = [os.path.splitext(os.path.basename(f))[0] for f in image_files]

    # 存储案例名称，格式为 "DatasetFolderName/ImageName"
    # Synapse_dataset 会使用 base_dir + case_name 来构建完整路径
    # 例如：base_dir/Kvasir-SEG/image001.jpg
    case_names = [f"{dataset_folder_name}/{name}" for name in image_names]

    if not case_names:
        logging.warning(f"No cases found in {dataset_folder_name}.")
        return [], []

    random.shuffle(case_names)

    split_idx = int(len(case_names) * split_ratio)
    train_cases.extend(case_names[:split_idx])
    test_cases.extend(case_names[split_idx:])

    logging.info(f"{dataset_folder_name}: {len(train_cases)} training cases, {len(test_cases)} testing cases.")

    os.makedirs(dataset_list_dir, exist_ok=True)

    # 写入训练列表
    train_list_path = os.path.join(dataset_list_dir, 'train.txt')
    with open(train_list_path, 'w') as f:
        for name in train_cases:
            f.write(f"{name}\n")

    # 写入测试列表
    test_list_path = os.path.join(dataset_list_dir, 'test_vol.txt')
    with open(test_list_path, 'w') as f:
        for name in test_cases:
            f.write(f"{name}\n")

    logging.info(f"Split files created for {dataset_folder_name} in {dataset_list_dir}")
    return train_cases, test_cases


# ----------------------------------------------------------------------
# --- 训练和推理函数 (已修改 num_workers 和数据集配置) ---
# ----------------------------------------------------------------------

def inference_during_training(args, net, epoch, snapshot_path, history_metrics):
    """
    在训练过程中进行推理测试，包含全面的评价指标和可视化
    """
    logging.info(f"Starting comprehensive inference for {args.dataset} at epoch {epoch}")

    # 简化数据集配置，直接使用 args 中的参数
    db_config = {
        'Dataset': Synapse_dataset,
        'volume_path': args.root_path, # 此时 args.root_path 已经是所有数据集的父目录 (e.g., data/data)
        'list_dir': args.list_dir,     # 此时 args.list_dir 已经是当前数据集的 list 目录
        'num_classes': args.num_classes,
        'z_spacing': 1, # 假设息肉数据集都是2D图像
    }

    # 设置测试日志目录
    test_log_folder = os.path.join(snapshot_path, 'test_logs', f'epoch_{epoch}')
    os.makedirs(test_log_folder, exist_ok=True)
    os.makedirs(os.path.join(test_log_folder, 'visualizations'), exist_ok=True)

    # 设置测试专用的日志记录器
    test_logger = logging.getLogger(f'test_epoch_{epoch}_{args.dataset}')
    test_logger.setLevel(logging.INFO)

    # 清除之前的处理器，避免重复写入
    for handler in test_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            test_logger.removeHandler(handler)

    # 添加文件处理器
    test_handler = logging.FileHandler(os.path.join(test_log_folder, f'test_epoch_{epoch}.log'))
    test_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    test_handler.setFormatter(formatter)
    test_logger.addHandler(test_handler)

    test_logger.info(f"Testing with comprehensive metrics for {args.dataset} at epoch {epoch}")

    try:
        # 创建测试数据集
        db_test = db_config['Dataset'](base_dir=db_config['volume_path'],
                                       split="test_vol",
                                       list_dir=db_config['list_dir'])

        if len(db_test) == 0:
            test_logger.warning(f"No test data found for {args.dataset}!")
            return None

        # num_workers=0 是因为 custom_test_single_volume 内部使用了 numpy/zoom 操作，
        # 在多进程环境下可能会导致数据拷贝开销或冲突，单进程测试更稳定。
        testloader = torch.utils.data.DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)

        # 设置网络为评估模式
        net.eval()
        all_metrics = []
        visualized_count = 0

        with torch.no_grad():
            for i_batch, sampled_batch in tqdm(enumerate(testloader), desc=f"Testing {args.dataset} Epoch {epoch}",
                                               total=len(testloader)):

                # 获取案例名称用于日志和可视化
                case_name = sampled_batch['case_name'][0] if 'case_name' in sampled_batch else f"case_{i_batch}"

                # 决定是否可视化当前样本
                visualize_flag = visualized_count < args.num_visualize

                try:
                    metrics_i = custom_test_single_volume(
                        sampled_batch['image'],
                        sampled_batch['label'],
                        net,
                        classes=args.num_classes,
                        patch_size=[args.img_size, args.img_size],
                        test_save_path=os.path.join(test_log_folder, 'visualizations'),
                        case=case_name,
                        z_spacing=db_config['z_spacing'],
                        visualize_sample=visualize_flag
                    )

                    if metrics_i is not None and len(metrics_i) > 0:
                        all_metrics.append(metrics_i)
                        if visualize_flag:
                            visualized_count += 1

                except Exception as e:
                    test_logger.error(f"Error processing batch {i_batch} ({case_name}) for {args.dataset}: {str(e)}")
                    continue

        if len(all_metrics) > 0:
            # 计算每个类别的平均指标
            metric_names = ['dice', 'iou', 'precision', 'recall', 'specificity',
                            'f1', 'accuracy', 'balanced_acc', 'mcc', 'hd', 'hd95', 'assd', 'rve']

            avg_metrics_epoch = {}

            # 对于息肉分割 (num_classes=2)，我们只关心 class_idx=0 (即前景)
            # metrics_i 包含的是从 class_idx=1 开始的指标，所以这里取 class_idx=0
            class_idx_for_foreground = 0
            test_logger.info(f"\n=== {args.dataset} - Class {class_idx_for_foreground + 1} (Foreground) Metrics ===")

            for metric_name in metric_names:
                values = [metrics[class_idx_for_foreground][metric_name] for metrics in all_metrics
                          if class_idx_for_foreground < len(metrics)]
                if values:
                    # 过滤掉极端的HD/ASSD值 (例如，100.0)
                    if 'hd' in metric_name or 'assd' in metric_name:
                        valid_values = [v for v in values if v < 99.0]
                        if not valid_values: # 如果所有值都是100.0，则平均值也设为100.0
                            mean_val = 100.0
                            std_val = 0.0
                        else:
                            mean_val = np.mean(valid_values)
                            std_val = np.std(valid_values)
                    else:
                        mean_val = np.mean(values)
                        std_val = np.std(values)

                    # 存储平均值用于历史跟踪
                    avg_metrics_epoch[metric_name] = mean_val

                    test_logger.info(f'{metric_name.upper()}: {mean_val:.6f} ± {std_val:.6f}')

            # 计算总体平均性能 (基于第一个类别)
            overall_dice = avg_metrics_epoch.get('dice', 0.0)
            overall_iou = avg_metrics_epoch.get('iou', 0.0)
            overall_hd95 = avg_metrics_epoch.get('hd95', 100.0)

            test_logger.info(f"\n=== {args.dataset} - Overall Performance at Epoch {epoch} ===")
            test_logger.info(f'Overall DICE: {overall_dice:.6f}')
            test_logger.info(f'Overall IoU: {overall_iou:.6f}')
            test_logger.info(f'Overall HD95: {overall_hd95:.6f}')

            # 绘制当前周期的指标分布图 (Plots 8, 9, 10)
            plot_metrics(None, all_metrics, epoch, test_log_folder, is_final=(epoch == args.max_epochs))

            # 保存详细结果到CSV文件
            csv_path = os.path.join(test_log_folder, f'detailed_metrics_epoch_{epoch}.csv')
            with open(csv_path, 'w', newline='') as csvfile:
                fieldnames = ['sample_id', 'case_name', 'class_id'] + metric_names
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                # 重新遍历数据加载器以获取 case_name
                testloader_for_logging = torch.utils.data.DataLoader(db_test, batch_size=1, shuffle=False,
                                                                     num_workers=0)

                for sample_idx, sampled_batch in enumerate(testloader_for_logging):
                    if sample_idx >= len(all_metrics):
                        break

                    case_name_log = sampled_batch['case_name'][
                        0] if 'case_name' in sampled_batch else f"case_{sample_idx}"
                    metrics = all_metrics[sample_idx]

                    for class_id_in_list, class_metrics_data in enumerate(metrics):
                        row = {'sample_id': sample_idx, 'case_name': case_name_log, 'class_id': class_id_in_list + 1}
                        row.update(class_metrics_data)
                        writer.writerow(row)

            test_logger.info(f"Detailed metrics saved to: {csv_path}")

        else:
            overall_dice, overall_iou, overall_hd95 = 0.0, 0.0, 100.0
            test_logger.warning(f"No valid metrics computed for {args.dataset}")

        # 将网络重新设置为训练模式
        net.train()

        # 返回主要指标用于历史跟踪
        return {
            'epoch': epoch,
            'dice': overall_dice,
            'hd95': overall_hd95,
            'iou': overall_iou
        }

    except Exception as e:
        test_logger.error(f"Error during testing for {args.dataset}: {str(e)}")
        import traceback
        test_logger.error(traceback.format_exc())
        net.train()
        return None


def modified_trainer_synapse(args, model, snapshot_path):
    """
    修改后的训练器，包含定期测试功能和历史记录
    **FIXED: Increased num_workers for faster data loading.**
    """
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    import torch.optim as optim
    from torch.nn.modules.loss import CrossEntropyLoss
    from torch.utils.data import DataLoader

    # 设置日志
    log_file_path = os.path.join(snapshot_path, f"log_{args.dataset}.txt") # 每个数据集独立的日志文件
    # 清除之前的日志处理器，确保每个数据集有独立的日志文件
    for handler in logging.getLogger().handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logging.getLogger().removeHandler(handler)
    logging.basicConfig(filename=log_file_path, level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')

    # 确保控制台输出也被添加
    if not any(isinstance(handler, logging.StreamHandler) for handler in logging.getLogger().handlers):
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info(f"Training parameters for {args.dataset}: {str(args)}")

    base_lr = args.base_lr # 使用传入的 args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    # --- FIX: Increase num_workers to 4 for parallel data loading ---
    NUM_WORKERS = 4
    # ---------------------------------------------------------------

    # 创建训练数据集 (使用当前数据集的 train.txt 列表和 root_path)
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=RandomGenerator(output_size=[args.img_size, args.img_size]))

    logging.info(f"The length of train set for {args.dataset} is: {len(db_train)}")

    # 使用 NUM_WORKERS 启用多进程加载
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True,
                             worker_init_fn=worker_init_fn if NUM_WORKERS > 0 else None)

    # 如果模型是DataParallel包装的，确保在每个训练周期开始时是正确的
    # 或者在循环外部处理DataParallel
    if args.n_gpu > 1 and not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)
    elif args.n_gpu == 1 and isinstance(model, nn.DataParallel):
        # 如果只有1个GPU，且模型被DataParallel包装了，解包
        model = model.module

    model.train()
    ce_loss = CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)

    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    best_performance = 0.0
    history_metrics = []  # 存储所有测试周期的平均指标

    iterator = tqdm(range(max_epoch), ncols=70, desc=f"Training {args.dataset}")
    for epoch_num in iterator:
        model.train()
        epoch_loss = 0.0

        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            outputs = model(image_batch)
            loss = ce_loss(outputs, label_batch[:].long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            epoch_loss += loss.item()

            if iter_num % 20 == 0:
                logging.info('iteration %d : loss : %f, lr : %f' % (iter_num, loss.item(), lr_))

        avg_epoch_loss = epoch_loss / len(trainloader)
        logging.info('Epoch %d : average loss : %f' % (epoch_num, avg_epoch_loss))

        # 保存检查点
        save_interval = 50
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            # 如果模型是DataParallel包装的，保存其module
            torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        # 定期测试
        if (epoch_num + 1) % args.test_interval == 0:
            logging.info(f"Performing comprehensive test for {args.dataset} at epoch {epoch_num + 1}")
            metrics_result = inference_during_training(args, model, epoch_num + 1, snapshot_path, history_metrics)

            if metrics_result is not None:
                history_metrics.append(metrics_result)
                performance = metrics_result['dice']

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
                    # 如果模型是DataParallel包装的，保存其module
                    torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), save_mode_path)
                    logging.info("save best model to {} with performance {}".format(save_mode_path, best_performance))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            # 如果模型是DataParallel包装的，保存其module
            torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    # 最终测试
    logging.info(f"Performing final comprehensive test for {args.dataset}")
    final_metrics_result = inference_during_training(args, model, max_epoch, snapshot_path, history_metrics)

    if final_metrics_result is not None:
        # 如果最后一个周期没有被测试过，则添加结果
        if not history_metrics or history_metrics[-1]['epoch'] != max_epoch:
            history_metrics.append(final_metrics_result)

    # 绘制历史趋势图 (Plots 5, 6, 7)
    plot_metrics(history_metrics, None, args.max_epochs, snapshot_path, is_final=True)
    logging.info(f"Training history plots saved to {snapshot_path}")

    return f"Training and Testing Finished for {args.dataset}!"


if __name__ == "__main__":
    # 添加CUDA检查
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")

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

    # 定义所有息肉数据集名称
    # ====================================================================
    # 修改点：只保留 ETIS-LaribPolypDB
    POLYP_DATASETS = [
        'ETIS-LaribPolypDB'
    ]
    # ====================================================================

    # ====================================================================
    # 路径修正点：现在直接从项目根目录进入 data/data
    BASE_DATA_DIR_RELATIVE = os.path.join('data', 'data')
    base_data_root = os.path.abspath(BASE_DATA_DIR_RELATIVE) # 这将是 Synapse_dataset 的 base_dir
    # ====================================================================

    # --- 数据集配置字典 ---
    # 包含所有单个数据集的配置。root_path 指向实际数据集文件夹的父目录。
    dataset_configs_individual = {
        'Kvasir-SEG': {
            'dataset_folder_name': 'Kvasir-SEG', # 实际数据集文件夹的名称
            'list_dir': os.path.abspath(os.path.join('./lists', 'lists_Kvasir-SEG')),
            'num_classes': 2,
        },
        'CVC-ClinicDB': {
            'dataset_folder_name': 'CVC-ClinicDB',
            'list_dir': os.path.abspath(os.path.join('./lists', 'lists_CVC-ClinicDB')),
            'num_classes': 2,
        },
        'CVC-ColonDB': {
            'dataset_folder_name': 'CVC-ColonDB',
            'list_dir': os.path.abspath(os.path.join('./lists', 'lists_CVC-ColonDB')),
            'num_classes': 2,
        },
        'CVC-300': {
            'dataset_folder_name': 'CVC-300',
            'list_dir': os.path.abspath(os.path.join('./lists', 'lists_CVC-300')),
            'num_classes': 2,
        },
        'ETIS-LaribPolypDB': {
            'dataset_folder_name': 'ETIS-LaribPolypDB',
            'list_dir': os.path.abspath(os.path.join('./lists', 'lists_ETIS-LaribPolypDB')),
            'num_classes': 2,
        },
    }

    all_datasets_final_results = {}

    # 存储原始的 base_lr，以便在每次循环开始时重置
    original_base_lr = args.base_lr

    for current_dataset_name in POLYP_DATASETS:
        print(f"\n=====================================================================")
        print(f"--- Starting Training and Testing for Dataset: {current_dataset_name} ---")
        print(f"=====================================================================\n")

        # 更新 args 以适应当前数据集
        args.dataset = current_dataset_name
        current_config = dataset_configs_individual[current_dataset_name]

        # args.root_path 应该是包含所有数据集文件夹的父目录 (例如 'data/data')
        args.root_path = base_data_root
        args.list_dir = current_config['list_dir']
        args.num_classes = current_config['num_classes']

        # --- 修改学习率逻辑 ---
        if current_dataset_name == 'CVC-300':
            args.base_lr = 0.001
            print(f"--- INFO: Learning rate for {current_dataset_name} set to {args.base_lr} ---")
        else:
            args.base_lr = original_base_lr # 重置为默认值 0.01
            print(f"--- INFO: Learning rate for {current_dataset_name} set to {args.base_lr} (default) ---")
        # ---------------------

        # 检查特定数据集文件夹是否存在于 base_data_root 中
        actual_dataset_path = os.path.join(args.root_path, current_config['dataset_folder_name'])
        if not os.path.isdir(actual_dataset_path):
            logging.error(f"Dataset folder not found for {current_dataset_name}: {actual_dataset_path}")
            logging.error("Please ensure the data is correctly placed in the 'data/data' directory relative to the script.")
            continue # 跳过此数据集并处理下一个

        # --- 执行单个数据集的 80/20 分割 ---
        print(f"--- Executing 80/20 split for {current_dataset_name} ---")
        create_single_dataset_split(current_config['dataset_folder_name'], # 传递实际文件夹名称作为列表文件中的前缀
                                    actual_dataset_path,                   # 传递特定数据集的完整路径用于 glob 图像
                                    args.list_dir,
                                    split_ratio=0.8)
        print("------------------------------------------------------------\n")

        print(f"--- Training Configuration for {current_dataset_name} ---")
        print(f"Dataset: {args.dataset}")
        print(f"Root Path (Base Dir for Synapse_dataset): {args.root_path}")
        print(f"List Dir: {args.list_dir}")
        print(f"Num Classes: {args.num_classes}")
        print(f"Base Learning Rate: {args.base_lr}") # 打印当前学习率
        print(f"Test Interval: {args.test_interval} epochs")
        print(f"--------------------------------------------------")

        args.is_pretrain = True
        # 生成每个数据集唯一的快照路径
        exp_name = args.dataset.replace(' ', '_')
        args.exp = 'TU_' + exp_name + str(args.img_size)

        dataset_snapshot_base = os.path.join("../model", exp_name, 'TU')
        snapshot_path = dataset_snapshot_base + '_pretrain' if args.is_pretrain else dataset_snapshot_base
        snapshot_path += '_' + args.vit_name
        snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
        snapshot_path = snapshot_path + '_vitpatch' + str(
            args.vit_patches_size) if args.vit_patches_size != 16 else snapshot_path
        snapshot_path = snapshot_path + '_' + str(args.max_iterations)[
                                              0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
        snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
        snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
        # 注意：这里 snapshot_path 的命名会包含当前设置的 args.base_lr
        snapshot_path = snapshot_path + '_lr' + str(args.base_lr).replace('.', '') if args.base_lr != 0.01 else snapshot_path
        snapshot_path = snapshot_path + '_' + str(args.img_size)
        snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path

        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)

        # 为每个数据集重新初始化模型，确保从头开始或加载相同的预训练权重
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

        # 在训练开始前检查模型设备
        print(f"Model device for {current_dataset_name}: {next(net.parameters()).device}")

        # 确保预训练权重路径存在
        if os.path.exists(config_vit.pretrained_path):
            net.load_from(weights=np.load(config_vit.pretrained_path))
            logging.info(f"Loaded pretrained weights for {current_dataset_name} from {config_vit.pretrained_path}")
        else:
            logging.warning(
                f"Pretrained weights not found at {config_vit.pretrained_path} for {current_dataset_name}. Starting training from scratch.")

        # ====================================================================
        # --- 打印参数量和 FLOPs ---
        # ====================================================================
        if HAS_TORCHINFO:
            try:
                # 创建一个假的输入张量 (Batch size = 1, 3 channels, H x W)
                dummy_input = torch.randn(1, 3, args.img_size, args.img_size).cuda()

                # 使用 logging.info 打印模型摘要
                logging.info(f"\n--- Model Summary for {current_dataset_name} (Parameters and FLOPs/MAdds) ---")

                # summary() 函数会直接打印到 stdout，我们通过 logging 记录
                model_summary = summary(net, input_data=dummy_input, verbose=0,
                                        col_names=["input_size", "output_size", "num_params", "mult_adds"],
                                        mode="train")  # 使用 train mode 来计算 MAdds/FLOPs

                # 打印摘要信息
                logging.info(f"Total Parameters: {model_summary.total_params:,}")
                # MAdds (Multiply-Accumulate operations) 通常用于估计 FLOPs
                logging.info(f"Total MAdds (FLOPs estimate): {model_summary.total_mult_adds:,}")
                logging.info("---------------------------------------------------\n")

            except Exception as e:
                logging.error(f"Error calculating model summary for {current_dataset_name}: {e}")
        # ====================================================================

        # 使用修改后的训练器
        print(f"Starting training for {current_dataset_name} with comprehensive evaluation metrics and visualizations...")
        result_for_current_dataset = modified_trainer_synapse(args, net, snapshot_path)
        print(f"Finished training for {current_dataset_name}: {result_for_current_dataset}")
        all_datasets_final_results[current_dataset_name] = result_for_current_dataset

    print("\n=====================================================================")
    print("--- All Datasets Training and Testing Completed ---")
    print("=====================================================================")
    for ds, res in all_datasets_final_results.items():
        print(f"Dataset: {ds}, Final Result: {res}")
    print("=====================================================================")

