# File: F:\danzi10\4.5w\daima\TransUNet-main\main.py (or main_rensnet50_vit_b.py)

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

# 设置日志级别 (提前设置，以便 set_seed 可以使用 logging)
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')


# ----------------------------------------------------------------------
# --- 随机种子固定函数 ---
# ----------------------------------------------------------------------

def set_seed(seed):
    """固定所有可能的随机源，确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # 强制 CUDNN 确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"Global seed set to {seed}. CUDNN set to deterministic (benchmark=False).")


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
            return 100.0  # 如果缺少边界点，返回最大距离

        # 计算双向Hausdorff距离
        hd1 = directed_hausdorff(pred_points, true_points)[0]
        hd2 = directed_hausdorff(true_points, pred_points)[0]

        return max(hd1, hd2)
    except:
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

        # 计算所有距离
        distances1 = []
        for pred_point in pred_points:
            min_dist = np.min(np.sqrt(np.sum((true_points - pred_point) ** 2, axis=1)))
            distances1.append(min_dist)

        distances2 = []
        for true_point in true_points:
            min_dist = np.min(np.sqrt(np.sum((pred_points - true_point) ** 2, axis=1)))
            distances2.append(min_dist)

        all_distances = distances1 + distances2
        if len(all_distances) == 0:
            return 100.0

        # 返回95%分位数
        return np.percentile(all_distances, 95)
    except:
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
        distances1 = []
        for pred_point in pred_points:
            min_dist = np.min(np.sqrt(np.sum((true_points - pred_point) ** 2, axis=1)))
            distances1.append(min_dist)

        # 计算从真值到预测的距离
        distances2 = []
        for true_point in true_points:
            min_dist = np.min(np.sqrt(np.sum((pred_points - true_point) ** 2, axis=1)))
            distances2.append(min_dist)

        # 计算平均对称表面距离
        all_distances = distances1 + distances2
        return np.mean(all_distances) if len(all_distances) > 0 else 100.0
    except:
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
            mcc = 0
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
            rve = 1.0 if pred_volume > 0 else 0.0

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
    overlay[..., 0] = pred_mask * 255  # Red for Prediction
    overlay[..., 1] = gt_mask * 255  # Green for Ground Truth

    # 将原图转换为灰度图以进行叠加
    if image.ndim == 3:
        if image.shape[2] == 3:
            # Simple grayscale conversion for visualization
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
    if overlay.shape[:2] != gray_image.shape:
        # Recreate overlay if dimensions mismatch (shouldn't happen if image_resized is used)
        overlay = np.zeros((*gray_image.shape, 3), dtype=np.uint8)
        overlay[..., 0] = pred_mask * 255
        overlay[..., 1] = gt_mask * 255

    axes[3].imshow(overlay, alpha=0.5)
    axes[3].set_title("4. Overlay (R: Pred, G: GT)")
    axes[3].axis('off')

    plt.suptitle(f"Segmentation Results for Case: {case_name}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Ensure case name is file-system safe
    safe_case_name = case_name.replace('/', '_')
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
        dice_values = [m[0]['dice'] for m in current_metrics_list if m and len(m) > 0]
        hd95_values = [m[0]['hd95'] for m in current_metrics_list if m and len(m) > 0]
        precision_values = [m[0]['precision'] for m in current_metrics_list if m and len(m) > 0]
        recall_values = [m[0]['recall'] for m in current_metrics_list if m and len(m) > 0]

        if not dice_values:
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 8. Dice Score Distribution (Histogram)
        axes[0].hist(dice_values, bins=15, range=(0, 1), color='skyblue', edgecolor='black')
        axes[0].set_title(f'8. Dice Score Distribution (E{epoch})')
        axes[0].set_xlabel('Dice Score')
        axes[0].set_ylabel('Frequency')
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
                case = "Unknown_Case"

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
        print(f"Error in custom_test_single_volume for case {case}: {str(e)}")
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
# logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s') # 已经提前设置

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
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
# --- MODIFICATION: 更改默认随机种子 ---
parser.add_argument('--seed', type=int,
                    default=42069, help='random seed')
# --------------------------------------
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
# --- MODIFICATION 1: Change default ViT backbone to R50-ViT-B_16 ---
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
# ----------------------------------------------------------------
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
    # 确保每个 worker 使用不同的种子，但基于固定的 args.seed
    set_seed(args.seed + worker_id)


# ----------------------------------------------------------------------
# --- 数据集分割函数 (已修改文件扩展名和 ETIS 命名) ---
# ----------------------------------------------------------------------

def create_combined_split(data_configs, base_data_dir, list_dir, split_ratio=0.9):
    """
    创建特定的组合训练和测试列表：
    训练: 90% Kvasir-SEG + 90% CVC-ClinicDB
    测试: 10% Kvasir-SEG + 10% CVC-ClinicDB + 100% CVC-ColonDB + 100% CVC-300 + 100% ETIS-LaribPolypDB
    """
    import glob
    import random

    # 使用固定的种子进行分割，确保分割结果稳定
    random.seed(args.seed)

    train_cases = []
    test_cases = []

    # Datasets for 90/10 split
    split_datasets = ['Kvasir-SEG', 'CVC-ClinicDB']
    # Datasets for 100% test (Assuming ETIS-LaribPolypDB is the correct folder name without space)
    full_test_datasets = ['CVC-ColonDB', 'CVC-300', 'ETIS-LaribPolypDB']

    all_dataset_names = split_datasets + full_test_datasets

    for ds_name in all_dataset_names:
        # 路径结构: base_data_dir / ds_name / images / *
        ds_root = os.path.join(base_data_dir, ds_name)
        images_dir = os.path.join(ds_root, 'images')

        if not os.path.exists(images_dir):
            logging.warning(f"Skipping {ds_name}: Image directory not found at {images_dir}")
            continue

        # 获取所有图像名称 (不带扩展名)
        # 假设图像是 .png, .jpg, .jpeg, .tif, 或 .tiff
        image_files_png = glob.glob(os.path.join(images_dir, '*.png'))
        image_files_jpg = glob.glob(os.path.join(images_dir, '*.jpg'))
        image_files_jpeg = glob.glob(os.path.join(images_dir, '*.jpeg'))
        image_files_tif = glob.glob(os.path.join(images_dir, '*.tif'))  # FIX: Added TIF support
        image_files_tiff = glob.glob(os.path.join(images_dir, '*.tiff'))  # FIX: Added TIFF support

        image_files = image_files_png + image_files_jpg + image_files_jpeg + image_files_tif + image_files_tiff

        image_names = [os.path.splitext(os.path.basename(f))[0] for f in image_files]

        # 存储案例名称，格式为 "DatasetName/ImageName"
        case_names = [f"{ds_name}/{name}" for name in image_names]

        if not case_names:
            logging.warning(f"No cases found in {ds_name}.")
            continue

        random.shuffle(case_names)

        if ds_name in split_datasets:
            # 90% train, 10% test
            split_idx = int(len(case_names) * split_ratio)
            train_cases.extend(case_names[:split_idx])
            test_cases.extend(case_names[split_idx:])
            logging.info(
                f"{ds_name}: {len(case_names[:split_idx])} training cases, {len(case_names[split_idx:])} testing cases (10% split).")
        elif ds_name in full_test_datasets:
            # 100% test
            test_cases.extend(case_names)
            logging.info(f"{ds_name}: {len(case_names)} testing cases (100% test).")

    os.makedirs(list_dir, exist_ok=True)

    # 写入训练列表
    train_list_path = os.path.join(list_dir, 'train.txt')
    with open(train_list_path, 'w') as f:
        for name in train_cases:
            f.write(f"{name}\n")

    # 写入测试列表
    test_list_path = os.path.join(list_dir, 'test_vol.txt')
    with open(test_list_path, 'w') as f:
        for name in test_cases:
            f.write(f"{name}\n")

    logging.info(f"Combined split completed:")
    logging.info(f"Total Training Cases: {len(train_cases)}")
    logging.info(f"Total Testing Cases: {len(test_cases)}")
    return train_cases, test_cases


# ----------------------------------------------------------------------
# --- 训练和推理函数 (已修改 num_workers) ---
# ----------------------------------------------------------------------

def inference_during_training(args, net, epoch, snapshot_path, history_metrics):
    """
    在训练过程中进行推理测试，包含全面的评价指标和可视化
    """
    logging.info(f"Starting comprehensive inference at epoch {epoch}")

    # 数据集配置 (必须包含组合数据集配置)
    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': args.root_path,  # 假设 Synapse 路径在 args.root_path 附近
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
            'z_spacing': 1,
        },
        # 组合数据集配置
        'Polyp_Combined_Cross_Validation': {
            'Dataset': Synapse_dataset,
            # volume_path 必须指向包含所有子数据集文件夹的父目录 (即 data/data)
            'volume_path': args.root_path,
            'list_dir': args.list_dir,  # 使用主程序设置的组合列表目录
            'num_classes': 2,
            'z_spacing': 1,
        },
        # 保持其他数据集配置，以防万一 (注意：这里也需要同步更新 ETIS 的命名)
        'Kvasir-SEG': {'Dataset': Synapse_dataset, 'volume_path': args.root_path, 'list_dir': args.list_dir,
                       'num_classes': 2, 'z_spacing': 1, },
        'CVC-ClinicDB': {'Dataset': Synapse_dataset, 'volume_path': args.root_path, 'list_dir': args.list_dir,
                         'num_classes': 2, 'z_spacing': 1, },
        'CVC-ColonDB': {'Dataset': Synapse_dataset, 'volume_path': args.root_path, 'list_dir': args.list_dir,
                        'num_classes': 2, 'z_spacing': 1, },
        'CVC-300': {'Dataset': Synapse_dataset, 'volume_path': args.root_path, 'list_dir': args.list_dir,
                    'num_classes': 2, 'z_spacing': 1, },
        'ETIS-LaribPolypDB': {'Dataset': Synapse_dataset, 'volume_path': args.root_path, 'list_dir': args.list_dir,
                              'num_classes': 2, 'z_spacing': 1, },  # FIX: Updated name
    }

    if args.dataset not in dataset_config:
        logging.error(f"Dataset {args.dataset} not configured for testing.")
        return None

    db_config = dataset_config[args.dataset]

    # 设置测试日志目录
    test_log_folder = os.path.join(snapshot_path, 'test_logs', f'epoch_{epoch}')
    os.makedirs(test_log_folder, exist_ok=True)
    os.makedirs(os.path.join(test_log_folder, 'visualizations'), exist_ok=True)

    # 设置测试专用的日志记录器
    test_logger = logging.getLogger(f'test_epoch_{epoch}')
    test_logger.setLevel(logging.INFO)

    # 清除之前的处理器
    for handler in test_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            test_logger.removeHandler(handler)

    # 添加文件处理器
    test_handler = logging.FileHandler(os.path.join(test_log_folder, f'test_epoch_{epoch}.log'))
    test_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    test_handler.setFormatter(formatter)
    test_logger.addHandler(test_handler)

    test_logger.info(f"Testing with comprehensive metrics at epoch {epoch}")

    try:
        # 创建测试数据集
        db_test = db_config['Dataset'](base_dir=db_config['volume_path'],
                                       split="test_vol",
                                       list_dir=db_config['list_dir'])

        if len(db_test) == 0:
            test_logger.warning("No test data found!")
            return None

        # num_workers=0 for testing to avoid potential issues with shared memory/CUDA context
        testloader = torch.utils.data.DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)

        # 设置网络为评估模式
        net.eval()
        all_metrics = []
        visualized_count = 0

        with torch.no_grad():
            for i_batch, sampled_batch in tqdm(enumerate(testloader), desc=f"Testing Epoch {epoch}",
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
                    test_logger.error(f"Error processing batch {i_batch} ({case_name}): {str(e)}")
                    continue

        if len(all_metrics) > 0:
            # 计算每个类别的平均指标
            metric_names = ['dice', 'iou', 'precision', 'recall', 'specificity',
                            'f1', 'accuracy', 'balanced_acc', 'mcc', 'hd', 'hd95', 'assd', 'rve']

            avg_metrics_epoch = {}

            # 对于息肉分割 (num_classes=2)，我们只关心 class_idx=0 (即前景)
            class_idx = 0
            test_logger.info(f"\n=== Class {class_idx + 1} (Foreground) Metrics ===")

            for metric_name in metric_names:
                values = [metrics[class_idx][metric_name] for metrics in all_metrics
                          if class_idx < len(metrics)]
                if values:
                    # 过滤掉极端的HD/ASSD值 (例如，100.0)
                    if 'hd' in metric_name or 'assd' in metric_name:
                        valid_values = [v for v in values if v < 99.0]
                        if not valid_values:
                            mean_val = 100.0
                            std_val = 0.0
                        else:
                            mean_val = np.mean(valid_values)
                            std_val = np.std(values)  # Use all values for std calculation
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

            test_logger.info(f"\n=== Overall Performance at Epoch {epoch} ===")
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

                    for class_idx, class_metrics in enumerate(metrics):
                        row = {'sample_id': sample_idx, 'case_name': case_name_log, 'class_id': class_idx + 1}
                        row.update(class_metrics)
                        writer.writerow(row)

            test_logger.info(f"Detailed metrics saved to: {csv_path}")

        else:
            overall_dice, overall_hd95 = 0.0, 100.0
            overall_iou = 0.0
            test_logger.warning("No valid metrics computed")

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
        test_logger.error(f"Error during testing: {str(e)}")
        import traceback
        test_logger.error(traceback.format_exc())
        net.train()
        return None


def modified_trainer_synapse(args, model, snapshot_path):
    """
    修改后的训练器。
    功能：仅在训练结束时进行一次全面的测试和评估。
    """
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    import torch.optim as optim
    from torch.nn.modules.loss import CrossEntropyLoss
    from torch.utils.data import DataLoader

    # 设置日志
    log_file_path = os.path.join(snapshot_path, "log.txt")

    # 确保日志配置正确
    root_logger = logging.getLogger()
    # 移除可能存在的重复文件处理器
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == log_file_path:
            root_logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)

    # 确保控制台输出也被添加
    if not any(isinstance(handler, logging.StreamHandler) for handler in root_logger.handlers):
        root_logger.addHandler(logging.StreamHandler(sys.stdout))

    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    # --- FIX: Increase num_workers to 4 for parallel data loading ---
    NUM_WORKERS = 4
    # ---------------------------------------------------------------

    # 创建训练数据集 (使用组合后的 train.txt 列表和 root_path)
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=RandomGenerator(output_size=[args.img_size, args.img_size]))

    logging.info("The length of train set is: {}".format(len(db_train)))

    # 使用 NUM_WORKERS 启用多进程加载，并使用 worker_init_fn 确保每个 worker 的数据增强是确定的
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True,
                             worker_init_fn=worker_init_fn if NUM_WORKERS > 0 else None)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    model.train()
    ce_loss = CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)

    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    history_metrics = []

    iterator = tqdm(range(max_epoch), ncols=70)
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

        # 保存检查点 (保持原有的定期保存逻辑)
        save_interval = 50
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            # 保存最终周期的模型
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save final epoch model to {}".format(save_mode_path))
            iterator.close()
            break

    # ----------------------------------------------------------------
    # 最终测试 (只在训练结束后执行一次)
    logging.info("Performing final comprehensive test")
    final_metrics_result = inference_during_training(args, model, max_epoch, snapshot_path, history_metrics)

    if final_metrics_result is not None:
        # 记录最终指标
        history_metrics.append(final_metrics_result)

        # 由于只测试了一次，将最终模型保存为 'best_model.pth'
        save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
        torch.save(model.state_dict(), save_mode_path)
        logging.info(f"save final model as best_model.pth (DICE: {final_metrics_result.get('dice', 0.0):.6f})")

    # 绘制历史趋势图 (如果 history_metrics 不为空，则绘制)
    # 此时 history_metrics 应该只包含一个点，即最终测试结果
    plot_metrics(history_metrics, None, args.max_epochs, snapshot_path, is_final=True)
    logging.info(f"Training history plots saved to {snapshot_path}")

    return "Training and Testing Finished!"


if __name__ == "__main__":
    # --- MODIFICATION: 定义 R50-ViT-B_16 预训练权重路径 ---
    # 假设该路径是用户系统中的绝对路径
    R50_PRETRAINED_PATH = r"F:\danzi10\4.5w\daima\TransUNet-main\model\vit_checkpoint\imagenet21k\R50+ViT-B_16.npz"

    # 解析参数

    # ----------------------------------------------------------------
    # --- 强制固定随机种子和确定性设置 ---
    # ----------------------------------------------------------------
    # 使用 args.seed (默认值已更改为 42069)
    set_seed(args.seed)
    # ----------------------------------------------------------------

    # 添加CUDA检查
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")

    # 强制使用组合数据集名称进行训练
    COMBINED_DATASET_NAME = 'Polyp_Combined_Cross_Validation'
    if args.dataset != COMBINED_DATASET_NAME:
        print(f"--- Overriding dataset argument to use combined split: {COMBINED_DATASET_NAME} ---")
        args.dataset = COMBINED_DATASET_NAME

    dataset_name = args.dataset

    # ====================================================================
    # --- 路径修正点：使用 '..' 导航到项目根目录下的 data/data ---
    # 假设脚本在 `TransUNet-main/networks/` 中运行，数据在 `TransUNet-main/data/data/`
    BASE_DATA_DIR_RELATIVE = os.path.join('..', 'data', 'data')
    # ====================================================================

    # --- 数据集配置字典 ---
    # 包含所有单个数据集的配置，用于分割逻辑
    dataset_config = {
        'Synapse': {
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
        # FIX: Renamed the key and path for ETIS
        'ETIS-LaribPolypDB': {
            'root_path': os.path.join(BASE_DATA_DIR_RELATIVE, 'ETIS-LaribPolypDB'),
            'list_dir': './lists/lists_ETIS-LaribPolypDB',
            'num_classes': 2,
        },
    }

    # 添加组合数据集的配置
    # args.root_path 必须是所有子数据集的父目录: F:\danzi10\4.5w\daima\TransUNet-main\data\data
    # 确保 args.root_path 是绝对路径，并且基于脚本的实际位置正确计算
    # 脚本路径: .../networks/main.py
    # 数据路径: .../data/data
    project_root = os.path.dirname(base_dir)  # base_dir = .../networks, project_root = .../TransUNet-main
    args.root_path = os.path.abspath(os.path.join(project_root, 'data', 'data'))

    # 列表路径通常相对于脚本运行目录，但为了安全，也使用绝对路径
    args.list_dir = os.path.abspath(os.path.join(project_root, 'lists', 'lists_Polyp_Combined'))

    dataset_config[COMBINED_DATASET_NAME] = {
        'root_path': args.root_path,
        'list_dir': args.list_dir,
        'num_classes': 2,
    }

    if dataset_name not in dataset_config:
        print(f"Error: Dataset '{dataset_name}' not configured.")
        sys.exit(1)

    # 动态加载配置
    config = dataset_config[dataset_name]
    args.num_classes = config['num_classes']

    # 检查数据根目录是否存在
    if not os.path.isdir(args.root_path):
        logging.error(f"Data root path not found: {args.root_path}")
        logging.error("Please ensure the data is correctly placed in the 'data/data' directory relative to the script.")
        # 如果路径仍然错误，打印出项目根目录供用户参考
        logging.error(f"Calculated Project Root: {project_root}")
        sys.exit(1)

    # --- 执行组合分割 ---
    if dataset_name == COMBINED_DATASET_NAME:
        print("\n--- Executing required combined 90/10 cross-dataset split ---")
        # 传入所有单个数据集的配置，以及共同的父目录
        create_combined_split(dataset_config, args.root_path, args.list_dir, split_ratio=0.9)
        print("------------------------------------------------------------\n")

    print(f"--- Training Configuration ---")
    print(f"Dataset: {dataset_name} (Combined)")
    print(f"Root Path (Base Dir): {args.root_path}")
    print(f"List Dir: {args.list_dir}")
    print(f"Num Classes: {args.num_classes}")
    print(f"ViT Model: {args.vit_name}")
    print(f"Test Interval: {args.test_interval} epochs (NOTE: Only final test is performed in this modified version)")
    print(f"-----------------------------")

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
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 42069 else snapshot_path

    # 确保快照路径相对于项目根目录是正确的
    # 如果脚本在 networks 中，../model 指向 TransUNet-main/model
    snapshot_path = os.path.join(project_root, snapshot_path)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    # 尝试获取配置
    config_vit = CONFIGS_ViT_seg[args.vit_name]

    # --- MODIFICATION 2: Override pretrained path for R50-ViT-B_16 ---
    if args.vit_name == 'R50-ViT-B_16':
        config_vit.pretrained_path = R50_PRETRAINED_PATH
        logging.warning(f"Overriding pretrained path for {args.vit_name} to: {config_vit.pretrained_path}")
    # ----------------------------------------------------------

    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        # R50-ViT-B_16 使用 ResNet-50 作为特征提取器，需要设置网格大小
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size),
                                   int(args.img_size / args.vit_patches_size))

    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    # 在训练开始前检查模型设备
    print(f"Model device: {next(net.parameters()).device}")

    # 确保预训练权重路径存在
    if os.path.exists(config_vit.pretrained_path):
        logging.info(f"Loading weights from: {config_vit.pretrained_path}")
        net.load_from(weights=np.load(config_vit.pretrained_path))
    else:
        logging.warning(
            f"Pretrained weights not found at {config_vit.pretrained_path}. Starting training from scratch.")

    # ====================================================================
    # --- 打印参数量和 FLOPs ---
    # ====================================================================
    if HAS_TORCHINFO:
        try:
            # 创建一个假的输入张量 (Batch size = 1, 3 channels, H x W)
            dummy_input = torch.randn(1, 3, args.img_size, args.img_size).cuda()

            # 使用 logging.info 打印模型摘要
            logging.info("\n--- Model Summary (Parameters and FLOPs/MAdds) ---")

            # summary() 函数会直接打印到 stdout, 我们通过 logging 记录
            model_summary = summary(net, input_data=dummy_input, verbose=0,
                                    col_names=["input_size", "output_size", "num_params", "mult_adds"],
                                    mode="train")  # 使用 train mode 来计算 MAdds/FLOPs

            # 打印摘要信息
            logging.info(f"Total Parameters: {model_summary.total_params:,}")
            # MAdds (Multiply-Accumulate operations) 通常用于估计 FLOPs
            logging.info(f"Total MAdds (FLOPs estimate): {model_summary.total_mult_adds:,}")
            logging.info("---------------------------------------------------\n")

        except Exception as e:
            logging.error(f"Error calculating model summary: {e}")
    # ====================================================================

    # 使用修改后的训练器
    print("Starting training with final evaluation only...")
    result = modified_trainer_synapse(args, net, snapshot_path)
    print(result)
