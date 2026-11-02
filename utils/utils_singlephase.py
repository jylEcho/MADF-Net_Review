import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
from torch.nn import functional as F
from torchvision import transforms
import  logging
import random
from scipy.spatial.distance import directed_hausdorff

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0
def calculate_metrics(prediction, labels, idx=2):
    # 只考虑label为2的区域
    print("current pred shape is ", prediction.shape)
    for ind in range(prediction.shape[-1]):
        pred = prediction[:, :, ind]
        label = labels[:, :, ind]
        pred_binary = (pred == idx).astype(int)
        label_binary = (label == idx).astype(int)

        # Calculate intersection and union for IoU
        intersection = np.sum(pred_binary * label_binary)
        union = np.sum(pred_binary) + np.sum(label_binary) - intersection

        # Dice Similarity Coefficient
        dsc = (2. * intersection) / (np.sum(pred_binary) + np.sum(label_binary) + 1e-6)

        # Relative Volume Difference
        rvd = (np.sum(pred_binary) - np.sum(label_binary)) / (np.sum(label_binary) + 1e-6)

        # Volume Overlap Error
        voe = 1 - (intersection / (np.sum(pred_binary) + np.sum(label_binary) - intersection + 1e-6))

        # Mean Intersection over Union
        miou = intersection / (union + 1e-6)

        # 95th percentile of the Hausdorff Distance
        # Find all the coordinates of the predicted and label binary images
        coords_pred = np.argwhere(pred_binary)
        coords_label = np.argwhere(label_binary)

        if coords_pred.shape[0] > 0 and coords_label.shape[0] > 0:
            # Calculate all pairwise distances between the coordinates in the two arrays
            hausdorff_distances = [directed_hausdorff(coords_pred, coords_label)[0],
                                   directed_hausdorff(coords_label, coords_pred)[0]]
            hd95 = np.percentile(hausdorff_distances, 95)
            hd95_ = metric.binary.hd95(pred_binary, label_binary)
        else:
            hd95 = np.inf  # Set to infinity if one of the objects is empty

        if dsc >= 0.60 and idx== 2:
            print(f"current id:{idx}, dsc is {dsc:.4f}, voe is {voe:.4f}, rvd is {rvd:.4f}, miou is {miou:.4f}, hd95 is {hd95_}")

# def random_crop(slice, box, case_box, output_size):
#     """在给定的肝脏框内进行随机裁剪，同时考虑整个病例的范围"""
#     min_x, max_x, min_y, max_y = box
#     min_x_case, max_x_case, min_y_case, max_y_case, _, _ = case_box
#
#     # 考虑整个病例的肝脏区域来决定裁剪的中心点
#     x_center = random.randint(max(min_x_case, min_x), min(max_x_case, max_x - output_size[0]))
#     y_center = random.randint(max(min_y_case, min_y), min(max_y_case, max_y - output_size[1]))
#
#     # 计算裁剪的起始和结束点
#     x_start = max(x_center - output_size[0] // 2, 0)
#     y_start = max(y_center - output_size[1] // 2, 0)
#     cropped_slice = slice[x_start:x_start + output_size[0], y_start:y_start + output_size[1]]
#
#     return cropped_slice

# def random_test_single_volume(image, label, box, case_box, net, classes, patch_size=[224, 224], test_save_path=None, case=None, z_spacing=1):
#     image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
#     box, case_box = box.cpu().detach().numpy(), case_box.cpu().detach().numpy()
#     prediction = np.zeros_like(label)
#
#     x_transforms = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize([0.5], [0.5])
#     ])
#
#     for ind in range(image.shape[-1]):  # 遍历每个切片
#         # 如果当前切片没有对应的肝脏框，直接跳过
#         if ind + 1 not in box:
#             continue
#
#         slice = image[:, :, ind]
#         crop_box = box[ind + 1]  # 深度是从1开始的
#         cropped_slice = random_crop(slice, crop_box, case_box, patch_size)
#
#         # 调整切片尺寸
#         x, y = cropped_slice.shape[0], cropped_slice.shape[1]
#         if x != patch_size[0] or y != patch_size[1]:
#             cropped_slice = zoom(cropped_slice, (patch_size[0] / x, patch_size[1] / y), order=3)
#
#         input_tensor = x_transforms(cropped_slice).unsqueeze(0).float().cuda()
#         net.eval()
#         with torch.no_grad():
#             outputs = net(input_tensor)
#             out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
#             out = out.cpu().detach().numpy()
#
#         # 将预测结果调整回原始大小
#         if x != patch_size[0] or y != patch_size[1]:
#             pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
#         else:
#             pred = out
#         prediction[:, :, ind] = pred
#
#     metric_list = []
#     for i in range(1, classes):
#         # metric_list.append(calculate_metric_percase(prediction == i, label == i))  只计算肿瘤的
#         if i ==2:
#             metric_list.append(calculate_metric_percase(prediction == i, label == i))
#             # print('case_metric:',metric_list)
#             logging.info('case_metric: %s'%metric_list)
#     if test_save_path is not None:
#         img_itk = sitk.GetImageFromArray(image.astype(np.float32))
#         prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
#         lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
#         img_itk.SetSpacing((1, 1, z_spacing))
#         prd_itk.SetSpacing((1, 1, z_spacing))
#         lab_itk.SetSpacing((1, 1, z_spacing))
#         sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
#         sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
#         sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
#
#     return metric_list
#

def random_crop(slice, box, case_box, output_size):
    """在给定的肝脏框内进行随机裁剪，并返回裁剪区域的坐标"""
    min_x, max_x, min_y, max_y = box
    min_x_case, max_x_case, min_y_case, max_y_case, _, _ = case_box

    # 计算裁剪中心点的范围
    x_center_min = max(min_x_case + output_size[0] // 2, min_x + output_size[0] // 2)
    x_center_max = min(max_x_case - output_size[0] // 2, max_x - output_size[0] // 2)
    y_center_min = max(min_y_case + output_size[1] // 2, min_y + output_size[1] // 2)
    y_center_max = min(max_y_case - output_size[1] // 2, max_y - output_size[1] // 2)

    # 随机选择裁剪中心点
    x_center = random.randint(x_center_min, max(x_center_min, x_center_max))
    y_center = random.randint(y_center_min, max(y_center_min, y_center_max))

    # 计算裁剪的起始点
    x_start = max(x_center - output_size[0] // 2, 0)
    y_start = max(y_center - output_size[1] // 2, 0)

    # 执行裁剪并返回裁剪区域及其坐标
    cropped_slice = slice[x_start:x_start + output_size[0], y_start:y_start + output_size[1]]
    return cropped_slice, (x_start, y_start, x_start + output_size[0], y_start + output_size[1])


def max_available_crop(slice, box, case_box, output_size):
    """在给定的肝脏框内进行最大可用区域裁剪"""
    min_x, max_x, min_y, max_y = box
    min_x_case, max_x_case, min_y_case, max_y_case, _, _ = case_box

    # 计算可用的裁剪区域范围
    x_min_limit = max(min_x_case, min_x)
    x_max_limit = min(max_x_case, max_x) - output_size[0]
    y_min_limit = max(min_y_case, min_y)
    y_max_limit = min(max_y_case, max_y) - output_size[1]

    # 确保裁剪区域的起始点不越界
    if x_max_limit < x_min_limit:
        x_start = x_min_limit
    else:
        x_start = (x_min_limit + x_max_limit) // 2  # 选择中间位置

    if y_max_limit < y_min_limit:
        y_start = y_min_limit
    else:
        y_start = (y_min_limit + y_max_limit) // 2  # 选择中间位置

    # 执行裁剪
    cropped_slice = slice[x_start:x_start + output_size[0], y_start:y_start + output_size[1]]

    return cropped_slice,(x_start, y_start, x_start + output_size[0], y_start + output_size[1])

def max_crop(slice, box, case_box, output_size):
    """在给定的肝脏框内进行最大可用区域裁剪"""
    _, _, _, _ = box  # 如果不使用box，这行可以省略
    min_x_case, max_x_case, min_y_case, max_y_case, _, _ = case_box

    # 考虑输出尺寸，避免裁剪区域超出图像边界
    x_start = max(min_x_case, 0)
    y_start = max(min_y_case, 0)
    x_end = max_x_case
    y_end = max_y_case

    # 执行裁剪
    cropped_slice = slice[x_start:x_end, y_start:y_end]

    return cropped_slice, (x_start, y_start, x_end, y_end)


def random_crop_inference(slice_v, slice_a, slice_d, net, box, case_box, patch_size, transforms, num_random_crops):
    slice_predictions = np.zeros((slice_v.shape[0], slice_v.shape[1], num_random_crops))

    for i in range(num_random_crops):
        cropped_slice_v, crop_coords = random_crop(slice_v, box, case_box, patch_size)
        x_start, y_start, x_end, y_end = crop_coords
        cropped_slice_a = slice_a[x_start:x_end, y_start:y_end]
        cropped_slice_d = slice_d[x_start:x_end, y_start:y_end]
        # 调整切片尺寸
        original_shape = cropped_slice_v.shape
        if original_shape[0] != patch_size[0] or original_shape[1] != patch_size[1]:
            cropped_slice_v = zoom(cropped_slice_v, (patch_size[0] / original_shape[0], patch_size[1] / original_shape[1]),
                                 order=3)
            cropped_slice_a = zoom(cropped_slice_a,
                                   (patch_size[0] / original_shape[0], patch_size[1] / original_shape[1]),
                                   order=3)
            cropped_slice_d = zoom(cropped_slice_d,
                                   (patch_size[0] / original_shape[0], patch_size[1] / original_shape[1]),
                                   order=3)
        input_tensor_v = transforms(cropped_slice_v).unsqueeze(0).float().cuda()
        input_tensor_a = transforms(cropped_slice_a).unsqueeze(0).float().cuda()
        input_tensor_d = transforms(cropped_slice_d).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            outputs = net(input_tensor_v, input_tensor_a, input_tensor_d) # 3-phase
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()

        # 将预测结果调整回原始大小
        if original_shape[0] != patch_size[0] or original_shape[1] != patch_size[1]:
            pred = zoom(out, (original_shape[0] / patch_size[0], original_shape[1] / patch_size[1]), order=0)
        else:
            pred = out

        # 将预测结果放置在原始尺寸的切片上
        slice_predictions[x_start:x_end, y_start:y_end, i] = pred

    # 合并预测结果
    return np.mean(slice_predictions, axis=2)


def standard_crop_inference(slice_v, slice_a, slice_d, net, box, case_box, patch_size, transforms, num_random_crops, args):
    slice_predictions = np.zeros((slice_v.shape[0], slice_v.shape[1]))


    # cropped_slice, crop_coords = max_available_crop(slice, box, case_box, patch_size)
    cropped_slice_v, crop_coords = max_crop(slice_v, box, case_box, patch_size)
    cropped_slice_a, _ = max_crop(slice_a, box, case_box, patch_size)
    cropped_slice_d, _ = max_crop(slice_d, box, case_box, patch_size)
    x_start, y_start, x_end, y_end = crop_coords

    # 调整切片尺寸
    original_shape = cropped_slice_v.shape
    if original_shape[0] != patch_size[0] or original_shape[1] != patch_size[1]:
        cropped_slice_v = zoom(cropped_slice_v, (patch_size[0] / original_shape[0], patch_size[1] / original_shape[1]),
                             order=3)
        cropped_slice_d = zoom(cropped_slice_d, (patch_size[0] / original_shape[0], patch_size[1] / original_shape[1]),
                             order=3)
        cropped_slice_a = zoom(cropped_slice_a, (patch_size[0] / original_shape[0], patch_size[1] / original_shape[1]),
                             order=3)

    input_tensor_v = transforms(cropped_slice_v).unsqueeze(0).float().cuda()
    input_tensor_a = transforms(cropped_slice_a).unsqueeze(0).float().cuda()
    input_tensor_d = transforms(cropped_slice_v).unsqueeze(0).float().cuda()
    net.eval()
    with torch.no_grad():
        if args.model_name == 'Pareseg':
            outputs = net(input_tensor_v, input_tensor_a) # 2-phase
        elif args.model_name == 'Acftransunet':
            outputs = net(input_tensor_v)
        else:
            outputs = net(input_tensor_a, input_tensor_v, input_tensor_d)
        out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()

    # 将预测结果调整回原始大小
    if original_shape[0] != patch_size[0] or original_shape[1] != patch_size[1]:
        pred = zoom(out, (original_shape[0] / patch_size[0], original_shape[1] / patch_size[1]), order=0)
    else:
        pred = out

    # 将预测结果放置在原始尺寸的切片上
    slice_predictions[x_start:x_end, y_start:y_end] = pred

    # 合并预测结果
    return slice_predictions

def resize_inference(slice, net, box, case_box, patch_size, transforms, num_random_crops):
    slice_predictions = np.zeros((slice.shape[0], slice.shape[1]))
    original_shape = slice_predictions.shape
    if original_shape[0] != patch_size[0] or original_shape[1] != patch_size[1]:
        cropped_slice = zoom(slice, (patch_size[0] / original_shape[0], patch_size[1] / original_shape[1]),
                             order=3)

    input_tensor = transforms(cropped_slice).unsqueeze(0).float().cuda()
    net.eval()
    with torch.no_grad():
        outputs = net(input_tensor)
        out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()

    # 将预测结果调整回原始大小
    if original_shape[0] != patch_size[0] or original_shape[1] != patch_size[1]:
        pred = zoom(out, (original_shape[0] / patch_size[0], original_shape[1] / patch_size[1]), order=0)
    else:
        pred = out

    # 将预测结果放置在原始尺寸的切片上
    slice_predictions = pred.copy()

    # 合并预测结果
    return slice_predictions

# todo: multiprocessing 推理
def merge_test_single_volume(image_v,image_a,image_d, label, box, case_box, net, classes, crop_inference=random_crop_inference ,patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1,  num_random_crops=1,args=None):
    image_v, image_a, image_d, label = image_v.squeeze(0).cpu().detach().numpy(), image_a.squeeze(0).cpu().detach().numpy(), image_d.squeeze(0).cpu().detach().numpy(),\
                   label.squeeze(0).cpu().detach().numpy()
    box, case_box = box.copy(), case_box.cpu().detach().numpy()
    prediction = np.zeros_like(label)

    x_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])   # maybe not necessary if training without normalize
    ])

    for ind in range(image_v.shape[-1]):  # 遍历每个切片
        if ind + 1 not in box:
            continue

        slice_v = image_v[:, :, ind]
        slice_a = image_a[:, :, ind]
        try:
            slice_d = image_d[:, :, ind]
        except Exception as e:
            print(f"发生错误: {e}")
        crop_box = box[ind + 1]
        slice_prediction = crop_inference(slice_v, slice_a, slice_d, net, crop_box[0], case_box[0], patch_size, x_transforms, num_random_crops=num_random_crops,args=args)
        prediction[:, :, ind] = slice_prediction

    # ... [评价指标和保存图像的代码，保持不变] ...
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
        calculate_metrics(prediction, label,i)
        if i ==2:
            # metric_list.append(calculate_metric_percase(prediction == i, label == i))
            # print('case_metric:',metric_list)
            logging.info('lesion_case_metric: %s' % (str(metric_list[1]),)) #只log肿瘤的
    if test_save_path is not None:
        image_transposed = np.transpose(image_v, (2, 1, 0))  # 将 (X, Y, Z) 转换为 (Z, Y, X)
        label_transposed = np.transpose(label, (2, 1, 0))
        prediction_transposed = np.transpose(prediction, (2, 1, 0))

        # 使用转置后的数据创建 SimpleITK 图像
        img_itk = sitk.GetImageFromArray(image_transposed.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label_transposed.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction_transposed.astype(np.float32))

        # 设置空间分辨率并写入文件
        img_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(img_itk, test_save_path + '/' + case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/' + case + "_gt.nii.gz")
        sitk.WriteImage(prd_itk, test_save_path + '/' + case + "_pred.nii.gz")

    return metric_list
