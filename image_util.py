# ////////////////////////////////////////////////////////////////////////////
# // This file is part of CRefNet. For more information
# // see <https://github.com/JundanLuo/CRefNet>.
# // If you use this code, please cite our paper as
# // listed on the above website.
# //
# // Licensed under the Apache License, Version 2.0 (the “License”);
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an “AS IS” BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.
# ////////////////////////////////////////////////////////////////////////////


import io
import os
import math
import warnings
from typing import Tuple, List

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import cv2
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def rgb_to_srgb(rgb, gamma=1.0/2.2):
    return rgb.clip(min=0.0) ** gamma


def srgb_to_rgb(srgb, gamma=1.0/2.2):
    return srgb.clip(min=0.0) ** (1.0 / gamma)


def valid_rgI(rgI, v_min=0.0, v_max=1.0):
    assert rgI.ndim == 4 and rgI.size(1) == 3
    assert v_min <= v_max
    ret = torch.ones_like(rgI)
    for i in range(rgI.size(1)):
        c = rgI[:, i:i+1, :, :]
        ret = ret.mul(c >= v_min).mul(c <= v_max)
    s = rgI[:, 0:1, :, :] + rgI[:, 1:2, :, :]
    ret = ret.mul(s >= v_min).mul(s <= v_max)
    return ret


def valid_rgb(rgb, v_min=0.0, v_max=1.0):
    assert rgb.ndim == 4 and rgb.size(1) == 3
    assert v_min <= v_max
    ret = torch.ones_like(rgb)
    for i in range(rgb.size(1)):
        c = rgb[:, i:i+1, :, :]
        ret = ret.mul(c >= v_min).mul(c <= v_max)
    return ret


def rgb_to_chromaticity(rgb):
    """ converts rgb to chromaticity """
    if rgb.ndim == 3:
        sum = torch.sum(rgb, dim=0, keepdim=True).clamp(min=1e-6)
    elif rgb.ndim == 4:
        sum = torch.sum(rgb, dim=1, keepdim=True).clamp(min=1e-6)
    else:
        raise Exception("Only supports image: [C, H, W] or [B, C, H, W]")
    chromat = rgb / sum
    return chromat


def save_srgb_image(image, path, filename):
    # Transform to PILImage
    image_np = np.transpose(image.to(torch.float32).cpu().numpy(), (1, 2, 0)) * 255.0
    image_np = image_np.astype(np.uint8)
    image_pil = Image.fromarray(image_np, mode='RGB')
    # Save Image
    if not os.path.exists(path):
        os.makedirs(path)
    image_pil.save(os.path.join(path,filename))


def tone_mapping(image, mask, rescale=True, trans2srgb=False, clip=True, gamma=1.0/2.2,
                 src_PERCENTILE=0.9,
                 dst_VALUE=0.8,
                 w_RGB=(0.3, 0.59, 0.11)):
    # Adapted from: https://github.com/snavely/pbrs_tonemapper
    assert image.ndim == mask.ndim == 3, "Only supports image: [C, H, W]"
    # MAX_SRGB = 1.077837  # SRGB 1.0 = RGB 1.077837
    if mask.sum() < 1.0:
        return image.detach(), 1.0

    vis = image.detach()
    vis = vis.clamp(min=0.0)
    if vis.size(0) == 1:
        vis = vis.repeat(3, 1, 1)
    mask = mask.min(dim=0, keepdim=True)[0]  # 1HW

    if rescale:
        brightness = w_RGB[0] * vis[0, :, :] + w_RGB[1] * vis[1, :, :] + w_RGB[2] * vis[2, :, :]  # HW
        # brightness = 0.3 * vis[0, :, :] + 0.59 * vis[1, :, :] + 0.11 * vis[2, :, :]  # HW
        brightness = brightness[mask[0] > 0.999]  # Apply the mask to the brightness tensor
        src_value = brightness.quantile(src_PERCENTILE)
        if src_value < 1.0e-4:
            scalar = 0.0
        else:
            scalar = math.exp(math.log(dst_VALUE) * (1.0/gamma) - math.log(src_value))
        vis = scalar * vis
        # s = np.percentile(vis.cpu(), 99.9)
        # # if mask is None:
        # #     s = np.percentile(vis.numpy(), 99.9)
        # # else:
        # #     s = np.percentile(vis[mask > 0.5].numpy(), 99.9)
        # if s > MAX_SRGB:
        #     vis = vis / s * MAX_SRGB
    else:
        scalar = 1.0

    vis = torch.clamp(vis, min=0)
    if trans2srgb:
        # vis[vis > MAX_SRGB] = MAX_SRGB
        vis = rgb_to_srgb(vis, gamma=gamma)
    if clip:
        vis = vis.clamp(min=0.0, max=1.0)
    return vis, scalar


def adjust_image_for_display(image: torch.tensor, rescale: bool, trans2srgb: bool, src_percentile=0.9999, dst_value=0.85,
                             clip: bool = True, gamma=1.0/2.2):
    assert image.ndim == 3 or image.ndim == 4, "Only supports image: [C, H, W] or [B, C, H, W]"
    vis = image.detach()
    if vis.ndim == 3:
        vis = vis.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
    if vis.size(1) == 1:
        vis = vis.repeat(1, 3, 1, 1)  # [1, 1, H, W] -> [1, 3, H, W]

    if rescale:
        src_value = vis.mean(dim=1, keepdim=True)
        src_value = src_value.view(src_value.size(0), -1).quantile(src_percentile, dim=1)
        src_value[src_value < 1e-5] = 1.0
        vis = vis / src_value.view(src_value.size(0), 1, 1, 1) * dst_value
    if trans2srgb:
        vis = rgb_to_srgb(vis, gamma=gamma)

    if image.ndim == 3:
        vis = vis.squeeze(0)
    return vis.clamp(min=0.0, max=1.0) if clip else vis


def get_scale_alpha(image: torch.tensor, mask: torch.tensor, src_percentile: float, dst_value: float):
    assert image.ndim == mask.ndim == 3, "Only supports image: [C, H, W] or [H, W, C]"
    if mask.sum() < 1.0:
        if torch.is_tensor(image):
            return torch.tensor([1.0]).to(image.device, torch.float32)
        elif isinstance(image, np.ndarray):
            return np.array([1.0], dtype=np.float32)
        else:
            assert False, f"image should be tensor or numpy array, but got {type(image)}."
    if torch.is_tensor(image):
        vis = image.detach()
        src_value = vis.mean(dim=0)  # HW
        mask = mask.min(dim=0)[0]  # HW
    else:
        vis = image
        src_value = vis.mean(axis=-1)
        mask = mask.min(axis=-1)
    assert src_value.shape == mask.shape, f"src_value shape {src_value.shape} != mask shape {mask.shape}."
    if torch.is_tensor(src_value):
        src_value = src_value[mask > 0.999].quantile(src_percentile)
    elif isinstance(src_value, np.ndarray):
        src_value = np.quantile(src_value[mask > 0.999], src_percentile)
    else:
        assert False, f"src_value should be tensor or numpy array, but got {type(src_value)}."
    alpha = 1.0 / src_value.clip(min=1e-5) * dst_value
    return alpha


def convert_plot_to_tensor(plot):
    """convert plt to tensor."""
    buf = io.BytesIO()
    plot.savefig(buf, format='jpeg')
    buf.seek(0)
    image = Image.open(buf)
    t = ToTensor()(image)
    return t


def generate_histogram_image(array, title, xlabel, ylabel, bins=100, range=None):
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if torch.is_tensor(array):
        array = array.cpu().numpy()
    plt.hist(array, histtype='bar', alpha=0.3, bins=bins, range=range)
    hist_img = convert_plot_to_tensor(plt)
    return hist_img


def compute_std_and_mean(image, mask, per_channel):
    """
    Compute the mean and standard deviation of an image in the masked region.
    :param image: Input image. H x W x C
    :param mask: Binary mask. H x W x C
    :param per_channel: Whether to compute the mean and std per channel.
    :return: Mean and standard deviation of the image in the masked region. np.float64. 1 X 1 X C
    """
    assert isinstance(image, np.ndarray) and isinstance(mask, np.ndarray), "Input should be numpy array."
    assert image.shape == mask.shape, f"Shape mismatch: {image.shape} vs {mask.shape}"
    assert image.ndim == 3, f"Invalid image shape: {image.shape}"
    mask = (mask > 0.5).astype(np.float64)  # Binary mask
    if np.sum(mask) < 1e-6:
        print("Empty mask. Skip computation.")
        return 0.0, 0.0
    image = image.astype(np.float64)
    if per_channel:
        axis = (0, 1)
    else:
        axis = (0, 1, 2)
    image_mean = np.sum(image * mask, axis=axis, keepdims=True) / np.sum(mask, axis=axis, keepdims=True).clip(min=1e-6)
    image_var = np.sum(((image - image_mean) * mask) ** 2, axis=axis, keepdims=True) / np.sum(mask, axis=axis, keepdims=True).clip(min=1e-6)
    image_std = np.sqrt(image_var)
    return image_std, image_mean


def align_using_std_and_mean(src_img, src_img_std, src_img_mean, ref_img_std, ref_img_mean, use_std):
    """
    Align the image using the mean of the source and reference images.
    If use_std is True, the standard deviation is also used for alignment.
    :param src_img:
    :param src_img_std:
    :param src_img_mean:
    :param ref_img_std:
    :param ref_img_mean:
    :param use_std:
    :return:
    """
    assert isinstance(src_img, np.ndarray) and isinstance(src_img_mean, np.ndarray), "Input should be numpy array."
    src_img = src_img.astype(np.float64)
    if src_img_std is not None:
        src_img_std = src_img_std.astype(np.float64)
    src_img_mean = src_img_mean.astype(np.float64)
    if ref_img_std is not None:
        ref_img_std = ref_img_std.astype(np.float64)
    ref_img_mean = ref_img_mean.astype(np.float64)
    if use_std:
        scale_factor = ref_img_std / src_img_std.clip(min=1e-6)
    else:
        scale_factor = ref_img_mean / src_img_mean.clip(min=1e-6)
    if use_std:
        mean_shift = ref_img_mean - (src_img_mean * scale_factor)
    else:
        mean_shift = 0
    src_img_aligned = (src_img * scale_factor) + mean_shift
    return src_img_aligned


def mask_valid_region(image, overexposed_threshold: float=None, underexposed_threshold: float=None):
    assert torch.is_tensor(image), "Input should be torch tensor."
    assert image.ndim == 3, "Input should be 3D tensor. C x H x W"

    mask = torch.ones_like(image)
    if overexposed_threshold is not None:
        mask_overexposed = (image > overexposed_threshold).to(torch.float32)
        mask_overexposed = mask_overexposed.max(dim=0, keepdim=True)[0]
        mask = mask * (1 - mask_overexposed)
    if underexposed_threshold is not None:
        mask_underexposed = (image < underexposed_threshold).to(torch.float32)
        mask_underexposed = mask_underexposed.max(dim=0, keepdim=True)[0]
        mask = mask * (1 - mask_underexposed)
    return mask


def numpy_to_tensor(img):
    assert isinstance(img, np.ndarray), "img should be np.ndarray"
    if img.ndim == 3:  # HWC
        img = np.transpose(img, (2, 0, 1))  # CHW
    elif img.ndim == 4:  # BHWC
        img = np.transpose(img, (0, 3, 1, 2))  # BCHW
    else:
        assert False
    return torch.from_numpy(img).contiguous().to(torch.float32)


def tensor_to_numpy(img):
    assert torch.is_tensor(img), "img should be torch.tensor"
    img = img.cpu().numpy()
    if img.ndim == 3:  # CHW
        img = np.transpose(img, (1, 2, 0))  # HWC
    elif img.ndim == 4:  # BCHW
        img = np.transpose(img, (0, 2, 3, 1))  # BHWC
    else:
        assert False
    return img


def split_tenors_from_dict(dict, size):
    a, b = {}, {}
    for key, item in dict.items():
        if isinstance(item, torch.Tensor):
            a[key], b[key] = item.split(size, dim=0)
        else:
            a[key] = b[key] = item
    return a, b


def plot_images(_images, titles=None, figsize_base=4, columns=3, cmap="gray",
                font_size=10, axis_label_size=8, HW_ratio=1.0, show=True):
    num_images = len(_images)
    rows = (num_images + columns - 1) // columns
    figsize = (figsize_base * columns, int(figsize_base * rows * HW_ratio))
    fig, axs = plt.subplots(rows, columns, figsize=figsize)
    axs = np.array(axs).ravel()  # Make sure axs is always a 1D array
    for i, img in enumerate(_images):
        if torch.is_tensor(img):
            img = img.detach()
            if img.ndim == 4:
                if img.shape[0] == 1:
                    img = img.squeeze(0)
                else:
                    assert False, f"img should be 3D tensor, but got {img.ndim}."
            img = img.cpu()
            img = img.numpy().transpose(1, 2, 0)
        if img.dtype == np.float32:
            img = img.clip(min=0.0, max=1.0)
        elif img.dtype == np.uint8:
            img = img.clip(min=0, max=255)
        axs[i].imshow(img, cmap=cmap)
        if titles is not None:
            axs[i].set_title(titles[i], fontsize=font_size)
        axs[i].tick_params(axis='both', which='major', labelsize=axis_label_size)
    for i in range(len(_images), rows * columns):
        axs[i].axis('off')
    plt.tight_layout()
    if show:
        plt.show()
    return plt


def read_image(path: str, type: str, inf_v=None, nan_v=None, preserve_alpha=False):
    """ Read image from path """
    MAX_8bit = 255.0
    MAX_16bit = 65535.0
    # Read image
    assert os.path.exists(path), f"image {path} not exists."
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED|cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)
    assert img is not None, f"read image {path} failed."
    # Convert to float32 [0, 1]
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / MAX_16bit
    elif img.dtype == np.uint8:
        img = img.astype(np.float32) / MAX_8bit
    elif img.dtype == np.float32:
        pass
    else:
        assert False, f"Not supported dtype {img.dtype} for image {path}."
    # check infinities
    if np.any(np.isinf(img)):
        if inf_v is None:
            warnings.warn(f"image {path} contains Inf values.")
        else:
            img[np.isinf(img)] = inf_v  # set inf to a specific value
    # check NaNs
    if np.any(np.isnan(img)):
        if nan_v is None:
            assert False, f"image {path} contains NaN values."
        else:
            img[np.isnan(img)] = nan_v  # set nan to a specific value
    # Check image shape and convert to RGB
    assert img.ndim < 4, f"Image should be 2D or 3D, but got {img.ndim}."
    if img.ndim == 3:
        if img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.shape[-1] == 4:
            if not preserve_alpha:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        assert img.shape[-1] == 3 or img.shape[-1] == 1 or (img.shape[-1] == 4 and preserve_alpha), \
            f"image {path} should be gray or RGB or RGBA({preserve_alpha}), but got shape {img.shape}."
    elif img.ndim == 2:
        img = img[:, :, np.newaxis]  # HW -> HWC
    else:
        assert False, f"image {path} has wrong dimension {img.ndim}."
    # Convert to specified type
    if type == "numpy":
        pass
    elif type == "tensor":
        img = torch.from_numpy(img.copy()).to(torch.float32).permute(2, 0, 1)  # HWC -> CHW
    else:
        raise NotImplementedError(f"Type {type} is not implemented.")
    return img


def save_image(img: torch.tensor or np.array or list, path: str, **kwargs):
    if torch.is_tensor(img):
        assert img.ndim in [2, 3, 4], f"img should be 2D, 3D or 4D tensor, but got {img.ndim}."
        torchvision.utils.save_image(img, path)
    elif isinstance(img, np.ndarray):
        assert False, f"not implemented yet."
    elif isinstance(img, list):
        _img = []
        for x in img:
            if x.ndim == 4 and x.shape[0] == 1:
                x = x.squeeze(0)
            assert x.ndim == 3, f"img should be 3D tensor, but got {x.ndim}."
            if x.shape[0] == 1:
                x = x.repeat(3, 1, 1)
            _img.append(x)
        torchvision.utils.save_image(_img, path, **kwargs)
    else:
        assert False, f"img should be torch.Tensor or np.ndarray, but got {type(img)}."


def save_image_opencv(img: torch.tensor or np.array, path: str,
                      convert_2_uint8: bool = False,
                      is_RGB_img: bool = True):
    """
    Save image using OpenCV.
    :param img:
    :param path:
    :param convert_2_uint8: Whether to convert the image to uint8.
    :param is_RGB_img: Whether the order of the color channels is RGB.
    :return:
    """

    assert img.ndim in [3], f"img should be 3D tensor, but got {img.ndim}."
    if torch.is_tensor(img):
        img = tensor_to_numpy(img)  # HWC
    elif isinstance(img, np.ndarray):
        pass
    else:
        assert False, f"img should be torch.Tensor or np.ndarray, but got {type(img)}."
    assert img.shape[-1] == 3, f"img should be RGB, but got {img.shape}."
    if is_RGB_img:
        # Convert RGB to BGR as OpenCV uses BGR format
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if convert_2_uint8:
        assert img.dtype == np.float32, f"img should be float32, but got {img.dtype}."
        img = (img * 255.0).clip(min=0, max=255).astype(np.uint8)
    cv2.imwrite(path, img)


def resize_tensor(x: torch.tensor, size, mode="bilinear"):
    """
    Resize tensor x to size.
    :param x:
    :param size:
    :param mode:
    :return:
    """
    if x.ndim == 3:
        return torch.nn.functional.interpolate(x[None], size, mode=mode, align_corners=True, antialias=True)[0]
    elif x.ndim == 4:
        return torch.nn.functional.interpolate(x, size, mode=mode, align_corners=True, antialias=True)
    else:
        assert False, f"Invalid tensor shape: {x.shape}"


def crop_around_center(
    image: torch.Tensor, centers: List[Tuple[int, int]], crop_radius: Tuple[int, int]
) -> List[torch.Tensor]:
    """
    Crops patches from the image tensor around specified center pixels with given crop radius.
    image tensor can be C x H x W  or B x C x H x W.

    Parameters:
        image (torch.Tensor): Input image tensor (C, H, W) or (B, C, H, W). Crop is performed along the last two dimensions.
        centers (List[Tuple[int, int]]): List of center pixel locations (y, x).
        crop_radius (Tuple[int, int]): Tuple indicating the crop radius (crop_y, crop_x).

    Returns:
        List[torch.Tensor]: A list of cropped patches as tensors.
    """
    assert torch.is_tensor(image), "Input should be torch tensor."
    assert image.ndim == 3 or image.ndim == 4, f"Invalid image shape: {image.shape}"
    with_batch = (image.ndim == 4)
    crop_y, crop_x = crop_radius
    assert crop_y >= 0 and crop_x >= 0, "Crop radius dimensions should be non-negative."
    cropped_patches = []

    for (center_y, center_x) in centers:
        # Calculate bounds for cropping
        start_y = max(center_y - crop_y, 0)
        end_y = min(center_y + crop_y + 1, image.shape[-2])  # Height
        start_x = max(center_x - crop_x, 0)
        end_x = min(center_x + crop_x + 1, image.shape[-1])  # Width
        # Crop the patch
        if with_batch:
            cropped_patch = image[:, :, start_y:end_y, start_x:end_x]  # for (B, C, H, W)
        else:
            cropped_patch = image[:, start_y:end_y, start_x:end_x]  # for (C, H, W)
        cropped_patches.append(cropped_patch)

    return cropped_patches
