# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os

import cv2
import torch
from natsort import natsorted
import numpy as np
from numpy import ndarray
import imgproc
import model
from utils import make_directory
from PIL import Image
import math
from skimage.metrics import structural_similarity
from typing import Any
from torch import Tensor
from torch import nn
from torch.nn import functional as F_torch
from torchvision import models
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor
class SRResNet(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            channels: int,
            num_rcb: int,
            upscale_factor: int
    ) -> None:
        super(SRResNet, self).__init__()
        # Low frequency information extraction layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, (9, 9), (1, 1), (4, 4)),
            nn.PReLU(),
        )

        # High frequency information extraction block
        trunk = []
        for _ in range(num_rcb):
            trunk.append(_ResidualConvBlock(channels))
        self.trunk = nn.Sequential(*trunk)

        # High-frequency information linear fusion layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )

        # zoom block
        upsampling = []
        if upscale_factor == 2 or upscale_factor == 4 or upscale_factor == 8:
            for _ in range(int(math.log(upscale_factor, 2))):
                upsampling.append(_UpsampleBlock(channels, 2))
        elif upscale_factor == 3:
            upsampling.append(_UpsampleBlock(channels, 3))
        self.upsampling = nn.Sequential(*upsampling)

        # reconstruction block
        self.conv3 = nn.Conv2d(channels, out_channels, (9, 9), (1, 1), (4, 4))

        # Initialize neural network weights
        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        out1 = self.conv1(x)
        out = self.trunk(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)

        out = torch.clamp_(out, 0.0, 1.0)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)



class _ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super(_ResidualConvBlock, self).__init__()
        self.rcb = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.rcb(x)

        out = torch.add(out, identity)

        return out


class _UpsampleBlock(nn.Module):
    def __init__(self, channels: int, upscale_factor: int) -> None:
        super(_UpsampleBlock, self).__init__()
        self.upsample_block = nn.Sequential(
            nn.Conv2d(channels, channels * upscale_factor * upscale_factor, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.upsample_block(x)

        return out




def srresnet_x4(**kwargs: Any) -> SRResNet:
    model = SRResNet(upscale_factor=4, **kwargs)

    return model



model_names = sorted(
    name for name in model.__dict__ if
    name.islower() and not name.startswith("__") and callable(model.__dict__[name]))
def tensor_to_image(tensor:torch.Tensor, range_norm: bool, half: bool) :
    if range_norm:
        tensor = tensor.add(1.0).div(2.0)
    if half:
        tensor = tensor.half()

    image = tensor.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")

    return image
def psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))
def image_to_tensor(image: ndarray, range_norm: bool, half: bool):
    # Convert image data type to Tensor data type
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()
    # Scale the image data from [0, 1] to [-1, 1]
    if range_norm:
        tensor = tensor.mul(2.0).sub(1.0)
    # Convert torch.float32 image data type to torch.half image data type
    if half:
        tensor = tensor.half()
    return tensor
def main() -> None:
    device = torch.device("cuda", 0)
    lr_dir = f"./data/Set5"
    sr_dir = f"./segan_results/"
    gt_dir = f"./data/Set5"
    g_model_weights_path='./srgan_model/SRGAN_x4-ImageNet-8c4a7569.pth.tar'
    # Initialize the super-resolution bsrgan_model
    g_model = SRResNet(in_channels=3,out_channels=3,
                                            channels=64,
                                            num_rcb=16)
    g_model = g_model.to(device=device)

    # Load the super-resolution bsrgan_model weights
    checkpoint = torch.load(g_model_weights_path, map_location=lambda storage, loc: storage)
    g_model.load_state_dict(checkpoint["state_dict"])

    # Create a folder of super-resolution experiment results
    make_directory(sr_dir)
    # Start the verification mode of the bsrgan_model.
    g_model.eval()

    psnr_metrics_all = 0.0
    ssim_metrics_all = 0.0
    # Get a list of test image file names.
    file_names = natsorted(os.listdir(lr_dir))
    # Get the number of test image files.
    total_files = len(file_names)

    for index in range(total_files):
        lr_image_path = os.path.join(lr_dir, file_names[index])
        sr_image_path = os.path.join(sr_dir, file_names[index])
        gt_image_path = os.path.join(gt_dir, file_names[index])

        gt_img = Image.open(gt_image_path)
        gt_image = np.array(gt_img).astype(np.float32) / 255.0
        gt_tensor=image_to_tensor(gt_image, False, False).unsqueeze_(0)
        gt_rgb_image = cv2.cvtColor(gt_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(sr_dir, f'GroundTruth_{ file_names[index]}'), gt_rgb_image*255)
        print(f"Processing `{os.path.abspath(lr_image_path)}`...")
        lr_img = Image.open(lr_image_path)
        size = np.min(lr_img.size)
        downscale = transforms.Resize(int(size / 4), interpolation=Image.BICUBIC)
        lr_img = downscale(lr_img)
        lr_image = np.array(lr_img)
        lr_rgb_image = cv2.cvtColor(lr_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(sr_dir, f'subsample_4_{file_names[index]}'), lr_rgb_image)
        lr_tensor = image_to_tensor(lr_image, False, False).unsqueeze_(0)

        lr_tensor = lr_tensor.to(device=device, memory_format=torch.channels_last, non_blocking=True)/255.0
        gt_tensor = gt_tensor.to(device=device, memory_format=torch.channels_last, non_blocking=True)
        # Only reconstruct the Y channel image data.
        with torch.no_grad():
            sr_tensor = g_model(lr_tensor)

        # Save image
        sr_image = tensor_to_image(sr_tensor, False, True)
        # Cal IQA metrics

        psnr_metrics = psnr(sr_tensor, gt_tensor)
        ssim_metrics = structural_similarity(sr_image.astype(np.float32) / 255.0, gt_image, win_size=11, gaussian_weights=True,
                                             multichannel=True, data_range=1.0, K1=0.01, K2=0.03, sigma=1.5)
        psnr_metrics_all += psnr_metrics
        ssim_metrics_all += ssim_metrics
        print(file_names[index], f' psnr:{psnr_metrics}')
        print(file_names[index], f' ssim:{ssim_metrics}')
        sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(sr_dir, f'super_resolution_{file_names[index]}'), sr_image)

    avg_psnr = 100 if psnr_metrics_all / total_files > 100 else psnr_metrics_all / total_files
    avg_ssim = 1 if ssim_metrics_all / total_files > 1 else ssim_metrics_all / total_files

    print(f"PSNR: {avg_psnr:4.2f} [dB]\n"
          f"SSIM: {avg_ssim:4.4f} [u]")


if __name__ == "__main__":
    main()
