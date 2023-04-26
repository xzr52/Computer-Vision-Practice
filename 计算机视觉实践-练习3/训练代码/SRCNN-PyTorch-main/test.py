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
import numpy as np
import torch
from natsort import natsorted
from torchvision.transforms import functional as F
import config
import imgproc
from image_quality_assessment import PSNR, SSIM
from torch import nn
import math
from skimage.metrics import structural_similarity
class SRCNN(nn.Module):
    def __init__(self) -> None:
        super(SRCNN, self).__init__()
        # Feature extraction layer.
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, (9, 9), (1, 1), (4, 4)),
            nn.ReLU(True)
        )

        # Non-linear mapping layer.
        self.map = nn.Sequential(
            nn.Conv2d(64, 32, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(True)
        )

        # Rebuild the layer.
        self.reconstruction = nn.Conv2d(32, 1, (5, 5), (1, 1), (2, 2))

        # Initialize model weights.
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # Support torch.script function.
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.map(out)
        out = self.reconstruction(out)

        return out

    # The filter weight of each layer is a Gaussian distribution with zero mean and
    # standard deviation initialized by random extraction 0.001 (deviation is 0)
    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight.data, 0.0, math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())))
                nn.init.zeros_(module.bias.data)

        nn.init.normal_(self.reconstruction.weight.data, 0.0, 0.001)
        nn.init.zeros_(self.reconstruction.bias.data)

def bgr2ycbcr_torch(tensor: torch.Tensor, only_use_y_channel: bool) -> torch.Tensor:
    if only_use_y_channel:
        weight = torch.Tensor([[24.966], [128.553], [65.481]]).to(tensor)
        tensor = torch.matmul(tensor.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = torch.Tensor([[24.966, 112.0, -18.214],
                               [128.553, -74.203, -93.786],
                               [65.481, -37.797, 112.0]]).to(tensor)
        bias = torch.Tensor([16, 128, 128]).view(1, 3, 1, 1).to(tensor)
        tensor = torch.matmul(tensor.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias

    tensor /= 255.

    return tensor
def ycbcr2rgb(image: np.ndarray) -> np.ndarray:
    image_dtype = image.dtype
    image *= 255.

    image = np.matmul(image, [[0.00456621, 0.00456621, 0.00456621],
                              [0, -0.00153632, 0.00791071],
                              [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]

    image /= 255.
    image = image.astype(image_dtype)

    return image
def image2tensor(image: np.ndarray, range_norm: bool, half: bool) -> torch.Tensor:

    # Convert image data type to Tensor data type
    tensor = F.to_tensor(image)

    # Scale the image data from [0, 1] to [-1, 1]
    if range_norm:
        tensor = tensor.mul(2.0).sub(1.0)

    # Convert torch.float32 image data type to torch.half image data type
    if half:
        tensor = tensor.half()

    return tensor
def bgr2ycbcr(image: np.ndarray, only_use_y_channel: bool) -> np.ndarray:
    if only_use_y_channel:
        image = np.dot(image, [24.966, 128.553, 65.481]) + 16.0
    else:
        image = np.matmul(image, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [
            16, 128, 128]

    image /= 255.
    image = image.astype(np.float32)

    return image
def ycbcr2bgr(image: np.ndarray) -> np.ndarray:

    image_dtype = image.dtype
    image *= 255.

    image = np.matmul(image, [[0.00456621, 0.00456621, 0.00456621],
                              [0.00791071, -0.00153632, 0],
                              [0, -0.00318811, 0.00625893]]) * 255.0 + [-276.836, 135.576, -222.921]

    image /= 255.
    image = image.astype(image_dtype)

    return image

def tensor2image(tensor: torch.Tensor, range_norm: bool, half: bool) :
    if range_norm:
        tensor = tensor.add(1.0).div(2.0)
    if half:
        tensor = tensor.half()

    image = tensor.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")

    return image
def psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))





def main() -> None:
    lr_dir=f"./data/Set5"
    hr_dir = f"./data/Set5"
    device = torch.device("cuda", 0)
    upscale_factor=2
    model_path='./srcnn_model/srcnn_x4-T91-7c460643.pth.tar'
    # Initialize the super-resolution model
    model = SRCNN().to(device=device, memory_format=torch.channels_last)
    print("Build SRCNN model successfully.")

    # Load the super-resolution model weights
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Load SRCNN model weights `{os.path.abspath(config.model_path)}` successfully.")

    # Create a folder of super-resolution experiment results
    results_dir = os.path.join("srcnn_results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Start the verification mode of the model.
    model.eval()
    # Turn on half-precision inference.
    model.half()

    # Initialize the sharpness evaluation function


    # Set the sharpness evaluation function calculation device to the specified model


    # Initialize IQA metrics
    psnr_metrics = 0.0
    ssim_metrics = 0.0

    # Get a list of test image file names.
    file_names = natsorted(os.listdir(lr_dir))
    # Get the number of test image files.
    total_files = len(file_names)

    for index in range(total_files):
        lr_image_path = os.path.join(lr_dir, file_names[index])
        sr_image_path = os.path.join(results_dir, file_names[index])
        hr_image_path = os.path.join(hr_dir, file_names[index])

        print(f"Processing `{os.path.abspath(lr_image_path)}`...")
        # Read LR image and HR image
        lr_image = cv2.imread(lr_image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        hr_image = cv2.imread(hr_image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0

        lr_image = cv2.resize(lr_image, (0, 0),fx=1 / upscale_factor,fy=1/upscale_factor,interpolation=cv2.INTER_CUBIC)
        lr_image = cv2.resize(lr_image, (0, 0),fx= upscale_factor,fy=upscale_factor,interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(results_dir,f'subsample_{ file_names[index]}'), lr_image * 255.0)
        # Get Y channel image data
        lr_y_image = bgr2ycbcr(lr_image, True)
        hr_y_image = bgr2ycbcr(hr_image, True)

        # Get Cb Cr image data from hr image
        hr_ycbcr_image = bgr2ycbcr(hr_image, False)
        _, hr_cb_image, hr_cr_image = cv2.split(hr_ycbcr_image)

        # Convert RGB channel image format data to Tensor channel image format data
        lr_y_tensor = image2tensor(lr_y_image, False, True).unsqueeze_(0)
        hr_y_tensor = image2tensor(hr_y_image, False, True).unsqueeze_(0)

        # Transfer Tensor channel image format data to CUDA device
        lr_y_tensor = lr_y_tensor.to(device=device, memory_format=torch.channels_last, non_blocking=True)
        hr_y_tensor = hr_y_tensor.to(device=device, memory_format=torch.channels_last, non_blocking=True)

        # Only reconstruct the Y channel image data.
        with torch.no_grad():
            sr_y_tensor = model(lr_y_tensor).clamp_(0, 1.0)



        # Save image
        sr_y_image = tensor2image(sr_y_tensor, False, True)
        sr_y_image = sr_y_image.astype(np.float32) / 255.0
        sr_ycbcr_image = cv2.merge([sr_y_image, hr_cb_image, hr_cr_image])
        sr_image = ycbcr2bgr(sr_ycbcr_image)
        hr_y_image = tensor2image(hr_y_tensor, False, True)
        hr_y_image = hr_y_image.astype(np.float32) / 255.0
        psnr_metrics = psnr(sr_y_tensor, hr_y_tensor).item()
        ssim_metrics = structural_similarity(sr_y_image, hr_y_image, win_size=11, gaussian_weights=True,
                                             multichannel=True, data_range=1.0, K1=0.01, K2=0.03, sigma=1.5)
        text='psnr:'+str(psnr_metrics)+' ssim:'+str(ssim_metrics)
        cv2.putText(sr_image, text, (40, 50), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0,0), 1)
        print(file_names[index],f' psnr:{psnr_metrics}')
        print(file_names[index], f' ssim:{ssim_metrics}')
        cv2.imwrite(sr_image_path, sr_image * 255.0)



    # Calculate the average value of the sharpness evaluation index,
    # and all index range values are cut according to the following values
    # PSNR range value is 0~100
    # SSIM range value is 0~1
    avg_ssim = 1 if ssim_metrics / total_files > 1 else ssim_metrics / total_files
    avg_psnr = 100 if psnr_metrics / total_files > 100 else psnr_metrics / total_files

    print(f"PSNR: {avg_psnr:4.2f} dB\n"
          f"SSIM: {avg_ssim:4.4f} u")


if __name__ == "__main__":
    main()
