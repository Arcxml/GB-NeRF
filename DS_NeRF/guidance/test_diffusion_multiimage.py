import os
import argparse
import dotenv
dotenv.load_dotenv(override=True)
import os
os.environ['TRANSFORMERS_OFFLINE'] = 'yes'
import PIL
import requests
import torch
from io import BytesIO
import torch.nn.functional as F
from PIL import Image, ImageDraw
import os
import cv2
from cal_metrics import calculate_metrics_mean
import numpy as np

from diffusers import StableDiffusionInpaintPipeline

# 设置命令行参数解析
parser = argparse.ArgumentParser(description="Generate images using Stable Diffusion")
parser.add_argument('--prompt_folder', type=str, required=True, help='Path to the prompt folder')
parser.add_argument('--image_folder', type=str, required=True, help='Path to the folder containings images')
parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
parser.add_argument('--save_image_folder', type=str, required=True, help='Path to the folder to save images')
parser.add_argument('--mask_path', type=str, required=True, help='Path to the folder to save masks')
parser.add_argument('--image_test_folder', type=str, required=True, help='Path to the folder containings png images')
args = parser.parse_args()

prompt_folder = args.prompt_folder
image_folder = args.image_folder
model_path = args.model_path
save_image_folder = args.save_image_folder
mask_path = args.mask_path
image_test_folder = args.image_test_folder

# 检查并创建image_folder
if not os.path.exists(save_image_folder):
    os.makedirs(save_image_folder)

# 加载模型
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
)
pipe.load_lora_weights(model_path, weight_name="pytorch_lora_weights.safetensors")
pipe.to("cuda")

prompt_files = sorted(os.listdir(prompt_folder))
init_images = sorted(os.listdir(image_folder))
mask_images = sorted(os.listdir(mask_path))  # 假设masks文件夹路径为'masks'

# 处理每个prompt文件
for prompt_file, init_image_file, mask_image_file in zip(prompt_files, init_images, mask_images):
    with open(os.path.join(prompt_folder, prompt_file), 'r') as f:
        prompt = f.read().strip()

    # 加载初始图像和掩码图像
    init_image_path = os.path.join(image_folder, init_image_file)
    if init_image_file.endswith('.npy'):
        init_image = np.load(init_image_path)
        init_image = (init_image + 1) / 2
        init_image = (init_image * 255).astype(np.uint8)
        print(f"数据范围: min={init_image.min()}, max={init_image.max()}")

# 检查数组维度
        print(f"数组维度: {init_image.shape}")
        init_image = PIL.Image.fromarray(init_image)
        
        
    else:
        init_image = PIL.Image.open(init_image_path)
        image_array = np.array(init_image)
        print(f"image range:{np.max(image_array)=},{np.min(image_array)=}")
    mask_image = PIL.Image.open(os.path.join(mask_path, mask_image_file))

    # 获取原始图像尺寸
    original_size = init_image.size

    # 生成图像
    image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]

    # 调整图像到原始尺寸
    image_g = image.resize(original_size, PIL.Image.ANTIALIAS)

    # 保存图像
    image_save_path = os.path.join(save_image_folder, f"{os.path.splitext(prompt_file)[0]}_diffusion.png")
    image_g.save(image_save_path)
avg_psnr, avg_lpips, fid_value = calculate_metrics_mean(image_test_folder, save_image_folder)
print(f'Average PSNR: {avg_psnr}')
print(f'Average LPIPS: {avg_lpips}')
print(f'FID: {fid_value}')