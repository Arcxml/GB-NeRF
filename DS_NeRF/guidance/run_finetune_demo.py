import dotenv
dotenv.load_dotenv(override=True)
import os
os.environ['TRANSFORMERS_OFFLINE'] = 'yes'
import PIL
import requests
import torch
from io import BytesIO
import torch.nn.functional as F

import os
import cv2

import numpy as np

from diffusers import StableDiffusionInpaintPipeline


def change_mask(file_path):
    save_path = "/data/Arc_xml/MVIP/MVIP-NeRF-main/DS_NeRF/guidance/data_test/mask_to_255.png"
    image = cv2.imread(file_path) 
    print(image.max(),image.min())
    image = (np.maximum(image, 0) / image.max()) * 255.0
    image = np.uint8(image)

    cv2.imwrite(save_path,image)  #save_path为保存路径
    return save_path

import torch
image_path = "/data/Arc_xml/MVIP/MVIP-NeRF-main/DS_NeRF/guidance/data_test/image.png"
mask_path = "/data/Arc_xml/MVIP/MVIP-NeRF-main/DS_NeRF/guidance/data_test/mask.png"
mask_path = change_mask(mask_path)
init_image = PIL.Image.open(image_path)
H = init_image.size[0]
W = init_image.size[1]
init_image = init_image.convert("RGB").resize((512, 512))
mask_image = PIL.Image.open(mask_path).convert("RGB").resize((512, 512))
model_path = "/data/Arc_xml/MVIP/MVIP-NeRF-main/DS_NeRF/guidance/ckpt_2/checkpoint-500"
pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16)
#pipe.unet.load_attn_procs(model_path, weight_name="pytorch_lora_weights.safetensors")
pipe.load_lora_weights(model_path, weight_name="pytorch_lora_weights.safetensors")

pipe.to("cuda")


prompt = "A photo of sks bench."
image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]
image_g = image.resize((H, W),PIL.Image.ANTIALIAS)
image_g.save("data_test/bench_finetune.png")


# def log_validation(
#     pipeline,
#     args,
#     accelerator,
#     pipeline_args,
#     epoch,
#     is_final_validation=False,
# ):
#     logger.info(
#         f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
#         f" {args.validation_prompt}."
#     )
#     # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
#     scheduler_args = {}

#     if "variance_type" in pipeline.scheduler.config:
#         variance_type = pipeline.scheduler.config.variance_type

#         if variance_type in ["learned", "learned_range"]:
#             variance_type = "fixed_small"

#         scheduler_args["variance_type"] = variance_type

#     pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, **scheduler_args)

#     pipeline = pipeline.to(accelerator.device)
#     pipeline.set_progress_bar_config(disable=True)

#     # run inference
#     generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None


#     with torch.cuda.amp.autocast():
#         image = pipeline(**pipeline_args, generator=generator).images[0]
 



#     del pipeline
#     torch.cuda.empty_cache()

#     return image









