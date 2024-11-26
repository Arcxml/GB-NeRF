import dotenv
dotenv.load_dotenv(override=True)
from diffusers import StableDiffusionInpaintPipeline
import torch
import os
print(f"{os.environ['HF_HOME']=}", flush=True)
pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            
            torch_dtype=torch.float16,
            safety_checker = None
        )
