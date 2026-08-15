import os
import logging
import warnings

os.environ["DIFFUSERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_VERBOSITY"] = "error"

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

import torch

from diffusers import StableDiffusionPipeline
from diffusers.utils import logging as diffusers_logging
from transformers.utils import logging as transformers_logging

diffusers_logging.set_verbosity_error()
diffusers_logging.disable_progress_bar()

transformers_logging.set_verbosity_error()
transformers_logging.disable_progress_bar()

model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)

pipe.enable_model_cpu_offload()

prompt = input("Enter your image prompt: ")

negative_prompt = (
    "blurry, out of focus, low quality, pixelated, "
    "cartoon, illustration, painting, 3d render, "
    "distorted face, deformed, unnatural anatomy"
)

image = pipe(
    prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=40,
    guidance_scale=7.5
).images[0]

image.save("generated_image.png")

print("Image generated successfully!")
print("Saved as generated_image.png")
os.startfile("generated_image.png")