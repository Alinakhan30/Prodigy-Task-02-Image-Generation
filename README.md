# Prodigy Task 2 - Image Generation

This project implements text-to-image generation using the pre-trained Stable Diffusion model as part of my Generative AI Internship at Prodigy InfoTech.

## Technologies Used

- Python
- PyTorch
- Hugging Face Diffusers
- Stable Diffusion v1.5
- CUDA
- NVIDIA GPU

## About the Project

The program takes a text prompt from the user and generates a realistic image based on the prompt using Stable Diffusion.

## How It Works

1. The pre-trained Stable Diffusion model is loaded.
2. The user enters an image prompt.
3. The model processes the prompt and generates an image.
4. The generated image is saved as `generated_image.png`.

## Learning Outcome

Through this task, I learned the basics of text-to-image generation, Stable Diffusion, prompt engineering, negative prompts, inference steps, guidance scale, GPU acceleration and the limitations of AI image-generation models.

I also gained practical experience in setting up the environment, solving errors and improving the generated results through experimentation.

## How to Run

Install the required libraries:

```bash
pip install torch diffusers transformers accelerate safetensors
