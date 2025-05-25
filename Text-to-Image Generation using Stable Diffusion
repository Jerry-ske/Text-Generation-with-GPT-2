from diffusers import StableDiffusionPipeline
import torch

# Load the pre-trained Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    revision="fp16"
).to("cuda")  # use 'cpu' if you don't have a GPU

# Generate image from prompt
prompt = "A futuristic city with flying cars and glowing neon lights"
image = pipe(prompt).images[0]

# Show the image
image.show()

# Save the image (optional)
image.save("generated_image.png")
