# set FAL_KEY = '${.env.FAL.AI_KEY}'

"""
flux_image_gen.py

Synthetic camera trap image generator using FAL.AI's FLUX Schnell diffusion model.
Creates realistic animal images with varied prompts and environments for data augmentation
of camera trap classification datasets.

Features:
- Batch processing with configurable batch sizes
- Multiple prompt variations for visual diversity (poses, environments, ages)
- Automatic image downloading and local storage
- Comprehensive metadata logging (prompts, batch info, generation parameters)
- Progress tracking and error handling
- Environment-based API key management
- Sequential filename generation with animal prefix

Current setup: Generates hare images with forest/field/hedge backgrounds.
Easily configurable for other species by changing animal variable and prompts.

Requires FAL.AI API key and sufficient credits for image generation.
"""

import fal_client
import os
from dotenv import load_dotenv, dotenv_values
from pathlib import Path
import json
import time
from tqdm import tqdm
import requests

load_dotenv()

FAL_KEY = os.environ.get('FAL_KEY')

#output_dir = "C:/Users/c0062193.CAMPUS/OneDrive - Newcastle University/General - Fox-AI/Processed/Cleaned/Fox/FLUX_API"
output_dir = "F:/Fox-AI/To process/Flux/newrabbit"
Path(output_dir).mkdir(exist_ok=True)

# Parameters
animal = "hare"
prompt = f"zoomed in, up close and realistic picture of a {animal} walking"
batch_size = 4  # Number of images per API call
total_images = 400  # Total images to generate
image_width = 512
image_height = 512

# Optional: Additional prompt variations for diversity
prompt_variations = [
    f"zoomed in, up close and realistic picture of a {animal} walking away in grass next to a wooden fence",
    f"zoomed in, up close and realistic picture of a {animal} walking away through forest",
	f"zoomed in, up close and realistic picture of a young {animal} running across a field in a hedge",
    f"zoomed in, up close and realistic picture of a young {animal} running across a field in scattered autumn leaves",
    f"zoomed in, up close and realistic picture of a young {animal} running across a field with a wooden fence",
    f"zoomed in, up close and realistic picture of a {animal} running across a field through forest",
    f"zoomed in, up close and realistic picture of a {animal} running across a field clearing",
    f"zoomed in, up close and realistic picture of a {animal} looking away running across a field with scattered autumn leaves",
    f"zoomed in, up close and realistic picture of a young {animal} looking away in a meadow",
    f"zoomed in, up close and realistic picture of a {animal} looking away through forest",
	f"zoomed in, up close and realistic picture of a {animal} looking away away along a hedge",
	f"zoomed in, up close and realistic picture of a baby {animal} running across a field in snow",
    f"zoomed in, up close and realistic picture of a baby {animal} running across a field in autumn leaves",
    f"zoomed in, up close and realistic picture of a baby {animal} running across a field in a meadow",
    f"zoomed in, up close and realistic picture of a baby {animal} running across a field through forest",
	f"zoomed in, up close and realistic picture of a baby {animal} running across a field away in a hedge",
    f"zoomed in, up close and realistic picture of a baby {animal} running across a field away in autumn leaves",
    f"zoomed in, up close and realistic picture of a baby {animal} running across a field away in grass next to a wooden fence",
    f"zoomed in, up close and realistic picture of a baby {animal} running across a field away through forest",
	f"zoomed in, up close and realistic picture of a baby {animal} running across a field across the shot in a hedge",
    f"zoomed in, up close and realistic picture of a baby {animal} running across a field across the shot on grass",
    f"zoomed in, up close and realistic picture of a baby {animal} running across a field across the shot in grass by a wooden fence",
    f"zoomed in, up close and realistic picture of a running {animal} walking across the shot through forest",
    f"zoomed in, up close and realistic picture of a running {animal} looking away in a clearing",
    f"zoomed in, up close and realistic picture of a running {animal} looking away on bare ground",
    f"zoomed in, up close and realistic picture of a running {animal} looking away in a meadow",
    f"zoomed in, up close and realistic picture of a running {animal} looking away through trees",
	f"zoomed in, up close and realistic picture of a running {animal} looking away away along a hedge",
]

# Track all generated images metadata
all_images_metadata = []

# Calculate number of batches needed
num_batches = (total_images + batch_size - 1) // batch_size  # Ceiling division

print(f"Generating {total_images} images in {num_batches} batches...")

# Process in batches
for batch in tqdm(range(num_batches)):
    # Calculate how many images to generate in this batch (handles final batch)
    images_in_batch = min(batch_size, total_images - batch * batch_size)
    
    # Optional: use different prompt variations for diversity
    current_prompt = prompt_variations[batch % len(prompt_variations)] if prompt_variations else prompt
    
    try:
        # Make the API call
        handler = fal_client.submit(
            "fal-ai/flux/schnell",
            arguments={
                "prompt": current_prompt,
                "image_size": {
                    "width": image_width,
                    "height": image_height
                },
                "num_images": images_in_batch
            },
        )
        result = handler.get()
        
        # Add batch info to result for tracking
        result["batch"] = batch + 1
        result["prompt_used"] = current_prompt
        
        # Download each image in the batch
        for i, image_data in enumerate(result['images']):
            url = image_data['url']
            
            # Create a unique filename
            global_index = batch * batch_size + i + 500
            filename = f"{global_index:03d}_{animal}.png"
            filepath = os.path.join(output_dir, filename)
            
            # Download the image
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                # Add filename to image data for reference
                image_data["local_filename"] = filename
                print(f"Downloaded image {global_index}/{total_images}: {filename}")
            else:
                print(f"Failed to download image {global_index}: Status code {response.status_code}")
                image_data["download_failed"] = True
        
        # Add to overall metadata
        all_images_metadata.append(result)
        
        # Optional: small delay between batches to avoid rate limiting
        if batch < num_batches - 1:
            time.sleep(1)
            
    except Exception as e:
        print(f"Error in batch {batch+1}: {str(e)}")
        # Optionally retry the batch or continue to the next one

# Save complete metadata
metadata_file = os.path.join(output_dir, f"generation_metadata_{animal}_2.json")
with open(metadata_file, 'w') as f:
    json.dump(all_images_metadata, f, indent=2)

print(f"\nGeneration complete! {total_images} {animal} images saved to '{output_dir}'")
print(f"Metadata saved to {metadata_file}")