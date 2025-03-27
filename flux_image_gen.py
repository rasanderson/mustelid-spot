# set FAL_KEY = '${.env.FAL.AI_KEY}'

import fal_client
import os
from dotenv import load_dotenv, dotenv_values
import json
import time
from tqdm import tqdm
import requests

load_dotenv()

FAL_KEY = os.environ.get('FAL_KEY')

output_dir = "C:/Users/c0062193.CAMPUS/OneDrive - Newcastle University/General - Fox-AI/Processed/Cleaned/Fox/FLUX_API"

# Parameters
animal = "fox"
prompt = "zoomed in, up close and realistic picture of a {animal} walking"
batch_size = 4  # Number of images per API call
total_images = 100  # Total images to generate
image_width = 512
image_height = 512

# Optional: Additional prompt variations for diversity
prompt_variations = [
    f"zoomed in, up close and realistic picture of a {animal} walking in snow",
    f"zoomed in, up close and realistic picture of a {animal} walking in autumn leaves",
    f"zoomed in, up close and realistic picture of a {animal} walking in a meadow",
    f"zoomed in, up close and realistic picture of a {animal} walking through forest",
	f"zoomed in, up close and realistic picture of a {animal} walking away in a hedge",
    f"zoomed in, up close and realistic picture of a {animal} walking away in autumn leaves",
    f"zoomed in, up close and realistic picture of a {animal} walking away in grass",
    f"zoomed in, up close and realistic picture of a {animal} walking away through forest",
	f"zoomed in, up close and realistic picture of a {animal} walking across the shot in a hedge",
    f"zoomed in, up close and realistic picture of a {animal} walking across the shot in autumn leaves",
    f"zoomed in, up close and realistic picture of a {animal} walking across the shot in grass",
    f"zoomed in, up close and realistic picture of a {animal} walking across the shot through forest"
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
            global_index = batch * batch_size + i + 1 + 100
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