# Get images from GBIF
from pygbif import species, occurrences
from pathlib import Path
import requests
from urllib.parse import urlparse
import os

# Get species key
def get_species_key(scientific_name):
    r = requests.get(
        "https://api.gbif.org/v1/species/match",
        params={"name": scientific_name}
    )
    r.raise_for_status()

    result = r.json()

    if "usageKey" not in result:
        raise ValueError(
            f"No GBIF match found for '{scientific_name}'"
        )

    return result["usageKey"]

# polecat
common_name = "polecat"
species_key = get_species_key("Mustela putorius")

print(species_key)

image_urls = set()

offset = 0
limit = 300

while len(image_urls) < 1000:

    print(f"Offset {offset}")

    result = occurrences.search(
        taxon_key=species_key,
        media_type="StillImage",
        limit=limit,
        offset=offset
    )

    records = result.get("results", [])

    if not records:
        break

    for rec in records:

        for media in rec.get("media", []):

            url = media.get("identifier")

            if url:
                image_urls.add(url)

    offset += limit

print(f"Unique image URLs found: {len(image_urls)}")

# First 1000
image_urls = list(image_urls)[:1000]

base_dir = Path("images/gbif")
output_dir = base_dir / common_name.replace(" ", "_")
output_dir.mkdir(parents=True, exist_ok=True)

for i, url in enumerate(image_urls, start=1):

    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()

        content_type = r.headers.get("Content-Type", "").lower()

        # Only save JPEG images
        if "image/jpeg" not in content_type:
            continue

        filename = output_dir / f"{common_name.replace(' ', '_')}_{i:04d}.jpg"

        with open(filename, "wb") as f:
            f.write(r.content)

    except Exception as e:
        print(f"Failed {url}: {e}")