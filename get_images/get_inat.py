# Download photos using pyinaturalise

import pathlib
import requests
from pyinaturalist import get_observations

# Great Britain place ID
# PLACE_ID = 6857

# TAXON_NAME
# - Lutra lutra (European otter)
# - Neovison vison (American mink)
# - Mustela putorius (European polecat)
# - Mustela erminea (stoat)
# - Mustela nivalis (weasel)
# - Martes martes (pinemarten)
# - Mustela furo (ferret)
TAXON_NAME = "Mustela furo"  # ferret
OUTPUT_DIR = pathlib.Path("images/inat/ferret")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_IMAGES = 1000
PER_PAGE = 200

downloaded = 0
page = 1

while downloaded < TARGET_IMAGES:
    print(
    f"taxon_name={TAXON_NAME}, "
    #f"place_id={PLACE_ID}, "
    f"per_page={PER_PAGE}, "
    f"page={page}"
    )

    response = get_observations(
        taxon_name=TAXON_NAME,
        # place_id=PLACE_ID,
        photos=True,
        per_page=PER_PAGE,
        page=page,
        quality_grade="research"
    )

    observations = response.get("results", [])
    print(f"Page {page}: {len(observations)} observations")

    if not observations:
        break

    for obs in observations:

        photos = obs.get("photos", [])

        for photo in photos:

            if downloaded >= TARGET_IMAGES:
                break

            # Use a higher-resolution image
            image_url = photo["url"].replace("square", "original")

            photo_id = photo["id"]
            obs_id = obs["id"]

            filename = OUTPUT_DIR / f"{obs_id}_{photo_id}.jpg"

            try:
                r = requests.get(image_url, timeout=30)
                r.raise_for_status()

                with open(filename, "wb") as f:
                    f.write(r.content)

                downloaded += 1

                if downloaded % 50 == 0:
                    print(f"Downloaded {downloaded} images")

            except Exception as e:
                print(f"Failed: {filename}: {e}")

    page += 1

print(f"Finished. Downloaded {downloaded} images.")