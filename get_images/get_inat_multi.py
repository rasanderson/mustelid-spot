# Multiple species download simultaneously using threads
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests
from pyinaturalist import get_observations

# Great Britain place ID
PLACE_ID = 6857
# Note GB-only gives less than 1000 images for some species. Many images are
# footprints, spraints, scats so unusable.
# Could omit place_id to get more images, or use a bounding box for north
# European countries etc by editing the get_observations() call below.
#response = get_observations(
#    taxon_name=taxon_name,
#    photos=True,
#    nelat=56.0,
#    nelng=16.0,
#    swlat=49.0,
#    swlng=-11.0,
#    quality_grade="research",
#    per_page=PER_PAGE,#
#    page=page
#)

# Taxa to download
SPECIES = {
    "Lutra lutra": "otter",
    "Neovison vison": "mink",
    "Mustela putorius": "polecat",
    "Mustela erminea": "stoat",
    "Mustela nivalis": "weasel",
    "Martes martes": "pine_marten",
}

TARGET_IMAGES = 1000
PER_PAGE = 200


def download_species(taxon_name, folder_name):

    output_dir = Path(f"images/images_raw/{folder_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    page = 1

    print(f"Starting: {taxon_name}")

    while downloaded < TARGET_IMAGES:

        try:
            response = get_observations(
                taxon_name=taxon_name,
                # place_id=PLACE_ID,
                nelat=56.0,
                nelng=16.0,
                swlat=49.0,
                swlng=-11.0,
                photos=True,
                per_page=PER_PAGE,
                page=page,
                quality_grade="research"
            )

        except Exception as e:
            print(f"{taxon_name}: API error: {e}")
            break

        observations = response.get("results", [])

        print(
            f"{taxon_name} | Page {page} | "
            f"{len(observations)} observations"
        )

        if not observations:
            break

        for obs in observations:

            photos = obs.get("photos", [])

            for photo in photos:

                if downloaded >= TARGET_IMAGES:
                    break

                image_url = photo["url"].replace(
                    "square",
                    "original"
                )

                photo_id = photo["id"]
                obs_id = obs["id"]

                filename = output_dir / f"{obs_id}_{photo_id}.jpg"

                # Skip if already downloaded
                if filename.exists():
                    downloaded += 1
                    continue

                try:
                    r = requests.get(image_url, timeout=30)
                    r.raise_for_status()

                    with open(filename, "wb") as f:
                        f.write(r.content)

                    downloaded += 1

                    if downloaded % 50 == 0:
                        print(
                            f"{taxon_name}: "
                            f"{downloaded} images downloaded"
                        )

                except Exception as e:
                    print(f"Failed: {filename}: {e}")

        page += 1

    print(
        f"Finished: {taxon_name} "
        f"({downloaded} images)"
    )


if __name__ == "__main__":

    with ThreadPoolExecutor(max_workers=len(SPECIES)) as executor:

        futures = [
            executor.submit(
                download_species,
                taxon_name,
                folder_name
            )
            for taxon_name, folder_name in SPECIES.items()
        ]

        # Wait for all species to finish
        for future in futures:
            future.result()

    print("All downloads complete.")
    print("\nImage counts by folder:")

    base_dir = Path("images/images_raw")


    for folder in sorted(base_dir.iterdir()):
        if folder.is_dir():
            count = len(list(folder.glob("*.jpg")))
            print(f"{folder.name}: {count} images")