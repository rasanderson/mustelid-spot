# Crop images with megadetector and save animal classes
# Crop images with MegaDetector (PyTorchWildlife)
from pathlib import Path

import torch
from PIL import Image
from PytorchWildlife.models import detection as pw_detection

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_ROOT = Path("images/lila")
OUTPUT_ROOT = Path("images/lila_crop")

SPECIES = [
    "ferret",
    "mink",
    "otter",
    "pinemarten",
    "polecat",
    "stoat",
    "weasel",
]

CONF_THRESHOLD = 0.20
PADDING = 0.10

# ============================================================
# LOAD MEGADETECTOR
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

detector = pw_detection.MegaDetectorV6(
    device=device,
    pretrained=True,
    version="MDV6-yolov10-e"
)

# ============================================================
# PROCESS FOLDERS
# ============================================================

for species in SPECIES:

    input_dir = INPUT_ROOT / species
    output_dir = OUTPUT_ROOT / species

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {species}")

    results = detector.batch_image_detection(str(input_dir))

    for result in results:

        try:
            image_path = Path(result["img_id"])

            img = Image.open(image_path)

            if img.mode != "RGB":
                img = img.convert("RGB")

            width, height = img.size

            detections = result["detections"]

            if len(detections.confidence) == 0:
                continue

            # detections above confidence threshold
            valid_indices = [
                i
                for i, conf in enumerate(detections.confidence)
                if float(conf) >= CONF_THRESHOLD
            ]

            if not valid_indices:
                continue

            # highest confidence detection
            best_idx = max(
                valid_indices,
                key=lambda i: float(detections.confidence[i])
            )

            # PyTorchWildlife normalised coordinates
            x1, y1, x2, y2 = result["normalized_coords"][best_idx]

            # add padding
            box_width = x2 - x1
            box_height = y2 - y1

            x1 -= box_width * PADDING
            y1 -= box_height * PADDING
            x2 += box_width * PADDING
            y2 += box_height * PADDING

            # clip to image bounds
            x1 = max(0.0, x1)
            y1 = max(0.0, y1)
            x2 = min(1.0, x2)
            y2 = min(1.0, y2)

            # normalised -> pixels
            left = int(x1 * width)
            top = int(y1 * height)
            right = int(x2 * width)
            bottom = int(y2 * height)

            crop = img.crop(
                (left, top, right, bottom)
            )

            output_file = output_dir / image_path.name

            crop.save(
                output_file,
                format="JPEG",
                quality=95
            )

            print(f"Saved {output_file}")

        except Exception as e:

            print(f"Failed: {image_path}")
            print(e)

print("\nFinished")