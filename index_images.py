import argparse
import os
from pathlib import Path

import lancedb
import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"
}


def find_images(root_folder: str):
    root = Path(root_folder)
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = "ViT-B-32"
    pretrained = "laion2b_s34b_b79k"

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer (model_name)

    model = model.to(device)
    model.eval()

    return model, preprocess, tokenizer, device


def encode_image_batch(model, preprocess, device, image_paths):
    images = []
    valid_paths = []

    for path in image_paths:
        try:
            image = Image.open(path).convert("RGB")
            image_tensor = preprocess(image)
            images.append(image_tensor)
            valid_paths.append(path)
        except Exception as e:
            print(f"Skipping damaged/unreadable image: {path} | Error: {e}")

    if not images:
        return [], []

    image_batch = torch.stack(images).to(device)

    with torch.no_grad():
        features = model.encode_image(image_batch)
        features = features / features.norm(dim=-1, keepdim=True)

    vectors = features.cpu().numpy().astype(np.float32)
    return valid_paths, vectors


def chunks(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-folder",
        required=True,
        help="Path to your main event decoration image folder"
    )
    parser.add_argument(
        "--db-path",
        default="./event_lancedb",
        help="Local LanceDB folder"
    )
    parser.add_argument(
        "--table-name",
        default="event_images",
        help="LanceDB table name"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size. Use 8  or 16 if you do not have GPU."
    )
    args = parser.parse_args()

    print("Finding images...")
    image_paths = list(find_images(args.image_folder))
    print(f"Found {len(image_paths)} images.")

    if len(image_paths) == 0:
        print("No images found. Check your folder path.")
        return

    print("Loading OpenCLIP model...")
    model, preprocess, tokenizer, device = load_model()
    print(f"Using device: {device}")

    print("Connecting to local LanceDB...")
    db = lancedb.connect(args.db_path)

    table = None
    first_batch = True

    for batch in tqdm(list(chunks(image_paths, args.batch_size)), desc="Indexing images"):
        valid_paths, vectors = encode_image_batch(model, preprocess, device, batch)

        records = []
        for path, vector in zip(valid_paths, vectors):
            try:
                stat = path.stat()
                records.append({
                    "path": str(path),
                    "filename": path.name,
                    "folder": str(path.parent),
                    "file_size": int(stat.st_size),
                    "modified_time": float(stat.st_mtime),
                    "vector": vector.tolist(),
                })
            except Exception as e:
                print(f"Could not create record for {path}: {e}")

        if not records:
            continue

        if first_batch:
            table = db.create_table(
                args.table_name,
                data=records,
                mode="overwrite"
            )
            first_batch =  False
        else:
            table.add(records)

    if table is not None:
        print("Creating vector index. This may take some time for a huge dataset...")
        try:
            table.create_index(vector_column_name="vector")
        except Exception as e:
            print(f"Index creation skipped or failed: {e}")

    print("Done.")
    print(f"Local LanceDB saved at: {os.path.abspath(args.db_path)}")


if __name__ == "__main__":
    main()
