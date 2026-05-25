import os
import platform
import subprocess
from pathlib import Path

import lancedb
import open_clip
import streamlit as st
import torch
from PIL import Image


DB_PATH =  "./event_lancedb"
TABLE_NAME = "event_images"


@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = "ViT-B-32"
    pretrained = "laion2b_s34b_b79k"

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(model_name)

    model = model.to(device)
    model.eval()

    return model, tokenizer, device


@st.cache_resource
def load_table():
    db = lancedb.connect(DB_PATH)
    return db.open_table(TABLE_NAME)


def encode_text(model, tokenizer, device, text):
    tokens = tokenizer([text]).to(device)

    with torch.no_grad():
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy().astype("float32")[0]


def open_file(path):
    system = platform.system()

    try:
        if system == "Windows":
            os.startfile(path)
        elif system == "Darwin":
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)
    except Exception as e:
        st.error(f"Could not open file: {e}")


st.set_page_config(
    page_title="Local Event Decoration AI Search",
    layout="wide"
)

st.title("Local Event Decoration AI Search")
st.write("Search your local event decoration photos using natural language.")

model, tokenizer, device = load_model()

try:
    table = load_table()
except Exception:
    st.error("Could not open LanceDB table. Please run index_images.py first.")
    st.stop()

st.sidebar.header("Search Settings")
top_k = st.sidebar.slider("Number of results",  5, 100, 24)
show_paths = st.sidebar.checkbox("Show full image paths", value=False)

query = st.text_input(
    "Search your images",
    placeholder="Example: bride in red lehenga, white background, golden wedding stage"
)

if query:
    query_vector = encode_text(model,  tokenizer, device, query)

    results = table.search(query_vector).limit(top_k).to_pandas()

    st.subheader(f"Results for: {query}")
    st.write(f"Found {len(results)} results")

    columns = st.columns(4)

    for i, row in results.iterrows():
        image_path = row["path"]
        filename = row["filename"]
        distance = row.get("_distance", None)

        col = columns[i  % 4]

        with col:
            if Path(image_path).exists():
                try:
                    image = Image.open(image_path)
                    caption = filename

                    if distance is not None:
                        caption += f"\nDistance: {distance:.4f}"

                    st.image(image, caption=caption, use_container_width=True)

                    if show_paths:
                        st.code(image_path)

                    if st.button("Open original", key=f"open_{i}"):
                        open_file(image_path)

                except Exception as e:
                    st.warning(f"Could not show image: {filename}")
                    st.caption(str(e))
            else:
                st.warning(f"File missing: {filename}")
                if show_paths:
                    st.code(image_path)
else:
    st.info("Type something like: white background, bride in red lehenga, floral wedding stage.")
