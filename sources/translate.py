import json
import os
import random
import shutil
from transformers import pipeline
from tqdm import tqdm

pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-ro")

captions_file = r"C:\Users\Vlad\ibd_image_captioning\data\coco\captions.json"
images_folder = r"C:\Users\Vlad\ibd_image_captioning\data\coco\train2014"
output_captions_file = r"C:\Users\Vlad\ibd_image_captioning\data\coco\output\captions.token"
output_images_folder = r"C:\Users\Vlad\ibd_image_captioning\data\coco\output\train2014"

subset_size = 30000

os.makedirs(output_images_folder, exist_ok=True)

with open(captions_file, 'r') as f:
    captions_data = json.load(f)

images = captions_data["images"]
annotations = captions_data["annotations"]

subset_images = random.sample(images, subset_size)

selected_image_ids = {image['id'] for image in subset_images}

subset_annotations = [anno for anno in annotations if anno["image_id"] in selected_image_ids]

id_to_filepath = {image["id"]: image["file_name"] for image in subset_images}

output_annotations = ""

for anno in tqdm(subset_annotations):
    image_id = anno["image_id"]
    if image_id in id_to_filepath:
        new_anno = pipe(f">>ron<< {anno["caption"]}")[0]["translation_text"]

        output_annotations += f"{id_to_filepath[image_id]}#0\t{new_anno}\n"

for image in subset_images:
    src_path = os.path.join(images_folder, image['file_name'])
    dst_path = os.path.join(output_images_folder, image['file_name'])
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
    else:
        print(f"Warning: Image file {image['file_name']} not found.")

with open(output_captions_file, 'w', encoding='utf-8') as f:
    f.write(output_annotations)

print("Subset dataset creation complete.")
