# import json
# import os
# import random
# import shutil
# from transformers import pipeline
# from tqdm import tqdm
#
# pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-ro")
#
# captions_file = r"C:\Users\Vlad\ibd_image_captioning\data\coco\captions.json"
# images_folder = r"C:\Users\Vlad\ibd_image_captioning\data\coco\train2014"
# output_captions_file = r"C:\Users\Vlad\ibd_image_captioning\data\coco\output\captions.token"
# output_images_folder = r"C:\Users\Vlad\ibd_image_captioning\data\coco\output\train2014"
#
# subset_size = 30000
#
# os.makedirs(output_images_folder, exist_ok=True)
#
# with open(captions_file, 'r') as f:
#     captions_data = json.load(f)
#
# images = captions_data["images"]
# annotations = captions_data["annotations"]
#
# subset_images = random.sample(images, subset_size)
#
# selected_image_ids = {image['id'] for image in subset_images}
#
# subset_annotations = [anno for anno in annotations if anno["image_id"] in selected_image_ids]
#
# id_to_filepath = {image["id"]: image["file_name"] for image in subset_images}
#
# output_annotations = ""
#
# for anno in tqdm(subset_annotations):
#     image_id = anno["image_id"]
#     if image_id in id_to_filepath:
#         new_anno = pipe(f">>ron<< {anno["caption"]}")[0]["translation_text"]
#
#         output_annotations += f"{id_to_filepath[image_id]}#0\t{new_anno}\n"
#
# for image in subset_images:
#     src_path = os.path.join(images_folder, image['file_name'])
#     dst_path = os.path.join(output_images_folder, image['file_name'])
#     if os.path.exists(src_path):
#         shutil.copy(src_path, dst_path)
#     else:
#         print(f"Warning: Image file {image['file_name']} not found.")
#
# with open(output_captions_file, 'w', encoding='utf-8') as f:
#     f.write(output_annotations)
#
# print("Subset dataset creation complete.")

# import os
# from PIL import Image
#
# def count_image_channels(folder_path):
#     # Initialize counters for each type of channel
#     channel_counts = {1: 0, 3: 0, 4: 0, 'unknown': 0}
#
#     # Loop through all files in the folder
#     for filename in os.listdir(folder_path):
#         file_path = os.path.join(folder_path, filename)
#
#         try:
#             # Open the image
#             with Image.open(file_path) as img:
#                 # Get the number of channels
#                 mode_to_channels = {
#                     "L": 1,  # Grayscale
#                     "RGB": 3,  # RGB
#                     "RGBA": 4  # RGBA
#                 }
#
#                 channels = mode_to_channels.get(img.mode, 'unknown')
#
#                 if channels != 'unknown':
#                     channel_counts[channels] += 1
#                 else:
#                     channel_counts['unknown'] += 1
#
#         except Exception as e:
#             print(f"Error processing file {filename}: {e}")
#
#     return channel_counts
#
# # Example usage
# folder_path = r"C:\Users\Vlad\ibd_image_captioning\data\coco\output\train2014"  # Replace with your folder path
# channel_counts = count_image_channels(folder_path)
#
# print("Image channel counts:")
# print(f"1-channel (Grayscale): {channel_counts[1]}")
# print(f"3-channel (RGB): {channel_counts[3]}")
# print(f"4-channel (RGBA): {channel_counts[4]}")
# print(f"Unknown or unsupported formats: {channel_counts['unknown']}")

import os
from PIL import Image

def process_and_copy_images(source_folder, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for filename in os.listdir(source_folder):
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(destination_folder, filename)

        try:
            with Image.open(source_path) as img:
                # Convert grayscale images to RGB
                if img.mode == "L":
                    img = img.convert("RGB")

                # Save the image to the destination folder
                img.save(destination_path)

        except Exception as e:
            print(f"Error processing file {filename}: {e}")

# Example usage
source_folder = r"C:\Users\Vlad\ibd_image_captioning\data\coco\output\train2014"  # Replace with your source folder path
destination_folder = r"C:\Users\Vlad\ibd_image_captioning\data\coco\output\images"  # Replace with your destination folder path

process_and_copy_images(source_folder, destination_folder)

print("All images have been processed and copied.")