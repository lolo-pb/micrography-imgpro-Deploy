import cv2
import os
import glob
from getmeresults import getMeResults

input_folder = 'preprodata'
output_folder = 'processed_results'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif']
files_to_process = []
for ext in image_extensions:
    files_to_process.extend(glob.glob(os.path.join(input_folder, ext)))

all_stats = []

print(f"Found {len(files_to_process)} images. Starting processing...")

i = 0
for file_path in files_to_process:
    print("Processing [{i}] ...")
    filename = os.path.basename(file_path)
    name_only = os.path.splitext(filename)[0]
    
    base_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if base_img is None:
        continue

    stats, segmentation, coloring = getMeResults(base_img)
    
    stats['filename'] = filename
    all_stats.append(stats)

    cv2.imwrite(os.path.join(output_folder, f"{name_only}_seg.png"), segmentation)
    
    cv2.imwrite(os.path.join(output_folder, f"{name_only}_color.png"), coloring)
    print("Done.")

print(f"Processing complete! Results saved in '{output_folder}'.")