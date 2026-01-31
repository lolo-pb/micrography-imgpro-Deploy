import cv2
import os
import glob
import argparse
from getmeresults import getMeResults

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--fibers", action="store_true") 
parser.add_argument("-fl", "--flashes", action="store_true")
parser.add_argument("-p", "--pores", action="store_true")
args = parser.parse_args()


input_folder = 'preprodata'
output_folder = 'processed_results'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

parameters = {
    'first_kernel_size': (5,5),
    'second_kernel_size': (3,3),
    'contours_mult': 2.5,
    'bh_ks': (7,7),
    'bhm_iter': 4,
    'bhm_mult': 60,
    'cont_mult': 2.5,
    'ws_ths_factor': 0.025,
    'ws_gl_vecinity': 15,
}

image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif']
files_to_process = []
for ext in image_extensions:
    files_to_process.extend(glob.glob(os.path.join(input_folder, ext)))

all_stats = []

print(f"Found {len(files_to_process)} images. Starting processing...")


if args.fibers or args.flashes or args.pores:
    if args.fibers:
        print("Processing Fibers...")

        # run getmefibers

    if args.flashes:
        print("Processing Flashes...")

        # run getmeflashe

    if args.pores:
        print("Processing Pores...")

        # run getme pores

else:
    print("Processing All Results...")

    i = 0
    for file_path in files_to_process:
        print(f"Processing [{i}] ...")
        filename = os.path.basename(file_path)
        name_only = os.path.splitext(filename)[0]
        
        base_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if base_img is None:
            continue
        
        stats, segmentation, coloring = getMeResults(base_img, parameters)
        
        stats['filename'] = filename
        all_stats.append(stats)
    
        cv2.imwrite(os.path.join(output_folder, f"{name_only}_seg.png"), segmentation)
        
        cv2.imwrite(os.path.join(output_folder, f"{name_only}_color.png"), coloring)
        print("Done.")
        i += 1
    
    print(f"Processing complete! Results saved in '{output_folder}'.")