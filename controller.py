import cv2
import os
import glob
import argparse

from getmeresults import getMeResults
import getmepores as gmp
import getmeflashes as gmfl
import getmefibers as gmf

## Checking for flags 
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--fibers", action="store_true") 
parser.add_argument("-fl", "--flashes", action="store_true")
parser.add_argument("-p", "--pores", action="store_true")
args = parser.parse_args()


## In/Out
input_folder = 'preprodata'

image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif']
files_to_process = []
for ext in image_extensions:
    files_to_process.extend(glob.glob(os.path.join(input_folder, ext)))
print(f"Found {len(files_to_process)} images. Starting processing...")

all_stats = []


## Filetr parameters, TODO : these are hardoded / they might not nees to be
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

if args.fibers or args.flashes or args.pores:
### Run only Fibers 
    if args.fibers:
        print("Processing Fibers...")
        
        output_folder = 'processed_fibers'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        i = 0
        for file_path in files_to_process:
            print(f"Processing [{i+1}] ...")
            filename = os.path.basename(file_path)
            name_only = os.path.splitext(filename)[0]
    
            base_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if base_img is None:
                continue
            print(f"            |{filename}")
            
            binary_mask, contours_filtered_img, list_masks = gmf.getMeFibers(base_img,bh_ks=parameters['bh_ks'],bhm_iter=parameters['bhm_iter'],bhm_mult=parameters['bhm_mult'],cont_mult=parameters['cont_mult'],ws_ths_factor=parameters['ws_ths_factor'],ws_gl_vecinity=parameters['ws_gl_vecinity'])
     
            cv2.imwrite(os.path.join(output_folder, f"{name_only}_fib.png"), binary_mask)
            print("Done.")
            i += 1
    
        print(f"Processing complete! Results saved in '{output_folder}'.")

### Run only Flashes 
    if args.flashes:
        print("Processing Flashes...")

        output_folder = 'processed_flashes'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        i = 0
        for file_path in files_to_process:
            print(f"Processing [{i+1}] ...")
            filename = os.path.basename(file_path)
            name_only = os.path.splitext(filename)[0]
    
            base_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if base_img is None:
                continue
            print(f"            |{filename}")
            
            mask_flashes = gmfl.getMeFlashes(base_img,cont_mult=parameters['cont_mult'])
     
            cv2.imwrite(os.path.join(output_folder, f"{name_only}_flash.png"), mask_flashes)
            print("Done.")
            i += 1
    
        print(f"Processing complete! Results saved in '{output_folder}'.")


### Run only Pores 
    if args.pores:
        print("Processing Pores...")

        output_folder = 'processed_pores'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        i = 0
        for file_path in files_to_process:
            print(f"Processing [{i+1}] ...")
            filename = os.path.basename(file_path)
            name_only = os.path.splitext(filename)[0]
    
            base_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if base_img is None:
                continue
            print(f"            |{filename}")
            
            mask_bubbles, undefined_region_mask = gmp.getMetPores(base_img,first_kernel_size=parameters['first_kernel_size'],second_kernel_size=parameters['second_kernel_size'])
     
            cv2.imwrite(os.path.join(output_folder, f"{name_only}_pore.png"), mask_bubbles)
            print("Done.")
            i += 1
    
        print(f"Processing complete! Results saved in '{output_folder}'.")

### Run only Results
else:
    print("Processing All Results...")

    output_folder = 'processed_results'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    i = 0
    for file_path in files_to_process:
        print(f"Processing [{i}] ...")
        filename = os.path.basename(file_path)
        name_only = os.path.splitext(filename)[0]

        base_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if base_img is None:
            continue
        print(f"     |{filename}")

        stats, segmentation, coloring = getMeResults(base_img, parameters)

        stats['filename'] = filename
        all_stats.append(stats)

        cv2.imwrite(os.path.join(output_folder, f"{name_only}_seg.png"), segmentation)
        cv2.imwrite(os.path.join(output_folder, f"{name_only}_color.png"), coloring)
        print("Done.")
        i += 1

    print(f"Processing complete! Results saved in '{output_folder}'.")