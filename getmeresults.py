import cv2
import numpy as np

import getmepores as gmp
import getmeflashes as gmfl
import getmefibers as gmf

def getMeResults(base_img, parameters = {
    'first_kernel_size': (5,5),
    'second_kernel_size': (3,3),
    'contours_mult': 2.5,
    'bh_ks': (7,7),
    'bhm_iter': 4,
    'bhm_mult': 60,
    'cont_mult': 2.5,
    'ws_ths_factor': 0.025,
    'ws_gl_vecinity': 15,
}):
    
    pores_mask, undefined_mask = gmp.getMetPores(base_img,
                                                first_kernel_size=parameters['first_kernel_size'],
                                                second_kernel_size=parameters['second_kernel_size'])
    
    flashes_mask = gmfl.getMeFlashes(base_img,cont_mult=parameters['cont_mult'])
    
    fibers_mask, _, _ = gmf.getMeFibers(base_img,bh_ks=parameters['bh_ks'],
                                        bhm_iter=parameters['bhm_iter'],
                                        bhm_mult=parameters['bhm_mult'],
                                        cont_mult=parameters['cont_mult'],
                                        ws_ths_factor=parameters['ws_ths_factor'],
                                        ws_gl_vecinity=parameters['ws_gl_vecinity'])

    # Se completa la máscara de objetos indefinidos
    undefined_mask_complete = np.zeros(np.shape(base_img), dtype=np.uint8)
    undefined_mask_complete[undefined_mask == 255] = 255
    undefined_mask_complete[flashes_mask == 255] = 255
    
    # Se refina la máscara de fibras
    fibers_mask_complete = fibers_mask.copy()
    fibers_mask_complete[pores_mask == 255] = 0
    fibers_mask_complete[undefined_mask_complete == 255] = 0

    # Se obtiene la máscara de resina y se refina
    resin_mask = cv2.bitwise_not(fibers_mask_complete)
    resin_mask[pores_mask == 255] = 0
    resin_mask[undefined_mask_complete == 255] = 0

    # Cálculos
    total_pixels = base_img.shape[0] * base_img.shape[1]
    total_pores = np.sum(pores_mask == 255)
    total_fibers = np.sum(fibers_mask_complete == 255)
    total_resin = np.sum(resin_mask == 255)
    total_undefined = np.sum(undefined_mask_complete == 255)

    percentages = {
        'pores': (total_pores / total_pixels) * 100,
        'fibers': (total_fibers / total_pixels) * 100,
        'resin': (total_resin / total_pixels) * 100,
        'undefined': (total_undefined / total_pixels) * 100,
        'sumcheck': (total_pores + total_fibers + total_resin + total_undefined) / total_pixels * 100
    }

    # 1. Convert grayscale base to BGR (OpenCV standard)
    coloring = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)

    # 2. Poros -> Pure RED
    coloring[pores_mask != 0] = [0, 0, 255] 

    # 3. Fibras -> Pure YELLOW
    coloring[fibers_mask_complete != 0] = [0, 255, 255]

    # 4. Resina -> Pure GREEN
    coloring[resin_mask != 0] = [0, 255, 0]

    # 5. Indefinidos -> CYAN
    coloring[undefined_mask_complete != 0] = [0, 255, 255]
    ## 5. Indefinidos -> BLACK
    #coloring[undefined_mask_complete != 0] = [0, 0, 0]

    # Imagen segmentada
    segmentation = np.zeros(np.shape(base_img), dtype=np.uint8)

    # 1. Poros
    segmentation[pores_mask != 0] = 60
    # 2. Fibras 
    segmentation[fibers_mask_complete != 0] = 250
    # 3. Resina
    segmentation[resin_mask != 0] = 150
    # 4. Indefinidos
    segmentation[undefined_mask_complete != 0] = 20

    return percentages, segmentation, coloring 