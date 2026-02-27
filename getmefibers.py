import cv2
import skimage as ski
import numpy as np

from common import getBlackHatMask, getContours, filterContours, getFirstElementOfContour, applyFlooding, applyWatershed

def getMeFibers(base_img,
                bh_ks=(7,7),
                bhm_iter=4,
                bhm_mult=60,
                cont_mult=2.5,
                ws_ths_factor=0.025,
                ws_gl_vecinity=15):
    
    # Eliminación de ruido y mejora de contraste
    test_1 = cv2.GaussianBlur(base_img, (7, 7), 0)
    
    # Mejorar contraste
    clahe = cv2.createCLAHE(clipLimit=2,tileGridSize=(10, 10))
    test_2 = cv2.GaussianBlur(clahe.apply(test_1), (11, 11), 0)

    # Aplicar Black Hat a la imagen seleccionada
    black_hat_img = getBlackHatMask(test_2, kernel_size=bh_ks)

    # Obtener máscara binaria mejorada de la imagen Black Hat
    kernel_bh = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]], dtype=np.uint8)

    black_hat_mask = cv2.morphologyEx(black_hat_img, cv2.MORPH_CLOSE, kernel_bh,iterations=bhm_iter)

    # Aplicar la máscara a la imagen original
    test_3 = np.int16(test_2) - bhm_mult*np.int16(black_hat_mask)
    test_3[test_3 < 0] = 0
    test_3 = np.uint8(test_3)

    # Aplicar umbralización multiotsu para segmentar test_3
    list_ts_test3 = ski.filters.threshold_multiotsu(test_3, classes=5)

    # Segementación de test_3 para separar la mayor cantidad de fibras
    thresh_1 = np.zeros(np.shape(test_3),dtype=np.uint8)
    thresh_1[test_3 > np.mean(list_ts_test3[2:])] = 255
    #thresh_1[test_3 > np.mean(list_ts_test3[2])] = 255
    test_4 = test_3.copy()
    test_4[~(thresh_1 == 255)] = 0
    test_4[test_4 > 0] = 255

    # Obtener contornos de la imagen binaria
    contours, contours_img = getContours(test_4)

    # Filtrar contornos por longitud
    contours_filtered, contours_filtered_img = filterContours(test_4, contours, mult=cont_mult)

    # Aplicar flooding para obtener la máscara de las fibras
    coordinates = getFirstElementOfContour(contours_filtered)
    mask_flood, list_masks = applyFlooding(test_4, coordinates)

    # Se mejora la máscara de las fibras con una operación morfológica de cierre
    kernel_mfl = np.array([[0, 1, 0],
                          [1, 0, 1],
                          [0, 1, 0]], dtype=np.uint8)
    
    test_5 = cv2.morphologyEx(mask_flood, cv2.MORPH_CLOSE, kernel_mfl,iterations=2)

    # Aplicar watershed para obtener la segmentación final
    regions_result, binary_mask = applyWatershed(test_2, test_5, threshold_factor=ws_ths_factor, gl_vecinity=ws_gl_vecinity)

    return binary_mask, contours_filtered_img, list_masks



