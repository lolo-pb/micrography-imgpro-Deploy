import cv2
import skimage as ski
import numpy as np
import matplotlib.pyplot as plt

def getBlackHatMask(base_img, kernel_size=(5,5)):
    # Crear kernel para operación morfológica
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    
    # Aplicar transformación Black Hat
    black_hat = cv2.morphologyEx(base_img, cv2.MORPH_BLACKHAT, kernel)
    
    return black_hat

def getContours(base_img):

    # Calcular contornos de la imagen
    contours, _ = cv2.findContours(image=base_img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    dim = np.shape(base_img)

    contours_img = cv2.drawContours(np.zeros((dim[0],dim[1],3)), contours, -1, (0,255,0), 1)
    contours_img = np.uint8(contours_img)

    return contours, contours_img

def filterContours(base_img, contours, mult=2.5, mode='DOWN'):
    # Filtrado de contornos por longitud
    contours_lenght = [len(c) for c in contours]

    cl_mean = np.mean(contours_lenght)
    cl_std = np.std(contours_lenght)

    if mode == 'DOWN':
        contours_filtered = [c for c in contours if len(c) < cl_mean + mult*cl_std]
    elif mode == 'UP':
        contours_filtered = [c for c in contours if len(c) > cl_mean + mult*cl_std]

    dim = np.shape(base_img)
    contours_filtered_img = cv2.drawContours(np.zeros((dim[0],dim[1],3)), contours_filtered, -1, (0,255,0), 1)
    contours_filtered_img = np.uint8(contours_filtered_img)

    return contours_filtered, contours_filtered_img

def getFirstElementOfContour(contours):
    
    # Obtener el primer elemento de cada contorno
    elements = np.array([np.uint32(contours[i][0][0]) for i in range(len(contours))])
    
    # Obtener coordenadas en formato correcto
    coordinates = [(np.uint16(p[1]),np.uint16(p[0])) for p in elements]

    return coordinates

def applyFlooding(base_img, coordinates):
    img_flood = base_img.copy()
    img_flood[img_flood != 0] = 255

    mask_flood = np.zeros(np.shape(base_img),dtype=np.uint8)

    list_masks = []
    for coordinate in coordinates:
        mask = ski.segmentation.flood(img_flood, coordinate, tolerance=1)
        list_masks.append(mask)
        mask_flood[mask] = 255

    return mask_flood, list_masks

def applyWatershed(base_img, mask, threshold_factor = 0.025, gl_vecinity=15):
    # 1. Define kernels adecuados para cada operación
    kernel_op = np.ones((3,3), np.uint8)  # Para opening
    kernel_dil = np.ones((3,3), np.uint8)  # Para dilatación

    # 2. Preprocesamiento mejorado
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_op, iterations=2)

    # 3. Área de fondo seguro con kernel correcto
    sure_bg = cv2.dilate(opening, kernel_dil, iterations=3)

    # 4. Área de primer plano seguro - ajusta el factor de threshold si es necesario
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_C, 5)
    threshold_factor = threshold_factor
    ret, sure_fg = cv2.threshold(dist_transform, threshold_factor*dist_transform.max(), 255, 0)

    # 5. Región desconocida
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 6. Etiquetado de marcadores
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown==255] = 0

    # 7. Aplicar watershed
    img_rgb = cv2.cvtColor(base_img, cv2.COLOR_GRAY2RGB)
    markers = cv2.watershed(img_rgb, markers)

    # 8. Utiliza los resultados de watershed como punto de partida
    regions_result = markers.copy()

    # Crear una máscara expandida
    expanded_mask = np.zeros_like(base_img, dtype=np.uint8)

    # Primero, identifica las regiones que ya fueron segmentadas (excluyendo bordes y fondo)
    valid_labels = np.unique(regions_result)
    valid_labels = valid_labels[valid_labels > 1]  # Ignorar fondo (0,1) y bordes (-1)

    # Para cada región segmentada por watershed, expande a píxeles adyacentes con intensidad similar
    for label in valid_labels:
        # Máscara de la región actual
        region_mask = (regions_result == label)
        
        # Calcular estadísticas de intensidad de esta región
        region_intensities = base_img[region_mask]
        mean_intensity = np.mean(region_intensities)
        std_intensity = np.std(region_intensities)
        
        # Dilatar la región para encontrar píxeles vecinos
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(region_mask.astype(np.uint8), kernel, iterations=2)
        
        # Encontrar píxeles vecinos (dilatados - originales)
        neighbors = dilated & ~region_mask
        
        # Para los vecinos, verificar si su intensidad es similar a la región
        for y, x in zip(*np.where(neighbors)):
            if abs(int(base_img[y, x]) - mean_intensity) < gl_vecinity * std_intensity:
                # Si el píxel tiene intensidad similar, asígnalo a esta región
                regions_result[y, x] = label
                
        # Añadir esta región al resultado expandido
        expanded_mask[regions_result == label] = 255

    binary_mask = np.zeros_like(regions_result, dtype=np.uint8)
    binary_mask[regions_result > 1] = 255  # Set foreground regions to 255 (white)

    return regions_result, binary_mask

# Funciones que vamos a usar

def getSegmentationFigure(segmentation, percentages, filename, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111)

    im = ax.imshow(segmentation)
    ax.axis('off')

    # Agregar la barra de color con etiquetas personalizadas
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", shrink=0.4)
    cbar.set_ticks([20,60,150,250])
    cbar.set_ticklabels([
        f"Indefinido - {percentages['undefined']:.2f}%",
        f"Poros - {percentages['pores']:.2f}%",
        f"Resina - {percentages['resin']:.2f}%",
        f"Fibra - {percentages['fibers']:.2f}%"
        ])
    ax.set_title(f"Segmentación - {filename}")
    plt.tight_layout()

def getColoringFigure(coloring, filename):
    fig = plt.figure(figsize=(14, 8))
    plt.imshow(coloring,vmin=0,vmax=255)
    plt.axis('off')
    plt.title(f"Coloración - {filename}")
    plt.tight_layout()