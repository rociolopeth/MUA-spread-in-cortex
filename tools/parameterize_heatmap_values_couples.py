# -*- coding: utf-8 -*-
"""
Created on Fri May 10 13:01:02 2024

@authors: Rocio Lopez Peco
@email: yrociro@gmail.es
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.filters import threshold_otsu
import matplotlib.patches as mpatches
from pathlib import Path



def histogram(array,pair_key, plot=True):
    counts, bin_edges = np.histogram(array, bins=20)
    x = np.diff(bin_edges)/2 + bin_edges[:-1]
    if plot:
        plt.figure()
        sns.histplot(array, bins=20, kde=False, edgecolor='black', alpha=0.7)
        plt.title(f'Activity histogram in {pair_key} couple')
        plt.xlabel("Activity value")
        plt.ylabel("Counts")
        plt.show()
    return counts, x

def ostu_threshold_method(array, counts, x, electrode_pair, path_to_save, save_otsu =False):
    otsu_threshold = threshold_otsu(array, hist=(counts, x))
    otsu_threshold = threshold_otsu(array,nbins=30,  hist=counts)
    otsu_threshold =  x[otsu_threshold]
    print(f"Otsu's threshold: {otsu_threshold}")
    if save_otsu:
        filename = f'otsu_threshold_{electrode_pair}.npy'
        path_complete = Path(path_to_save, electrode_pair)
        ruta = path_complete / filename
        np.save(ruta, otsu_threshold)
 
    return otsu_threshold

def generate_binary_image(array, otsu_threshold, electrode_pair, electrode_mapping, path_to_save, intensity, save_binary_image = False):
    matrix = np.full((10, 10), np.nan)
    # Llenar la matriz usando el mapeo
    for idx, value in enumerate(array):
        position = electrode_mapping[str(idx + 1)]
        matrix[position] = value
    binary_image = matrix > otsu_threshold 
    if save_binary_image:
        filename = f'binary_image_{electrode_pair}_{intensity}.npy'
        path_complete = Path(path_to_save, electrode_pair)
        ruta = path_complete /  filename
        np.save(ruta, binary_image)

    # Crear y devolver una imagen binaria basada en el umbral de Otsu
    return binary_image

def plot_binary_image(array, binary_image, otsu_threshold, electrode_pair, intensity, electrode_mapping):
        
    plt.figure(figsize=(8, 8))
    plt.imshow(binary_image, cmap='gray_r', interpolation='none')  # Usamos 'gray_r' para invertir el colormap 
    
    # Añadir números a los electrodos activados
    for idx, value in enumerate(array):
        position = electrode_mapping[str(idx + 1)]
        if binary_image[position]:
            plt.text(position[1], position[0], str(idx + 1), ha='center', va='center', color='white', fontsize=8)
    
    
    # Crear "falsos" artistas para la leyenda
    inactive_patch = mpatches.Patch(color='white', label='Inactive')
    active_patch = mpatches.Patch(color='black', label='Active')
    
    plt.legend(handles=[inactive_patch, active_patch], loc='upper right')
    plt.title(f"Electrode Activity (Above Otsu's Threshold: {otsu_threshold:.2f}), stim couple{electrode_pair}, intensity {intensity} ")
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("Utah array")
    
def count_active_electrodes(binary_image, target_array, electrode_mapping, electrode_pair, path_to_save, intensity, save_active_electrodes = True):
    
     # Crear una matriz 2D basada en el mapeo de electrodos para alinear con la imagen binaria
    activity_matrix = np.full(binary_image.shape, np.nan)  
    for key, (row, col) in electrode_mapping.items():
        idx = int(key) - 1
        activity_matrix[row, col] = target_array[idx] 
    # Ahora aplicamos la máscara binaria a la matriz de actividad para obtener los valores de actividad
    active_electrode_values = activity_matrix[binary_image]
 
    # Contar electrodos activados
    num_active_electrodes = np.count_nonzero(active_electrode_values)
 
    print(f"Número de electrodos activos: {num_active_electrodes}")
    print(f"Valores de electrodos activos: {active_electrode_values}")



    if save_active_electrodes:
        filename = f'active_electrode_values_{electrode_pair}_{intensity}.npy'
        path_complete = Path(path_to_save, electrode_pair)
        ruta = path_complete /  filename
        np.save(ruta, active_electrode_values)

    return active_electrode_values

def amount_activity_couples(path, filename_pattern, plot_scatter= True, plot_errorbar = True):
    
    carpetas_de_electrodos = [e for e in path.iterdir() if e.is_dir()]

    top_electrode_values = []
    stim_electrodes = []


    for carpeta in carpetas_de_electrodos:
        numero_electrodo = carpeta.name
    
        # Buscar todos los archivos que coincidan con el patrón 'pattern_*.npy'
        archivos = list(carpeta.glob(filename_pattern.format(numero_electrodo=numero_electrodo)))
        
        # Iterar a través de los archivos encontrados
        for archivo in archivos:
            # Cargar los datos del archivo y agregarlos a la lista
            datos = np.load(archivo, allow_pickle=True)
            top_electrode_values.append(datos)
            # Agregar el número del electrodo a la lista de nombres
            stim_electrodes.append(numero_electrodo)

            
    return top_electrode_values, stim_electrodes
                    
