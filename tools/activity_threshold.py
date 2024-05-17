# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 16:40:44 2024

@authors: Rocio Lopez Peco
@email: yrociro@gmail.es
"""


import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from tools.load_data import electrode_mapping
import matplotlib.patches as mpatches



def otsu_threshold(all_values, path_to_save, all_data, plot_histogram =True, save_otsu = False):
    
    counts, bin_edges = np.histogram(all_values, bins=20)
    x = np.diff(bin_edges)/2 + bin_edges[:-1]
    
    if plot_histogram:
        plt.figure()
        sns.histplot(all_values, bins=20, kde=False, edgecolor='black', alpha=0.7)
        plt.title("Activity histogram")
        plt.xlabel("Activity value")
        plt.ylabel("Counts")

    '''Activity threshold with otsu and histogram'''
    
    
    # Aplicar el método de Otsu
    otsu_threshold = threshold_otsu(all_data, hist=(counts, x))
    otsu_threshold = threshold_otsu(all_data,nbins=30,  hist=counts)
    x[otsu_threshold]
    
    print(x[otsu_threshold])
    print(f"Otsu's threshold: {otsu_threshold}")
    otsu_threshold =  x[otsu_threshold]
    
    if save_otsu:
        filename = 'otsu_threshold.npy'
        ruta = str(path_to_save) +  filename
        np.save(ruta, otsu_threshold)
    
    return otsu_threshold
    
def generate_binary_image(data_array, otsu_threshold):
    matrix = np.full((10, 10), np.nan)
    # Llenar la matriz usando el mapeo
    for idx, value in enumerate(data_array):
        position = electrode_mapping[str(idx + 1)]
        matrix[position] = value
    
    # Crear y devolver una imagen binaria basada en el umbral de Otsu
    return matrix > otsu_threshold
    
def plot_otsu_binary_image (all_values, otsu_threshold, path_to_save, save_binary_image =False):
    


    for i, data_array in enumerate(all_values):
        binary_image = generate_binary_image(data_array, otsu_threshold)
        
        #figura
        plt.figure(figsize=(8, 8))
        plt.imshow(binary_image, cmap='gray_r', interpolation='none')  # Usamos 'gray_r' para invertir el colormap 
        
        # Añadir números a los electrodos activados
        for idx, value in enumerate(data_array):
            position = electrode_mapping[str(idx + 1)]
            if binary_image[position]:
                plt.text(position[1], position[0], str(idx + 1), ha='center', va='center', color='white', fontsize=8)
        
        
        # Crear "falsos" artistas para la leyenda
        inactive_patch = mpatches.Patch(color='white', label='Inactive')
        active_patch = mpatches.Patch(color='black', label='Active')
        
        plt.legend(handles=[inactive_patch, active_patch], loc='upper right')
        plt.title(f"Electrode Activity (Above Otsu's Threshold: {otsu_threshold:.2f})")
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("Utah array")
        
        if save_binary_image:
            filename = f'binary_image_{i}.npy'
            ruta = str(path_to_save) +  filename
            np.save(ruta, binary_image)