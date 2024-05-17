# -*- coding: utf-8 -*-
"""
Created on Tue May  7 15:48:14 2024

@authors: Rocio Lopez Peco
@email: yrociro@gmail.es
"""

#%% load data

from tools.load_data import load_values_couples, electrode_mapping
from pathlib import Path

mainfolder = Path('C:/Users/yroci/Desktop/xabier_spread/')
folder = 'values_couples'
intensities = [21,41,61,81]
path_to_load = mainfolder / folder 


electrode_data, pair_keys = load_values_couples(path_to_load)



# Crear una subcarpeta para 'electrodes_data' si no existe
electrodes_data_path = mainfolder / 'electrodes_data_couples'
electrodes_data_path.mkdir(exist_ok=True)

for electrode_pair in pair_keys:    
    electrodes_path = electrodes_data_path / electrode_pair
    electrodes_path.mkdir(exist_ok=True)

path_to_save = electrodes_data_path
#%% extract info

from tools.parameterize_heatmap_values import activity_area, activity_amount
from tools.parameterize_heatmap_values_couples import histogram, ostu_threshold_method, generate_binary_image, plot_binary_image, count_active_electrodes
import numpy as np

for pair_key, arrays in electrode_data.items():
    path_to_save_2 = path_to_save / pair_key
    # Concatena los cuatro arrays para obtener el histograma y calcular el umbral de Otsu
    combined_array = np.concatenate(arrays)
    # Calcula el histograma y el umbral de Otsu para los datos combinados
    counts, x = histogram(combined_array,pair_key, plot=True)
    otsu_threshold = ostu_threshold_method(combined_array, counts, x, pair_key, path_to_save, save_otsu=False)
    active_electrodes_count, active_areas_mm2 = activity_area(arrays, otsu_threshold , path_to_save_2, plot_areas =True, save_areas = False)
    top_electrode_values, top_electrode_values_medias = activity_amount(arrays, active_electrodes_count, active_areas_mm2, path_to_save_2, plot_activities =True, save_activities = True)

    
    for intensity_array, intensity in zip(arrays, intensities):

        target_array = intensity_array
        
        # Genera la imagen binaria usando el umbral de Otsu calculado de los datos combinados
        binary_image = generate_binary_image(target_array, otsu_threshold, pair_key, electrode_mapping, path_to_save, intensity, save_binary_image=False)
        
        # Dibuja la imagen binaria para el tercer array
        plot_binary_image(target_array, binary_image, otsu_threshold, pair_key, intensity, electrode_mapping)
        
        active_values = count_active_electrodes(binary_image, target_array, electrode_mapping, pair_key, path_to_save, intensity, save_active_electrodes = False)
    
