# -*- coding: utf-8 -*-
"""
Created on Fri May 10 13:59:12 2024

@authors: Rocio Lopez Peco
@email: yrociro@gmail.es
"""

#%% load data

from tools.load_data import load_binary_data
from tools.parameterize_heatmap_values_couples import amount_activity_couples

from pathlib import Path
import numpy as np

mainfolder = Path('C:/Users/yroci/Desktop/xabier_spread/')
folder = 'electrodes_data_couples'
folder_2 = 'electrodes_data_repes'
intensities_list = [21,41,61,81]
path_to_load = mainfolder / folder 
path_to_load_2 = mainfolder / folder_2 



filename_pattern = "binary_image_{numero_electrodo}_*.npy"  #patron de parejas 
filename_pattern_2 = "{numero_electrodo}binary_image_*.npy"  # Patrón original de electrodos individuales


all_binary_data_couples, stim_electrodes_couples, couples_data_map = load_binary_data(path_to_load, filename_pattern)
        
all_binary_data_singles, stim_electrodes, singles_data_map = load_binary_data(path_to_load_2, filename_pattern_2)

for key in singles_data_map:
    if len(singles_data_map[key]) > 0:  
        singles_data_map[key] = singles_data_map[key][1:]  



filename_pattern_3 = "top_electrode_values.npy"  # Patrón original de electrodos individuales
top_electrode_values_couples, stim_electrodes_couples = amount_activity_couples(path_to_load, filename_pattern_3, plot_scatter= True, plot_errorbar = True)


filename_pattern_4 = "{numero_electrodo}top_electrode_values.npy"  # Patrón original de electrodos individuales
top_electrode_values_indv, stim_electrodes_indv = amount_activity_couples(path_to_load_2, filename_pattern_4, plot_scatter= True, plot_errorbar = True)

for i in range(len(top_electrode_values_indv)):
    if len(top_electrode_values_indv[i]) > 0:
        top_electrode_values_indv[i] = top_electrode_values_indv[i][1:]

#%% paired singles and couples data, and statistics

from tools.shape_analysis_couples import check_similarlity

for pair_key in couples_data_map.keys():
    individual_keys = pair_key.split('_')
    # Obtener las imágenes binarias de cada electrodo individual
    binary_image_individual_1 = np.array(singles_data_map[individual_keys[0]])
    binary_image_individual_2 = np.array(singles_data_map[individual_keys[1]])
    # Obtener la imagen binaria del par
    binary_image_pair = np.array(couples_data_map[pair_key])
    
    # Realiza la suma lineal de las imágenes individuales y la compara con la imagen binaria del par
    # Asumiendo que cada `binary_image_*` es una lista de arrays y queremos sumar todos los arrays correspondientes
    linear_sum = np.clip(sum(binary_image_individual_1) + sum(binary_image_individual_2), 0, 1)
    check_similarlity(linear_sum, sum(binary_image_pair))

#%% extract images
from tools.shape_analysis_couples import plot_combined_electrode_activity, plot_activity_areas, plot_electrode_activity_paired, calculate_histogram_area
import matplotlib.pyplot as plt

individual_electrode_keys = list(singles_data_map.keys())
couples_electrode_keys = list(couples_data_map.keys())

# Bucle para recorrer todas las parejas y visualizar las activaciones
for pair_key, intensities in couples_data_map.items():
    # Separar los números de electrodos basados en el separador '_'
    individual_keys = pair_key.split('_')
    individual_key1, individual_key2 = individual_keys[0], individual_keys[1]
    
    for intensity_index, intensity_array in enumerate(intensities):
        
        electrode_data1 = singles_data_map[individual_key1][intensity_index]
        electrode_data2 = singles_data_map[individual_key2][intensity_index]
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Ajusta el tamaño según sea necesario
        
        current_intensity = intensities_list[intensity_index]

        # Plotea la actividad combinada de ambos electrodos individuales
        plot_combined_electrode_activity(axs[0], electrode_data1, electrode_data2, f'Individual Activity {individual_key1} and {individual_key2}')
        # Plotea la actividad de la pareja en otro subplot
        plot_electrode_activity_paired(axs[1], intensity_array, f'Combined Activity {pair_key}', 'Greens')
    
        # Personalización adicional de la figura
        plt.suptitle(f'Stimulation in couples at Intensity {current_intensity}', fontsize=16)
        plt.subplots_adjust(top=0.85)  # Ajusta si es necesario
        plt.show()
        
        #sacar las barras de activacion
        active_values_indv = top_electrode_values_indv
        active_values_couples = top_electrode_values_couples
        
        active_areas_indv = calculate_histogram_area(top_electrode_values_indv,individual_electrode_keys, plot_hist=False)
        active_areas_couples = calculate_histogram_area(top_electrode_values_couples,couples_electrode_keys, plot_hist =False)

        plot_activity_areas(active_areas_indv, active_areas_couples, pair_key, individual_key1, individual_key2, intensity_index)  

#%% estadistical analysis
from tools.shape_analysis_couples import comparison_graph

comparison_graph(active_areas_indv, active_areas_couples,intensities_list)


