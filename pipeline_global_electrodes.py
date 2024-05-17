# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:12:52 2024

@authors: Rocio Lopez Peco
@email: yrociro@gmail.es
"""

from tools.load_data import load_all_data
from pathlib import Path

'''------------console-----------------'''

electrodes_data_path =Path ('C:/Users/yroci/Desktop/spread/max_spread/electrodes_data')
'''------------console-----------------'''

all_electrodes_data = load_all_data(electrodes_data_path, plot_data_histogram = True)


#%% visualize data distribution

from tools.visualisations import all_data_distribution, activated_electrodes_distribution, activated_electrodes_distribution_per_electrode

all_data_distribution(all_electrodes_data)

all_activated_electrodes_data, electrodo_nombres = activated_electrodes_distribution(electrodes_data_path)

activated_electrodes_distribution_per_electrode(all_activated_electrodes_data, electrodo_nombres)

#%% areas activity analysis

from tools.activity_analysis import areas_analysis, areas_analysis_statistics, statistics_heatmap

# intensities = [1,11,21,31,41,51,61,71,81,91]
intensities = [1,21,41,61,81]

active_areas_mm2, active_areas_data_map = areas_analysis(electrodes_data_path, all_activated_electrodes_data,electrodo_nombres, intensities)


bonf_df = areas_analysis_statistics(intensities, active_areas_mm2)

heatmap_matrix = statistics_heatmap(bonf_df)


# falta el heatmap

#%% amount activity analysis

from tools.activity_analysis import amount_activity, amount_analysis_statistics, statistics_heatmap

# intensities = [1,11,21,31,41,51,61,71,81,91]


top_electrode_values_mean = amount_activity(electrodes_data_path, electrodo_nombres, intensities, plot_scatter= True, plot_errorbar = True)

bonf_df = amount_analysis_statistics(intensities, top_electrode_values_mean)


statistics_heatmap(bonf_df)


#%% shape and compactness

from tools.load_data import electrode_mapping, load_binary_data
from tools.shape_analysis import extract_compactness, plot_combined_activity_pattern 

filename_pattern = "{numero_electrodo}binary_image_*.npy" 
all_binary_data, stim_electrodes, electrode_data_map = load_binary_data(electrodes_data_path, filename_pattern)

compactaciones_map = extract_compactness(electrode_data_map, active_areas_data_map)

low_intensity = 21
high_intensity = 81

# Asume que estos son los índices que has determinado para las intensidades que quieres mostrar
low_index = intensities.index(low_intensity)
low_index = int(low_index)
high_index = intensities.index(high_intensity)
high_index = int(high_index)

'''------------console-----------------'''
electrodes_to_plot = [ '16']
'''------------console-----------------'''


for electrode in electrodes_to_plot:
    data_for_electrode = electrode_data_map[electrode]  # Lista de matrices de actividad por intensidad

    # Llamamos a la función de dibujo con los valores específicos
    plot_combined_activity_pattern(electrode, data_for_electrode, electrode_mapping, compactaciones_map, low_index, high_index)


#%% centroids
from tools.centroid_analysis import plot_distance_to_centroid, extract_centroid_distances_mean, centroids_statistics
from tools.load_data import electrode_mapping

intensity_index = 9  # replace with the actual index for the intensity you're interested in
sorted_distances = plot_distance_to_centroid(electrode_data_map, intensity_index, electrode_mapping)

all_distances, mean_distances, std_devs = extract_centroid_distances_mean(intensities, electrode_data_map, electrode_mapping)
tukey_results = centroids_statistics(all_distances, intensities)

