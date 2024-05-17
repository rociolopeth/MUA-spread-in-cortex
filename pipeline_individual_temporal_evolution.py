# -*- coding: utf-8 -*-
"""
Created on Fri May  3 09:59:45 2024

@authors: Rocio Lopez Peco
@email: yrociro@gmail.es
"""

from tools.load_data import load_values, electrode_mapping
from pathlib import Path


mainfolder = Path('C:/Users/yroci/Desktop/miguel_spread/')
folder = 'values'
subfolder = '93_evolution'
fs = 30000

path_to_load = mainfolder / folder / subfolder
path_to_save = mainfolder / folder / subfolder



all_values, all_data, files = load_values(path_to_load, path_to_save, save_all_data = False)

#%% improve inhibition and activation thresholding (thresholds:otsu, min_value)
# from tools.load_data import load_otsu_threshold, load_data
# import numpy as np


# mainfolder = Path('C:/Users/yroci/Desktop/miguel_spread/')
# folder = 'electrodes_data'
# subfolder = '30'


# path_to_load = mainfolder / folder / subfolder


# otsu_threshold = load_otsu_threshold(path_to_load, subfolder)

# values = load_data(path_to_load, subfolder)
# smallest_values = np.partition(values, 10)[:10]
# min_value = np.mean(smallest_values)

#%%  extract histogram basal with values from intensity 1uA
from tools.temporal_analysis import load_values_for_histogram,characterize_gaussian


max_folders = 10 #to have an equal amount of values for the basal histogram 

path_to_load = mainfolder / folder

all_electrodes_data = load_values_for_histogram(path_to_load, max_folders, plot_data_histogram = False)

mean, std = characterize_gaussian(all_electrodes_data, plot = False)

#%% median and iqr
from tools.temporal_analysis import characterize_median_iqr



median, iqr, lower_threshold, upper_threshold = characterize_median_iqr(all_electrodes_data, plot =False)



#%% visualize data distribution and order data
from tools.visualisations import histogram_intersections
from tools.temporal_analysis import labeled_values, extract_intensity_from_filename, organize_data_by_intensity, extract_window_time

# intersections = [min_value,otsu_threshold]
# intersections = [0.5,1.5]
# intersections = [mean-std,mean+std]
intersections = [lower_threshold,upper_threshold]



histogram_intersections(all_data,intersections)

all_values_labels = labeled_values(all_values,intersections)

all_values_dict = organize_data_by_intensity(files, all_values)

all_values_labels_dict = organize_data_by_intensity(files, all_values_labels)

times = extract_window_time(files, fs)

#%% analysis and pattern visualisation
from tools.visualisations import temporal_pattern


# Recorrer el diccionario por intensidad
for intensity, intensity_values in all_values_labels_dict.items():
    temporal_pattern(files, times, electrode_mapping, intensity_values, intensity)


#%% graph
from tools.temporal_analysis import extract_percentage_labels, save_percentages


percentages_dict  = {}

for intensity in sorted(all_values_labels_dict.keys()):
    percentages = extract_percentage_labels(all_values_labels_dict, intensity, times, max_times=6,  plot_graphs =False)
    percentages_dict[intensity] = percentages
    
    
save_percentages(percentages_dict, mainfolder, folder, subfolder)    
    
    
    
    