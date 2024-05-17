# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 16:35:16 2024

@authors: Rocio Lopez Peco,
@email: yrociro@gmail.es,
"""

#%% load data

from tools.load_data import load_values, electrode_mapping
from pathlib import Path

'''------------console-----------------'''
mainfolder = Path('C:/Users/yroci/Desktop/spread/max_spread')
folder = 'values'
subfolder = '73'
'''------------console-----------------'''


path_to_load = mainfolder / folder / subfolder


# Crear una subcarpeta para 'electrodes_data' si no existe
electrodes_data_path = mainfolder / 'electrodes_data'
electrodes_data_path.mkdir(exist_ok=True)

# Crear una subcarpeta para 'electrode' si no existe
electrodes_path = electrodes_data_path / subfolder
electrodes_path.mkdir(exist_ok=True)

path_to_save = electrodes_path / subfolder


all_values, all_data, files = load_values(path_to_load, path_to_save, save_all_data = True)


#%% look for activity threshold

from tools.activity_threshold import otsu_threshold, plot_otsu_binary_image

otsu_threshold = otsu_threshold (all_values, path_to_save, all_data, plot_histogram =True, save_otsu = True)

#extract and plot binary images

plot_otsu_binary_image(all_values, otsu_threshold, path_to_save, save_binary_image =True)

#%% extract area and amount of activity

from tools.parameterize_heatmap_values import activity_area, activity_amount

active_electrodes_count, active_areas_mm2 = activity_area(all_values, otsu_threshold , path_to_save, plot_areas =True, save_areas = True)

top_electrode_values, top_electrode_values_medias = activity_amount(all_values, active_electrodes_count, active_areas_mm2, path_to_save, plot_activities =True, save_activities = True)




