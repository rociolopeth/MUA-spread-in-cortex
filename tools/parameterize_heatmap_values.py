# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 10:27:15 2024

@authors: Rocio Lopez Peco
@email: yrociro@gmail.es
"""
import numpy as np
from tools.load_data import electrode_mapping
import matplotlib.pyplot as plt



def activity_area(all_values, otsu_threshold, path_to_save, plot_areas =True, save_areas = True):
    
    electrode_area = 400 * 400
    active_electrodes_count = []
    active_areas = []

    for specific_array in all_values:
        # Llenar la matriz usando el mapeo
        matrix = np.full((10, 10), np.nan)
        for idx, value in enumerate(specific_array):
            position = electrode_mapping[str(idx + 1)]
            matrix[position] = value

        # Crear una imagen binaria basada en el umbral de Otsu
        binary_image = matrix > otsu_threshold

        # Contar electrodos activados y agregar a la lista
        active_electrodes_count.append(np.sum(binary_image))
        active_areas.append(np.sum(binary_image) * electrode_area)
    active_areas_mm2 = [area / 1_000_000 for area in active_areas]
    print(active_areas_mm2)
    
    
    if plot_areas:
        # Crear el gráfico de barras area
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(active_areas_mm2)), active_areas_mm2, color='skyblue', edgecolor='black')

        # Añadir etiquetas y título
        intensities_labels = ["Intensity 1", "Intensity 11", "Intensity 21", "Intensity 31", "Intensity 41" , "Intensity 51" , "Intensity 61" , "Intensity 71" , "Intensity 81" , "Intensity 91"]
        plt.xticks(range(len(intensities_labels)), intensities_labels)
        plt.ylabel("Activated area (mm^2)")
        plt.title("Activated areas per intensities")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
    if save_areas:
        
        filename = 'active_areas_mm2.npy'
        ruta = str(path_to_save) +  filename
        np.save(ruta, active_areas_mm2)


    return active_electrodes_count, active_areas_mm2


def activity_amount(all_values, active_electrodes_count, active_areas_mm2, path_to_save, plot_activities =True, save_activities = False):
    '''Extraer los valores de actividad de los electrodos activados'''
    # 1. Extraer los valores de actividad de los n electrodos activados para cada intensidad
    top_electrode_values = []

    for idx, values in enumerate(all_values):
        n = active_electrodes_count[idx]
        if n == 0:
            top_values = np.array([])  # Array vacío
        else:
            top_values = np.sort(values)[-n:]
        top_electrode_values.append(top_values)
        
    if plot_activities:
        
        # Crear el gráfico de barras cantidad
        top_electrode_values_medias = [np.mean(array) for array in top_electrode_values]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(top_electrode_values_medias)), top_electrode_values_medias, color='skyblue', edgecolor='black')
        
        # Añadir etiquetas y título
        # intensities_labels = ["Intensity 1", "Intensity 11", "Intensity 21", "Intensity 31", "Intensity 41" , "Intensity 51" , "Intensity 61" , "Intensity 71" , "Intensity 81" , "Intensity 91"]
        # plt.xticks(range(len(intensities_labels)), intensities_labels)
        plt.ylabel("Activity amount")
        plt.title("Activity amount per intensities")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
    if save_activities:
        filename = 'top_electrode_values.npy'
        ruta = str(path_to_save) +  filename
        np.save(ruta, top_electrode_values)

        top_electrode_values_medias = [np.mean(array) for array in top_electrode_values]
        filename = 'top_electrode_values_mean.npy'
        ruta = str(path_to_save) +  filename
        np.save(ruta, top_electrode_values_medias)
        
        

    

    return top_electrode_values, top_electrode_values_medias
    
    