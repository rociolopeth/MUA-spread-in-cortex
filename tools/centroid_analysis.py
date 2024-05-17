# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 11:37:42 2024

@authors: Rocio Lopez Peco
@email: yrociro@gmail.es

"""



import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
from matplotlib import lines as mlines
import scipy.stats as stats
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd



def calculate_centroid(matrix):
    activated_electrodes = np.argwhere(matrix)
    return np.mean(activated_electrodes, axis=0)

def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def generate_distance_matrix(distance_mapping, electrode_mapping):
    matrix = np.full((10, 10), np.nan)
    for electrode, distance in distance_mapping.items():
        position = electrode_mapping[electrode]
        matrix[position] = distance
    return matrix

def plot_distance_to_centroid(electrode_data_map, intensity_index, electrode_mapping):
    plt.figure(figsize=(8, 8))
    
    # Calcular distancias para cada electrodo en la intensidad específica
    distances = []
    distance_mapping = {}
    for electrode, binary_data_list in electrode_data_map.items():
        binary_data = binary_data_list[intensity_index]
        centroid = calculate_centroid(binary_data)
        stimulation_position = electrode_mapping[electrode]
        distance = calculate_distance(centroid, stimulation_position)
        distance_micras = distance * 400
        distances.append(distance_micras)
        distance_mapping[electrode] = distance_micras
        
        # Draw arrow from stimulation position to centroid
        dx = centroid[1] - stimulation_position[1]
        dy = centroid[0] - stimulation_position[0]
        plt.arrow(stimulation_position[1], stimulation_position[0], dx, dy, head_width=0.3, head_length=0.5, fc='lightgreen', ec='lightgreen')

        # Mark the electrode with its number
        plt.text(stimulation_position[1], stimulation_position[0], electrode, ha='center', va='center', color='red', fontsize=8)

    # Generar y dibujar la matriz de distancias
    distance_matrix = generate_distance_matrix(distance_mapping, electrode_mapping)
    cax = plt.imshow(distance_matrix, cmap='Blues')

    # Ordenar las distancias y crear etiquetas para la colorbar
    sorted_distances = np.array(sorted(distances))
    labels = [f"{dist:.2f} ({electrode})" for electrode, dist in sorted(distance_mapping.items(), key=lambda item: item[1])]

    cbar = plt.colorbar(cax, ticks=sorted_distances)
    cbar.set_label('Distance to Centroid (µm)', fontsize=12)
    cbar.ax.set_yticklabels(labels)



    plt.title(f"Distance to Centroid at Intensity Index {intensity_index}", fontsize=15)
    
    # Configure axis ticks and labels
    ticks_positions = np.arange(-0.5, 10.5, 1)  # Extra tick for the last edge
    tick_labels_x = np.arange(0, 11, 1) * 400  # Labels in microns
    tick_labels_y = np.arange(10, -1, -1) * 400  # Labels in microns

    plt.xticks(ticks_positions, tick_labels_x)
    plt.yticks(ticks_positions, tick_labels_y)
    plt.xlabel("Utah array (µm)", fontsize=12)
    plt.ylabel("Utah array (µm)", fontsize=12)

    plt.show()
    
    return sorted_distances

def extract_centroid_distances_mean(intensities, electrode_data_map, electrode_mapping):
    
    
    mean_distances = []
    all_distances = []
    std_devs = []
    
    for intensity_index in range(len(intensities)):
        distances_at_intensity = []
        
        for electrode, binary_data_list in electrode_data_map.items():
            binary_data = binary_data_list[intensity_index]
            centroid = calculate_centroid(binary_data)
            stimulation_position = electrode_mapping[electrode]
            distance = calculate_distance(centroid, stimulation_position)
            distances_at_intensity.append(distance * 400)  # Asumiendo 400 como factor de conversión a micras
    
        mean_distances.append(np.mean(distances_at_intensity))
        all_distances.append(distances_at_intensity)
        std_devs.append(np.std(distances_at_intensity))
    
    # Graficar la distancia media y la desviación estándar para cada intensidad
    plt.figure(figsize=(10, 6))
    plt.errorbar(intensities, mean_distances, yerr=std_devs, fmt='-o', color='blue', ecolor='lightblue', elinewidth=3, capsize=0, label='Mean Distance')
    plt.xticks(intensities)
    plt.fill_between(intensities, np.array(mean_distances) - np.array(std_devs), np.array(mean_distances) + np.array(std_devs), color='lightblue', alpha=0.5)
    plt.title('Electrodes Distance to Centroid')
    plt.xlabel('Stimulation Intensity (µA)')
    plt.ylabel('Distance to Centroid (µm)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return all_distances, mean_distances, std_devs

def centroids_statistics(all_distances, intensities):
    
    all_distances = all_distances[1:]
    intensities = intensities[1:]
    
    flat_distances = [dist for sublist in all_distances for dist in sublist]
    labels = [intensity for intensity, sublist in zip(intensities, all_distances) for _ in sublist]
    
    # Realizar ANOVA
    f_value, p_value = f_oneway(*all_distances)
    print("ANOVA results: F = {:.2f}, p = {:.3f}".format(f_value, p_value))
    
    # Si p_value es menor que el umbral de significación (por ejemplo, 0.05),
    # entonces se procede a realizar pruebas post-hoc.
    if p_value < 0.05:
        # Realizar pruebas post-hoc como la prueba de Tukey
        tukey_results = pairwise_tukeyhsd(endog=flat_distances, groups=labels, alpha=0.05)
        print(tukey_results)
        # tukey_results.plot_simultaneous()  # Esto generará el gráfico de la prueba de Tukey

    return tukey_results

