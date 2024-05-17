# -*- coding: utf-8 -*-
"""
Created on Fri May 10 13:48:10 2024

@authors: Rocio Lopez Peco
@email: yrociro@gmail.es
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd



def check_similarlity(linear_sum, binary_image_pair):
    # Calcula la similitud de Jaccard
    intersection = np.logical_and(linear_sum, binary_image_pair).sum()
    union = np.logical_or(linear_sum, binary_image_pair).sum()
    jaccard_similarity = intersection / union
    
    print(f"Similitud de Jaccard: {jaccard_similarity:.2f}")
    
    # Calcula la similitud de Dice
    dice_similarity = 2 * intersection / (linear_sum.sum() + binary_image_pair.sum())
    
    print(f"Similitud de Dice: {dice_similarity:.2f}")
     
    
def calculate_histogram_area(arrays, electrode_keys, plot_hist=True):
    active_areas_dict = {}  # Diccionario para almacenar las áreas por clave de electrodo

    # Asegurarse de que la longitud de arrays y electrode_keys coincida
    if len(arrays) != len(electrode_keys):
        raise ValueError("La longitud de 'arrays' y 'electrode_keys' debe coincidir.")

    # Iterar sobre cada array y su correspondiente clave de electrodo
    for array, key in zip(arrays, electrode_keys):
        electrode_areas = []  # Lista para almacenar las áreas de los sub-arrays de un electrodo
        for sub_index, sub_array in enumerate(array):
            hist, bins = np.histogram(sub_array, bins='auto')  # Calcular histograma para el sub-array
            bin_widths = np.diff(bins)
            area = np.sum(hist * bin_widths)
            electrode_areas.append(area)  # Añadir el área calculada a la lista del electrodo específico
            
            if plot_hist:
                plt.figure()
                plt.hist(sub_array, bins=bins)  # Usar los bins calculados por numpy
                plt.xlabel('Valor')
                plt.ylabel('Frecuencia')
                plt.title(f'Histograma para {key}, Sub-Array {sub_index}. Área: {area:.2f}')
                plt.grid(True)
                plt.show()

        active_areas_dict[key] = electrode_areas  # Añadir las áreas del electrodo al diccionario

    return active_areas_dict

def plot_activity_areas(active_areas_indv, active_areas_couples, pair_key, individual_key1, individual_key2, intensity_index):
    # Asumimos que pair_key es una tupla con los nombres de los electrodos individuales, por ejemplo ('83', '35')
    fig, axs = plt.subplots(3, 1, figsize=(6, 4))  # 3 filas para las barras de umbral de los dos electrodos individuales y la pareja

    # Extraer todos los valores de los diccionarios y aplanarlos a una sola lista
    all_indv_values = [val for sublist in active_areas_indv.values() for val in sublist]
    all_couples_values = [val for sublist in active_areas_couples.values() for val in sublist]
    
    # Calcular el máximo de estos valores aplanados
    max_threshold = max(max(all_indv_values), max(all_couples_values))

    # max_threshold_intensity = max_threshold[intensity_index]
    colors = ['goldenrod', 'cornflowerblue', 'darkgreen']  # Colores para los umbrales de cada electrodo y la pareja
    # thresholds = [individual_key1, individual_key2, active_areas_couples[pair_key]]
    thresholds = [active_areas_indv[individual_key1][intensity_index], active_areas_indv[individual_key2][intensity_index], active_areas_couples[pair_key][intensity_index]]

    thresholds = np.round(thresholds,2)

    # Dibuja las barras de umbral
    for i, ax in enumerate(axs):
        ax.barh([0], [max_threshold], color='white', edgecolor='white')
        ax.barh([0], [thresholds[i]], color=colors[i], edgecolor='white')
        ax.set_xlim(0, max_threshold)
        ax.set_xticks([0, thresholds[i]])  # Establece marcas solo para 0 y el umbral
        ax.set_yticks([])
        ax.set_xticklabels([0, thresholds[i]])  # Establece etiquetas solo para 0 y el umbral


    # Personalización adicional de la figura
    plt.suptitle(f'Activity Thresholds for Electrodes {pair_key[0]} and {pair_key[1]} and their Pair', fontsize=16)
    plt.tight_layout()
    plt.show()


 
def plot_combined_electrode_activity(ax, binary_image1, binary_image2, title):
    # Combinar las imágenes binarias
    combined_image = binary_image1 + binary_image2 * 2  # Asumiendo que binary_image1 y binary_image2 son binarios (0 o 1)
    # Usar un mapa de colores personalizado
    cmap = mcolors.ListedColormap(['white', 'goldenrod', 'cornflowerblue', 'olivedrab'])
    bounds = [0, 1, 2, 3, 4]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    ax.imshow(combined_image, cmap=cmap, interpolation='nearest', norm=norm)
    ax.set_title(title)
    ax.axis('off')  # Desactiva los ejes
    
def plot_electrode_activity_paired(ax, binary_image, title, color):
    ax.imshow(binary_image, cmap=color, interpolation='nearest', vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis('off')  # Desactiva los ejes
    
def comparison_graph(active_areas_indv, active_areas_couples,intensities_list):

    
    # Convertir los diccionarios en DataFrames
    indv_df = pd.DataFrame(active_areas_indv)
    couples_df = pd.DataFrame(active_areas_couples)
    
    # Crear figuras
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))
    axs = axs.flatten()  # Aplanar el arreglo de ejes para un manejo más sencillo en el loop
    for i, (pair_key, values) in enumerate(couples_df.items()):
        # Extraer los individuos de la clave de la pareja
        individual_keys = pair_key.split('_')
        ind1, ind2 = individual_keys[0], individual_keys[1]
        
        # Datos para graficar
        x = np.arange(len(values))  # Índices de las intensidades
        width = 0.35  # Ancho de las barras
        
        # Graficar barras apiladas para los individuos
        axs[i].bar(x - width/2, indv_df[ind1], width, label=f'Individual {ind1}', color='goldenrod')
        axs[i].bar(x - width/2, indv_df[ind2], width, bottom=indv_df[ind1], label=f'Individual {ind2}', color='cornflowerblue')
        
        # Graficar barras para las parejas
        axs[i].bar(x + width/2, values, width, label='Área de Pareja', color='darkgreen')
    
        axs[i].set_ylabel('Activity areas (AUC)')
        axs[i].set_ylabel('Intensity values (uA)')
        axs[i].set_title(f'Activación de {pair_key}')
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(intensities_list)
        axs[i].legend()
    
    plt.tight_layout()
    plt.show()
    
    
    
        