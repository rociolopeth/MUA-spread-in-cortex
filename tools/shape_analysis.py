# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 12:05:04 2024

@authors: Rocio Lopez Peco
@email: yrociro@gmail.es
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from tools.load_data import electrode_mapping
from skimage import measure





def plot_electrode_matrix(electrode_mapping):
    """Dibuja la matriz de electrodos como referencia."""
    plt.figure(figsize=(8, 8))
    matrix = np.full((10, 10), np.nan)  # Matriz vacía
    plt.imshow(matrix, cmap='YlGnBu', alpha=0)  # alpha=0 hace que la matriz sea transparente
    
    for electrode, position in electrode_mapping.items():
        plt.text(position[1], position[0], electrode, ha='center', va='center', fontsize=10, alpha = 1)

    plt.axis('off')
    plt.title("Electrode Matrix")
    plt.show()

# def plot_combined_activity_pattern(electrode, data_for_electrode, electrode_mapping, compactness_values):
#     # plt.figure()  
#     plot_electrode_matrix(electrode_mapping)

#     colors = ['cornflowerblue','lightcoral']
#     alpha_value = 0.5
#     intensities = [61, 21]


#     # Indices correspondientes en data_for_electrode para 21 y 81
#     intensity_indices = [3, 1]

#     for idx, intensity in zip(intensity_indices, intensities):
#         data_matrix = data_for_electrode[idx]
#         if np.any(data_matrix):
#             for i in range(data_matrix.shape[0]):
#                 for j in range(data_matrix.shape[1]):
#                     if data_matrix[i, j]:  # Si hay actividad, rellena el electrodo
#                         plt.fill([j - 0.5, j - 0.5, j + 0.5, j + 0.5], 
#                                  [i - 0.5, i + 0.5, i + 0.5, i - 0.5], 
#                                  color=colors[intensities.index(intensity)], alpha= alpha_value)

#     # Resaltar el electrodo de estimulación
#     position = electrode_mapping[electrode]
#     rect = mpatches.Rectangle((position[1]-0.5, position[0]-0.5), 1, 1, fill=False, edgecolor='tomato', linewidth=2)
#     plt.gca().add_patch(rect)
   
    
#     # Crear leyendas personalizadas
#     legend_elements = [
#         mpatches.Patch(color='cornflowerblue', label=f'Intensity: 61uA (Compactation value: {compactness_values[0]})'),
#         mpatches.Patch(color='lightcoral', label=f'Intensity: 21uA (Compactation value: {compactness_values[1]})' ),
#         mlines.Line2D([], [], color='none', marker='s', markerfacecolor='none', markeredgewidth=2, markeredgecolor='tomato', label='Stimulation electrode')
#     ]
    
#     plt.legend(handles=legend_elements, loc='upper right', fontsize=13)
    
        
#     plt.title(f"Activity Patterns for stim electrode {electrode}", fontsize = 18)
    
    

def calcular_perimetro(data_matrix):
    contornos = measure.find_contours(data_matrix, 0.5)
    perimetro = 0
    for contorno in contornos:
        # Asegúrate de que el contorno tiene más de un punto
        if contorno.shape[0] > 1:
            perimetro += np.sqrt(np.sum(np.diff(contorno, axis=0)**2, axis=1)).sum()
    return perimetro

def calcular_compactacion(area, perimetro):
    if perimetro == 0:  # Evita la división por cero
        return 0
    area_circulo = (perimetro ** 2) / (4 * np.pi)
    return area / area_circulo

def generate_compactness_matrix(compactaciones, electrode_mapping):
    # Inicializa una matriz de 10x10 con NaNs
    matrix = np.full((10, 10), np.nan)
    # Rellena la matriz con los valores de compactación
    for electrode, compactness in compactaciones.items():
        position = electrode_mapping[electrode]
        matrix[position[0], position[1]] = compactness
    return matrix
 
 
# def compactations(intensity_index, electrodo_nombres, active_areas_mm2, stim_electrodes, electrode_data_map, plot_compactness = True ):
#     # Vamos a iterar a través de los electrodos de estimulación y obtener el data_matrix correspondiente a la intensidad 81
#     for electrode in stim_electrodes:
#         # Obtén la lista de matrices para el electrodo actual
#         all_intensities_data = electrode_data_map[electrode]
        
#         data_matrix = all_intensities_data[intensity_index]
        
#         perimetro = calcular_perimetro(data_matrix)
#         print(f"Perímetro para el electrodo {electrode} a intensidad  {intensity_index}: {perimetro}")
    
#     '''3. comparar con un circulo de las mimsmas dimensiones'''
#     #Calcular la compactacion
    
#     # Ahora, para calcular la compactación, debes asegurarte de emparejar cada área con su electrodo correspondiente
#     compactaciones = {}
#     for nombre_electrodo in electrodo_nombres:
#         # Índice del electrodo actual
#         index_electrodo = electrodo_nombres.index(nombre_electrodo)
        
#         all_intensities_data = electrode_data_map[nombre_electrodo]
#         data_matrix = all_intensities_data[intensity_index]
#         perimetro = calcular_perimetro(data_matrix)
#         areas_intensity = [areas[intensity_index] for areas in active_areas_mm2]  # 4 es el índice para el quinto elemento
#         area = areas_intensity[index_electrodo]
#         compactacion = calcular_compactacion(area, perimetro)
#         compactaciones[nombre_electrodo] = compactacion
    
#     # Imprime las compactaciones para cada electrodo
#     for nombre_electrodo, compactacion in compactaciones.items():
#         print(f"Compactación para el electrodo {nombre_electrodo} a intensidad 81: {compactacion}")
    
#     if plot_compactness:
     
#         compactness_matrix = generate_compactness_matrix(compactaciones, electrode_mapping)
        
#         plt.figure(figsize=(8, 8))
#         cax = plt.imshow(compactness_matrix, cmap='YlGnBu')
#         # Coloca etiquetas con los valores de compactación sobre los electrodos
#         for electrode, compactness in compactaciones.items():
#             position = electrode_mapping[electrode]
#             color = 'white' if compactness > np.nanmedian(compactness_matrix) else 'black'
#             plt.text(position[1], position[0], f"{compactness:.2f}", ha='center', va='center', color=color, fontsize=8)
        
#         # Configura la barra de color para mostrar la compactación
#         cbar = plt.colorbar(cax)
#         cbar.set_label('Compactness')
#         # Puedes ajustar las etiquetas de la barra de color según sea necesario
        
#         plt.title(f"Compactness of Stimulation Electrodes at 81uA Intensity")
        
        
    
#     return compactness


# def extract_compactations(intensities, electrodo_nombres, active_areas_mm2, stim_electrodes, electrode_data_map):
    
#     all_compactations =[]
    
    
#     for i in range(len(intensities)):
#         intensity_index = intensities[i]
#         compactaciones = compactations(intensity_index, electrodo_nombres, active_areas_mm2, stim_electrodes, electrode_data_map, plot_compactness = True )
#         all_compactations.append(compactaciones)
        
        
#     return all_compactations
    

# def extract_all_compactations(electrodo_nombres, active_areas_mm2, stim_electrodes, electrode_data_map, intensities, electrode_mapping):
#     all_compactations = {}

#     # Iterar a través de cada intensidad
#     for intensity_index in intensities:
#         # Diccionario temporal para almacenar compactaciones de esta intensidad
#         compactaciones = {}

#         # Iterar a través de los electrodos de estimulación para calcular compactaciones
#         for nombre_electrodo in electrodo_nombres:
#             # Obtener la lista de matrices para el electrodo actual
#             all_intensities_data = electrode_data_map[nombre_electrodo]
#             data_matrix = all_intensities_data[intensity_index]

#             # Calcular perimetro y área para la compactación
#             perimetro = calcular_perimetro(data_matrix)
#             areas_intensity = [areas[intensity_index] for areas in active_areas_mm2]
#             area = areas_intensity[electrodo_nombres.index(nombre_electrodo)]
#             compactacion = calcular_compactacion(area, perimetro)

#             # Almacenar el valor de compactación en el diccionario
#             compactaciones[nombre_electrodo] = compactacion

#         # # Opcional: plotear la matriz de compactaciones
#         # if True:  # Reemplazar por una condición si es necesario
#         #     compactness_matrix = generate_compactness_matrix(compactaciones, electrode_mapping)
#         #     plot_compactness_matrix(compactness_matrix, electrode_mapping, intensity_index)

#         # Almacenar las compactaciones de esta intensidad en el diccionario general
#         all_compactations[intensity_index] = compactaciones
    
#     return all_compactations

# # Nota: Asegúrate de definir la función `plot_compactness_matrix` si deseas visualizar los resultados.



def extract_compactness(electrode_data_map, active_areas_data_map):
    # Crear un diccionario para almacenar las compactaciones
    compactaciones_map = {}

    # Iterar sobre los electrodos y sus datos de actividad
    for electrode, data_list in electrode_data_map.items():
        # Asegúrate de inicializar la lista para este electrodo en el diccionario
        compactaciones_map[electrode] = []
        
        # Iterar sobre cada intensidad para este electrodo
        for i, data_matrix in enumerate(data_list):
            # Obtener el área correspondiente a esta intensidad
            area = active_areas_data_map[electrode][i]
            # Calcular el perímetro
            perimetro = calcular_perimetro(data_matrix)
            # Calcular la compactación
            compactacion = calcular_compactacion(area, perimetro)
            # Agregar la compactación a la lista correspondiente en el diccionario
            compactaciones_map[electrode].append(compactacion)

    return compactaciones_map



def plot_combined_activity_pattern(electrode, data_for_electrode, electrode_mapping, compactaciones_map, low_index, high_index):
    # Preparar la figura
    # plt.figure(figsize=(8, 8))  
    plot_electrode_matrix(electrode_mapping) 

    # Definir colores y opacidad para los electrodos activos
    colors = ['cornflowerblue', 'lightcoral']
    alpha_value = 0.5

    # Obtener los valores de compactación para las intensidades seleccionadas
    compactness_value_bajo = round(compactaciones_map[electrode][low_index], 2)
    compactness_value_alto = round(compactaciones_map[electrode][high_index], 2)
    compactness_values = [compactness_value_bajo, compactness_value_alto]

    # Dibujar los patrones de actividad
    for idx, compactness_value in zip([low_index, high_index], compactness_values):
        data_matrix = data_for_electrode[idx]
        if np.any(data_matrix):
            for i in range(data_matrix.shape[0]):
                for j in range(data_matrix.shape[1]):
                    if data_matrix[i, j]:  # Si hay actividad, rellena el electrodo
                        color_idx = 0 if idx == low_index else 1
                        plt.fill([j - 0.5, j - 0.5, j + 0.5, j + 0.5], 
                                 [i - 0.5, i + 0.5, i + 0.5, i - 0.5], 
                                 color=colors[color_idx], alpha=alpha_value)

    # Resaltar el electrodo de estimulación
    position = electrode_mapping[electrode]
    rect = mpatches.Rectangle((position[1]-0.5, position[0]-0.5), 1, 1, fill=False, edgecolor='tomato', linewidth=2)
    plt.gca().add_patch(rect)

    # Crear leyendas personalizadas
    legend_elements = [
        mpatches.Patch(color='cornflowerblue', label=f'Lower Intensity (Compactness value: {compactness_value_bajo})'),
        mpatches.Patch(color='lightcoral', label=f'Higher Intensity (Compactness value: {compactness_value_alto})'),
        mlines.Line2D([], [], color='none', marker='s', markerfacecolor='none', markeredgewidth=2, markeredgecolor='tomato', label='Stimulation electrode')
    ]

    plt.legend(handles=legend_elements, loc='upper right', fontsize=13)

    plt.title(f"Activity Patterns for stim electrode {electrode}", fontsize=18)

# Ahora puedes llamar a esta función para cada electrodo que desees visualizar.
    