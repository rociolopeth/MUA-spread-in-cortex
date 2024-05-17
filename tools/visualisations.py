# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 13:44:24 2024

@authors: Rocio Lopez Peco
@email: yrociro@gmail.es

"""

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy.stats import norm



'''Ver la distribucion de todos los datos'''
def all_data_distribution(all_electrodes_data):
    
        
    # Supongamos que tienes tus datos en la variable "todos_los_datos"
    
    # Crear un modelo K-Means con 2 clusters
    kmeans = KMeans(n_clusters=2, random_state=0)
    
    # Ajustar el modelo a tus datos
    kmeans.fit(all_electrodes_data.reshape(-1, 1))
    
    # Obtener las etiquetas de cluster asignadas a cada dato
    labels = kmeans.labels_
    
    # Separar los datos en dos grupos basados en las etiquetas
    grupo1 = all_electrodes_data[labels == 0]
    grupo2 = all_electrodes_data[labels == 1]
    
    # Graficar los dos grupos
    plt.figure(figsize=(8, 6))
    plt.scatter(grupo1, np.zeros_like(grupo1), color='blue', label='Grupo 1', alpha=0.7)
    plt.scatter(grupo2, np.zeros_like(grupo2), color='red', label='Grupo 2', alpha=0.7)
    plt.xlabel('Valor')
    plt.ylabel('Cluster')
    plt.title('2-Groups separation with k-means')
    plt.legend()
    plt.grid(False)
    plt.show()
    
    # Crear histogramas para cada grupo
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(grupo1, bins=30, color='blue', alpha=0.7)
    sns.histplot(grupo1, bins=30, kde=True, edgecolor='black', alpha=0.7)
    plt.xlabel('Activity value')
    plt.ylabel('Counts')
    plt.title('Group 1 histogram')
    
    plt.subplot(1, 2, 2)
    plt.hist(grupo2, bins=30, color='red', alpha=0.7)
    sns.histplot(grupo2, bins=30, kde=True, edgecolor='black', alpha=0.7)
    plt.xlabel('Activity value')
    plt.ylabel('Counts')
    plt.title('Group 2 histogram')
    
    plt.tight_layout()
    plt.show()



'''Valores de electrodos activados histogram'''

    
def activated_electrodes_distribution(electrodes_data_path):

    
    # Obtener la lista de todas las carpetas dentro del directorio principal
    carpetas_de_electrodos = [e for e in electrodes_data_path.iterdir() if e.is_dir()]
    
    # Definir una lista para almacenar los datos cargados
    all_activated_electrodes_data = []
    electrodo_nombres = []
    
    # Iterar a través de las carpetas de electrodos
    for carpeta in carpetas_de_electrodos:
        # Extraer el número de la carpeta, asumiendo que el nombre de la carpeta es el número
        numero_electrodo = carpeta.name
        
        # Construir el nombre del archivo con el prefijo correspondiente al número del electrodo
        nombre_archivo = carpeta / f"{numero_electrodo}top_electrode_values.npy"
        
        # Comprobar si el archivo existe
        if nombre_archivo.exists():
            # Cargar los datos del archivo y agregarlos a la lista
            datos = np.load(nombre_archivo,  allow_pickle=True)
            all_activated_electrodes_data.append(datos)
            # Extraer el nombre del electrodo de la carpeta y agregarlo a la lista de nombres
            nombre_electrodo = numero_electrodo  # Puedes ajustar esto según cómo estén nombradas tus carpetas
            electrodo_nombres.append(nombre_electrodo)
        else:
            print(f"El archivo {nombre_archivo} no existe.")
    
        
    all_activated_electrodes_data_concatenated = np.concatenate(all_activated_electrodes_data)
    all_activated_electrodes_data_flattened = np.concatenate(all_activated_electrodes_data_concatenated)
    all_activated_electrodes_data_flattened = [x for x in all_activated_electrodes_data_flattened if x >= 1]
    
    
    # Crear un histograma
    plt.figure(figsize=(8, 6))
    plt.hist(all_activated_electrodes_data_flattened, bins=50, alpha=0.7)
    plt.xlabel('Activity value')
    sns.histplot(all_activated_electrodes_data_flattened, bins=50, kde=True, edgecolor='black', alpha=0.7)
    plt.ylabel('Counts')
    plt.title('Histogram for activated electrodes')
    plt.grid(False)
    plt.show()
    
    return all_activated_electrodes_data, electrodo_nombres



'''Valores de electrodos activados histogram por cada electrodo'''

def activated_electrodes_distribution_per_electrode(all_activated_electrodes_data, electrodo_nombres):
    
    
    plt.figure(figsize=(10, 6))
    
    # Colores para las curvas de los electrodos
    colores = sns.color_palette("husl", len(all_activated_electrodes_data))
    
    for datos, nombre, color in zip(all_activated_electrodes_data, electrodo_nombres, colores):
        # Flatten y filtra los datos
        datos_flattened = np.concatenate(datos)
        datos_flattened = [x for x in datos_flattened if x >= 1]
        
        # Crea un histograma de densidad de probabilidad (KDE)
        sns.kdeplot(datos_flattened, color=color, label=nombre)
    
        # Anota el número del electrodo encima de la línea
        max_density = max(datos_flattened)  # Puedes ajustar esto según tus datos
        plt.annotate(nombre, xy=(max_density, 0), xytext=(5, 5), textcoords='offset points', color=color, fontsize=8)
    
    plt.xlabel('Activity value')
    plt.ylabel('Density')
    plt.title('Density Plot for Activated Electrodes')
    plt.legend(title='Electrode')
    plt.grid(False)
    plt.show()
    
def histogram_intersections(all_data, intersections):
    plt.figure()

    # Suponiendo que all_data es tu conjunto de datos
    all_data = all_data.reshape(-1, 1)  # Concatena y da forma a los datos
    plt.hist(all_data.flatten(), bins=100, alpha=0.4, label='Datos', density=False)

    plt.axvline(intersections[0], color='red', linestyle='--', label=f'Intersection {intersections[0]}')
    plt.axvline(intersections[1], color='red', linestyle='--', label=f'Intersection {intersections[1]}')
    

# def extract_basal_gaussian(all_data):
    
#     plt.figure()
#     data = all_data.reshape(-1, 1)  # Concatena y da forma a los datos
    
#     mu, std = norm.fit(data)  # Esto debería ajustarse solo a la parte basal idealmente

#     # Definición de umbrales
#     threshold_low = mu - 1.5*std
#     threshold_high = mu + 1.5*std
    
#     # Clasificación de los datos
#     basal = data[(data > threshold_low) & (data < threshold_high)]
#     activated = data[data > threshold_high]
#     inhibited = data[data < threshold_low]
    
#     # Visualización
#     plt.hist(data, bins=100, alpha=0.6, color='g', label='Total Distribution')
#     plt.hist(basal, bins=100, alpha=0.6, color='b', label='Basal Activity')
#     plt.hist(activated, bins=100, alpha=0.6, color='r', label='Activated')
#     plt.hist(inhibited, bins=100, alpha=0.6, color='y', label='Inhibited')
#     plt.axvline(threshold_low, color='k', linestyle='dashed', linewidth=1)
#     plt.axvline(threshold_high, color='k', linestyle='dashed', linewidth=1)
#     plt.legend()
        

def temporal_pattern(files, times, electrode_mapping, intensity_values,intensity):
    
    fig, axs = plt.subplots(1, len(times), figsize=(20, 5))  

    n_rows, n_cols = 10, 10
    # Define the color map and the normalization
    cmap = ListedColormap(['slategray', 'lightsteelblue', 'cornflowerblue'])
    bounds = [0, 1, 2, 3]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)


    for i, (data, time) in enumerate(zip(intensity_values, times)):
        # Crear una matriz vacía con NaNs
        matrix = np.full((n_rows, n_cols), np.nan)
        unique_labels = set()

        # Rellenar la matriz con los datos etiquetados usando el mapeo de electrodos
        for j, label in enumerate(data):
            row, col = electrode_mapping[str(j+1)]  # asumiendo que los electrodos están etiquetados del 1 al 96
            matrix[row, col] = label
            unique_labels.add(label)

        # Imprimir la matriz antes de visualizarla
        # print("Matrix for Time", i)

        # print(matrix)
        # Colores para la visualización
        if unique_labels == {0, 1}:
            cmap = ListedColormap(['slategray', 'lightsteelblue'])  # Solo dos colores
        elif unique_labels == {1, 2}:
            cmap = ListedColormap(['lightsteelblue', 'cornflowerblue'])   # Solo dos colores
        else:
            cmap = ListedColormap(['slategray', 'lightsteelblue', 'cornflowerblue'])  # Tres colores
       
        # Mostrar la matriz en el subplot correspondiente
        axs[i].imshow(matrix, cmap=cmap, interpolation='nearest')
        axs[i].axis('off')  # Ocultar los ejes
        axs[i].set_title(f'Time {time} ms')  # Poner título con el tiempo correspondiente

    # Create an axes for color bar
    cbar_ax = fig.add_axes([0.04, 0.3, 0.1, 0.02])  # Adjust the position and size as needed
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), 
                      cax=cbar_ax, 
                      orientation='horizontal',
                      ticks=[0.5, 1.5, 2.5])
    cb.ax.set_xticklabels(['Inhibition', 'Basal', 'Activation'])  # Set the labels

    # Annotate each subplot with the corresponding time
    for i, (ax, time) in enumerate(zip(axs, times)):
        ax.set_title(f'{time} ')
        
    fig.suptitle('Time window (ms)', fontsize=15, y=0.7, x=0.17)  # Adjust the y position as needed
    plt.title(f'Pattern evolution in Intensity {intensity} uA', y =28, x = 3, fontsize = 18)

    plt.tight_layout(rect=[0, 0.01, 1, 0.95])
    plt.show()
    

    
def plot_percentages(percentages):
    
    all_data = np.concatenate(percentages)
    # Inicializar diccionarios para almacenar resultados
    mean_dict = {}
    std_dict = {}
    
    # Lista de claves únicas (intensidades de estimulación)
    keys = np.unique(all_data['key'])
    
    # Calcular las estadísticas para cada key
    for key in keys:
        mask = all_data['key'] == key
        mean_dict[key] = {
            'data1_mean': np.mean(all_data[mask]['data1'], axis=0),
            'data2_mean': np.mean(all_data[mask]['data2'], axis=0),
            'data3_mean': np.mean(all_data[mask]['data3'], axis=0),
            'data1_std': np.std(all_data[mask]['data1'], axis=0),
            'data2_std': np.std(all_data[mask]['data2'], axis=0),
            'data3_std': np.std(all_data[mask]['data3'], axis=0)
        }
        
    times = np.arange(1, 7)  # 6 momentos temporales, ajustar si tienes otros tiempos específicos

    percent_inhibition = []
    std_inhibition = []
    percent_basal = []
    std_basal = []
    percent_activation = []
    std_activation = []

    # Calcular medias y desviaciones estándar para cada tipo de dato en cada tiempo
    for t in range(6):  # 6 momentos temporales
        percent_inhibition.append(np.mean(all_data['data1'][:, t]))
        std_inhibition.append(np.std(all_data['data1'][:, t]))
        percent_basal.append(np.mean(all_data['data2'][:, t]))
        std_basal.append(np.std(all_data['data2'][:, t]))
        percent_activation.append(np.mean(all_data['data3'][:, t]))
        std_activation.append(np.std(all_data['data3'][:, t]))

    # Creación de la figura y los ejes
    plt.figure(figsize=(10, 6))
    plt.errorbar(times, percent_inhibition, yerr=std_inhibition, fmt='-o', color='slategray', label='Inhibition', capsize=5)
    plt.errorbar(times, percent_basal, yerr=std_basal, fmt='-o', color='lightsteelblue', label='Basal activity', capsize=5)
    plt.errorbar(times, percent_activation, yerr=std_activation, fmt='-o', color='cornflowerblue', label='Activation', capsize=5)

    # Configuración de los ejes y la grilla
    plt.xticks(times)
    plt.xlabel('Time (ms)')
    plt.ylabel('Percentage (%)')
    plt.title('Percentage evolution along time with error bars')
    plt.legend()
    plt.grid(False)
        