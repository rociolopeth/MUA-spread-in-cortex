# -*- coding: utf-8 -*-
"""
Created on Fri May  3 10:37:01 2024

@authors: Rocio Lopez Peco
@email: yrociro@gmail.es
"""
import numpy as np
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import glob



def labeled_values(all_values,intersections):
    all_values_labels = []


    for values in all_values:
        # np.digitize devuelve índices de los intervalos a los que pertenecen los valores
        # bins se define como [-np.inf] + intersections + [np.inf] para abarcar todos los posibles valores
        labels = np.digitize(values, bins=[-np.inf] + intersections + [np.inf]) - 1
        # Restamos 1 porque np.digitize comienza la cuenta en 1
        all_values_labels.append(labels)
        
    return all_values_labels


def extract_intensity_from_filename(filename):
    pattern = r"values_\d+_(\d+)_\d+_\d+.npy"
    match = re.match(pattern, filename)
    if match:
        intensity = int(match.group(1))  # Convertimos a entero para manejar como número
        return intensity
    
    return None

def organize_data_by_intensity(files, all_values):
    data_dict = defaultdict(list)
    for file, values in zip(files, all_values):
        intensity = extract_intensity_from_filename(file)
        if intensity is not None:
            data_dict[intensity].append(values)
    
    return dict(data_dict)



def extract_window_from_filename(filename, fs):
    pattern = r"values_\d+_\d+_\d+_(\d+).npy"
    match = re.match(pattern, filename)
    if match:
        num_samples = int(match.group(1))  # Extraemos el número de muestras del nombre del archivo
        # Calculamos el tiempo en milisegundos usando la frecuencia de muestreo
        window_time_ms = (num_samples / fs) * 1000  # Convertimos muestras a tiempo en ms (30000 muestras/seg)
        return int(window_time_ms)  # Retornamos como entero, si es necesario
    
    return None

def extract_window_time(files, fs):
    times = []
    for file in files:
        window_time =  extract_window_from_filename(file, fs)
        times.append(window_time)
        
    times = sorted(set(times))
        
    return times
from scipy.stats import norm

    
# def extract_inhibition_activation(all_data, otsu_threshold, min_value):
    
#     data = all_data.reshape(-1, 1)  # Concatena y da forma a los datos
#     inhibited_data = data[data < min_value]
#     activated_data = data[data > otsu_threshold]
#     basal_data = data[(data > min_value) & (data < otsu_threshold)]

    
#     # Ajuste de gaussianas a cada conjunto de datos
#     mu_basal, std_basal = norm.fit(basal_data)
#     mu_activ, std_activ = norm.fit(activated_data)
#     mu_inhib, std_inhib = norm.fit(inhibited_data)
    
#     # Estimación de las distribuciones gaussianas ajustadas
#     x_basal = np.linspace(basal_data.min(), basal_data.max(), 100)
#     pdf_basal = norm.pdf(x_basal, mu_basal, std_basal)
    
#     x_activ = np.linspace(activated_data.min(), activated_data.max(), 100)
#     pdf_activ = norm.pdf(x_activ, mu_activ, std_activ)
    
#     x_inhib = np.linspace(inhibited_data.min(), inhibited_data.max(), 100)
#     pdf_inhib = norm.pdf(x_inhib, mu_inhib, std_inhib)
    
#     # Visualización de los ajustes de distribuciones gaussianas
#     plt.figure(figsize=(12, 6))
#     plt.hist(data, bins=100, alpha=0.3, label='Datos totales', density=True)
#     plt.plot(x_basal, pdf_basal, 'r-', label='Gaussiana Basal')
#     plt.plot(x_activ, pdf_activ, 'g-', label='Gaussiana Activación')
#     plt.plot(x_inhib, pdf_inhib, 'b-', label='Gaussiana Inhibición')
#     plt.legend()
#     plt.show()
    
        
def load_values_for_histogram(path_to_load, max_folders, plot_data_histogram = True):

        
    # Obtener la lista de todas las carpetas dentro del directorio principal
    carpetas_de_electrodos = [e for e in path_to_load.iterdir() if e.is_dir()]
    
    # Definir una lista para almacenar los datos cargados
    all_electrodes_data = []
    count = 0
    count_electrodes = 0
    # Iterar a través de las carpetas de electrodos
    for carpeta in carpetas_de_electrodos:
        if count_electrodes < max_folders:
            # Extraer el número de la carpeta, asumiendo que el nombre de la carpeta es el número
            numero_electrodo = carpeta.name[:2]
            
            # Construir el nombre del archivo con el prefijo correspondiente al número del electrodo
            pattern = f"values_{numero_electrodo}_01_100_*.npy"
            archivos_encontrados = glob.glob(str(carpeta / pattern)) 
            print(archivos_encontrados)
            
            if archivos_encontrados:

                for archivo in archivos_encontrados:
                    # Cargar los datos del archivo y agregarlos a la lista
                    datos = np.load(archivo)
                    all_electrodes_data.append(datos)
                    print(f"El archivo {archivo} cargado")
                    count = count +1
                count_electrodes += 1
                print(f"Procesada carpeta {count_electrodes} con {len(archivos_encontrados)} archivos.")
            else:
                print(f"No se encontraron archivos en la carpeta {carpeta.name}.")
        else:
            break  # Salir del bucle si se alcanza el número máximo de carpetas
            
    print(f"Total de carpetas procesadas: {count_electrodes}")
    print(f"Total de archivos cargados: {count}")
    # Combinar todos los datos en un solo arreglo
    all_electrodes_data = np.concatenate(all_electrodes_data)
    
    if plot_data_histogram:
        # Crear un histograma
        plt.figure(figsize=(8, 6))
        plt.hist(all_electrodes_data, bins=100, color='blue', alpha=0.7)
        plt.xlabel('Activity value')
        plt.ylabel('Counts')
        plt.title('Histogram for all data')
        plt.grid(False)
        plt.show()
        
    return all_electrodes_data        
    
    

def characterize_gaussian(data, plot = False):
    # Calcular la media y la desviación estándar de los datos
    mean = np.mean(data)
    std = np.std(data)
    
    # Mostrar los resultados
    print(f"Media de los datos: {mean}")
    print(f"Desviación estándar de los datos: {std}")
    
    
    if plot:
   
        # Dibujar los datos y la curva gaussiana ajustada
        plt.figure(figsize=(8, 6))
        plt.hist(data, bins=100, color='blue', alpha=0.7, density=True)
        plt.title('Fit of Gaussian Distribution')
        plt.xlabel('Activity value')
        plt.axvline(mean+std, color = 'red')
        plt.axvline(mean-std, color = 'red')
        plt.axvline(mean, color = 'black')
        plt.grid(False)
        plt.show()

    return mean, std

def characterize_median_iqr(data, plot = False):
    median = np.median(data)
    q25 = np.percentile(data, 25)
    q75 = np.percentile(data, 75)
    iqr = q75 - q25

    # Definir umbrales utilizando el IQR
    # Ajuste común es usar 1.5 * IQR para detectar valores atípicos
    lower_threshold = q25 - 1.5 * iqr
    upper_threshold = q75 + 1.5 * iqr

    if plot:
        # Visualizar los resultados
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=100, alpha=0.7, label='Data Distribution')
        plt.axvline(x=median, color='red', linestyle='-', label='Median')
        plt.axvline(x=lower_threshold, color='green', linestyle='--', label='Lower Threshold (1.5*IQR)')
        plt.axvline(x=upper_threshold, color='blue', linestyle='--', label='Upper Threshold (1.5*IQR)')
        plt.legend()
        plt.xlabel('Data Values')
        plt.ylabel('Frequency')
        plt.title('Data Analysis with Median and IQR')
        plt.grid(True)
        plt.show()

    return median, iqr, lower_threshold, upper_threshold

def extract_percentage_labels(all_values_labels_dict, intensity, times, max_times, plot_graphs =True):
    
    time_arrays = all_values_labels_dict[intensity]
    percentage_arrays = []
    
    # Procesar cada array de momento temporal
    for array in time_arrays:
        # Inicializar conteos para cada etiqueta en 0
        label_counts = {0: 0, 1: 0, 2: 0}
        
        # Contar la frecuencia de cada etiqueta y actualizar el diccionario
        labels, counts = np.unique(array, return_counts=True)
        label_counts.update(dict(zip(labels, counts)))
        
        # Calcular porcentajes y agregarlos en el orden de las etiquetas
        total_counts = sum(label_counts.values())
        percentages = [label_counts[label] / total_counts * 100 for label in [0, 1, 2]]
        
        # Agregar a la lista como array
        percentage_arrays.append(percentages)

    ### Visualización
    max_times = min(max_times, len(time_arrays))

    
    percent_0 = [round(percentage[0], 2) for percentage in percentage_arrays[:max_times]]  # Inhibición
    percent_1 = [round(percentage[1], 2) for percentage in percentage_arrays[:max_times]]  # Basal
    percent_2 = [round(percentage[2], 2) for percentage in percentage_arrays[:max_times]] # Activación
    
    if plot_graphs: 
        plt.figure(figsize=(10, 6))
        plt.plot(times[:max_times], percent_0, color='slategray', marker='o', linestyle='-', label='Inhibition')
        plt.plot(times[:max_times], percent_1, color='lightsteelblue', marker='o', linestyle='-', label='Basal activity')
        plt.plot(times[:max_times], percent_2, color='cornflowerblue', marker='o', linestyle='-', label='Activation')
        plt.xticks(times[:max_times])
        
        plt.xlabel('Time (sequence index)')
        plt.ylabel('Percentage (%)')
        plt.title(f'Percentage evolution along time for intensity {intensity}')
        plt.legend()
        plt.grid(True)
        plt.show()
                
    return percent_0, percent_1, percent_2


def save_percentages(percentages_dict, mainfolder, folder, subfolder):
    
    
    dtype = [('key', int), ('data1', float, (6,)), ('data2', float, (6,)), ('data3', float, (6,))]
    percentages_array = np.array([(key, *map(np.array, value)) for key, value in percentages_dict.items()], dtype=dtype)
    
    save_folder = mainfolder / 'electrodes_data_evolution' / subfolder
    save_folder.mkdir(parents=True, exist_ok=True)
    electrode_number = subfolder[:2]  # Asume que los dos primeros caracteres son el número
    file_name = f"{electrode_number}_percentages_by_intensity.npy"
    save_path = save_folder / file_name
    
    np.save(save_path, percentages_array)
    print(f"Datos guardados en {save_path}")

    

    
    
    
    
    
    
    
    
    
    
    
    
    