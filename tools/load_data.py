# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 16:36:23 2024

@authors: Rocio Lopez Peco
@email: yrociro@gmail.es
"""

import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re



# Mapeo de electrodos a posiciones
electrode_mapping = {             '88':(0,1),'78':(0,2),'68':(0,3),'58':(0,4),'48':(0,5),'38':(0,6),'28':(0,7),'18':(0,8), 
                          '96':(1,0),'87':(1,1),'77':(1,2),'67':(1,3),'57':(1,4),'47':(1,5),'37':(1,6),'27':(1,7),'17':(1,8),'8':(1,9),
                          '95':(2,0),'86':(2,1),'76':(2,2),'66':(2,3),'56':(2,4),'46':(2,5),'36':(2,6),'26':(2,7),'16':(2,8),'7':(2,9),
                          '94':(3,0),'85':(3,1),'75':(3,2),'65':(3,3),'55':(3,4),'45':(3,5),'35':(3,6),'25':(3,7),'15':(3,8),'6':(3,9),
                          '93':(4,0),'84':(4,1),'74':(4,2),'64':(4,3),'54':(4,4),'44':(4,5),'34':(4,6),'24':(4,7),'14':(4,8),'5':(4,9),
                          '92':(5,0),'83':(5,1),'73':(5,2),'63':(5,3),'53':(5,4),'43':(5,5),'33':(5,6),'23':(5,7),'13':(5,8),'4':(5,9),
                          '91':(6,0),'82':(6,1),'72':(6,2),'62':(6,3),'52':(6,4),'42':(6,5),'32':(6,6),'22':(6,7),'12':(6,8),'3':(6,9),
                          '90':(7,0),'81':(7,1),'71':(7,2),'61':(7,3),'51':(7,4),'41':(7,5),'31':(7,6),'21':(7,7),'11':(7,8),'2':(7,9),
                          '89':(8,0),'80':(8,1),'70':(8,2),'60':(8,3),'50':(8,4),'40':(8,5),'30':(8,6),'20':(8,7),'10':(8,8),'1':(8,9),
                                     '79':(9,1),'69':(9,2),'59':(9,3),'49':(9,4),'39':(9,5),'29':(9,6),'19':(9,7),'9':(9,8)}




def load_values(path_to_load, path_to_save, save_all_data = False):

    
    all_values = []
    files = []
    
    # Iterar sobre todos los archivos en el directorio
    for file in path_to_load.iterdir():
        if file.suffix == '.npy':
            values = np.load(file)
            all_values.append(values)
            print(file.name)
            files.append(file.name)
            
    all_data = np.concatenate(all_values)  # Concatena todos los arrays en un solo array
    
    if save_all_data:
        filename = 'all_data.npy'
        ruta = str(path_to_save) +  filename
        np.save(ruta, all_data)

            
    return all_values, all_data, files


def load_all_data(electrodes_data_path, plot_data_histogram = True):
        
    
    # Obtener la lista de todas las carpetas dentro del directorio principal
    carpetas_de_electrodos = [e for e in electrodes_data_path.iterdir() if e.is_dir()]
    
    # Definir una lista para almacenar los datos cargados
    all_electrodes_data = []
    
    # Iterar a través de las carpetas de electrodos
    for carpeta in carpetas_de_electrodos:
        # Extraer el número de la carpeta, asumiendo que el nombre de la carpeta es el número
        numero_electrodo = carpeta.name
        
        # Construir el nombre del archivo con el prefijo correspondiente al número del electrodo
        nombre_archivo = carpeta / f"{numero_electrodo}all_data.npy"
        
        # Comprobar si el archivo existe
        if nombre_archivo.exists():
            # Cargar los datos del archivo y agregarlos a la lista
            datos = np.load(nombre_archivo)
            all_electrodes_data.append(datos)
        else:
            print(f"El archivo {nombre_archivo} no existe.")
    
    # Combinar todos los datos en un solo arreglo
    all_electrodes_data = np.concatenate(all_electrodes_data)
    
    if plot_data_histogram:
        # Crear un histograma
        plt.figure(figsize=(8, 6))
        plt.hist(all_electrodes_data, bins=50, color='blue', alpha=0.7)
        plt.xlabel('Activity value')
        sns.histplot(all_electrodes_data, bins=50, kde=True, edgecolor='black', alpha=0.7)
        plt.ylabel('Counts')
        plt.title('Histogram for all data')
        plt.grid(False)
        plt.show()
        
    return all_electrodes_data
        
        
def load_binary_data(electrodes_data_path, filename_pattern):
    
    
    carpetas_de_electrodos = [e for e in electrodes_data_path.iterdir() if e.is_dir()]
    
    all_binary_data = []
    stim_electrodes = []
            
    for carpeta in carpetas_de_electrodos:
        numero_electrodo = carpeta.name

        # Buscar todos los archivos que coincidan con el patrón 'binary_image_*.npy'
        archivos = list(carpeta.glob(filename_pattern.format(numero_electrodo=numero_electrodo)))
        
        # Iterar a través de los archivos encontrados
        for archivo in archivos:
            # Cargar los datos del archivo y agregarlos a la lista
            datos = np.load(archivo, allow_pickle=True)
            all_binary_data.append(datos)
            # Agregar el número del electrodo a la lista de nombres
            stim_electrodes.append(numero_electrodo)
            
            
    electrode_data_map = {}
    
    # Inicializar el diccionario con listas vacías
    for electrode in set(stim_electrodes):
        electrode_data_map[electrode] = []
    
    # Asignar las matrices a su respectivo electrodo
    for i, electrode in enumerate(stim_electrodes):
        electrode_data_map[electrode].append(all_binary_data[i])
    
    # Asegurarse de que cada lista contenga solo diez matrices
    for electrode, data in electrode_data_map.items():
        electrode_data_map[electrode] = data[:10]

            
    return all_binary_data, sorted(set(stim_electrodes)), electrode_data_map
    

def load_otsu_threshold(path_to_load, subfolder):
    
  # Obtener el número de electrodo del nombre de la carpeta subfolder
    numero_electrodo = subfolder
    
    # Construir la ruta completa al archivo
    nombre_archivo = path_to_load / f"{numero_electrodo}otsu_threshold.npy"
    
    # Comprobar si el archivo existe
    if nombre_archivo.exists():
        # Cargar los datos del archivo y agregarlos a la lista
        otsu_threshold = np.load(nombre_archivo, allow_pickle=True)
        
    else:
        print(f"El archivo {nombre_archivo} no existe.")
    
    return otsu_threshold
    

def load_data(path_to_load, subfolder):
    
  # Obtener el número de electrodo del nombre de la carpeta subfolder
    numero_electrodo = subfolder
    
    # Construir la ruta completa al archivo
    nombre_archivo = path_to_load / f"{numero_electrodo}all_data.npy"
    
    # Comprobar si el archivo existe
    if nombre_archivo.exists():
        # Cargar los datos del archivo y agregarlos a la lista
        values = np.load(nombre_archivo, allow_pickle=True)
        
    else:
        print(f"El archivo {nombre_archivo} no existe.")
    
    return values
    

def load_electrodes_data_evolution(path_to_load):
    
    carpetas_de_electrodos = [e for e in path_to_load.iterdir() if e.is_dir()]
    
    percentages = []
    stim_electrodes = []
            
    for carpeta in carpetas_de_electrodos:
        numero_electrodo = carpeta.name[:2]

        # Buscar todos los archivos que coincidan con el patrón 'binary_image_*.npy'
        percentage_file = carpeta / f"{numero_electrodo}_percentages_by_intensity.npy"

        # Comprobar si el archivo existe
        if percentage_file.exists():
            # Cargar los datos del archivo y agregarlos a la lista
            datos = np.load(percentage_file, allow_pickle = True)
            percentages.append(datos)
        else:
            print(f"El archivo {percentage_file} no existe.")
            
        
    
    return percentages, stim_electrodes



def load_values_couples(path_to_load):
    
        
    # Regular expression to match electrode pairs in the format ['83', '70']
    pattern = re.compile(r"\['(\d+)', '(\d+)'\]")
    
    # Diccionario para almacenar los valores de cada par de electrodos
    electrode_data = {}
    pair_keys = []
    
    # Iterar sobre todos los archivos en el directorio
    for file in path_to_load.glob('**/values*.npy'):
        # Use la expresión regular para buscar el par de electrodos en el nombre del archivo
        match = pattern.search(file.stem)  # Usamos file.stem para obtener el nombre del archivo sin la extensión
        if match:
            # Si se encuentra una coincidencia, extrae los números de los electrodos
            electrodes = match.groups()
            
            # Ordena el par de electrodos para que '83' siempre aparezca primero
            sorted_electrodes = sorted(electrodes, key=lambda x: x != '83')
            pair_key = f"{sorted_electrodes[0]}_{sorted_electrodes[1]}"
            pair_keys.append(pair_key)
            # Carga los datos del archivo
            values = np.load(file)
            
            # Si el par de electrodos ya está en el diccionario, agrega los datos a la lista existente
            if pair_key in electrode_data:
                electrode_data[pair_key].append(values)
            else:
                # Si el par de electrodos no está en el diccionario, inicia una nueva lista
                electrode_data[pair_key] = [values]
    sorted(set(pair_keys))
    
    return electrode_data, sorted(set(pair_keys))