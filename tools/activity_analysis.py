# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 16:17:52 2024

@authors: Rocio Lopez Peco
@email: yrociro@gmail.es
"""


import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import f_oneway
import pandas as pd
from statsmodels.stats.multicomp import MultiComparison
import seaborn as sns



'''Areas analysis'''

def areas_analysis(directory, all_activated_electrodes_data,electrodo_nombres, intensities, plot_scatter = True, plot_errorbar =  True):
    
    
    # Obtener la lista de todas las carpetas dentro del directorio principal
    carpetas_de_electrodos = [e for e in directory.iterdir() if e.is_dir()]
    
    # Definir una lista para almacenar los datos cargados
    active_areas_mm2 = []
    
    
    # Iterar a través de las carpetas de electrodos
    for carpeta in carpetas_de_electrodos:
        # Extraer el número de la carpeta, asumiendo que el nombre de la carpeta es el número
        numero_electrodo = carpeta.name
        
        # Construir el nombre del archivo con el prefijo correspondiente al número del electrodo
        nombre_archivo = carpeta / f"{numero_electrodo}active_areas_mm2.npy"
        
        # Comprobar si el archivo existe
        if nombre_archivo.exists():
            # Cargar los datos del archivo y agregarlos a la lista
            datos = np.load(nombre_archivo,  allow_pickle=True)
            active_areas_mm2.append(datos)
        else:
            print(f"El archivo {nombre_archivo} no existe.")
            
            
             
    if plot_scatter:
        plt.figure()
        
        
        # Definir un colormap
        colormap = plt.cm.nipy_spectral  # Este colormap tiene una amplia gama de colores
        num_electrodos = len(electrodo_nombres)
        colors = colormap(np.linspace(0, 1, num_electrodos))
        
        
        # Crear un gráfico de dispersión con colores distintos para cada electrodo
        for (nombre_electrodo, media_actividad_electrodo, color) in zip(electrodo_nombres, active_areas_mm2, colors):
            plt.scatter(intensities, media_actividad_electrodo, color=color, label=nombre_electrodo)
            plt.plot(intensities, media_actividad_electrodo, marker='o', color=color)  # Se asume que cada línea tiene un único color
        
        # Configurar etiquetas y título
        plt.xlabel('Stimulation intensity')
        plt.ylabel('Area of activation (mm2)')
        plt.title('Area of activation')
        plt.xticks(intensities)
        
        # Mostrar leyenda
        plt.legend()
        
        
        
    if plot_errorbar:
        
        plt.figure()
        

        # Calcular la media y la desviación estándar de todas las medias de actividad
        media_total = np.mean(active_areas_mm2, axis=0)
        desviacion_estandar_total = np.std(active_areas_mm2, axis=0)
        
        # Crear un gráfico de barras para la media y la desviación estándar
        plt.errorbar(intensities, media_total, yerr=desviacion_estandar_total, fmt='o-', capsize=5, label='Mean ± Standard deviation')
        plt.xlabel('Stimulation Intensity (uA)')
        plt.ylabel('Area of activation (mm2)')
        plt.title('Activation area in different intensities of stimulation')
        plt.legend()
        plt.grid(False)
        plt.xticks(intensities)
        plt.show()
    
        
    active_areas_data_map = {}

    for i, electrode in enumerate(electrodo_nombres):
        active_areas_data_map[electrode] = active_areas_mm2[i]

    # Imprimir para verificar que todo está guardado como se espera
    for key, value in active_areas_data_map.items():
        print(key, value)
        
    
    return active_areas_mm2, active_areas_data_map
        
'''Estadistica'''

def areas_analysis_statistics(intensities, active_areas_mm2):


    areas_intensity_data = {}

    # Iterar sobre la lista de intensidades
    for index, intensity in enumerate(intensities):
        # Crear una lista para cada intensidad y guardarla en el diccionario
        areas_intensity_data[intensity] = [electrodo[index] for electrodo in active_areas_mm2]
        
    
    # Realizar ANOVA:
    intensity_areas = list(areas_intensity_data.values())
    f_stat, p_value = f_oneway(*intensity_areas)
    print(f"ANOVA F-stat: {f_stat}, P-value: {p_value}")  
    
    
    # Si el p-value es significativo, proceder con pruebas post-hoc
    if p_value < 0.05:
        # Combinar todos los datos en una sola lista y crear etiquetas correspondientes
        data = np.concatenate(intensity_areas)
        labels = []
        for intensity in intensities:
            labels += [intensity] * len(areas_intensity_data[intensity])
        
        # Realizar la comparación múltiple
        mc = MultiComparison(data, labels)
        result = mc.tukeyhsd()
        print(result.summary())
    
        # Para el Bonferroni específicamente:
        bonf_result = mc.allpairtest(f_oneway, method='bonf')
        print(bonf_result[0])
        
        simple_table = bonf_result[0]

        # Extraer los datos de la SimpleTable y convertir a DataFrame
        data_rows = simple_table.data[1:]  # Excluir el encabezado
        header = simple_table.data[0]  # Los encabezados de la tabla
        bonf_df = pd.DataFrame(data_rows, columns=header)
        
        # Convertir las columnas necesarias a tipo numérico. Si conoces qué columnas son, puedes hacerlo directamente.
        bonf_df = bonf_df.apply(pd.to_numeric, errors='ignore')
        
        # Ahora puedes configurar los nombres de las columnas según corresponda
        bonf_df.columns = ['group1', 'group2', 'stat', 'pval', 'pval_corr', 'reject']
        
        # Asegurarse de que las columnas 'group1' y 'group2' sean enteros si esos son los identificadores de los grupos
        bonf_df['group1'] = bonf_df['group1'].astype(int)
        bonf_df['group2'] = bonf_df['group2'].astype(int)
        bonf_df['pval_corr'] = bonf_df['pval_corr'].astype(float)  # Asegurarse de que 'pval_corr' sea float
        
        return bonf_df


# '''en un heatmap para que se vea mas claro''' queda pendiente

            
        
def amount_activity(electrodes_data_path, electrodo_nombres, intensities, plot_scatter= True, plot_errorbar = True):

    top_electrode_values_mean = []
    
    # Obtener la lista de todas las carpetas dentro del directorio principal
    carpetas_de_electrodos = [e for e in electrodes_data_path.iterdir() if e.is_dir()]
     
    # Iterar a través de las carpetas de electrodos
    for carpeta in carpetas_de_electrodos:
        # Extraer el número de la carpeta, asumiendo que el nombre de la carpeta es el número
        numero_electrodo = carpeta.name
        
        # Construir el nombre del archivo con el prefijo correspondiente al número del electrodo
        nombre_archivo = carpeta / f"{numero_electrodo}top_electrode_values_mean.npy"
        
        # Comprobar si el archivo existe
        if nombre_archivo.exists():
            # Cargar los datos del archivo y agregarlos a la lista
            datos = np.load(nombre_archivo,  allow_pickle=True)
            top_electrode_values_mean.append(datos)
        else:
            print(f"El archivo {nombre_archivo} no existe.")
            
    # Supongamos que 'top_electrode_values_mean' es tu lista de arrays
    for i in range(len(top_electrode_values_mean)):
        top_electrode_values_mean[i] = np.nan_to_num(top_electrode_values_mean[i])

    
    if plot_scatter:
        
        plt.figure()
        
        # Definir un colormap
        colormap = plt.cm.nipy_spectral  # Este colormap tiene una amplia gama de colores
        num_electrodos = len(electrodo_nombres)
        colors = colormap(np.linspace(0, 1, num_electrodos))
        
        # Crear un gráfico de dispersión con colores distintos para cada electrodo
        for (nombre_electrodo, media_actividad_electrodo, color) in zip(electrodo_nombres, top_electrode_values_mean, colors):
            plt.scatter(intensities, media_actividad_electrodo, color=color, label=nombre_electrodo)
            plt.plot(intensities, media_actividad_electrodo, marker='o', color=color)  # Se asume que cada línea tiene un único color
        
        # Configurar etiquetas y título
        plt.xlabel('Stimulation intensity')
        plt.ylabel('Media de Actividad')
        plt.title('Actividad vs. Intensidad para Electrodos')
        plt.xticks(intensities)
        
        # Mostrar leyenda
        plt.legend()

        
    if plot_errorbar:
        
        plt.figure()
            
        # Calcular la media y la desviación estándar de todas las medias de actividad
        media_total = np.mean(top_electrode_values_mean, axis=0)
        media_total = np.nan_to_num(media_total)
        
        desviacion_estandar_total = np.std(top_electrode_values_mean, axis=0)
        desviacion_estandar_total = np.nan_to_num(desviacion_estandar_total)
        
        
        # Crear un gráfico de barras para la media y la desviación estándar
        plt.errorbar(intensities, media_total, yerr=desviacion_estandar_total, fmt='o-', capsize=5, label='Mean ± Standard deviation')
        plt.xlabel('Stimulation Intensity (uA)')
        plt.ylabel('Activity value (Frequency Hz)')
        plt.title('Amount of activation in different intensities of stimulation')
        plt.legend()
        plt.grid(False)
        plt.xticks(intensities)
        
        return top_electrode_values_mean
                
                
            
def amount_analysis_statistics(intensities, top_electrode_values_mean):
    
    
    amount_intensity_data = {}

    # Iterar sobre la lista de intensidades
    for index, intensity in enumerate(intensities):
        # Crear una lista para cada intensidad y guardarla en el diccionario
        amount_intensity_data[intensity] = [electrodo[index] for electrodo in top_electrode_values_mean]
        
    
    # Realizar ANOVA:
    intensity_areas = list(amount_intensity_data.values())
    f_stat, p_value = f_oneway(*intensity_areas)
    print(f"ANOVA F-stat: {f_stat}, P-value: {p_value}")  
    # Inicializar DataFrame para resultados Bonferroni
    bonf_df = pd.DataFrame()
    
    
    # Si el p-value es significativo, proceder con pruebas post-hoc
    if p_value < 0.05:
        # Combinar todos los datos en una sola lista y crear etiquetas correspondientes
        data = np.concatenate(intensity_areas)
        labels = []
        for intensity in intensities:
            labels += [intensity] * len(amount_intensity_data[intensity])
        
        # Realizar la comparación múltiple
        mc = MultiComparison(data, labels)
        result = mc.tukeyhsd()
        print(result.summary())
    
        # Para el Bonferroni específicamente:
        bonf_result = mc.allpairtest(f_oneway, method='bonf')
        print(bonf_result[0])
        simple_table = bonf_result[0]

        # Extraer los datos de la SimpleTable y convertir a DataFrame
        data_rows = simple_table.data[1:]  # Excluir el encabezado
        header = simple_table.data[0]  # Los encabezados de la tabla
        bonf_df = pd.DataFrame(data_rows, columns=header)
        
        # Convertir las columnas necesarias a tipo numérico. Si conoces qué columnas son, puedes hacerlo directamente.
        bonf_df = bonf_df.apply(pd.to_numeric, errors='ignore')
        
        # Ahora puedes configurar los nombres de las columnas según corresponda
        bonf_df.columns = ['group1', 'group2', 'stat', 'pval', 'pval_corr', 'reject']
        
        # Asegurarse de que las columnas 'group1' y 'group2' sean enteros si esos son los identificadores de los grupos
        bonf_df['group1'] = bonf_df['group1'].astype(int)
        bonf_df['group2'] = bonf_df['group2'].astype(int)
        bonf_df['pval_corr'] = bonf_df['pval_corr'].astype(float)  # Asegurarse de que 'pval_corr' sea float
                



    return bonf_df
  
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches


      
def statistics_heatmap(bonf_df):

    
    # Asegúrate de que 'group1' y 'group2' son numéricos y de tipo int, y 'pval_corr' es un float
    bonf_df['group1'] = bonf_df['group1'].astype(int)
    bonf_df['group2'] = bonf_df['group2'].astype(int)
    bonf_df['pval_corr'] = bonf_df['pval_corr'].astype(float)
    
    # Paso 1: Crear una matriz cuadrada con 'group1' y 'group2' como índices y columnas
    unique_groups = sorted(set(bonf_df['group1']).union(bonf_df['group2']))
    heatmap_matrix = pd.DataFrame(index=unique_groups, columns=unique_groups, data=np.nan)  # Inicializar con NaN
    
    # Paso 2: Rellenar la matriz con los valores p ajustados
    for _, row in bonf_df.iterrows():
        g1 = row['group1']
        g2 = row['group2']
        pval = row['pval_corr']
        
        heatmap_matrix.loc[g1, g2] = pval
        heatmap_matrix.loc[g2, g1] = pval  # Espejar la matriz para tener simetría
    

    # Definir una función para aplicar el color según los rangos de valores p
    def apply_color_to_values(val):
        if val < 0.001:
            return 'darkgoldenrod'
        elif val < 0.002:
            return 'goldenrod'
        elif val < 0.01:
            return 'palegoldenrod'
        elif val < 0.05:
            return 'white'
        else:
            return 'darkcyan'
    
    # Aplicar la función a los valores del DataFrame para obtener una matriz de colores
    color_matrix = heatmap_matrix.applymap(apply_color_to_values)
    
    # Crear una máscara para la parte superior de la matriz
    mask = np.triu(np.ones_like(heatmap_matrix, dtype=bool))
    
    # Configurar el tamaño del gráfico
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Dibujar el heatmap con los colores asignados
    sns.heatmap(heatmap_matrix, mask=mask, annot=True, fmt=".3f",
                linewidths=.5, square=True, ax=ax, cbar=False,
                annot_kws={'color': 'black'})
    
    # Aplicar los colores de la matriz de colores a las celdas
    for i, j in zip(*np.where(~mask)):
        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color=color_matrix.iloc[i, j]))
        
        
   
    
    # Poner títulos y etiquetas
    plt.title('Mapa de Calor de Valores P Ajustados (Bonferroni)')
    plt.xlabel('Group 1')
    plt.ylabel('Group 2')
    
    # Crear la leyenda con parches
    legend_elements = [mpatches.Patch(color='darkgoldenrod', label='<0.001'),
                   mpatches.Patch(color='palegoldenrod', label='0.001–0.01'),
                   mpatches.Patch(color='white', label='0.01–0.05'),
                   mpatches.Patch(color='darkcyan', label='>0.05')]

    # Añadir la leyenda al gráfico
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=len(legend_elements))
    
    # Invertir el eje y para que la parte triangular superior esté vacía
    
    # Mostrar el gráfico
    plt.show()
    
    return heatmap_matrix