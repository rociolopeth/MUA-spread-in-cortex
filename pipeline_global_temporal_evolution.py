# -*- coding: utf-8 -*-
"""
Created on Tue May  7 12:56:51 2024

@authors: Rocio Lopez Peco
@email: yrociro@gmail.es
"""

from tools.load_data import load_electrodes_data_evolution
from pathlib import Path

mainfolder = Path('C:/Users/yroci/Desktop/miguel_spread/')
folder = 'electrodes_data_evolution'

path_to_load = mainfolder / folder

percentages, stim_electrodes = load_electrodes_data_evolution(path_to_load)


#%%
from tools.visualisations import plot_percentages

plot_percentages(percentages)