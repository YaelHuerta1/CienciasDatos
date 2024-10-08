import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import math

from seaborn import heatmap
from sklearn.datasets import load_breast_cancer

from pandas import pivot_table
import pandas as pd

output_path = 'output/'


def generate_histogram(data, column=None):
    output_path = 'output/histograms/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if column is None:
        num_columns = len(data.columns)
        fig, axes = plt.subplots(num_columns, 1, figsize=(10, num_columns * 5), squeeze=False)

        for i, column in enumerate(data.columns):
            sns.histplot(data[column], kde=True, ax=axes[i, 0])
            axes[i, 0].set_title(f'Histogram of {column}')

        plt.tight_layout()
        plt.savefig(f'{output_path}all_histograms.png')
        plt.close()
    else:
        plt.figure(figsize=(10, 5))
        sns.histplot(data[column], kde=True)
        plt.title(f'Histogram of {column}')
        plt.savefig(f'{output_path}{column}_histogram.png')
        plt.close()


def generate_scatterplot(data, x=None, y=None):
    output_path = 'output/scatterplots/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    num_columns = len(data.columns)

    if x is None and y is None:
        # Genera un gráfico de dispersión para todas las combinaciones de columnas
        fig, axes = plt.subplots(num_columns, num_columns, figsize=(15, 15), squeeze=False)

        for i in range(num_columns):
            for j in range(num_columns):
                if i != j:
                    x_col = data.columns[j]
                    y_col = data.columns[i]
                    sns.scatterplot(x=x_col, y=y_col, data=data, ax=axes[i, j])
                    correlation = data[[x_col, y_col]].corr().iloc[0, 1]
                    axes[i, j].set_title(f'Corr: {correlation:.3f}')
                else:
                    axes[i, j].axis('off')  # Desactiva la diagonal principal
                axes[i, j].set_xlabel('')
                axes[i, j].set_ylabel('')

        plt.tight_layout()
        plt.savefig(f'{output_path}all_scatterplots.png')
        plt.close()
    elif y is None:
        # Genera gráficos de dispersión para todas las columnas en función de x
        num_plots = len(data.columns) - 1
        fig, axes = plt.subplots(num_plots, 1, figsize=(10, num_plots * 5), squeeze=False)

        for i, y_col in enumerate(data.columns):
            if y_col != x:
                sns.scatterplot(x=x, y=y_col, data=data, ax=axes[i, 0])
                correlation = data[[x, y_col]].corr().iloc[0, 1]
                axes[i, 0].set_title(f'Correlation of {x} with {y_col}: {correlation:.3f}')

        plt.tight_layout()
        plt.savefig(f'{output_path}{x}_scatterplots.png')
        plt.close()
    elif x is None:
        # Genera gráficos de dispersión para todas las columnas en función de y
        num_plots = len(data.columns) - 1
        fig, axes = plt.subplots(num_plots, 1, figsize=(10, num_plots * 5), squeeze=False)

        for i, x_col in enumerate(data.columns):
            if x_col != y:
                sns.scatterplot(x=x_col, y=y, data=data, ax=axes[i, 0])
                correlation = data[[x_col, y]].corr().iloc[0, 1]
                axes[i, 0].set_title(f'Correlation of {x_col} with {y}: {correlation:.3f}')

        plt.tight_layout()
        plt.savefig(f'{output_path}{y}_scatterplots.png')
        plt.close()
    else:
        # Genera un solo gráfico de dispersión para las columnas especificadas
        plt.figure(figsize=(10, 5))
        sns.scatterplot(x=x, y=y, data=data)
        correlation = data[[x, y]].corr().iloc[0, 1]
        plt.title(f'Correlation of {x} with {y}: {correlation:.3f}')
        plt.savefig(f'{output_path}{x}_{y}_scatterplot.png')
        plt.close()

def generate_heatmap(data, heat):
    output_path = f'output/heatmaps/{heat}/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    ocurrencias = pd.concat([data['edad_padn'], data['edad_madn']]).value_counts()
    data['ocurrencias_totales'] = data['edad_padn'].map(ocurrencias) + data['edad_madn'].map(ocurrencias)

    ocurren = data['ocurrencias_totales']

    pivot = pd.pivot_table(data, values='ocurrencias_totales', index='edad_padn', columns='edad_madn', aggfunc='mean') # VALUES = NUMERO DE OCURRENCIAS

    plt.figure(figsize=(8,6))
    sns.heatmap(pivot, annot=False, cmap="YlGnBu", cbar=True, linewidths=0.5)
    plt.title(f"Mapa de calor de {heat}")
    plt.savefig(f'{output_path}{heat}_heatmap.png')
    plt.close()

def normalize_ (data):
    columns_to_average = data.columns[:10]

    means = data[columns_to_average].mean()
    std_devs = data[columns_to_average].std()

    normlized_ = data[columns_to_average].apply(lambda x: ((x - means[x.name]) / std_devs[x.name]) * 1 / math.sqrt(442))

    # print(normlized_)
    return normlized_

def checkNormalized_ (dataFrames,):
    mean = dataFrames.mean()
    std = dataFrames.std()

    is_nomalized = np.allclose(mean, 0, atol=0.1) and np.allclose(std, 1, atol=0.1)

    if is_nomalized:
        print("Los datos estan Normalizados")
    else:
       dataNormalized = normalize_(dataFrames)
       return dataNormalized
