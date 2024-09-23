from utils import imports, visuals, clear
import os

from utils.visuals import normalize_

filepath = os.path.join('data', 'diabetes.tab.txt')
filepath2 = os.path.join('data', 'conjunto_de_datos_natalidad_2022.csv') # Estamos usando el conjunto_de_datos/conjunto_de_datos_nataidad_2020.csv

if __name__ == '__main__':
    clear.clear_outputs('scatterplots')
    data = imports.read_data(filepath)
    dataINE = imports.read_dataNO(filepath2)
    visuals.generate_histogram(data, 'AGE')
    visuals.generate_histogram(data)
    correlations = imports.correlations(data)
    visuals.generate_scatterplot(correlations)
    #visuals.normalize_(data)
    imports.linear_regresion(visuals.normalize_(data), data)

    visuals.generate_heatmap(dataINE)

    
    