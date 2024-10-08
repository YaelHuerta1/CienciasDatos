import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt


def read_data(file_path):
    return pd.read_csv(file_path, sep='\t')

def read_dataNO(file_path):
    return pd.read_csv(file_path)

def correlations(data):
    return data.corr()

def linear_regresion(data_normalized, data):

    X = data_normalized
    y = data['Y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=442)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R2 Score: {r2}')

def logistic_regresion(data_normalized):
    output_path = f'output/heatmaps/logic_heat/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    X = data_normalized[['mean radius', 'mean texture', 'mean perimeter', 'mean area']]
    y = data_normalized['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=30)

    scaler = StandardScaler()
    X_train_Scaled = scaler.fit_transform(X_train)
    X_test_Scaled = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_Scaled, y_train)

    y_pred = model.predict(X_test_Scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precision: {accuracy}")

    cm = confusion_matrix(y_test, y_pred)
    print("Matriz de confusion:\n", cm)

    print(f"Reporte de Clasificación:\n", classification_report(y_test, y_pred))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Clase 0', 'Clase 1'],
                yticklabels=['Clase 0', 'Clase 1'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - Logistic Regression')
    plt.savefig(f'{output_path}logicRegresion_heatmap.png')
    plt.close()


def load_data_sets():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df
