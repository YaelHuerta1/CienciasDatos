
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from utils.visuals import normalize_


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