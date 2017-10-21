from proactive_forest.estimator import DecisionTreeClassifier, DecisionForestClassifier, ProactiveForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import time
import pandas as pd

if __name__ == '__main__':

    dataset = pd.read_csv('../data/car.csv')

    data = pd.get_dummies(dataset['buying'], prefix=True)

    data.info()

    y = dataset['class']
    X = dataset.drop('class', axis=1)