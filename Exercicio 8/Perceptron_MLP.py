import numpy as np
import pandas as pd
import plot
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split

def main():
    print("a")
    # carrega dados da base de iris
    X, y = load_iris(return_X_y=True)

    #data=iris.data
    #features = iris.feature_names
    #print(iris.target_names)
    #iris_df = pd.DataFrame(data, columns = features)
    #iris_df['class'] = iris['target']   
#
    ## Definição das dimensões e dos respecticos resutantes
    #X = iris_df.iloc[:,0:4].values
    #y = iris_df['class'].values
#
    ##Divisão do dataset entre dados para Treinamento e para Teste
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=True)
    #
    #test = train_test_split(X, y, test_size=0.40, random_state=True)

    clf = Perceptron(tol=1e-3, random_state=0)
    clf.fit(X, y)
    clf.predict()
    print(clf.score(X, y))

    #data=iris.data
    #features = iris.feature_names
    #print(iris.target_names)
    #iris_df = pd.DataFrame(data, columns = features)
    #iris_df['class'] = iris['target']

    #classes = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}

    



if __name__ == "__main__": main()


