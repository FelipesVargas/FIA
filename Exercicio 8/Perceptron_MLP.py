import numpy as np
import pandas as pd
import plot
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from sklearn.neural_network import MLPRegressor


def percepetron(X, y, target_names, label_names):
    #Divisão do dataset entre dados para Treinamento e para Teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.12, random_state=True)

    #Definição do Classificador
    clf = Perceptron(tol=1e-3, random_state=0)
    #Treinamanto com os Dados definidos anteriormente
    X_r2 = clf.fit(X_train, y_train)

    print("Precisao do classificador Perceptron: ", clf.score(X_train, y_train))

    plt.figure()
    
    #Área onde o gráfico será plotado
    X_r2_min, X_r2_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_percpt_min, y_percpt_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    #Arrays com os dados do Perceptron para x e y
    xp_percept, yp_percept = np.meshgrid(np.arange(X_r2_min, X_r2_max, .02), np.arange(y_percpt_min, y_percpt_max, .02))

    #Predição do Perceptron
    pred_Percept = clf.predict(np.c_[xp_percept.ravel(), yp_percept.ravel()])
    pred_Percept = pred_Percept.reshape(xp_percept.shape)

    #Preenchimento das área do gráfico
    colors = ('navy', 'turquoise', 'darkorange')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    plt.contourf(xp_percept, yp_percept, pred_Percept, alpha = 0.3, cmap=cmap)
    
    # Plot do gráfico resultante do Perceptron
    for color, i, target_name in zip(colors, [0, 1, 2], label_names):
        plt.scatter(X[y == i, 0], X[y == i, 1], alpha=.8, color=color,
                    label=target_name, edgecolors='black')
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('Classificacao utilizando Perceptron no dataset IRIS')

def mlp(X, y, target_names, label_names):
    #Divisão do dataset entre dados para Treinamento e para Teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.12, random_state=True)

    #Definição do Classificador
    clf = MLPRegressor(hidden_layer_sizes=(100, 100), tol=1e-2, max_iter=500, random_state=0) 
    #Treinamanto com os Dados definidos anteriormente
    X_r2 = clf.fit(X_train, y_train)
    
    print("Precisao do classificador MLP: ", clf.score(X_train, y_train))

    plt.figure()
    
    #Área onde o gráfico derá plotado
    X_r2_min, X_r2_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_percpt_min, y_percpt_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    #Arrays com os dados do MLP para x e y
    xp_percept, yp_percept = np.meshgrid(np.arange(X_r2_min, X_r2_max, .02), np.arange(y_percpt_min, y_percpt_max, .02))

    #Predição do MPL
    pred_Percept = clf.predict(np.c_[xp_percept.ravel(), yp_percept.ravel()])
    pred_Percept = pred_Percept.reshape(xp_percept.shape)

    #Preenchimento das area do gráfico
    colors = ('navy', 'turquoise', 'darkorange')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    plt.contourf(xp_percept, yp_percept, pred_Percept, alpha = 0.3, cmap=cmap)
    
    # Plot do gráfico resultante do MPL
    for color, i, target_name in zip(colors, [0, 1, 2], label_names):
        plt.scatter(X[y == i, 0], X[y == i, 1], alpha=.8, color=color,
                    label=target_name, edgecolors='black')
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('Classificacao utilizando MLP no dataset IRIS')

def main():
    iris=load_iris()

    data=iris.data
    features = iris.feature_names

    iris_df = pd.DataFrame(data, columns = features)
    iris_df['class'] = iris['target']

    X_2 = iris_df.iloc[:100,1:3].values
    y_2 = iris_df['class'].values[:100]
    X_3 = iris_df.iloc[:,1:3].values
    y_3 = iris_df['class'].values
    
    percepetron(X_2, y_2, features,iris.target_names[:2])
    percepetron(X_3, y_3, features[:3],iris.target_names)
    mlp(X_3, y_3, features[:4],iris.target_names)

    plt.show()
    

if __name__ == "__main__": main()


