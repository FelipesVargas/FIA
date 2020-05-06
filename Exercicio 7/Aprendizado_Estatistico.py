import numpy as np 
import pandas as pd
import plot
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sk-learn import linear_model
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from matplotlib.colors import ListedColormap
import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

###########3############ Minimos Quadrados #########################
def last_square(X_train, X_test, y_train, y_test):
    print("Minimos quadrados:")
    plt.figure()
    
    lin_reg_model = linear_model.LinearRegression()

    # Treinamento do modelo
    lin_reg_model = lin_reg_model.fit(X_train, y_train)

    # Predicao a partir dos dados de teste utilizando o modelo definido
    y_pred = lin_reg_model.predict(X_test)

    # Plot do gráfico resultante da Regressão
    plt.scatter(y_test, y_pred, marker = 'o', c = y_test, edgecolors = 'navy')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], lw=1, c = 'black')
    plt.xlabel('Verdadeiro', fontsize = 10)
    plt.ylabel('Estimado', fontsize = 10)



############### PCA Utilizando dois componentes principais ######################
def pca(X, y,features, label_names):
    print("PCA")
    plt.figure()

    pca = PCA(n_components=2)
    X_r = pca.fit(X).transform(X)
    
    #Valores dos dois componentes principais
    X_r1 = X_r[:,0]
    X_r2 = X_r[:,1]

    #Transposta dos coeficiente em pares ordenados
    coeficiente = np.transpose(pca.components_[0:2, :])

    #Numero de pares
    n = coeficiente.shape[0]

    # Plot do gráfico resultante do PCA
    for color, i, target_name in zip(['navy', 'turquoise', 'darkorange'], [0, 1, 2], label_names):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1],  alpha=.8, color=color, lw=2,
                label=target_name,edgecolors='black')
    
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of IRIS dataset')

    # Impressão das retas de cada atributo do coeficiente
    for i in range(n):
        # Reta de cada atributo x e y do coeficiente
        plt.arrow(0, 0, coeficiente[i,0], coeficiente[i,1], color = 'r')

        # Texto de discrição da reta
        plt.text(coeficiente[i,0] , coeficiente[i,1] , features[i], color = 'r')


############### LDA Utilizando dois componentes principais ######################
def lda(X, y, target_names, label_names):
    print("LDA")
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_r2 = lda.fit(X, y)

    plt.figure()
    
    #Área onde o gráfico derá plotado
    X_r2_min, X_r2_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_lda_min, y_lda_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    #Arrays com os dados do LDA para x e y
    xp_lda, yp_lda = np.meshgrid(np.arange(X_r2_min, X_r2_max, .02), np.arange(y_lda_min, y_lda_max, .02))

    #Predição do LDA
    pred_lda = lda.predict(np.c_[xp_lda.ravel(), yp_lda.ravel()]).reshape(xp_lda.shape)

    #Preenchimento das area do gráfico
    colors = ('navy', 'turquoise', 'darkorange')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    plt.contourf(xp_lda, yp_lda, pred_lda, alpha = 0.3, cmap=cmap)
    
    # Plot do gráfico resultante do LDA
    for color, i, target_name in zip(colors, [0, 1, 2], label_names):
        plt.scatter(X[y == i, 0], X[y == i, 1], alpha=.8, color=color,
                    label=target_name, edgecolors='black')
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LDA of IRIS dataset')
    
############### SVM Utilizando dois componentes principais ######################   
def svm_classifier(X, y, target_names, label_names):
    print("SVM")
    plt.figure()

    # Definição do modelo SVM Linear
    SVM = svm.LinearSVC(C=1).fit(X, y)

    #Área onde o gráfico derá plotado
    x_svm_min, x_svm_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_svm_min, y_svm_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    #Arrays com os dados do SVM para x e y
    xp_svm, yp_svm = np.meshgrid(np.arange(x_svm_min, x_svm_max, .02), np.arange(y_svm_min, y_svm_max, .02))

    #Predição do SVM
    pred_svm = SVM.predict(np.c_[xp_svm.ravel(), yp_svm.ravel()]).reshape(xp_svm.shape)

    #Preenchimento das area do gráfico
    colors = ('navy', 'turquoise', 'darkorange')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    plt.contourf(xp_svm, yp_svm, pred_svm, alpha = 0.3, cmap=cmap)


    for color, i, target_name in zip(colors, [0, 1, 2], label_names):
        plt.scatter(X[y == i, 0], X[y == i, 1],  color=color, label=target_name, edgecolors='black')

    plt.xlabel(target_names[2])
    plt.ylabel(target_names[3])

    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('SVM of IRIS dataset')


def main():
    # carrega dados da base de iris
    iris = load_iris()
    
    data=iris.data
    features = iris.feature_names
    print(iris.target_names)
    iris_df = pd.DataFrame(data, columns = features)
    iris_df['class'] = iris['target']

    classes = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
    
    # Definição das dimensões e dos respecticos resutantes
    X = iris_df.iloc[:,0:4].values
    y = iris_df['class'].values

    #Divisão do dataset entre dados para Treinamento e para Teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=True)


    last_square(X_train, X_test, y_train, y_test)
    pca(X_train, y_train, features,iris.target_names)
    lda(iris_df.iloc[:,2:4].values, iris_df['class'].values, features,iris.target_names)
    svm_classifier(iris_df.iloc[:,2:4].values, iris_df['class'].values, features,iris.target_names)
    
    plt.show()


if __name__ == "__main__": main()


