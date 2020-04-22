## Exercicio Decision Tree utilziando o Algoritmo ID3
## Autor: Felipe Souza Vargas

import pandas as pd
import numpy as np
import operator
from sklearn.datasets import load_iris
from numpy import log2

#Calculo da entropia
def calc_entropy(base):
    data = base.iloc[:, - 1].unique()
    p = dict()

    for example in data:
        p[example] = base[base.iloc[:, - 1] == example].count()[0]
    
    entropy = 0
    f = 0
    for i in p:
        f = f + p[i]
    
    for i in p:
        entropy += ((-p[i]/f) * log2((p[i]/f)))

    return entropy, p

#Calculo da entropia de cada valor apartado
def entropy(base):    
    column = base.iloc[:, - 1].name
    features = base.columns.tolist()[:-1]
    
    entropy = dict()

    for i in features:
        entropy[i] = dict()
        
        column_act = base[[i, column]]
        
        for column_act, sub_base in column_act.groupby(column_act.columns[0]):
            aux = sub_base.iloc[0,0]
            entropy[i][aux] = dict()
            entropy[i][aux]['entropy'], entropy[i][aux]['pn'] = calc_entropy(sub_base)

    return entropy

#Retorna o ganho informacional médio
def avarage_IG(base):
    ig = dict()
    pn = 0

    x, base_pn = calc_entropy(base)
    base_details = entropy(base)
    
    for i in base_pn:
        pn = pn + base_pn[i]
    
    for col in base_details:
        auxiliar = list()
        values = base_details.get(col)

        for i in values:    
            iteration = values.get(i)
            summ = 0
            
            for c in iteration['pn']:
                summ = summ + iteration['pn'][c] 
                
            auxiliar.append(( abs(summ) / abs(pn) )* iteration['entropy'] )
            ig[col] = sum(auxiliar)
            
    return ig

#Retorna o ganho informacionção do Database
def IG(base):
    informaional_gain = dict()
    
    IG_avrg = avarage_IG(base)
    tot_entropy, x = calc_entropy(base)
    
    for i in IG_avrg:
        informaional_gain[i] = tot_entropy - IG_avrg.get(i)
    
    return dict(sorted(informaional_gain.items(), key=operator.itemgetter(1),reverse=True))

#Algoritmo de criação da Arvore de Decisão ID3
def ID3(base):    
    target = base.iloc[:, - 1].name
    base_gain = IG(base)
    col = list(base_gain.keys())[0]    
    
    ID3_tree = dict()  
    ID3_tree[col] = dict()
    
    values = base[col].unique().tolist()
    
    tb = base.groupby([col, target]).size().reset_index(name='c')      
   
    for i in values:
        if tb[col].tolist().count(i) == 1:
            ID3_tree[col][i] = tb[tb[col] == i][target].values[0]
        else:
            ID3_tree[col][i] = ID3(base[base[col] == i].reset_index(drop=True))
            
    return ID3_tree

#Utilizada a base iris fornecida pelo scikit_learn com 150 amostras
def main():
    base = load_iris()
    ID3_decisiontree = ID3(pd.DataFrame(data= np.c_[base['data'], base['target']], columns= base['feature_names'] + ['target']))

    print("Arvore de Decisao gerada pelo ID3 a partir da base IRISÇ")
    print(ID3_decisiontree)

if __name__ == "__main__": main()
