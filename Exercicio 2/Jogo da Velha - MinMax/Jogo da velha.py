## Exercicio Jogo da Velha MinMax
## Autor: Felipe Souza Vargas
## Exercício: Como obter 2 litros de água no jarro 4.
## Saída: Lista de estados necessários para se obter 2 litros no jarro de 4.
#Posições das casa
#_1_|_2_|_3_
#_4_|_5_|_6_
# 7 | 8 | 9 
#
#
#Pesos 
#_2_|_1_|_2_
#_1_|_3_|_1_
# 2 | 1 | 2 

import math

maquina=""
pesos=(2,1,2,1,3,1,2,1,2)
#      1  2  3  4  5  6  7  8  9
jogo=["","","","","","X","","",""]

def valor_jogada(estado):
    soma=0
    for i in range(len(estado)):
        if(estado[i] == "X"):
            soma+=pesos[i]
        elif(estado[i] == "O"):
            soma-=pesos[i]  
    
    return soma

def proximas_Jogadas(estado, atual):
    estados=[]

    if(atual == "X"):
        for i in range(len(estado)):
            novo_estado=estado.copy()
            if(estado[i] == "" ):
                novo_estado[i]="O"
                estados.append(novo_estado)
    else:
        for i in range(len(estado)):
            novo_estado=estado.copy()
            if(estado[i] == "" ):
                novo_estado[i]="X"
                estados.append(novo_estado)
    return estados            

def min_value(estado, atual):
    v = math.inf
    y=[estado, v]
    for i in proximas_Jogadas(estado, atual):
        valor=max_value(i, atual)
        z=[i, valor]
        print(z)
        if(y[1] > z[1]):
            y=z

    print(y)
    return y[0]

def max_value(estado, atual): 
    if(atual == "X"):
        atual = "O"
    else:
        atual = "X"
    v = -math.inf
    for i in proximas_Jogadas(jogo, atual):
        #v = max(v, valor_jogada(estado))
        valor=min_value(i, atual)
        z=[i, valor]
        print(z)
        if(y[1] < z[1]):
            y=z

    return v

#    retorn valor_jogada(estado)

jogada = "X"
maquina = "O"
print(min_value(jogo, jogada))