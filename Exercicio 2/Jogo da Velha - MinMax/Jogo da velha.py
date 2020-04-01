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
jogo=["","","","","","","","",""]


def valor_jogada(estado):
    soma=0
    if(maquina == "O"):
        teste="X"
        teste2="O"
    else:
        teste="O"
        teste2="X"
    if((estado[0]==teste and estado[1]==teste and estado[2]==teste) or (estado[3]==teste and estado[4]==teste and estado[5]==teste) or (estado[6]==teste and estado[7]==teste and estado[8]==teste)  or (estado[0]==teste and estado[3]==teste and estado[6]==teste) or (estado[1]==teste and estado[4]==teste and estado[7]==teste) or (estado[2]==teste and estado[5]==teste and estado[8]==teste)  or (estado[0]==teste and estado[4]==teste and estado[8]==teste) or (estado[2]==teste and estado[4]==teste and estado[6]==teste)):
        return -math.inf      
    elif( (estado[0]==teste2 and estado[1]==teste2 and estado[2]==teste2) or (estado[3]==teste2 and estado[4]==teste2 and estado[5]==teste2) or (estado[6]==teste2 and estado[7]==teste2 and estado[8]==teste2) or (estado[0]==teste2 and estado[3]==teste2 and estado[6]==teste2) or (estado[1]==teste2 and estado[4]==teste2 and estado[7]==teste2) or (estado[2]==teste2 and estado[5]==teste2 and estado[8]==teste2) or (estado[0]==teste2 and estado[4]==teste2 and estado[8]==teste2) or (estado[2]==teste2 and estado[4]==teste2 and estado[6]==teste2) ):
        return math.inf
    peso=[]
    
    for i in range(len(estado)):
        if(estado[i] == teste2):
            soma+=pesos[i]
            peso.append(pesos[i])
        elif(estado[i] == teste):
            soma-=pesos[i]
            peso.append(-pesos[i])
        else:
            peso.append(0)     

    return soma

def proximas_Jogadas(estado, atual):
    estados=[]

    if(atual == "O"):
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

    if(atual == "X"):
        atual = "O"
    else:
        atual = "X"

    for i in proximas_Jogadas(estado, atual):
        v = min(v, valor_jogada(i))

    return v

def max_value(estado, atual): 
    v = -math.inf
    y=[estado, v]
    for i in proximas_Jogadas(jogo, atual):
        valor=min_value(i, atual)
        z=[i, valor]
        if(y[1] < z[1]):
            y=z

    return y[0]

#    retorn valor_jogada(estado)


jogada = "X"
maquina = "X"

#print(valor_jogada(jogo))
print(max_value(jogo, jogada))