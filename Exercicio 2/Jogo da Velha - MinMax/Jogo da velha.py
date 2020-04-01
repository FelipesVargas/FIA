#####################################################################################
## Exercicio Jogo da Velha                                                         ##  
## Autor: Felipe Souza Vargas                                                      ##
#####################################################################################
## Exercício: Jogo da Velha utilizando Algoritmo MinMax para IA do computador.     ##
## Saída: A saida consite das respostas do computados as jogadas de entrada.       ##
##       Sendo que com previsão de uma jogada para frente não é possível ganhar    ##
##       do computador.                                                            ## 
#####################################################################################


#Posições das casas
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
import time

maquina=""
pesos=(2,1,2,1,3,1,2,1,2)
#      1   2   3   4   5   6   7   8   9
jogo=[" "," "," "," "," "," "," "," "," "]


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
            if(estado[i] == " " ):
                novo_estado[i]="O"
                estados.append(novo_estado)
    else:
        for i in range(len(estado)):
            novo_estado=estado.copy()
            if(estado[i] == " " ):
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

def mostrar_jogo(jogo):
    print(" "+jogo[0]+" | "+jogo[1]+" | "+jogo[2]+"  ")
    print("-----------")
    print(" "+jogo[3]+" | "+jogo[4]+" | "+jogo[5]+"  ")
    print("-----------")
    print(" "+jogo[6]+" | "+jogo[7]+" | "+jogo[8]+"  ")

def mostrar_velha():
    print("")
    print("jogo atual:"+ " "+jogo[0]+" | "+jogo[1]+" | "+jogo[2]+"  "+ "Referencia posicoes"  +"    0 | 1 | 2 ")
    print("           "+ "-----------"                               + "                   "  +"    -----------")
    print("           "+ " "+jogo[3]+" | "+jogo[4]+" | "+jogo[5]+"  "+ "                   "  +"    3 | 4 | 5 ")
    print("           "+ "-----------"                               + "                   "  +"    -----------")
    print("           "+ " "+jogo[6]+" | "+jogo[7]+" | "+jogo[8]+"  "+ "                   "  +"    6 | 7 | 8 ")
    print("")



print("Bem vindo ao Jogo da Velha")
jogador=input("Insira o simbolo que voce deseja jogar (X ou O): ")
 
if(jogador == "X"):
    maquina="O"
else:
    maquina="X"

empate=0
placar=0

while(empate<5):
    empate+=1
    mostrar_velha()
    jogada=input("Deseja colocar o "+jogador+" em qual posição: ")
    
    jogo[int(jogada)]=jogador
    
    placar=valor_jogada(jogo)
    if(placar==math.inf or placar==-math.inf):
        print("Voce ganhou!!!")
        empate=9   

    print("Vez do computador...")
    time.sleep(1)
    jogo=max_value(jogo, maquina)
    placar=valor_jogada(jogo)
    mostrar_velha()
    if(placar==math.inf or placar==-math.inf):
        print("Voce perdeu!!!")
        empate=9      

  
print("Empate, tente novamente!!")