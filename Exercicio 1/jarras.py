## Exercicio Jarra
## Autor: Felipe Souza Vargas
## Exercício: Como obter 2 litros de água no jarro 4.
## Saída: Lista de estados necessários para se obter 2 litros no jarro de 4.



abertos=[]           #Lista de controle de qual nó já foi aberto
visitados=[]         #Lista de controle de qual nó já foi visitado
pai={}               #Dicionario de identificação dos ancestrais do nós, seguindo a lógica de pai[nó]=PAI

#Função responsável por encher a jarra de 3 litros
def enche_a(x):    
    x[0]=3
    return x

#Função responsável por encher a jarra de 4 litros
def enche_b(x):
    x[1]=4
    return x

#Função responsável por esvaziar a jarra de 3 litros
def esvazia_a(x):
    x[0]=0
    return x

#Função responsável por esvaziar a jarra de 4 litros
def esvazia_b(x):
    x[1]=0
    return x



#Função responsável por tranferir o volume de água na jarra de 3 litros para a de 4 litros
def tranfere_a_b(x):
    a=x[0]
    b=x[1]
    if(b<4):
        b+=a
        if(b>4):
            a=b-4
            b=4
        else:
            a=0
        x[0]=a
        x[1]=b
    return x

#Função responsável por tranferir o volume de água na jarra de 4 litros para a de 3 litros
def tranfere_b_a(x):
    a=x[0]
    b=x[1]
    if(a<3):
        a+=b
        if(a>3):
            b=a-3
            a=3
        else:
            b=0
        x[0]=a
        x[1]=b
    return x


#Execução da busca utilizando o algoritmo DFS recursivo
def DFS(u):
    if(u == [0,2]):
        return 1

    global abertos
    global visitados
    abertos.append(u)
    adjacentes=[]
    
#Criação dos 6 possíveis estados à partir do estado atual    
    x = enche_a(u[:])
    adjacentes.append(x[:])
 
    x=enche_b(u[:])
    adjacentes.append(x[:])

    x=esvazia_a(u[:])
    adjacentes.append(x[:])

    x=esvazia_b(u[:])
    adjacentes.append(x[:])

    x=tranfere_a_b(u[:])
    adjacentes.append(x[:])

    x =tranfere_b_a(u[:])
    adjacentes.append(x[:])

#Contorle para caso o estado desejado tenha sido gerado ele para a recursão
    if([0,2] in adjacentes):
        pai[(0,2)]=u[:]
        return 1

    for x in adjacentes:
        if(x not in abertos and x not in visitados):
            adjacentes.remove(x)
            pai[tuple(x[:]) ]=u[:]
            if(DFS(x) == 1):
                return 1
            
    abertos.remove(u)
    visitados.append(u)


jarras = [0, 0]
pai={tuple(jarras):None}

DFS(jarras)

saida=[]
filho=(0,2)

#Montagem da saida a partir da lista de nós pais
while(pai[tuple(filho)] != None):
    saida.append(tuple(filho))
    filho=pai[tuple(filho)]

saida.append(tuple(jarras))

#Apresentação da lista sequencial de estados para obter 2 litros no jarro 2
print(saida[::-1])
