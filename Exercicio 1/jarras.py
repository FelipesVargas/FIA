## Exercicio Jarra
## Autor: Felipe Souza Vargas

abertos=[]
visitados=[]
pai={}

def enche_a(x):  
    x[0]=3
    return x

def enche_b(x):
    x[1]=4
    return x

def esvazia_a(x):
    x[0]=0
    return x

def esvazia_b(x):
    x[1]=0
    return x

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

def DFS(u):
    if(u == [0,2]):
        return 1

    global abertos
    global visitados
    abertos.append(u)
    adjacentes=[]
    
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

auxiliar=[]
filho=(0,2)


while(pai[tuple(filho)] != None):
    auxiliar.append(tuple(filho))
    filho=pai[tuple(filho)]

auxiliar.append(tuple(jarras))

print(auxiliar[::-1])
