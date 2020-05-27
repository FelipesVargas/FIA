import numpy as np
import pandas as pd
import random


lr       = 0.1     # taxa de aprendizado (learning rate)
gamma    = 0.9     # taxa de desconto (gamma)
ep       = 0.2     # taxa de exploração

goal     = (0, 2)  # posição final
iteractions = 75  # número de iterações do agente

grid = np.zeros((2,3))
directions = np.zeros((2,3))
politics = [['','',''],['','','']]


for num in range(iteractions): #number of times we will go through the whole grid
    if (random.uniform(0, 1) < ep):
        i = random.randint(0, 2 - 1)
        j = random.randint(0, 3 - 1)    
        # se não, a ação escolhida é a de maior valor na tabela Q

    else:
        a=grid.max(0)
        aux=[max(grid[0]),max(grid[1])]

        i = aux.index(max(aux))
        j = np.where(grid[i] == aux[i])
        j = j[0][0]


    up_grid = grid[i-1][j] if i > 0 else -10000   #if going up takes us out of the grid then its value be 0
    down_grid = grid[i+1][j] if i < 1 else -10000  #if going down takes us out of the grid then its value be 0
    left_grid = grid[i][j-1] if j > 0 else -10000  #if going left takes us out of the grid then its value be 0
    right_grid = grid[i][j+1] if j < 2 else -10000  #if going right takes us out of the grid then its value be 
    loop = grid[i][j]
    
    all_dirs = [up_grid, down_grid, left_grid, right_grid, loop]  
    direc = max(all_dirs)
    index= all_dirs.index(max(all_dirs))

    if(index == 0):
        politics[i][j] = ' ^ ' 
    elif(index == 1):
        politics[i][j] = " \/ " 
    elif(index == 2):
        politics[i][j] = " < " 
    elif(index == 3):
        politics[i][j] = " > " 
    elif(index == 4):
        politics[i][j] = " - " 

    if i==goal[0] and j==goal[1]: # the position of A
        grid[i][j] = grid[i][j] + lr * (1 + gamma * grid[i][j])
    else:
        grid[i][j] = grid[i][j] + lr * (gamma * direc - grid[i][j])


grid = np.round(grid,0)

print("")
print(grid[0][0], "\t",grid[0][1], "\t",grid[0][2])
print(grid[1][0], "\t",grid[1][1], "\t",grid[1][2])
print(politics[0])
print(politics[1])