import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
import pandas as pd
import random

style.use("ggplot")



classes = 10
batch_size = 64
population = 20
generations = 1000
threshold = 1000

df = pd.read_csv('C:/Users/Octavio/Desktop/oddseprobs.csv')
ml_result = df.loc[10:39,'v':'d'].values.tolist()
probs = df.loc[10:39,'odd_v':'odd_d'].values.tolist()
#probs [v,e,d]
results_g = [3.25,2.1,1.44,1.35,2.7] # [e,v,v]
results_g = df.loc[10:39,'odd_venc'].values.tolist()

invest = 2
games = len(probs)
carteira = invest*games
time_h = 0
lucro0 = 0

def bet(idx_game,value):
    if value == results_g[idx_game]:
        return 1
    elif value == 0:
        return 0
    else:
        return -1

def strategy(aposta,tipo,time_h):
    result = [0]
    odds = [1]
    v = aposta
    #for i,v in enumerate(aposta):
    i = time_h
    if tipo == 'min':
        odd = min(probs[i]) #odd escolhida
    elif tipo == 'rand':
        odd = np.random.randint(0,3) # escolha randomica
    elif tipo == 'ml':
        odd_idx = ml_result[i].index(max(ml_result[i])) # escolha do ML
        odd = probs[i][odd_idx]
    elif tipo == 'max':
        odd = max(probs[i])
            
    r = bet(i,odd*v)
    result[0] = r
    odds[0] = odd
            
    total = (result,odds)
    return total


choices = {0:"ml",1:"max",2:"min"}

def action(choice,time):
    # apostar ou nao apostar
    result = [0]
    odds = [1]
    #total = 0
    if choice == 0:
        aposta = (0)
        total = strategy(aposta,'ml',time)       
    elif choice == 1:
        aposta = (1)
        total = strategy(aposta,'max',time) 
    elif choice == 2:
        aposta = (1)
        total = strategy(aposta,'min',time)
    #nao apostar
    elif choice == 3:
        aposta = (0)
        total = strategy(aposta,'rand',time) 
        
    #total = (result,odds)
    return total #result = (1,1,0) acertou, errou a escolha ou nao escolheu

class Network():
    def __init__(self):
        self.actions = []

        for i in range(games):
            self.action = np.random.randint(0,3)
            self.actions.append(self.action)

        self.lucro = 0

    def get_actions(self):
        return self.actions

def init_networks(population):
    return [Network() for _ in range(population)]



def fitness(networks):
    for network in networks:
        actions = network.get_actions()
        strategies = []
        carteira = invest*games
        lucro_tot = 0
        try:
            time_h = 0
            for act in actions:
                result = action(act,time_h)
                erros = 0
                l_tot = 0
                j = result[0][0]
                if j == -1:
                    erros += 1
                    l_tot = 0
                elif j == 0:
                    l_tot = 0  
                elif j == 1:
                    l_tot = result[1][0]
                    
                #lucro = (l_tot*invest) - erros*invest
                lucro = (l_tot*invest) - invest
                reward = lucro
                lucro_tot += lucro
                time_h += 1
                
            network.lucro = lucro_tot
            print('Lucro Total: {}'.format(network.lucro))
            
        except:
            network.lucro = 0
            print('Build failed.')

    return networks

def selection(networks):
    networks = sorted(networks, key=lambda network: network.lucro, reverse=True)
    networks = networks[:int(0.2 * len(networks))]

    return networks


def crossover(networks):
    offspring = []
    '''
    parent1 = networks[0]
    parent2 = networks[1]
    child1 = Network()

    # Crossing over parent params
    p1 = int(len(parent1.actions)/2)
    p2 = int(len(parent2.actions)/2)
    a = parent1.actions[:p1]
    b = parent2.actions[:p2]
    new_actions = a + b 
    child1.actions = new_actions
    #child1.actions[p2:] = parent2.actions[:p2]
    
    offspring.append(child1)'''

    
    for _ in range(int((population - len(networks)) / 2)):
        parent1 = random.choice(networks)
        parent2 = random.choice(networks)
        #parent1 = networks[0]
        #parent2 = networks[1]
        child1 = Network()
        child2 = Network()
        #child3 = Network()
        #child4 = Network()

        # Crossing over parent params
        p1 = int(len(parent1.actions)/2)
        p2 = int(len(parent2.actions)/2)
        a = parent1.actions[:p1]
        b = parent2.actions[:p2]
        new_actions = a + b 
        child1.actions = new_actions
        
        new_actions2 = b + a
        child2.actions = new_actions2

        offspring.append(child1)
        offspring.append(child2)
        #offspring.append(child3)
        #offspring.append(child4)
        
    networks.extend(offspring)
    return networks


def mutate(networks):
    
    for network in networks[2:]:
        if np.random.uniform(0, 1) <= 0.1:
            idx = np.random.randint(0,len(network.actions))
            network.actions[idx] = np.random.randint(0,3)

    return networks


def main():
    lucro_nets = []
    best_lucro_nets = []
    best_networks = []
    
    networks = init_networks(population)

    for gen in range(generations):
        print ('Generation {}'.format(gen+1))

        networks = fitness(networks)
        networks = selection(networks)
        #best_networks.append(networks)
        networks = crossover(networks)
        networks = mutate(networks)
        
        network_lucro = []
        for network in networks:
            network_lucro.append(network.lucro)
            if network.lucro > threshold:
                print ('Threshold met')
                print (network.get_actions())
                print ('Best lucro: {}'.format(network.lucro))
                exit(0)
                
        print ('Best lucro: {}'.format(max(network_lucro)))
        best_lucro_nets.append(max(network_lucro))
        media = sum(network_lucro)/len(network_lucro)
        print ('Média lucro: {}'.format(media))
        lucro_nets.append(media)
        print("")

    plt.subplot(211)
    plt.plot([i for i in range(len(lucro_nets))], lucro_nets)
    plt.ylabel(f"Lucro médio geração")
    plt.xlabel("geração #")

    plt.subplot(212)
    plt.plot([i for i in range(len(best_lucro_nets))], best_lucro_nets)
    plt.plot([i for i in range(len(lucro_nets))], lucro_nets)
    #plt.plot([i for i in range(len(lucro_time))], lucro_time)

    plt.show()
                    

if __name__ == '__main__':
    main()
