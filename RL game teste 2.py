import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
import pandas as pd

style.use("ggplot")

#SIZE = 10
HM_EPISODES = 25000
MOVE_PENALTY = 1
ENEMY_PENALTY = 10
FOOD_REWARD = 10
epsilon = 0.9
EPS_DECAY = 0.9998
SHOW_EVERY = 3000
invest = 2
games = 20
carteira = invest*games
time_h = 0
lucro0 = 0

# MAX, MIN, MED

start_q_table = None # or filename saved

LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3

ml_result = [[0.2755,0.226295,0.498205],
             [0.364,0.3025,0.3335],
             [0.515,0.262,0.223],
             [0.543,0.1535,0.3035],
             [0.362,0.1455,0.4925]]

probs = [[4.33,3.25,1.85],
         [2.1,3.1,3.6],
         [1.44,4,7.25],
         [1.35,4.4,9],
         [2.6,3.1,2.7]]
df = pd.read_csv('C:/Users/Octavio/Desktop/oddseprobs.csv')
ml_result = df.loc[10:29,'v':'d'].values.tolist()
probs = df.loc[10:29,'odd_v':'odd_d'].values.tolist()
#probs [v,e,d]
results_g = [3.25,2.1,1.44,1.35,2.7] # [e,v,v]
results_g = df.loc[10:29,'odd_venc'].values.tolist()

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

#choices = {0:(0,0,0),1:(1,0,0),2:(1,1,0),3:(1,1,1),4:(0,1,0),5:(0,1,1),6:(0,0,1),7:(1,0,1)}
choices = {0:"ml",1:"max",2:"min"}

def action(choice,time):
    # apostar ou nao apostar
    result = [0]
    odds = [1]
    #total = 0
    if choice == 0:
        aposta = (1)
        total = strategy(aposta,'ml',time)       
    elif choice == 1:
        aposta = (1)
        total = strategy(aposta,'max',time) 
    elif choice == 2:
        aposta = (1)
        total = strategy(aposta,'min',time) 
        
    #total = (result,odds)
    return total #result = (1,1,0) acertou, errou a escolha ou nao escolheu

        
# Q table
val = 3
if start_q_table is None:
    #val = 2**games
    val = 3
    print("generating new table....")
    q_table = {}
    for j1 in range(0,carteira*2):
        for j2 in range(-4,carteira):
            # [(balance,lucro)]
            #q_table[(j2,j3)] = [np.random.uniform(0,1) for i in range(2^3)]
            q_table[(j1,j2)] = [0 for i in range(val)]
            #q_table[(j1,j2,j3)] = [0 for i in range(3)]
    #print(q_table)

else:
    with open(start_q_table,"rb") as f:
        q_table = pickle.load(f)

# run
episode_rewards = []
carteira_time = []
lucro_time = []
choice_prog = 0
not_choice = 0
full_strat = []
lucro = 0
for episode in range(HM_EPISODES):
    time_h = 0
    if episode % SHOW_EVERY == 0:
        print(f"on # {episode}, epsilon: {epsilon}")
        print(f"{SHOW_EVERY} ep mean {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False
        
    # run action
    episode_reward = 0
    strategies = []
    carteira = invest*games
    lucro0 = 0
    for i in range(0,games):
        # aposta dependendo da odds dos jogos (escolha aleatoria)
        #obs = (player-food, player-enemy)
        #obs = (0,0,0) # 3 jogos, 0 neg e 1 pos
        obs = (carteira,lucro0)
        if np.random.random() > epsilon:
            action_n = np.argmax(q_table[obs])
            #not_choice +=1
        else:
            action_n = np.random.randint(0,high=2)
            #choice_prog += 1
        strategies.append(choices[action_n])  
            
        result = action(action_n,time_h) #result = [(,),(,)]resultado de cada jogada e odds
        #contar os zeros
        '''jogos = [0,0,0,0,0] #odds ganhas
        for idx,j in enumerate(result[0]):
            if j == -1:
                erros += 1
                jogos[idx] = 0
            elif j == 0:
                jogos[idx] = 0
            elif j == 1:
                jogos[idx] = result[1][idx]
                
        l_tot = 0
        for i in range(0,len(result[0])):
            l_tot += invest*jogos[i]'''

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
            
        lucro = (l_tot*invest) - erros*invest
        #lucro = result[0][0]*invest*result[1][0] + result[0][1]*invest*result[1][1] + result[0][2]*invest*result[1][2] - erros*invest
        
        reward = lucro
        
        #new_obs = (player-food, player-enemy)
        lucro0 += lucro
        new_obs = (carteira,lucro0) # get new state
        try:
            max_future_q = np.max(q_table[new_obs])
        except:
            #val = 2**games
            #val = 3
            q_table[new_obs] = [0 for i in range(val)]
            max_future_q = np.max(q_table[new_obs])
            
        
        current_q = q_table[obs][action_n] 
 
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        
        q_table[obs][action_n] = new_q #update actual q
        #carteira = carteira + lucro # update carteira
        if episode == HM_EPISODES-1:
            print('lucro',lucro,'lucro ac',lucro0,'erros',erros,'action',action_n,'...............')
        episode_reward += reward
        time_h += 1
         
        #stop loss ou take profit
        if reward > 0.9*carteira:
            break

    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY
    carteira_time.append(obs[0])
    lucro_time.append(obs[1])
    full_strat.append(strategies)
    
    

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode = "valid")

plt.subplot(211)
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"reward {SHOW_EVERY}ma")
plt.xlabel("episode #")

plt.subplot(212)
plt.plot([i for i in range(len(carteira_time))], carteira_time)
plt.plot([i for i in range(len(lucro_time))], lucro_time)

plt.show()
lucro_final_ac = obs[1] + lucro
carteira_final = obs[0] + lucro_final_ac
print(f"Obs:{obs},Q table:{q_table[obs][action_n]},Action:{choices[action_n]},Lucro: {lucro}")
#print(f"Carteira: {obs[0]},Lucro: {obs[1]}")
print(f"Carteira: {carteira_final},Lucro: {lucro_final_ac}")
print(full_strat[-1])
with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)      
