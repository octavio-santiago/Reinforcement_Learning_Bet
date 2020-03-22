import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
import pandas as pd

style.use("ggplot")

HM_EPISODES = 45000
epsilon = 0.9
EPS_DECAY = 0.9998
SHOW_EVERY = 3000
invest = 2
games = 30
carteira = invest*games
time_h = 0
lucro0 = 0


start_q_table = None # or filename saved

LEARNING_RATE = 0.1
DISCOUNT = 0.95

df = pd.read_csv('oddseprobs.csv')
ml_result = df.loc[10:39,'v':'d'].values.tolist()
probs = df.loc[10:39,'odd_v':'odd_d'].values.tolist()
#probs [v,e,d]
results_g = df.loc[10:39,'odd_venc'].values.tolist()

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
    val = 3
    print("generating new table....")
    q_table = {}
    for j1 in range(0,carteira*2):
        for j2 in range(-4,carteira):
            # [(balance,lucro)]
            q_table[(j1,j2)] = [0 for i in range(val)]
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
        obs = (carteira,lucro0)
        if np.random.random() > epsilon:
            action_n = np.argmax(q_table[obs])
        else:
            action_n = np.random.randint(0,high=2)
        strategies.append(choices[action_n])  
            
        result = action(action_n,time_h) 

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
            
        lucro = (l_tot*invest) - invest        
        reward = lucro

        lucro0 += lucro
        new_obs = (carteira,lucro0) # get new state
        try:
            max_future_q = np.max(q_table[new_obs])
        except:
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
        if reward > 10*carteira:
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
idx = lucro_time.index(max(lucro_time))
print(full_strat[idx])
#print(full_strat[-1])

with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)      
