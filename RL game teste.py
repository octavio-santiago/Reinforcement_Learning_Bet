import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

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
carteira = 6
games = 3

# MAX, MIN, MED

start_q_table = None # or filename saved

LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3

ml_result = [[0.2755,0.226295,0.498205],
             [0.364,0.3025,0.3335],
             [0.515,0.262,0.223]]

probs = [[4.33,3.25,1.85],
         [2.1,3.1,3.6],
         [1.44,4,7.25]]
#probs [v,e,d]
results_g = [3.25,2.1,1.44] # [e,v,v]

def bet(idx_game,value):
    if value == results_g[idx_game]:
        return 1
    elif value == 0:
        return 0
    else:
        return -1
def strategy(aposta):
    result = [0,0,0]
    odds = [1,1,1]
    for i,v in enumerate(aposta):
            odd = min(probs[i]) #odd escolhida
            odd = np.random.randint(0,3) # escolha randomica
            odd_idx = ml_result[i].index(max(ml_result[i])) # escolha do ML
            odd = probs[i][odd_idx]
            
            r = bet(i,odd*v)
            result[i] = r
            odds[i] = odd
            
    total = (result,odds)
    return total

choices = {0:(0,0,0),1:(1,0,0),2:(1,1,0),3:(1,1,1),4:(0,1,0),5:(0,1,1),6:(0,0,1),7:(1,0,1)}

def action(choice):
    # apostar ou nao apostar
    result = [0,0,0]
    odds = [1,1,1]
    if choice == 0:
        aposta = (0,0,0)
        total = strategy(aposta)       
    elif choice == 1:
        aposta = (1,0,0)
        total = strategy(aposta) 
    elif choice == 2:
        aposta = (1,1,0)
        total = strategy(aposta) 
    elif choice == 3:
        aposta = (1,1,1)
        total = strategy(aposta) 
    elif choice == 4:
        aposta = (0,1,0)
        total = strategy(aposta) 
    elif choice == 5:
        aposta = (0,1,1)
        total = strategy(aposta) 
    elif choice == 6:
        aposta = (0,0,1)
        total = strategy(aposta) 
    elif choice == 7:
        aposta = (1,0,1)
        total = strategy(aposta) 
        
    #total = (result,odds)
    return total #result = (1,1,0) acertou, errou a escolha ou nao escolheu

        
# Q table
if start_q_table is None:
    val = 2**games
    print("generating new table....")
    q_table = {}
    for j1 in range(0,10):
        for j2 in range(-1,6):
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
lucro = 0
carteira_time = []
lucro_time = []
for episode in range(HM_EPISODES):

    if episode % SHOW_EVERY == 0:
        print(f"on # {episode}, epsilon: {epsilon}")
        print(f"{SHOW_EVERY} ep mean {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False
        
    # run action
    episode_reward = 0
    for i in range(200):
        # aposta dependendo da odds dos jogos (escolha aleatoria)
        #obs = (player-food, player-enemy)
        #obs = (0,0,0) # 3 jogos, 0 neg e 1 pos
        obs = (carteira+lucro,lucro)
        if np.random.random() > epsilon:
            action_n = np.argmax(q_table[obs])
        else:
            action_n = np.random.randint(0,8)
            
        result = action(action_n) #result = [(,),(,)]resultado de cada jogada e odds
        erros = 0
        #contar os zeros
        jogos = [0,0,0] #odds ganhas
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
            l_tot += invest*jogos[i]
  
        lucro = l_tot - erros*invest
        #lucro = result[0][0]*invest*result[1][0] + result[0][1]*invest*result[1][1] + result[0][2]*invest*result[1][2] - erros*invest
        '''if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY'''
        
        reward = lucro
        '''if lucro <= 0:
            #reward = -ENEMY_PENALTY
            reward = lucro
        elif lucro > 0:
            #reward = FOOD_REWARD
            reward = lucro'''

        #new_obs = (player-food, player-enemy)
        new_obs = (carteira+lucro,lucro) # get new state
        try:
            max_future_q = np.max(q_table[new_obs])
        except:
            val = 2**games
            q_table[new_obs] = [0 for i in range(val)]
            max_future_q = np.max(q_table[new_obs])
            
        current_q = q_table[obs][action_n] 

        '''if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        elif reward == -ENEMY_PENALTY:
            new_q = -ENEMY_PENALTY
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q) #formula q learning
        '''
        
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        
        q_table[obs][action_n] = new_q #update actual q

        episode_reward += reward
        
        #stop loss ou take profit
        if reward > 0.3*carteira:
            break

    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY
    
    carteira_time.append(obs[0])
    lucro_time.append(obs[1])

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode = "valid")

plt.subplot(211)
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"reward {SHOW_EVERY}ma")
plt.xlabel("episode #")

plt.subplot(212)
plt.plot([i for i in range(len(carteira_time))], carteira_time)
plt.plot([i for i in range(len(lucro_time))], lucro_time)

plt.show()

print(f"Obs:{obs},Q table:{q_table[obs][action_n]},Action:{choices[action_n]},Lucro: {lucro}")
print(f"Carteira: {obs[0]},Lucro: {obs[1]}")

with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)      
