# Reinforcement Learning — Teaching the Machine to Gamble with Q-learning

Reinforcement Learning is an area of Artificial Intelligence and Machine Learning that involves simulating many scenarios in order to optimize the outcomes. One of the most used approaches in Reinforcement Learning is the Q-learning method. In Q-learning, a simulation environment is created and the algorithm involves a set of ‘S’ states for each simulating scenario, a set of ‘A’ actions, and an agent that takes these actions to permeate through the states.

Each time the agent takes an action ‘a’ within the set ‘A’ it transitions from one state to another into the environment. Performing an action in a specific state in the environment returns a reward for the agent, which can be good or bad. The model’s objective is always to find a set of actions that maximize the reward and evolve in the best possible way for the environment. There are several different techniques within a group of Reinforcement Learning algorithms, from mathematical models with defined policies to more complex models such as evolutionary models and deep learning models such as Deep Reinforcement Learning.

## Q-Learning

Q-learning is a Reinforcement Learning off policy model that aims to find the best action to take based on the current state, in this case without a defined action policy. This model is considered an off-policy model because the Q-learning function learns through actions that are outside the current policy, in other words, its learning follows in an exploratory way by taking random actions to create an action policy that maximizes the total reward of the episode.
Why Q? And what would this policy be?

The letter Q stands for Quality and the learning model is based on a Q table (Quality table) which is the policy of actions that the model can use in the environment for each state. Thus, we have a table [state, action] that represents a policy where each action has a quality value (Q value) for each state.

Actions are part of the environment, as in the example above we have actions of walking to the North, South, East, West, etc. The concept of state is a set of variables that represent the evolution of the model through the simulated environment.

With each action taken, the Q value is updated by the concept of Value Iteration following the decision-making process known as the Markov Decision Process (MDP) and the Bellman equation. The equation consists of the old Q value for the action taken along with the action’s reward and the maximum Q value for the new state, both discounted from the “learning rate” that weighs the quality between the current value and the new value. Therefore, the model depends only on the [state, action] and the reward observed in the action taken into the environment.

## Model Development — Code

The libraries used for development were, according to their imports:
````
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

import pandas as pd
````
In the beginning, we have the definition of some values ​​for the Bellman equation, the episodes, and the learning rate, in addition to initial values ​​for the simulation environment of the bets with the investment value per game and the initial portfolio value.

````
HM_EPISODES = 45000
epsilon = 0.9
EPS_DECAY = 0.9998
LEARNING_RATE = 0.1
DISCOUNT = 0.95
````

* Learning rate defines the weight of the reward in the exploration with the update of the quality (Q) value.
* At first I used 45000 episodes which is the number of iterations that the model will follow.
* Epsilon and eps decay indicates the initial exploitation factor and its decay with iterations
* Discount represents the weight of the future value of Q after the action

After definition, the model loads a previous Q table if necessary or starts a new Q table to start the iterations.
Iterations are made with each episode that goes through each of the games previously selected.

## Creating the Simulation Environment

The simulation environment has 3 different functions:
* action() that performs the action defined by the model by selecting a strategy and a bet
* strategy() that receives the action from action() function and effectively performs the previously defined strategy. We have 3 strategies in this model: “Min” who chooses the team with the lowest odd, “Max” who chooses the highest odd, and “ML” who chooses the team according to a machine learning model developed at another time.
* bet() that receives action() and strategy() values and applies the strategy in the dataset with the real values of the game, thus returning the result value of the bet, whether it is a hit or a miss.

Right after, we have within the loops of episodes and games the definition of the exploitation policy:

````
if np.random.random() > epsilon:
   action_n = np.argmax(q_table[obs])
else:
   action_n = np.random.randint(0,high=2)
````

We have a section that performs the profit and reward calculation of each bet, updates the accumulated profit, and sets up a new observation state, a state that is represented by the value of the portfolio and the accumulated profit.

````
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
````

Finally, the application of the Bellman equation to update the Q-value for that state.

````
try:
	  max_future_q = np.max(q_table[new_obs])
except:
	  q_table[new_obs] = [0 for i in range(val)]
	  max_future_q = np.max(q_table[new_obs])
	            
	        
current_q = q_table[obs][action_n] 
	 
new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

q_table[obs][action_n] = new_q #update actual q
````

## Conclusion

According to the results presented, it can be concluded that the Reinforcement Learning model successfully learns to dominate the betting environment and choose the best strategy for each match. For me, it is very interesting to use machine learning to bet on football games without any previous policy programming and still finishing the simulation with great success in betting. For a real case scenario, we will deal with probabilities since we do not have the results for the matches before it happens, however, due to simulation we could find a more optimal way to do our bets.
There are still several more complex strategies in the area of ​​RL that I will bring in the future, my goal with this article was to bring a little of my experience in the area to help those interested and contribute to the community, I hope I have helped in understanding the subject and I am available for any contact. Thanks for reading.

If you want to know more, feel free to contact me on LinkedIn! https://www.linkedin.com/in/octavio-b-santiago/

