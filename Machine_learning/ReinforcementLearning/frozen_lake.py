import gym
import numpy as np
import random
import matplotlib.pyplot as plt

#call the environment
env = gym.make('FrozenLake-v0')

#create a Q table
q_table = np.zeros([env.observation_space.n, env.action_space.n])

#Hyperparameter
gamma = 0.95
alpha = 0.80 #temporal difference weighting
epsilon = 0.10

#Create a list to visualize the rewards.
reward_list = []

#Start to learn
episode_number = 100000 # Let's suppose becoming 100000 episodes
for i in range(1,episode_number):
    state = env.reset() #reset the environment at the beginning of each chapter
    reward_count = 0
    while True:
        #10% exploration rate, 90%exploitation rate
        if random.uniform(0,1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        
        #take an action in the environment, in return for the next status, the reward earned and whether it was done
        next_state, reward, done, _ = env.step(action)

        #Q learning function
        old_value = q_table[state,action] #old_value
        next_max = np.max(q_table[next_state]) #next_max

        next_value = (1-alpha)*old_value + alpha(reward + gamma*next_max)

        #Update Q table
        q_table[state, action] = next_value

        #update state
        state = next_state

        #calculate total reward
        reward_count += reward

        #if the episode is finished, break and start the new one
        if done:
            break
    if i%5000 == 0:
        print(f"Episode: {i}")
    if i%1000 == 0:
        reward_list.append(reward_count)

plt.figure()
plt.plot(reward_list)
plt.xlabel("Numbe of Episodes")
plt.ylabel("Reward")
plt.show()