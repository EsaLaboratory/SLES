import random
import numpy as np 
import pandas as pd
from tqdm import tqdm
import os
import pickle
import time
import tensorflow as tf
from tensorflow.python.keras import backend as K

from Env import Env
from Agent import Agent

tf.compat.v1.disable_eager_execution()

# Total number of episodes
EPISODES = 5

# Exploration settings
sigma = 1
decay = 0.95 #0.998


# For more repetitive results
random.seed(1)
np.random.seed(1)

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')


#sess = tf.Session()
sess = tf.compat.v1.Session()
K.set_session(sess)
#tf.compat.v1.keras.backend.set_session(sess)
#if __name__ == '__main__': 
#network_type = '4Bus-YY-Bal' 
env = Env(network_type = '13BusOxEmf')
state_shape = env.state_shape
#state_shape = env.state_shape
agent = Agent(state_shape, sess, 0.01, 0.001, 0.125, 0.98)

def run_episode(sigma):
	done = False
	current_state = env.reset()
	reward_ep, duration_ep, violations_ep, revenue_ep, revenue_pro_ep = [], [], [], [], []

	while(not done):
		start_step = time.time()
		actions = agent.get_action(current_state, sigma) #current_state[t] (3N,2)
		next_state, reward_step, violations_step, revenue_step, revenue_pro_step, done = env.step(np.array(actions))

		agent.update_replay_buffer([current_state, actions[0], reward_step, next_state, done]) 
		agent.train()
		agent.update_targets()
		current_state = next_state

		duration_step = time.time() - start_step
		duration_ep.append(duration_step)
		reward_ep.append(reward_step)
		violations_ep.append(violations_step)
		revenue_ep.append(revenue_step)
		revenue_pro_ep.append(revenue_pro_step)

	return reward_ep, duration_ep, violations_ep, revenue_ep, revenue_pro_ep
	
# Iterate over episodes
reward, duration, violations, revenue, revenue_pro = [], [], [], [], []
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
	reward_ep, duration_ep, violations_ep, revenue_ep, revenue_pro_ep = run_episode(sigma)
	sigma *= decay 
	reward.append(reward_ep)
	duration.append(duration_ep)
	violations.append(violations_ep)
	revenue.append(revenue_ep)
	revenue_pro.append(revenue_pro_ep)

	if (not(episode // 5)):
		agent.actor.save(f'models\\actor__ep_{episode:_>7.2f}__R{np.sum(reward_ep):_>7.2f}__{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.h5')
		agent.critic.save(f'models\\critic__ep_{episode:_>7.2f}__R{np.sum(reward_ep):_>7.2f}__{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.h5')

results ={'reward': reward,
		  'duration': duration,
		  'violations': violations,
		  'revenue': revenue,
		  'revenue_pro': revenue_pro}
pickle.dump((results), open( "Data/Save_Data/RL_5Episodes_3pros_4bus.p", "wb" ) )
