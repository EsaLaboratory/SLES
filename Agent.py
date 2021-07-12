import tensorflow.compat.v1 as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Conv1D, Concatenate, BatchNormalization, Reshape, Add, Subtract
from tensorflow.keras.constraints import Constraint
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.python.keras.backend import set_session

#from Env import Env
import numpy as np
import random
from collections import deque

tf.disable_v2_behavior() 

# For more repetitive results
np.random.seed(1)
random.seed(1)

class DiagonalWeight(Constraint):
    """Constrains the weights to be diagonal.
    """
    def __init__(self, min_value=-1.0, max_value=1.0):
    	self.min_value=min_value
    	self.max_value=max_value

    def __call__(self, w):
        N = K.int_shape(w)[-1]
        m = tf.eye(N)
        min_cond = tf.math.maximum(w*m, self.min_value)
        max_cond = tf.math.minimum(min_cond, self.max_value)
        return w*m #max_cond

#env=Env()
class Agent():
	def __init__(self, state_shape, sess, LRA, LRC, TAU, gamma,
		         min_replay=100, size_replay=10_000, mini_batch =32):
		self.sess = sess                     #session
		self.state_shape = state_shape       
		
		"""hyperparameters related to DN"""
		self.LRA = LRA                                   #learning rate \alpha for actor(generally = 0.01, 0.001 or 0.0001)
		self.LRC = LRC                                   #learning rate \alpha for critic(generally = 0.01, 0.001 or 0.0001)
		self.TAU = TAU                                   #parameter of targets' update (should be << 1 e.g, 0.125)

		self.min_replay = min_replay                     #the minimum samples in the buffer before we start the training
		self.size_replay = size_replay                   #size of the replay buffer
		self.replay_buffer = deque(maxlen=self.size_replay)   #buffer

		"""hyperparameters related to RL"""
		self.gamma = gamma

		self.graph = tf.get_default_graph()
		set_session(self.sess)
		# training actor and target actor 
		self.actor, self.input_actor = self.create_actor()
		plot_model(self.actor, to_file='actor_architecture.png', show_shapes=True, show_layer_names=True)
		self.target_actor, _ = self.create_actor()
		###initialize the weights of the target with the weights of training actor
		self.target_actor.set_weights(self.actor.get_weights())

		#training critic and target critic
		self.critic, self.critic_state_input, self.critic_action_input = self.create_critic()
		self.target_critic, _, _  = self.create_critic()
		plot_model(self.critic, to_file='critic_architecture.png', show_shapes=True, show_layer_names=True)
		###initialize the weights of the target with the weights of training critic
		self.target_critic.set_weights(self.critic.get_weights())


		######################## Part of the code that is bit tricky ########################################################
		# Temporary placeholder action gradient

		# where we will feed d(error)/d(critic) (from critic)
		self.actor_critic_grad = tf.placeholder(tf.float32, [None, self.state_shape[0], self.state_shape[0]+2])
			
		actor_weights = self.actor.trainable_weights
		# d(critic)/d(actor) (from actor)
		self.actor_grads = tf.gradients(self.actor.output, actor_weights, -self.actor_critic_grad) 
		#	                           [tf.math.negative(x) for x in self.actor_critic_grad]) 
		grads = zip(self.actor_grads, actor_weights)
		self.optimize = tf.train.AdamOptimizer(self.LRA).apply_gradients(grads)
		#AdamOptimizer
		
		self.critic_grads = tf.gradients(self.critic.output, self.critic_action_input)

		# Initialize for later gradient calculations
		self.sess.run(tf.global_variables_initializer())
		#self.sess.run(tf.initialize_all_variables())
		#######################################################################################################################

	def create_actor(self):
		actor_input = Input(shape=self.state_shape, name='state_input') #state_shape = (3(N-1) x 3)    
		h0 =  Conv1D(32, 3, padding='same', name='h0')(actor_input) #, activation='relu'
		h0 = BatchNormalization(name='h0_BN')(h0)
		#pooling?
		h1 = Conv1D(64, 3, padding='same', name='h1')(h0) #, activation='relu'
		h1 = BatchNormalization(name='h1_BN')(h1)
		h2 = Conv1D(128, 3, padding='same', name='h2')(h1) #, activation='relu'
		h2 = BatchNormalization(name='h2_BN')(h2)
		h3 = Conv1D(128, 3, padding='same', name='h3')(h2) #activation='relu', 
		h3 = BatchNormalization(name='h3_BN')(h3)

		fees = Conv1D(self.state_shape[0], 3, padding='same', activation='relu', name='fees')(h3)
		lambda_avg = Conv1D(1, 3, padding='same', activation='relu', name='lambda_avg')(h3)
		lambda_gap = Conv1D(1, 3, padding='same', activation='relu', name='lambda_gap')(h3)
		lambda_import = Add()([lambda_avg, lambda_gap])
		lambda_export = Subtract()([lambda_avg, lambda_gap])

		fees_final = Concatenate(axis=-1)([lambda_import, lambda_export, fees]) 

		model = Model(actor_input, fees_final)
		adam  = Adam(lr=self.LRA)
		model.compile(loss="mse", optimizer=adam)
		return model, actor_input

	def create_critic(self):
		state_input = Input(shape=self.state_shape, name='state_input')     
		action_input = Input(shape=(self.state_shape[0],self.state_shape[0]+2), name='action_input')			  		             #state (3N x 2)
		critic_input = Concatenate(axis=-1)([state_input, action_input]) #input (3N x 3N+4)

		h0 =  Conv1D(32, 3, padding='same', name='h0')(critic_input)
		h0 = BatchNormalization(name='h0_BN')(h0)
		#pooling?
		h1 = Conv1D(64, 3, padding='same', name='h1')(h0)
		h1 = BatchNormalization(name='h1_BN')(h1)
		h2 = Conv1D(128, 3, padding='same', name='h2')(h1)
		h2 = BatchNormalization(name='h2_BN')(h2)
		h3 = Conv1D(128, 3, padding='same', name='h3')(h2)
		h3 = BatchNormalization(name='h3_BN')(h3)

		Q = Conv1D(self.state_shape[0]+2, 3, padding='same', name='Q')(h3)
	
		model = Model([state_input, action_input], Q)
		adam  = Adam(lr=self.LRC)
		model.compile(loss="mse", optimizer=adam)
		return model, state_input, action_input

	def train(self):
		# Start training only if certain number of samples is already saved
		if len(self.replay_buffer) < self.min_replay:
			return

		# Get a minibatch of random samples from memory replay table
		minibatch = random.sample(self.replay_buffer, self.mini_batch)
		self._train_actor(minibatch)
		self._train_critic(minibatch)

	def _train_actor(self, minibatch):
		current_states = np.array([experience[0] for experience in minibatch])
		current_actions = np.array([experience[1] for experience in minibatch])

		grads = self.sess.run(self.critic_grads, feed_dict={
			self.critic_state_input:  current_states,
			self.critic_action_input: current_actions
		})[0]
		self.sess.run(self.optimize, feed_dict={
			self.actor_state_input: current_states,
			self.actor_critic_grad: grads
		})

	def _train_critic(self, minibatch):
		current_states = np.array([experience[0] for experience in minibatch])
		current_actions = np.array([experience[1] for experience in minibatch])
		#current_Q = self.critic.predict(current_states, current_actions)

		next_states = np.array([experience[3] for experience in minibatch])
		next_actions = self.target_actor.predict(next_states)
		next_Q = self.target_critic.predict([next_states, next_actions])

		#Elaborate the target r_t + gamma*Q(s_t+1, a_t+1)
		target, tar =[], np.zeros((self.state_shape[0],self.state_shape[0]+2))
		for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
			if not done:
				tar[:,0] = reward[:,0] + self.gamma*next_Q[index][:,0]
				tar[:,1] = reward[:,1] + self.gamma*next_Q[index][:,1]
				tar[:,2] = reward[:,2:] + self.gamma*next_Q[index][:,2:]
			else:
				tar = reward
			target.append(tar)
		self.critic.fit([current_states, current_actions], np.array(target), verbose=0)

	def update_targets(self):
		self._update_actor()
		self._update_critic()

	def _update_actor(self):
		actor_weights  = self.actor.get_weights()
		target_actor_weights = self.target_actor.get_weights()
	
		for i in range(len(target_actor_weights)):
			target_actor_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* target_actor_weights[i]
		self.target_actor.set_weights(target_actor_weights)

	def _update_critic(self):
		critic_weights  = self.critic.get_weights()
		target_critic_weights = self.target_critic.get_weights()
	
		for i in range(len(target_critic_weights)):
			target_critic_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* target_critic_weights[i]
		self.target_critic.set_weights(target_critic_weights)



	def get_action(self, state, sigma):
		action = self.actor.predict(state)
		action = action + np.random.normal(0,sigma, action.shape) 
		action[:,:,0] = np.max(action[:,:,0], 0) 
		for i in range(action.shape[0]):
			for j in range(action.shape[1]):
				if action[i, j, 1] > action[i,j,0]:
					action[i, j, 1] = action[i,j,0]
		action[:,:,2:] = np.max(action[:,:,2:], 0) 
		return action

	def update_replay_buffer(self, experience):
		self.replay_buffer.append(experience)
