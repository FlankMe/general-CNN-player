# -*- coding: utf-8 -*-
"""
General CNN player

Convolutional neural network that solves OpenAI gym environments 
(https://gym.openai.com) from raw pixel data via Q-learning 

Download the code, assign to game_name the name of environment you wish to 
run, and let the script learn how to solve it.  
Note the code only works for environments with discrete action space and 
3-dimensional observation space representing the raw pixel data of the screen.

https://github.com/FlankMe/general-CNN-player
@author: Riccardo Rossi
"""

# Choice of the game and definition of the goal
game_name = 'Pong-v0'
MAX_EPISODES = 2500
CONSECUTIVE_EPISODES = 100   # Number of trials' rewards to average for solving
IS_RECORDING = True 

# Fine-tuning the EPSILON_DECAY parameters will lead to better results for 
# some environments and worse for others. As this code is a go at a 
# general player, it is neater to treat it as a global constant 
EPSILON_DECAY = 0.99

# Import basic libraries
import numpy as np


"""
Main loop for the game's initialization and runs
"""
def main():

    # Import gym and launch the game
    import gym
    env = gym.make(game_name)
    
    assert isinstance(env.action_space, gym.spaces.discrete.Discrete), (
        'env.action_space is not Discrete and is currently unsupported')
    assert isinstance(env.observation_space, gym.spaces.box.Box), (
        'env.observation_space is not continuous and is currently unsupported')
    assert len(env.observation_space.shape) == 3, (
        '''CNN not required for this environment
        Download and run general-gym-player from my github page
        https://github.com/FlankMe/general-gym-player''')
    if IS_RECORDING:
        env.monitor.start('results-' + game_name, force=True)
    
    # Create a gym instance and initialize the agent
    agent = Agent(env.action_space.n,
                  env.observation_space.shape)
    reward, done, = 0.0, False
        
    
    # Start the game loop
    for episode in range(1, MAX_EPISODES + 1):
        obs, done = env.reset(), False
        action = agent.act(obs, reward, done, episode)
        
        while not done:
            
            # Un-comment to show the game on screen 
            #env.render()
            
            # Decide next action and feed the decision to the environment         
            obs, reward, done, _ = env.step(action)  
            action = agent.act(obs, reward, done, episode)        
            
    # Save info and shut activities
    env.close()
    agent.close()
    if IS_RECORDING:
        env.monitor.close()
    

"""
The general game player implements the Q-learning method with minibatches
"""
class Agent():

    def __init__(self, n_actions, obs_space):

        # Initialization of useful variables and constants
        self._N_ACTIONS = n_actions
        self._nn = FeedForwardNeuralNetwork(n_actions, obs_space)
        self._STATE_FRAMES = self._nn._STATE_FRAMES
        self._close = lambda : self._nn.close()
        
        # Hyperparameters of the training
        self._DISCOUNT_FACTOR = 0.99    # discount of future rewards
        self._TRAINING_PER_STAGE = 1
        self._MINIBATCH_SIZE = 32      
        self._REPLAY_MEMORY = 20000     

        # Exploration/exploitations parameters
        self._epsilon = 1.
        self._EPSILON_DECAY = EPSILON_DECAY
        self._EPISODES_PURE_EXPLORATION = 10
        self._MIN_EPSILON = 0.1

        # Define useful variables
        self._total_reward, self._list_rewards = 0.0, []
        self._last_action = np.zeros(self._N_ACTIONS)
        self._previous_observations, self._last_state = [], None
        self._LAST_STATE_IDX = 0
        self._ACTION_IDX = 1 
        self._REWARD_IDX = 2
        self._CURR_STATE_IDX = 3
        self._TERMINAL_IDX = 4
        
        
    def act(self, obs, reward, done, episode):
        
        self._total_reward += reward
        
        if done:
            self._list_rewards.append(self._total_reward)
            average = np.mean(self._list_rewards[-CONSECUTIVE_EPISODES:])
            print ('Episode', episode, 'Reward', self._total_reward, 
                   'Average Reward', round(average, 2))           
            self._total_reward = 0.0
            self._epsilon = max(self._epsilon * self._EPSILON_DECAY,
                                self._MIN_EPSILON)       
                        
        # Compress the input image into a pre-set format
        compressedImage = self._nn._compressImage(obs)                                    

        # If the CNN has no last state, fill it by using the current state,
        # choose a random action, and return the action to the game
        if self._last_state is None:
            self._last_state = compressedImage.copy()
            for _ in range(self._STATE_FRAMES - 1):
                self._last_state = np.append(self._last_state, 
                                             compressedImage,
                                             axis=3)
            value_per_action = self._nn.predict(self._last_state)
            chosen_action_index = np.argmax(value_per_action)  
            self._last_action = np.zeros(self._N_ACTIONS)
            self._last_action[chosen_action_index] = 1
            return (chosen_action_index)
                
        # Update the current state 
        # current_state is made by (STATE_FRAMES) reduced images
        current_state = np.append(compressedImage, 
                                  self._last_state[:,:,:,:-3], 
                                  axis=3)
         
        # Store the last transition
        new_observation = [0 for _ in range(5)]
        new_observation[self._LAST_STATE_IDX] = self._last_state.copy()
        new_observation[self._ACTION_IDX] = self._last_action.copy()
        new_observation[self._REWARD_IDX] = reward
        new_observation[self._CURR_STATE_IDX] = current_state.copy()
        new_observation[self._TERMINAL_IDX] = done
        self._previous_observations.append(new_observation)
        self._last_state = current_state.copy()
            
        # If the memory is full, pop the oldest stored transition
        while len(self._previous_observations) >= self._REPLAY_MEMORY:
            self._previous_observations.pop(0)
        
        # Only train and decide after enough episodes of random play
        if episode > self._EPISODES_PURE_EXPLORATION:
  
            for _ in range(self._TRAINING_PER_STAGE):
                self._train()       
                
            # Chose the next action with an epsilon-greedy approach
            if np.random.random() > self._epsilon:
                value_per_action = self._nn.predict(self._last_state)
                chosen_action_index = np.argmax(value_per_action)  
            else:
                chosen_action_index = np.random.randint(0, self._N_ACTIONS)
        
        else:
            chosen_action_index = np.random.randint(0, self._N_ACTIONS)
    
        next_action_vector = np.zeros([self._N_ACTIONS])
        next_action_vector[chosen_action_index] = 1
        self._last_action = next_action_vector
          
        return (chosen_action_index)
        

    def _train(self):
        
        # Sample a mini_batch to train on
        permutations = np.random.permutation(
            len(self._previous_observations))[:self._MINIBATCH_SIZE] 
        previous_states = np.concatenate(
            [self._previous_observations[i][self._LAST_STATE_IDX]
            for i in permutations], 
            axis=0)
        actions = np.concatenate(
            [[self._previous_observations[i][self._ACTION_IDX]] 
            for i in permutations], 
            axis=0)
        rewards = np.array(
            [self._previous_observations[i][self._REWARD_IDX] 
            for i in permutations]).astype('float')
        current_states = np.concatenate(
            [self._previous_observations[i][self._CURR_STATE_IDX] 
            for i in permutations], 
            axis=0)
        done = np.array(
            [self._previous_observations[i][self._TERMINAL_IDX] 
            for i in permutations]).astype('bool')

        # Calculates the value of the current_states (per action)
        valueCurrentstates = self._nn.predict(current_states)
        
        # Calculate the empirical target value for the previous_states
        valuePreviousstates = rewards.copy()
        valuePreviousstates += ((1. - done) * 
                                self._DISCOUNT_FACTOR * 
                                valueCurrentstates.max(axis=1))

        # Run a training step
        self._nn.fit(previous_states,
                          actions, 
                          valuePreviousstates)


"""
Plain Feed Forward Neural Network
The chosen activation function is the Leaky ReLU function
"""
class FeedForwardNeuralNetwork:
    
    def __init__(self, n_actions, obs_space):

        # NN variables
        self._generateNetwork(n_actions, obs_space)
        self._previous_observations = []
        self._PARAMETERS_FILE_PATH = 'Parameters_CNN.ckpt'


    def _generateNetwork(self, n_actions, obs_space):
        """
        The network is implemented in TensorFlow
        Change this method if you wish to use a different library
        """
        
        import tensorflow as tf   
        self._ALPHA = 1e-3              # learning rate    
        RESIZED_SCREEN = 84
        self._STATE_FRAMES = 3         # states/images used for taking a decision
        
        # Graph for compressing the input image 
        x, y, z = obs_space
        self._image_input_layer = tf.placeholder("float", 
            [None, x, y, z])
        image_step_size_x = int(np.ceil(float(x / RESIZED_SCREEN)))
        image_step_size_y = int(np.ceil(float(y / RESIZED_SCREEN)))
        extra_pad_x = RESIZED_SCREEN - int(x / image_step_size_x)
        extra_pad_y = RESIZED_SCREEN - int(y / image_step_size_y)
        self._image_output_layer = tf.nn.max_pool(
                self._image_input_layer, 
                ksize=[1, image_step_size_x, image_step_size_y, 1],
                strides=[1, image_step_size_x, image_step_size_y, 1], 
                padding="VALID")                                         
        
        # Function for compressing (and reshaping) the image
        self._compressImage = lambda obs : np.pad(
            self._session.run(
                self._image_output_layer, 
                feed_dict={self._image_input_layer: np.array([obs])})/255.0,    
            ((0,0), (0,extra_pad_x), (0,extra_pad_y), (0,0)),
            mode='constant')   

        CONVOLUTION_FILTER_VECTOR = [6, 6, 4]
        CONVOLUTION_STRIDE_VECTOR = [3, 3, 2]
        CONVOLUTION_KERNEL_VECTOR = [16, 16, 36]
        CONVOLUTION_INPUT_VECTOR = ([z * self._STATE_FRAMES] + 
                                    CONVOLUTION_KERNEL_VECTOR[:-1])
        FEED_FWD_VECTOR = [(3**2) * CONVOLUTION_KERNEL_VECTOR[-1], 64, 
                           n_actions]      
        
        # The chosen activation function is the Leaky ReLU function
        self._activation = lambda x : tf.maximum(0.01*x, x)

            
        # Initialization parameters
        INITIALIZATION_STDDEV = 0.1
        INITIALIZATION_MEAN = 0.00
        INITIALIZATION_BIAS = -0.001

        # Convolutional layers
        self._input_layer = tf.placeholder("float", 
                                           [None, 
                                            RESIZED_SCREEN, 
                                            RESIZED_SCREEN, 
                                            z * self._STATE_FRAMES])
        self._convolutional_weights = []
        self._convolutional_bias = []
        self._hidden_convolutional_layer = [self._input_layer]

        for i in range(len(CONVOLUTION_FILTER_VECTOR)):
            self._convolutional_weights.append(tf.Variable(tf.truncated_normal(
                [CONVOLUTION_FILTER_VECTOR[i], 
                 CONVOLUTION_FILTER_VECTOR[i], 
                 CONVOLUTION_INPUT_VECTOR[i], 
                 CONVOLUTION_KERNEL_VECTOR[i]], 
                mean=INITIALIZATION_MEAN, 
                stddev=INITIALIZATION_STDDEV)))
            self._convolutional_bias.append(tf.Variable(tf.constant(
                INITIALIZATION_BIAS, 
                shape=[CONVOLUTION_KERNEL_VECTOR[i]])))
            self._hidden_convolutional_layer.append(
                self._activation(tf.nn.conv2d(
                                    self._hidden_convolutional_layer[i], 
                                    self._convolutional_weights[i], 
                                    strides=[1, 
                                             CONVOLUTION_STRIDE_VECTOR[i],
                                             CONVOLUTION_STRIDE_VECTOR[i], 
                                             1], 
                                    padding="VALID") 
                                + self._convolutional_bias[i]))
                                
        # Feed forward layers
        self._hidden_activation_layer = [tf.reshape(
            self._hidden_convolutional_layer[-1], 
            [-1, FEED_FWD_VECTOR[0]])]
        self._feed_forward_weights = []
        self._feed_forward_bias = []

        for i in range(len(FEED_FWD_VECTOR) - 2):
            self._feed_forward_weights.append(tf.Variable(tf.truncated_normal(
                [FEED_FWD_VECTOR[i], 
                 FEED_FWD_VECTOR[i+1]], 
                mean=INITIALIZATION_MEAN, 
                stddev=INITIALIZATION_STDDEV)))
            self._feed_forward_bias.append(tf.Variable(tf.constant(
                INITIALIZATION_BIAS, shape=[FEED_FWD_VECTOR[i+1]])))
            self._hidden_activation_layer.append(self._activation(
                    tf.matmul(self._hidden_activation_layer[i], 
                              self._feed_forward_weights[i]) 
                    + self._feed_forward_bias[i])
                    )
                    
        # The calculation of the state-action value function does not 
        # require the neurons' activation function
        self._feed_forward_weights.append(tf.Variable(tf.truncated_normal(
            [FEED_FWD_VECTOR[-2], 
             FEED_FWD_VECTOR[-1]], 
            mean=INITIALIZATION_MEAN, 
            stddev=INITIALIZATION_STDDEV)))
        self._feed_forward_bias.append(tf.Variable(tf.constant(
            INITIALIZATION_BIAS, 
            shape=[FEED_FWD_VECTOR[-1]])))
        self._state_value_layer = (tf.matmul(self._hidden_activation_layer[-1], 
                                             self._feed_forward_weights[-1]) 
                                    + self._feed_forward_bias[-1])

        # Define the logic of the optimization
        self._action = tf.placeholder("float", [None, n_actions])
        self._target = tf.placeholder("float", [None])
        self._action_value_vector = tf.reduce_sum(tf.mul(
            self._state_value_layer, self._action), reduction_indices=1)
        self._cost = tf.reduce_sum(tf.square(
            self._target - self._action_value_vector))
        self._alpha = tf.placeholder('float')
        self._train_operation = tf.train.AdamOptimizer(
            self._alpha).minimize(self._cost)
        self._session = tf.Session()

        operation_intizializer = tf.initialize_all_variables()
        self._saver = tf.train.Saver()

        try:
            self._saver.restore(self._session, self._PARAMETERS_FILE_PATH)
            print ('Calibrated parameters SUCCESSFULLY LOADED.',
                   flush=True)
        except:
            self._session.run(operation_intizializer)
            print ('It was not possible to load calibrated parameters.',
                   flush=True)
   
        # Definition of feed_forward and optimization functions
        self._feedFwd = lambda state : self._session.run(
                            self._state_value_layer, 
                            feed_dict={self._input_layer: state})
                            
        self._backProp = lambda valueStates, actions, valueTarget : (
            self._session.run(self._train_operation, 
            feed_dict={self._input_layer: valueStates,
                       self._action: actions,
                       self._target: valueTarget,
                       self._alpha : self._ALPHA}))
                                         
    def close(self):

        # If training, save the RAM memory to file
        if self._is_training:
            self._saver.save(self._session, self._PARAMETERS_FILE_PATH)
            
        # Close the session and clear TensorFlow's graphs             
        from tensorflow.python.framework import ops
        ops.reset_default_graph() 
        self._session.close()
                                         
    def predict(self, state):    
        return(self._feedFwd(state))
       
    def fit(self, valueStates, actions, valueTarget):                      
        self._backProp(valueStates, actions, valueTarget)


if __name__=="__main__":
   main()

        
