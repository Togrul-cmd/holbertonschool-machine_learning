import gymnasium as gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Permute
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

class KerasRLCompatWrapper(gym.Wrapper):
    """
    Wrapper to make Gymnasium compatible with keras-rl2.
    Adjusts reset(), step(), and render() to match the legacy Gym API.
    """
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Combine terminated and truncated into a single 'done' boolean
        return obs, reward, terminated or truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # keras-rl expects only the observation, not the info dict
        return obs

    def render(self, mode='human'):
        return self.env.render()

def build_model(window_length, nb_actions):
    """Builds a basic Convolutional Neural Network for Atari games."""
    model = Sequential()
    # Breakout images are (210, 160, 3). Permute to match channels if needed.
    model.add(Permute((2, 3, 1), input_shape=(window_length, 210, 160, 3)))
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))
    return model

if __name__ == '__main__':
    # Initialize the Breakout environment
    env = gym.make('ALE/Breakout-v5')
    env = KerasRLCompatWrapper(env)
    
    nb_actions = env.action_space.n
    window_length = 4
    
    # Initialize model, memory, and policy
    model = build_model(window_length, nb_actions)
    memory = SequentialMemory(limit=100000, window_length=window_length)
    policy = EpsGreedyQPolicy()
    
    # Set up the DQNAgent
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, 
                   nb_steps_warmup=1000, target_model_update=10000, policy=policy)
    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])
    
    # Train the agent
    print("Starting training...")
    dqn.fit(env, nb_steps=10000, visualize=False, verbose=2)
    
    # Save the policy network
    dqn.save_weights('policy.h5', overwrite=True)
    print("Model saved to policy.h5")
