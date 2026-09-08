import gymnasium as gym
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory
from train import KerasRLCompatWrapper, build_model

if __name__ == '__main__':
    # Initialize environment with rendering enabled
    env = gym.make('ALE/Breakout-v5', render_mode='human')
    env = KerasRLCompatWrapper(env)
    
    nb_actions = env.action_space.n
    window_length = 4
    
    # Rebuild the model architecture
    model = build_model(window_length, nb_actions)
    memory = SequentialMemory(limit=10000, window_length=window_length)
    
    # Use GreedyQPolicy for evaluation
    policy = GreedyQPolicy()
    
    # Reconstruct the agent
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, 
                   nb_steps_warmup=10, target_model_update=10000, policy=policy)
    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])
    
    # Load the trained policy network
    dqn.load_weights('policy.h5')
    
    # Play 5 episodes
    print("Starting agent evaluation...")
    dqn.test(env, nb_episodes=5, visualize=True)
