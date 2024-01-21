import numpy as np
import gym
import matplotlib.pyplot as plt

# Initialize the FrozenLake environment
env = gym.make('FrozenLake-v1', is_slippery=True,render_mode='human')

# Initialize Q-table with zeros
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.8   # Learning rate
gamma = 0.3   # Discount factor
epsilon = 1.0 # Epsilon-greedy probability
num_episodes = 5000
max_steps = 100 # Maximum steps per episode
decay_rate = 0.1 # Decay rate for epsilon
epsilon_end = 0.01 # Minimum epsilon
epsilon_decay = np.exp(np.log(epsilon_end) / num_episodes) # Decay rate as a function of number of episodes
# For stats
rewards_per_episode = []
steps_per_episode = []
q_tables_at_steps = {} # To store Q-tables at specific steps

# Q-learning algorithm
for episode in range(num_episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0
    steps = 0

    for _ in range(max_steps):
        # Epsilon-greedy action selection
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(q_table[state])  # Exploit learned values

        next_state, reward, done,_,_ = env.step(action)
        total_reward += reward
        steps += 1

        # Q-table update
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        state = next_state

        if done:
            break

    # Decaying epsilon
    epsilon = max(epsilon_end, epsilon_decay * epsilon)

    # Record rewards and steps
    rewards_per_episode.append(total_reward)
    steps_per_episode.append(steps if done else max_steps)

    # Store Q-tables at specific episodes
    if episode in [500, 2000, num_episodes - 1]:
        q_tables_at_steps[episode] = q_table.copy()

# Plot of the reward per episode
plt.plot(rewards_per_episode)
plt.title('Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()

# Plot of the average number of steps to the goal over the last 100 episodes
average_steps = [np.mean(steps_per_episode[i:i+100]) for i in range(0, len(steps_per_episode), 100)]
plt.plot(average_steps)
plt.title('Average Number of Steps to Goal (per 100 episodes)')
plt.xlabel('Episode (in hundreds)')
plt.ylabel('Average Steps')
plt.show()
