import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import random
from collections import deque
import gymnasium as gym
from tensorflow.summary import create_file_writer
# Remove warnings
import os
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from IPython.display import clear_output

if tf.test.is_gpu_available():
    print("GPU is available")
else:
    print("No GPU found; using CPU")

RENDER = False

hidden_layers_3 = [150, 50, 100]  # For the 3 hidden layer network
hidden_layers_5 = [64, 64, 64, 64, 64]  # For the 5 hidden layer network
n_episodes = 3000
batch_size = 256
gamma = 0.98
epsilon_start = 1
epsilon_end = 0.05
epsilon_decay = 0.99
learning_rate = 0.001
update_freq = 1000
import datetime
log_dir = f"""logs/DuelingDQN_5Layer_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}"""

class DDQNAgent:
    def __init__(self, input_size, output_size, hidden_layers):
        self.policy_net = self.create_dueling_model(input_size, output_size, hidden_layers)
        self.target_net = self.create_dueling_model(input_size, output_size, hidden_layers)
        self.update_target_net()

    def create_dueling_model(self, input_size, output_size, hidden_layers):
        inputs = tf.keras.Input(shape=(input_size,))
        x = inputs
        for units in hidden_layers:
            x = Dense(units=units, activation="relu")(x)
        # Value stream
        value_stream = Dense(units=1, activation=None)(x)
        # Advantage stream
        advantage_stream = Dense(units=output_size, activation=None)(x)
        # Combine streams
        q_values = value_stream + (advantage_stream - tf.reduce_mean(advantage_stream, axis=1, keepdims=True))
        
        model = tf.keras.Model(inputs=inputs, outputs=q_values)
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=learning_rate))
        return model
    
    def update_target_net(self):
        self.target_net.set_weights(self.policy_net.get_weights())

    def train(self, states, actions, updated_q_values, n_actions):
        masks = tf.one_hot(actions, n_actions)

        with tf.GradientTape() as tape:
            q_values = self.policy_net(states)
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            loss = tf.keras.losses.MSE(updated_q_values, q_action)
        grads = tape.gradient(loss, self.policy_net.trainable_variables)
        return grads, loss

def plot_rewards(rewards_list, episode, epsilon):
    if episode % 5 == 0:
        clear_output(wait=False)
        plt.figure(figsize=(10, 5))
        plt.plot(rewards_list, label="Episode Rewards")
        
        # Calculate and plot the moving average
        if len(rewards_list) >= 100:
            moving_average = np.mean(rewards_list[-100:])
            plt.axhline(y=moving_average, color='r', linestyle='-', label="Moving Average (100 episodes)")
        
        plt.title(f"Episode {episode}, Epsilon {epsilon} - Latest Total Reward {rewards_list[-1]}")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.legend()  # Add legend to the plot
        plt.grid()
        plt.show()
    else:
        pass


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def sample_action(state, policy_net, epsilon, n_actions):
    if random.random() > epsilon:
        return np.argmax(policy_net.predict(state,verbose=0))
    else:
        return random.randint(0, n_actions - 1)


def train_agent(
    env,
    agent,
    optimizer,
    experience_replay,
    n_episodes,
    batch_size,
    gamma,
    epsilon_start,
    epsilon_end,
    epsilon_decay,
    update_freq,
    writer,
):
    global_step = 0
    update_counter = 0
    rewards_list = []
    epsilon = epsilon_start
    MAX_STEPS = 3000
    
    for episode in range(n_episodes):
        print(f"Episode {episode},")
        total_reward = 0
        state = env.reset()[0]
        state = np.expand_dims(state, axis=0)
        # epsilon = epsilon_start

        while True:
            action = sample_action(state, agent.policy_net, epsilon, env.action_space.n)
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.expand_dims(next_state, axis=0)
            experience_replay.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            update_counter += 1
            
            if total_reward > MAX_STEPS:
                done = True
                
            if done:
                break
            
            if update_counter % 50 == 0:
                # print(f"Update counter: {update_counter}")
                pass

            if len(experience_replay) > batch_size:
                sample_batch = experience_replay.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*sample_batch)

                states = np.vstack(states)
                actions = np.array(actions)
                rewards = np.array(rewards, dtype=np.float32)
                next_states = np.vstack(next_states)
                dones = np.array(dones, dtype=np.float32)
                future_q_values = agent.target_net.predict(next_states,verbose=0)
                updated_q_values = rewards + gamma * np.max(future_q_values, axis=1) * (1 - dones)

                grads, loss = agent.train(states, actions, updated_q_values, env.action_space.n)
                optimizer.apply_gradients(zip(grads, agent.policy_net.trainable_variables))

                # Log training loss
                with writer.as_default():
                    tf.summary.scalar("Loss", loss, step=global_step)
                global_step += 1
                
        if len(experience_replay) > batch_size:
            epsilon = max(epsilon_end, epsilon_decay * epsilon)

        # Log total reward after each episode
        with writer.as_default():
            tf.summary.scalar("Total Reward", total_reward, step=episode)

        print(f"Reward {total_reward}")
        # Update the target network
        # if episode % update_freq == 0:
        #     agent.update_target_net()
        if update_counter > update_freq:
            agent.update_target_net()
            update_counter = 0
            print(f"Target network updated, episode {episode}")

        # print(f"Global step: {global_step}")

        rewards_list.append(total_reward)
        # print(f"Current Episode {episode}: Total Reward: {total_reward}")
        with writer.as_default():
            tf.summary.scalar("100 Moving Average Reward", np.mean(rewards_list[-100:]), step=episode)
            
        # Print moving average of last 10 episodes
        if len(rewards_list) % 10 == 0:
            print(f"Average of last 100 episodes: {np.mean(rewards_list[-100:])}")

        # print(f"Epsilon: {epsilon}")
        # Update the live plot
        # plot_rewards(rewards_list, episode,epsilon)

def test_agent(env, agent, n_episodes):
    for episode in range(n_episodes):
        print(f"Testing episode {episode}")
        state = env.reset()[0]
        state = np.expand_dims(state, axis=0)
        total_reward = 0

        while True:
            action = sample_action(state, agent.policy_net, 0, env.action_space.n)
            next_state, reward, done, _, _ = env.step(action)
            state = np.expand_dims(next_state, axis=0)
            total_reward += reward

            if done:
                print(f"Test Episode {episode} Total Reward: {total_reward}")
                break


# Hyperparameters and Environment Setup
if RENDER == True:
    env = gym.make("CartPole-v1", render_mode="human")
else:
    env = gym.make("CartPole-v1")

n_actions = env.action_space.n
n_states = env.observation_space.shape[0]

agent = DDQNAgent(n_states, n_actions, hidden_layers_5)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
experience_replay = ReplayBuffer(12500)
writer = create_file_writer(log_dir)

train_agent(
    env,
    agent,
    optimizer,
    experience_replay,
    n_episodes,
    batch_size,
    gamma,
    epsilon_start,
    epsilon_end,
    epsilon_decay,
    update_freq,
    writer,
)

# Save the trained model
model_save_path = "DuelingDQNModel_5Layers"
agent.policy_net.save(model_save_path, save_format="tf")

test_agent(env, agent, 10)
env.close()
