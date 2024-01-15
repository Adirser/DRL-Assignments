import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import random
from collections import deque
import gymnasium as gym
from tensorflow.summary import create_file_writer
from datetime import datetime

RENDER = False

hidden_layers_3 = [64, 64, 64]  # For the 3 hidden layer network
hidden_layers_5 = [64, 64, 64, 64, 64]  # For the 5 hidden layer network
n_episodes = 1000
batch_size = 64
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 0.993
learning_rate = 0.001
update_freq = 1000
log_dir = f"logs/DQN_{datetime.now().strftime('%d%m%Y%H%M%S')}"


class DQNAgent:
    def __init__(self, input_size, output_size, hidden_layers):
        self.policy_net = self.create_model(input_size, output_size, hidden_layers)
        self.target_net = self.create_model(input_size, output_size, hidden_layers)
        self.update_target_net()

    def create_model(self, input_size, output_size, hidden_layers):
        model = Sequential()
        model.add(Dense(units=hidden_layers[0], activation="relu", input_shape=(input_size,)))
        for units in hidden_layers[1:]:
            model.add(Dense(units=units, activation="relu"))
        model.add(Dense(units=output_size, activation=None))
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
        return np.argmax(policy_net.predict(state, verbose=0))
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
    for episode in range(n_episodes):
        total_reward = 0
        step = 0
        state = env.reset()[0]
        state = np.expand_dims(state, axis=0)
        epsilon = epsilon_start

        while step < 1000:
            action = sample_action(state, agent.policy_net, epsilon, env.action_space.n)
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.expand_dims(next_state, axis=0)
            experience_replay.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step += 1

            if done:
                break

            if len(experience_replay) > batch_size:
                sample_batch = experience_replay.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*sample_batch)

                states = np.vstack(states)
                actions = np.array(actions)
                rewards = np.array(rewards, dtype=np.float32)
                next_states = np.vstack(next_states)
                dones = np.array(dones, dtype=np.float32)

                future_q_values = agent.target_net.predict(next_states, verbose=0)
                updated_q_values = rewards + gamma * np.max(future_q_values, axis=1) * (1 - dones)

                grads, loss = agent.train(states, actions, updated_q_values, env.action_space.n)
                optimizer.apply_gradients(zip(grads, agent.policy_net.trainable_variables))

                # Log training loss
                with writer.as_default():
                    tf.summary.scalar("Loss", loss, step=global_step)
                global_step += 1

                # Update the target network
                if global_step % update_freq == 0:
                    agent.update_target_net()

            epsilon = max(epsilon_end, epsilon_decay * epsilon)

        print(f"Episode {episode} - {step} steps")

        # Log total reward after each episode
        with writer.as_default():
            tf.summary.scalar("Total Reward", total_reward, step=episode)


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

agent = DQNAgent(n_states, n_actions, hidden_layers_3)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
experience_replay = ReplayBuffer(100000)
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
model_save_path = "DQNModel"
agent.policy_net.save(model_save_path, save_format="tf")

test_agent(env, agent, 10)
env.close()
