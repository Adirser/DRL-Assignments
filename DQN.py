import tensorflow as tf
import numpy as np
import random
from collections import deque
import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class DQN(tf.keras.Model):
    def __init__(self, input_size, output_size, hidden_layers):
        super(DQN, self).__init__()
        self.dense_layers = [
            tf.keras.layers.Dense(
                units=hidden_layers[0], activation="relu", input_shape=(input_size,)
            )
        ]
        self.dense_layers.extend(
            [tf.keras.layers.Dense(units=units, activation="relu") for units in hidden_layers[1:]]
        )
        self.output_layer = tf.keras.layers.Dense(units=output_size, activation=None)

    def call(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        return self.output_layer(x)

    def build_model(self, input_shape):
        self.build(input_shape)


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
        return np.argmax(policy_net.predict(state))
    else:
        return random.randint(0, n_actions - 1)


def train_agent(
    env,
    policy_net,
    target_net,
    optimizer,
    experience_replay,
    n_episodes,
    batch_size,
    gamma,
    epsilon_start,
    epsilon_end,
    epsilon_decay,
    update_freq,
):
    for episode in range(n_episodes):
        print(f"Episode {episode}")
        state = env.reset()[0]
        state = np.expand_dims(state, axis=0)
        epsilon = epsilon_start

        while True:
            action = sample_action(state, policy_net, epsilon, env.action_space.n)
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.expand_dims(next_state, axis=0)

            experience_replay.push(state, action, reward, next_state, done)

            state = next_state
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

                future_q_values = target_net.predict(next_states)
                updated_q_values = rewards + gamma * np.max(future_q_values, axis=1) * (1 - dones)

                masks = tf.one_hot(actions, env.action_space.n)

                with tf.GradientTape() as tape:
                    q_values = policy_net(states)
                    q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                    loss = tf.keras.losses.MSE(updated_q_values, q_action)

                grads = tape.gradient(loss, policy_net.trainable_variables)
                optimizer.apply_gradients(zip(grads, policy_net.trainable_variables))

            epsilon = max(epsilon_end, epsilon_decay * epsilon)

        # Update the target network
        if episode % update_freq == 0:
            target_net.set_weights(policy_net.get_weights())


def test_agent(env, policy_net, n_episodes):
    for episode in range(n_episodes):
        print(f"Testing episode {episode}")
        state = env.reset()[0]
        state = np.expand_dims(state, axis=0)
        total_reward = 0

        while True:
            action = sample_action(state, policy_net, 0, env.action_space.n)
            next_state, reward, done, _, _ = env.step(action)
            state = np.expand_dims(next_state, axis=0)
            total_reward += reward
            env.render(mode="human")

            if done:
                print(f"Test Episode {episode} Total Reward: {total_reward}")
                break
    env.close()


# Hyperparameters and Environment Setup
env = gym.make("CartPole-v1")
n_actions = env.action_space.n
n_states = env.observation_space.shape[0]

hidden_layers_3 = [64, 64, 64]  # For the 3 hidden layer network
hidden_layers_5 = [64, 64, 64, 64, 64]  # For the 5 hidden layer network

policy_net = DQN(n_states, n_actions, hidden_layers_3)
target_net = DQN(n_states, n_actions, hidden_layers_3)

# Explicitly build the models
policy_net.build_model((None, n_states))
target_net.build_model((None, n_states))

# Now, you can safely set the weights
target_net.set_weights(policy_net.get_weights())


n_episodes = 500
batch_size = 128
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 0.995
learning_rate = 0.001
update_freq = 25

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
experience_replay = ReplayBuffer(10000)


train_agent(
    env,
    policy_net,
    target_net,
    optimizer,
    experience_replay,
    n_episodes,
    batch_size,
    gamma,
    epsilon_start,
    epsilon_end,
    epsilon_decay,
    update_freq,
)

# Save the trained model
model_save_path = "DQNModel"
policy_net.save(model_save_path, save_format="tf")

test_agent(env, policy_net, 10)
env.close()
