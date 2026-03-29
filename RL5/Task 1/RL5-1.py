import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L

if type(os.environ.get("DISPLAY")) is not str or len(os.environ.get("DISPLAY")) == 0:
    os.environ['DISPLAY'] = ':1'

env = gym.make("CartPole-v0", render_mode="rgb_array")  # Создаем среду CartPole
state_dim = env.observation_space.shape[0]              # Размер состояния (4)
n_actions = int(env.action_space.n)                     # Количество действий (2)

s, _ = env.reset()                                      # Сбрасываем среду
plt.imshow(env.render())                                # Отображаем картинку
plt.show()                                              # Показываем график

network = keras.models.Sequential()                     # Последовательная модель Keras
network.add(L.InputLayer(input_shape=(state_dim,)))     # Входной слой под состояние
network.add(L.Dense(64, activation='relu'))            # Скрытый слой с 64 нейронами
network.add(L.Dense(64, activation='relu'))            # Еще один скрытый слой
network.add(L.Dense(n_actions, activation='linear'))   # Выходной слой: Q для каждого действия

optimizer = keras.optimizers.Adam(learning_rate=1e-4)  # Adam оптимизатор
loss_fn = keras.losses.MeanSquaredError()             # MSE для TD ошибки

def get_action(state, epsilon=0.0):

    if np.random.rand() < epsilon:                     # С вероятностью epsilon
        return np.random.randint(n_actions)            # выбираем случайное действие
    state_t = tf.convert_to_tensor(state[None], dtype=tf.float32)  # Преобразуем в тензор
    q_values = network(state_t)                        # Предсказываем Q для всех действий
    return int(tf.argmax(q_values[0]))                 # Выбираем действие с max Q

def train_step(states, actions, rewards, next_states, done):

    states_t = tf.convert_to_tensor(states, dtype=tf.float32)
    next_states_t = tf.convert_to_tensor(next_states, dtype=tf.float32)
    actions_t = tf.convert_to_tensor(actions, dtype=tf.int32)
    rewards_t = tf.convert_to_tensor(rewards, dtype=tf.float32)
    done_t = tf.convert_to_tensor(done, dtype=tf.float32)

    with tf.GradientTape() as tape:                     # Автоматическое вычисление градиентов
        q_pred = tf.reduce_sum(network(states_t) * tf.one_hot(actions_t, n_actions), axis=1)
        q_next = tf.reduce_max(network(next_states_t), axis=1)
        gamma = 0.99
        q_target = rewards_t + gamma * q_next * (1 - done_t)
        loss = loss_fn(q_target, q_pred)               # MSE между предсказанным и целевым Q
    grads = tape.gradient(loss, network.trainable_variables)  # Градиенты по весам
    optimizer.apply_gradients(zip(grads, network.trainable_variables))  # Обновляем веса

def generate_session(env, t_max=1000, epsilon=0, train=False):
    total_reward = 0
    s, _ = env.reset()                                 # Начальное состояние
    for t in range(t_max):
        a = get_action(s, epsilon)                     # Выбираем действие
        next_s, r, terminated, truncated, _ = env.step(a)  # Делаем шаг
        done = terminated or truncated

        if train:                                      # Если нужно обучать
            train_step([s], [a], [r], [next_s], [float(done)])  # TD update

        total_reward += r                              # Суммируем вознаграждение
        s = next_s                                     # Переходим к следующему состоянию
        if done:
            break
    return total_reward

epsilon = 0.5                                          # Начальное значение эпсилон
for i in range(1000):
    session_rewards = [generate_session(env, epsilon=epsilon, train=True) for _ in range(100)]
    mean_reward = np.mean(session_rewards)
    print(f"epoch #{i}\tmean reward = {mean_reward:.3f}\tepsilon = {epsilon:.3f}")
    epsilon *= 0.99                                    # Снижение эпсилон
    epsilon = max(epsilon, 1e-4)                       # Минимальный порог
    if mean_reward > 300:                              # Если агент научился
        print("You Win!")
        break
