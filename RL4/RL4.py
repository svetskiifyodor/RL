import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import gymnasium as gym

# создаем класс QLearningAgent
class QLearningAgent:
    # инициализация агента
    def __init__(self, alpha, epsilon, discount, get_legal_actions):
        # функция возвращающая список разрешенных действий для состояния
        self.get_legal_actions = get_legal_actions
        # словарь для хранения Q-значений по состояниям и действиям
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))
        # скорость обучения
        self.alpha = alpha
        # вероятность случайного выбора действия (exploration)
        self.epsilon = epsilon
        # коэффициент дисконтирования будущих наград
        self.discount = discount

    # возвращает текущее Q-значение для состояния и действия
    def get_qvalue(self, state, action):
        return self._qvalues[state][action]

    # устанавливает Q-значение для состояния и действия
    def set_qvalue(self, state, action, value):
        self._qvalues[state][action] = value

    # вычисляем V(s) = максимум по Q(s, a) для всех действий
    def get_value(self, state):
        possible_actions = self.get_legal_actions(state)  # получаем доступные действия
        if len(possible_actions) == 0:  # если действий нет, возвращаем 0
            return 0.0
        # возвращаем максимум Q(s,a) по всем действиям
        return max([self.get_qvalue(state, a) for a in possible_actions])

    # обновление Q-значения на основе формулы Q-learning
    def update(self, state, action, reward, next_state):
        gamma = self.discount  # коэффициент дисконтирования
        alpha = self.alpha  # скорость обучения
        q_current = self.get_qvalue(state, action)  # текущее Q
        q_next = self.get_value(next_state)  # V(s') — максимум будущих Q
        new_q = (1 - alpha) * q_current + alpha * (reward + gamma * q_next)  # формула Q-learning
        self.set_qvalue(state, action, new_q)  # сохраняем новое значение

    # выбираем наилучшее действие по текущим Q-значениям
    def get_best_action(self, state):
        possible_actions = self.get_legal_actions(state)  # получаем возможные действия
        if len(possible_actions) == 0:  # если действий нет, возвращаем None
            return None
        # выбираем действие с максимальным Q-значением
        return max(possible_actions, key=lambda a: self.get_qvalue(state, a))

    # выбираем действие с epsilon-greedy стратегией
    def get_action(self, state):
        possible_actions = self.get_legal_actions(state)
        if len(possible_actions) == 0:
            return None
        if random.random() < self.epsilon:  # с вероятностью epsilon выбираем случайное действие
            return random.choice(possible_actions)
        else:  # иначе выбираем лучшее действие
            return self.get_best_action(state)

# создаем среду Taxi-v3
env = gym.make("Taxi-v3", render_mode=None)
# количество действий в среде
n_actions = env.action_space.n

# создаем агента Q-learning
agent = QLearningAgent(
    alpha=0.5,  # скорость обучения
    epsilon=0.25,  # вероятность случайного действия
    discount=0.99,  # коэффициент дисконтирования
    get_legal_actions=lambda s: range(n_actions)  # функция получения действий
)

# функция для игры и обучения агента
def play_and_train(env, agent, t_max=10**4):
    total_reward = 0.0  # общая награда за эпизод
    s, _ = env.reset()  # сбрасываем среду и получаем начальное состояние
    for t in range(t_max):  # ограничение на шаги
        a = agent.get_action(s)  # агент выбирает действие
        next_s, r, terminated, truncated, _ = env.step(a)  # делаем шаг в среде
        done = terminated or truncated  # проверяем завершение эпизода
        agent.update(s, a, r, next_s)  # обновляем Q-значение
        s = next_s  # переходим к следующему состоянию
        total_reward += r  # суммируем награды
        if done:  # если эпизод завершен, выходим из цикла
            break
    return total_reward  # возвращаем суммарную награду

# импортируем clear_output для обновления графиков
from IPython.display import clear_output

rewards = []  # список для хранения наград каждого эпизода
for i in range(1000):  # обучаем агента 1000 эпизодов
    rewards.append(play_and_train(env, agent))  # запускаем один эпизод
    agent.epsilon *= 0.99  # уменьшаем вероятность случайного действия
    if i % 100 == 0:  # каждые 100 эпизодов рисуем график
        clear_output(True)  # очищаем старый график
        plt.title('eps = {:e}, mean reward = {:.1f}'.format(agent.epsilon, np.mean(rewards[-10:])))
        plt.plot(rewards)  # строим график
        plt.show()  # показываем график

from gym.core import ObservationWrapper  # импортируем ObservationWrapper для модификации состояний

# создаем класс для бинаризации состояний
class Binarizer(ObservationWrapper):
    def observation(self, state):  # метод, вызываемый для каждого состояния
        # округляем каждое измерение состояния до 1-2 цифр
        rounded_state = np.array([round(state[0], 1),
                                  round(state[1], 1),
                                  round(state[2], 2),
                                  round(state[3], 2)])
        return tuple(rounded_state)  # возвращаем кортеж, пригодный для Q-learning

# функция для создания среды CartPole-v1
def make_env():
    return gym.make('CartPole-v1', render_mode=None).env  # .env убирает ограничение по времени

env2 = Binarizer(make_env()).env

seen_observations = []  # список для хранения всех наблюдений
for _ in range(1000):  # собираем 1000 эпизодов
    s, _ = env2.reset()  # сбрасываем среду
    done = False  # флаг завершения эпизода
    seen_observations.append(s)  # сохраняем начальное состояние
    while not done:
        action = env2.action_space.sample()  # выбираем случайное действие
        s, r, terminated, truncated, _ = env2.step(action)  # делаем шаг
        done = terminated or truncated  # проверяем завершение
        seen_observations.append(s)  # сохраняем текущее состояние

# преобразуем список наблюдений в массив numpy для анализа
seen_observations = np.array(seen_observations)
# строим гистограммы по каждому измерению состояния
for obs_i in range(env2.observation_space.shape[0]):
    plt.hist(seen_observations[:, obs_i], bins=20)  # строим гистограмму
    plt.show()
