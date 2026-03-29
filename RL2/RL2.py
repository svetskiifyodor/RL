import sys, os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Создаём игровую среду
env = gym.make("Taxi-v3")
env.reset()
env.render()

# Получаем количество состояний и действий
n_states = env.observation_space.n  # Общее число состояний среды
n_actions = env.action_space.n  # Общее число действий, которые можно сделать
print("n_states=%i, n_actions=%i" % (n_states, n_actions))

def initialize_policy(n_states, n_actions):
    # Создаём равномерное распределение для всех действий в каждом состоянии
    policy = np.ones((n_states, n_actions)) / n_actions
    return policy

policy = initialize_policy(n_states, n_actions)

# Проверяем корректность политики
assert type(policy) in (np.ndarray, np.matrix)
assert np.allclose(policy, 1./n_actions)  # Все вероятности должны быть равны
assert np.allclose(np.sum(policy, axis=1), 1)  # Сумма вероятностей для каждого состояния = 1

def generate_session(env, policy, t_max=10**4):
    states, actions = [], []  # Списки для хранения состояний и действий
    total_reward = 0.  # Общая награда для этой сессии

    s, _ = env.reset()  # Получаем начальное состояние

    for t in range(t_max):
        a = np.random.choice(np.arange(n_actions), p=policy[s])  
        # Случайно выбираем действие a из доступных действий с вероятностями из policy

        new_s, r, terminated, truncated, info = env.step(a)  
        # Делаем шаг в среде
        done = terminated or truncated  # Если эпизод завершился по цели или лимиту шагов

        states.append(s)  # Записываем текущее состояние
        actions.append(a)  # Записываем выбранное действие
        total_reward += r  # Добавляем награду

        s = new_s  # Переходим в новое состояние

        if done:  # Если эпизод закончился, выходим из цикла
            break

    return states, actions, total_reward  # Возвращаем данные сессии

# Тестируем генерацию одной сессии
s, a, r = generate_session(env, policy)  

# Проверяем корректность возвращаемых данных
assert type(s) == type(a) == list
assert len(s) == len(a)
assert type(r) in [float, np.float64]

# Генерируем 200 сессий и берём их итоговую награду
sample_rewards = [generate_session(env, policy, t_max=1000)[-1] for _ in range(200)]  

plt.hist(sample_rewards, bins=20)  # Строим гистограмму наград
plt.vlines([np.percentile(sample_rewards, 50)], [0], [100], label="50'th percentile", color='green')  
plt.vlines([np.percentile(sample_rewards, 90)], [0], [100], label="90'th percentile", color='red')  
plt.legend()
plt.show()


def select_elites(states_batch, actions_batch, rewards_batch, percentile):
    reward_threshold = np.percentile(rewards_batch, percentile)  
    # Определяем минимальную награду, чтобы попасть в топ-процентиль

    elite_states = []  # Список состояний лучших сессий
    elite_actions = []  # Список действий лучших сессий

    for i in range(len(states_batch)):
        if rewards_batch[i] >= reward_threshold:
            elite_states.extend(states_batch[i])  # Добавляем все состояния этой сессии
            elite_actions.extend(actions_batch[i])  # Добавляем все действия этой сессии

    return elite_states, elite_actions

# Тестовые данные для проверки функции select_elites
states_batch = [[1, 2, 3], [4, 2, 0, 2], [3, 1]]
actions_batch = [[0, 2, 4], [3, 2, 0, 1], [3, 3]]
rewards_batch = [3, 4, 5]

# Проверяем работу функции на разных процентилях
test_result_0 = select_elites(states_batch, actions_batch, rewards_batch, percentile=0)
test_result_30 = select_elites(states_batch, actions_batch, rewards_batch, percentile=30)
test_result_90 = select_elites(states_batch, actions_batch, rewards_batch, percentile=90)
test_result_100 = select_elites(states_batch, actions_batch, rewards_batch, percentile=100)

# Проверяем правильность выбора элит
assert np.all(test_result_0[0] == [1, 2, 3, 4, 2, 0, 2, 3, 1])
assert np.all(test_result_0[1] == [0, 2, 4, 3, 2, 0, 1, 3, 3])
assert np.all(test_result_30[0] == [4, 2, 0, 2, 3, 1])
assert np.all(test_result_30[1] == [3, 2, 0, 1, 3, 3])
assert np.all(test_result_90[0] == [3, 1])
assert np.all(test_result_90[1] == [3, 3])
assert np.all(test_result_100[0] == [3, 1])
assert np.all(test_result_100[1] == [3, 3])


def get_new_policy(elite_states, elite_actions):
    new_policy = np.zeros([n_states, n_actions])  # Создаём пустую матрицу вероятностей

    for s, a in zip(elite_states, elite_actions):  # Для каждой пары состояние-действие
        new_policy[s][a] += 1  # Считаем количество раз, когда действие было в элите

    for s in range(n_states):  # Для каждого состояния
        if new_policy[s].sum() == 0:  # Если состояние не встречалось в элитах
            new_policy[s] = np.ones(n_actions) / n_actions  # Делаем равномерное распределение
        else:
            new_policy[s] /= new_policy[s].sum()  # Нормализуем, чтобы сумма = 1

    return new_policy  # Возвращаем новую политику

def show_progress(rewards_batch, log, percentile, reward_range=[-990, +10]):
    mean_reward = np.mean(rewards_batch)  # Средняя награда всех сессий
    threshold = np.percentile(rewards_batch, percentile)  # Порог для элит

    log.append([mean_reward, threshold])  # Сохраняем значения для графика

    plt.figure(figsize=[8, 4])  # Создаём окно графика

    plt.subplot(1, 2, 1)  # Левая часть — график средних наград
    plt.plot(list(zip(*log))[0], label='Mean rewards')  # Линия средних наград
    plt.plot(list(zip(*log))[1], label='Reward thresholds')  # Линия порога
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)  # Правая часть — гистограмма наград
    plt.hist(rewards_batch, range=reward_range)
    plt.vlines([threshold], [0], [100], label="percentile", color='red')
    plt.legend()
    plt.grid()

    clear_output(True)  # Очищаем вывод, чтобы график обновлялся
    print("mean reward = %.3f, threshold=%.3f" % (mean_reward, threshold))
    plt.show()

policy = initialize_policy(n_states, n_actions)  # Сбрасываем политику на равномерную

n_sessions = 250  # Количество сессий за одну итерацию
percentile = 50  # Процентиль для выбора элит
learning_rate = 0.5  # Насколько быстро обновляется политика

log = []  # Список для хранения прогресса

for i in range(100):  # 100 итераций обучения
    sessions = [generate_session(env, policy) for _ in range(n_sessions)]  
    # Генерируем n_sessions новых сессий

    states_batch, actions_batch, rewards_batch = zip(*sessions)  
    # Разделяем сессии на состояния, действия и награды

    elite_states, elite_actions = select_elites(
        states_batch, actions_batch, rewards_batch, percentile
    )  # Выбираем элитные сессии

    new_policy = get_new_policy(elite_states, elite_actions)  # Создаём новую политику

    policy = learning_rate * new_policy + (1 - learning_rate) * policy  
    # Обновляем политику частично (смешиваем старую и новую)

    show_progress(rewards_batch, log, percentile)  # Показываем прогресс обучения