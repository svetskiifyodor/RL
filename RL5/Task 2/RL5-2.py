import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

if type(os.environ.get("DISPLAY")) is not str or len(os.environ.get("DISPLAY")) == 0:
    os.environ['DISPLAY'] = ':1'

# Создаем среду CartPole с возможностью получения изображений
env = gym.make("CartPole-v0", render_mode="rgb_array")
state_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

# Сбрасываем среду и получаем начальное состояние
s, _ = env.reset()
# Отображаем текущее состояние среды в виде картинки
plt.imshow(env.render())
plt.show()

# Создаем класс нейронной сети для Q-обучения
class QNetwork(nn.Module):
    # Конструктор сети принимает размер состояния и количество действий
    def __init__(self, state_dim, n_actions):
        # Вызываем конструктор родительского класса
        super().__init__()
        # Определяем последовательность слоев сети
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),  # Полносвязный слой с 64 нейронами
            nn.ReLU(),                  # Активация ReLU
            nn.Linear(64, 64),          # Еще один полносвязный слой
            nn.ReLU(),                  # Активация ReLU
            nn.Linear(64, n_actions)    # Выходной слой, один нейрон на каждое действие
        )
    # Метод forward определяет, как сеть вычисляет выход
    def forward(self, x):
        return self.model(x)

# Создаем экземпляр нашей сети
network = QNetwork(state_dim, n_actions)
# Создаем оптимизатор Adam для обновления весов сети
optimizer = optim.Adam(network.parameters(), lr=1e-4)
# Создаем функцию потерь MSE (среднеквадратичная ошибка)
loss_fn = nn.MSELoss()

# Функция для выбора действия по эпсилон-жадной политике
def get_action(state, epsilon=0.0):
    # С вероятностью epsilon выбираем случайное действие
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    else:
        # Преобразуем состояние в тензор pytorch и добавляем измерение batch
        state_t = torch.FloatTensor(state).unsqueeze(0)
        # Получаем Q-значения от сети для текущего состояния
        q_values = network(state_t)
        # Выбираем действие с наибольшим Q-значением
        return int(torch.argmax(q_values))

# Функция для одной игровой сессии агента
def generate_session(env, t_max=1000, epsilon=0, train=False):
    # Инициализируем общее вознаграждение
    total_reward = 0
    # Сбрасываем среду и получаем начальное состояние
    s, _ = env.reset()
    # Проходим по максимуму t_max шагов
    for t in range(t_max):
        # Выбираем действие с использованием эпсилон-жадной стратегии
        a = get_action(s, epsilon=epsilon)
        # Выполняем действие в среде и получаем новые данные
        next_s, r, terminated, truncated, _ = env.step(a)
        # Проверяем, завершилась ли игра
        done = terminated or truncated
        # Если тренируем сеть, обновляем веса
        if train:
            # Преобразуем состояния, действия, вознаграждения и флаг done в тензоры
            s_t = torch.FloatTensor(s).unsqueeze(0)
            next_s_t = torch.FloatTensor(next_s).unsqueeze(0)
            a_t = torch.LongTensor([a])
            r_t = torch.FloatTensor([r])
            done_t = torch.BoolTensor([done])

            # Выбираем предсказанное Q для выбранного действия
            q_pred = network(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
            # Вычисляем Q для следующего состояния без градиента
            with torch.no_grad():
                q_next = network(next_s_t).max(1)[0]
                q_target = r_t + 0.99 * q_next * (~done_t)  # TD-целевое значение
            # Вычисляем ошибку между предсказанным и целевым Q
            loss = loss_fn(q_pred, q_target)
            # Обнуляем градиенты
            optimizer.zero_grad()
            # Вычисляем градиенты
            loss.backward()
            # Обновляем веса сети
            optimizer.step()

        # Добавляем текущее вознаграждение к общему
        total_reward += r
        # Переходим к следующему состоянию
        s = next_s
        # Если игра завершена, выходим из цикла
        if done:
            break
    # Возвращаем общее вознаграждение за сессию
    return total_reward

# Начальное значение эпсилон (вероятность случайного действия)
epsilon = 0.5
# Цикл обучения на 1000 эпох
for i in range(1000):
    # Генерируем 100 игровых сессий и считаем среднее вознаграждение
    session_rewards = [generate_session(env, epsilon=epsilon, train=True) for _ in range(100)]
    # Выводим номер эпохи, среднее вознаграждение и значение эпсилон
    print(f"epoch #{i}\tmean reward = {np.mean(session_rewards):.3f}\tepsilon = {epsilon:.3f}")
    # Уменьшаем эпсилон постепенно (агент исследует меньше со временем)
    epsilon *= 0.99
    # Убедимся, что эпсилон не слишком мал
    epsilon = max(epsilon, 1e-4)
    # Если агент хорошо играет, заканчиваем обучение
    if np.mean(session_rewards) > 300:
        print("You Win!")
        break
