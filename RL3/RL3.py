import numpy as np
from mdp import MDP, FrozenLakeEnv, has_graphviz
from IPython.display import display
import matplotlib.pyplot as plt
from time import sleep
from IPython.display import clear_output

# Определяем вероятности переходов для каждой пары состояние-действие
transition_probs = {
    's0': {  # состояние s0
        'a0': {'s0': 0.5, 's2': 0.5},  # действие a0: 50% остаться в s0, 50% перейти в s2
        'a1': {'s2': 1}  # действие a1: всегда перейти в s2
    },
    's1': {  # состояние s1
        'a0': {'s0': 0.7, 's1': 0.1, 's2': 0.2},  # действие a0: вероятности перехода в s0, s1, s2
        'a1': {'s1': 0.95, 's2': 0.05}  # действие a1: вероятности перехода в s1 и s2
    },
    's2': {  # состояние s2
        'a0': {'s0': 0.4, 's2': 0.6},  # действие a0: 40% в s0, 60% в s2
        'a1': {'s0': 0.3, 's1': 0.3, 's2': 0.4}  # действие a1: вероятности перехода в s0, s1, s2
    }
}

# Определяем вознаграждения для некоторых переходов
rewards = {
    's1': {'a0': {'s0': +5}},  # за переход s1->a0->s0 дается +5
    's2': {'a1': {'s0': -1}}  # за переход s2->a1->s0 дается -1
}

# Создаем объект MDP с начальными параметрами
mdp = MDP(transition_probs, rewards, initial_state='s0')

# Сбрасываем MDP в начальное состояние и выводим его
print('initial state =', mdp.reset())

# Делаем один шаг с действием 'a1' и выводим результат
next_state, reward, done, info = mdp.step('a1')
print('next_state = %s, reward = %s, done = %s' % (next_state, reward, done))

# Функция для вычисления Q(s,a) = ожидаемое вознаграждение за действие в состоянии
def get_action_value(mdp, state_values, state, action, gamma):
    """Вычисляет Q(s,a)"""
    q = 0.0  # начальное значение Q
    # Для всех возможных следующих состояний и вероятностей перехода
    for next_state, prob in mdp.get_next_states(state, action).items():
        reward = mdp.get_reward(state, action, next_state)  # получаем награду
        reward = reward if reward is not None else 0.0  # если награды нет, считаем 0
        # прибавляем к Q взвешенное по вероятности вознаграждение + дисконтированное значение следующего состояния
        q += prob * (reward + gamma * state_values[next_state])
    return q  # возвращаем Q(s,a)

# Функция для вычисления нового V(s) для итерации
def get_new_state_value(mdp, state_values, state, gamma):
    """Вычисляет V(s) на следующей итерации"""
    if mdp.is_terminal(state):  # если состояние терминальное, V=0
        return 0.0
    # Вычисляем Q для всех возможных действий в состоянии
    action_values = [get_action_value(mdp, state_values, state, a, gamma)
                     for a in mdp.get_possible_actions(state)]
    return max(action_values)  # возвращаем максимум, это новое V(s)

# Функция для нахождения оптимального действия pi*(s)
def get_optimal_action(mdp, state_values, state, gamma=0.9):
    """Возвращает оптимальное действие pi*(s)"""
    if mdp.is_terminal(state):  # если состояние терминальное, действия нет
        return None
    actions = mdp.get_possible_actions(state)  # список доступных действий
    q_values = [get_action_value(mdp, state_values, state, a, gamma) for a in actions]  # вычисляем Q
    max_index = np.argmax(q_values)  # индекс действия с максимальным Q
    return actions[max_index]  # возвращаем оптимальное действие

# Функция полной итерации значения
def value_iteration(mdp, state_values=None, gamma=0.9, num_iter=1000, min_difference=1e-5):
    """Полная функция Value Iteration"""
    state_values = state_values or {s: 0.0 for s in mdp.get_all_states()}  # инициализация V(s)=0
    for i in range(num_iter):
        # вычисляем новые значения для всех состояний
        new_state_values = {s: get_new_state_value(mdp, state_values, s, gamma)
                            for s in mdp.get_all_states()}
        # находим максимальное отклонение между старым и новым V
        diff = max(abs(new_state_values[s] - state_values[s]) for s in mdp.get_all_states())
        # печатаем информацию о итерации
        print("iter %4i | diff: %6.5f | V(start): %.3f " %
              (i, diff, new_state_values[mdp._initial_state]))
        state_values = new_state_values  # обновляем значения
        if diff < min_difference:  # если изменения очень маленькие, выходим
            break
    return state_values  # возвращаем финальные V(s)


gamma = 0.9  # задаем коэффициент дисконтирования
state_values = value_iteration(mdp, gamma=gamma, num_iter=100)  # запускаем VI
print("Final state values:", state_values)

# Проверка оптимальных действий для каждого состояния
for s in mdp.get_all_states():
    print(f"Optimal action in {s}:", get_optimal_action(mdp, state_values, s, gamma))



s = mdp.reset()  # сбрасываем MDP
rewards_list = []
for _ in range(10000):  # 10000 шагов
    a = get_optimal_action(mdp, state_values, s, gamma)  # выбираем оптимальное действие
    s, r, done, _ = mdp.step(a)  # делаем шаг
    rewards_list.append(r)  # сохраняем вознаграждение
print("average reward: ", np.mean(rewards_list))



mdp = FrozenLakeEnv(slip_chance=0)  # создаем среду Frozen Lake без проскальзывания
mdp.render()  # отображаем карту

# Запускаем VI на Frozen Lake
state_values = value_iteration(mdp, gamma=gamma, num_iter=1000)

# Прогон одного эпизода
s = mdp.reset()  # сброс среды
for t in range(100):
    a = get_optimal_action(mdp, state_values, s, gamma)  # выбираем оптимальное действие
    s, r, done, _ = mdp.step(a)  # делаем шаг
    mdp.render()  # показываем состояние
    if done:  # если эпизод закончился
        break


def draw_policy(mdp, state_values):
    plt.figure(figsize=(3, 3))  # создаем новую фигуру для графика
    h, w = mdp.desc.shape  # размеры карты
    states = sorted(mdp.get_all_states())  # все состояния в порядке
    V = np.array([state_values[s] for s in states])  # массив V(s)
    Pi = {s: get_optimal_action(mdp, state_values, s, gamma) for s in states}  # оптимальная политика
    plt.imshow(V.reshape(h, w), cmap='gray', interpolation='none', clim=(0, 1))  # рисуем V(s)
    ax = plt.gca()  # текущие оси
    ax.set_xticks(np.arange(w)-.5)  # делаем сетку
    ax.set_yticks(np.arange(h)-.5)
    ax.set_xticklabels([])  # убираем подписи
    ax.set_yticklabels([])
    a2uv = {'left': (-1, 0), 'down': (0, 1), 'right': (1, 0), 'up': (0, -1)}  # смещения стрелок
    for y in range(h):  # по строкам
        for x in range(w):  # по столбцам
            plt.text(x, y, str(mdp.desc[y, x].item()),  # пишем символ состояния на карте
                     color='g', size=12, verticalalignment='center',
                     horizontalalignment='center', fontweight='bold')
            state_index = y * w + x  # индекс состояния
            state = states[state_index]  # состояние
            a = Pi[state]  # оптимальное действие
            if a is None:
                continue
            u, v = a2uv[a]  # смещение для стрелки
            plt.arrow(x, y, u*0.3, v*0.3, color='m', head_width=0.1, head_length=0.1)  # рисуем стрелку
    plt.grid(color='b', lw=2, ls='-')  # рисуем сетку
    plt.show()  # отображаем график

draw_policy(mdp, state_values)

def evaluate_agent(mdp, state_values, gamma=0.9, num_games=1000):
    total_rewards = []  # список для накопления суммарных наград
    for _ in range(num_games):  # повторяем num_games раз
        s = mdp.reset()  # сброс среды
        rewards_list = []  # временный список для одного эпизода
        for t in range(100):  # максимум 100 шагов
            a = get_optimal_action(mdp, state_values, s, gamma)  # оптимальное действие
            s, r, done, _ = mdp.step(a)  # делаем шаг
            rewards_list.append(r)  # сохраняем награду
            if done:  # если эпизод закончился
                break
        total_rewards.append(np.sum(rewards_list))  # суммируем награды за эпизод
    print("average reward: ", np.mean(total_rewards))  # печатаем среднее
    return np.mean(total_rewards)


# Примеры оценки агента на разных средах
mdp = FrozenLakeEnv(slip_chance=0)  # среда без проскальзывания
state_values = value_iteration(mdp)
evaluate_agent(mdp, state_values)

mdp = FrozenLakeEnv(slip_chance=0.1)  # среда с небольшим проскальзыванием
state_values = value_iteration(mdp)
evaluate_agent(mdp, state_values)

mdp = FrozenLakeEnv(slip_chance=0.25)  # среда с большим проскальзыванием
state_values = value_iteration(mdp)
evaluate_agent(mdp, state_values)

mdp = FrozenLakeEnv(slip_chance=0.2, map_name='8x8')  # карта 8x8
state_values = value_iteration(mdp)
evaluate_agent(mdp, state_values)