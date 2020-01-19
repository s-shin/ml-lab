import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt
import signal
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from collections import deque
import copy

NUM_STEPS = 200
OK_THREASHOLD = 195
CONSECUTIVE_OK_COUNT = 100
NUM_ACTIONS = 2


def random_action():
    return np.random.choice([0, 1])


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.d = deque()

    def __len__(self):
        return len(self.d)

    def append(self, entry):
        if len(self.d) >= self.capacity:
            self.d.popleft()
        self.d.append(entry)

    def sample(self, n):
        for i in np.random.randint(0, len(self.d), n):
            yield self.d[i]


class BaseAgent:
    def save(self):
        pass

    def load(self):
        pass

    def best_action(self, observation):
        return 0

    def start_episode(self, n_episode, observation):
        return self.best_action(observation)

    def update(self, n_step, action, observation, reward, done, info):
        return 0


class QTableAgent(BaseAgent):
    def __init__(self, state_granularity=8, alpha=0.125, gamma=0.999,
                 qtable_file='tmp/qtable.csv'):
        self.state_granularity = state_granularity
        self.alpha = alpha
        self.gamma = gamma
        self.qtable_file = qtable_file
        self.num_states = state_granularity**4
        self.bins = [
            np.linspace(-3.0, 3.0, state_granularity+1)[1:-1],
            np.linspace(-5.0, 5.0, state_granularity+1)[1:-1],
            np.linspace(-0.5, 0.5, state_granularity+1)[1:-1],
            np.linspace(-5.0, 5.0, state_granularity+1)[1:-1],
        ]
        # Q(s[t], a[t]) = qtable[state_id, action]
        self.qtable = np.random.uniform(
            low=-1.0, high=1.0, size=(self.num_states, NUM_ACTIONS))

    def digitize_observation(self, observation):
        return sum([
            np.digitize(observation[i], bins=bins) * self.state_granularity**i
            for i, bins in enumerate(self.bins)
        ])

    # override
    def save(self):
        np.savetxt(self.qtable_file, self.qtable)

    # override
    def load(self):
        self.qtable = np.loadtxt(self.qtable_file)

    # override
    def best_action(self, observation):
        return np.argmax(self.qtable[self.digitize_observation(observation)])

    # override
    def start_episode(self, n_episode, observation):
        self.n_episode = n_episode
        self.state_id = self.digitize_observation(observation)
        # select current best action
        return np.argmax(self.qtable[self.state_id])

    # override
    def update(self, n_step, action, observation, reward, done, info):
        if done and n_step < OK_THREASHOLD:
            reward = -200

        next_state_id = self.digitize_observation(observation)
        next_best_action = np.argmax(self.qtable[next_state_id])

        # Update qtable.
        # Q(s[t], a[t]) = (1 - alpha) * Q(s[t], a[t])
        #   + alpha * (reward + gamma * max(Q(s[t+1], any))
        old = self.qtable[self.state_id, action]
        learned = reward + self.gamma * self.qtable[next_state_id].max()
        self.qtable[self.state_id, action] = \
            (1 - self.alpha) * old + self.alpha * learned

        # Decide next action by epsilon-greedy algorighm.
        epsilon = 0.2 * 0.999**self.n_episode
        if epsilon <= np.random.uniform(0, 1):
            action = next_best_action
        else:
            action = random_action()

        self.state_id = next_state_id
        return action


class KerasDQNAgent:
    def __init__(self, gamma=0.999, batch_size=32, model_file='tmp/model.h5',
                 replay_memory_size=50000):
        self.gamma = gamma
        self.batch_size = batch_size
        self.model_file = model_file
        self.model = tf.keras.Sequential([
            layers.Flatten(input_shape=(4,)),
            layers.Dense(16, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(NUM_ACTIONS, activation='linear'),
        ])
        self.model.compile(loss=tf.keras.losses.Huber(),
                           optimizer=optimizers.RMSprop(),
                           metrics=['accuracy'])
        self.target_model = copy.copy(self.model)
        self.replay_memory = ReplayMemory(replay_memory_size)
        self.total_steps = 0
        self.current_state = None
        self.is_preparing = True
        self.epsilon_factor = 0

    # override
    def save(self):
        self.model.save(self.model_file)

    # override
    def load(self):
        self.model = tf.keras.models.load_model(self.model_file)

    @classmethod
    def __q_values(cls, model, state):
        return model.predict(np.array([state]))[0]

    @classmethod
    def __best_action(cls, model, state):
        return np.argmax(cls.__q_values(model, state))

    # override
    def best_action(self, observation):
        return self.__best_action(self.model, observation)

    def __get_action(self, observation):
        epsilon = 0.5 * 0.9**self.epsilon_factor
        return self.best_action(observation) \
            if epsilon <= np.random.uniform(0, 1) else random_action()

    # override
    def start_episode(self, n_episode, observation):
        self.current_state = observation
        if self.is_preparing:
            return random_action()
        self.epsilon_factor += 1
        return self.__get_action(observation)

    # override
    def update(self, n_step, action, observation, reward, done, info):
        self.total_steps += 1

        if done and n_step < OK_THREASHOLD:
            reward = -1

        self.replay_memory.append(
            (self.current_state, action, reward, done, observation))
        self.current_state = observation

        # Before training, generate samples in replay memory by random actions.
        self.is_preparing = len(self.replay_memory) \
            < (self.replay_memory.capacity * 0.1)
        if self.is_preparing:
            return random_action()

        TRAINING_INTERVAL = 5
        if self.total_steps % TRAINING_INTERVAL == 0:
            # NOTE: For performance, we should use vectorize implementation.
            x_values = []
            y_values = []
            for entry in self.replay_memory.sample(self.batch_size):
                s1, a, r, d, s2 = entry
                x_values.append(s1)
                ya = r if d else r + self.gamma * \
                    self.__q_values(self.target_model, s2).max()
                y = self.__q_values(self.model, s2)
                y[a] = ya
                y_values.append(y)
            self.model.fit(np.array(x_values), np.array(y_values))

        TARGET_UPDATE_INTERVAL = 50
        if self.total_steps % TARGET_UPDATE_INTERVAL == 0:
            self.target_model = copy.copy(self.model)

        return self.__get_action(observation)


should_exit = False


def fit(agent, env, num_episodes=10000, log_style=0):
    global should_exit
    ok_count = 0
    max_consecutive_ok_count = 0
    consecutive_ok_count = 0
    episode_steps = np.zeros(num_episodes)
    min_obs = env.reset()
    max_obs = env.reset()

    for n_episode in range(num_episodes):
        if should_exit:
            break

        if log_style == 0:
            if n_episode % 10 == 0:
                if n_episode > 0:
                    print()
                print('Episode {:5} - {:5} ... '
                      .format(n_episode, n_episode+10),
                      end='')
        elif log_style == 1:
            print('=== Episode {} ==='.format(n_episode))

        observation = env.reset()
        action = agent.start_episode(n_episode, observation)

        for n_step in range(NUM_STEPS):
            if should_exit:
                break

            observation, reward, done, info = env.step(action)
            min_obs = np.minimum(min_obs, observation)
            max_obs = np.maximum(max_obs, observation)

            action = agent.update(
                n_step, action, observation, reward, done, info)

            if done:
                break

        if log_style == 0:
            print('{:<4.0f} '.format(n_step), end='')
        elif log_style == 1:
            print('steps => {:<4.0f}'.format(n_step))

        episode_steps[n_episode] = n_step

        if n_step >= OK_THREASHOLD:
            ok_count += 1
            consecutive_ok_count += 1
            if consecutive_ok_count > max_consecutive_ok_count:
                max_consecutive_ok_count = consecutive_ok_count
        else:
            consecutive_ok_count = 0

        if consecutive_ok_count >= CONSECUTIVE_OK_COUNT:
            break

    if log_style == 0:
        print()
    print('---')
    print('ok: total = {}, max_consecutive = {}'.format(
        ok_count, max_consecutive_ok_count))
    print('observation: min =  {}, max = {}'.format(min_obs, max_obs))

    t = episode_steps[:n_episode]
    plt.subplot(211)
    plt.xlabel('episode')
    plt.ylabel('steps')
    plt.plot(np.arange(len(t)), t)
    plt.subplot(212)
    plt.xlabel('steps')
    plt.ylabel('counts')
    plt.hist(t)
    plt.show()


def test(agent, env, num_episodes=10):
    global should_exit
    episode_steps = np.zeros(num_episodes)

    for n_episode in range(num_episodes):
        if should_exit:
            break

        if n_episode % 10 == 0:
            if n_episode > 0:
                print()
            print('Episode {:5} - {:5} ... '.format(n_episode, n_episode+10),
                  end='')

        observation = env.reset()
        action = agent.best_action(observation)

        for n_step in range(NUM_STEPS):
            if should_exit:
                break

            observation, reward, done, info = env.step(action)
            action = agent.best_action(observation)

            if done:
                break

        print('{:<4.0f} '.format(n_step), end='')
        episode_steps[n_episode] = n_step

    print()

    t = episode_steps[:n_episode]
    plt.subplot(211)
    plt.xlabel('episode')
    plt.ylabel('steps')
    plt.plot(np.arange(len(t)), t)
    plt.subplot(212)
    plt.xlabel('steps')
    plt.ylabel('counts')
    plt.hist(t)
    plt.show()
    pass


def signal_handler(signum, frame):
    global should_exit
    if signum == signal.SIGINT or signum == signal.SIGTERM:
        should_exit = True


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

np.random.seed(0)
env = gym.make('CartPole-v0')
if False:
    env = wrappers.Monitor(env, 'tmp/videos', force=True)
# agent = QTableAgent()
# fit(agent, env)
agent = KerasDQNAgent()
fit(agent, env, log_style=1)
agent.save()
# should_exit = False
# test(agent, env)
