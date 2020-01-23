import argparse
import math
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from typing import NamedTuple, Optional

IS_COLAB = 'google.colab' in sys.modules
if not IS_COLAB:
    from ml_lab.tetris import Environment, State, Action


class TetrisModel(nn.Module):
    def __init__(self):
        super(TetrisModel, self).__init__()
        # 10x10x40
        self.conv1 = nn.Conv2d(10, 16, 3, padding=1)
        # => 16x10x40
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        # # => 16x10x40
        # self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        # => 32x10x40
        self.fc1 = nn.Linear(16*10*40, len(Action))
        # => 7

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        x = F.softmax(self.fc1(x.view(x.size(0), -1)), 1)
        return x


def action_to_tensor(action: Action, device=None) -> torch.Tensor:
    # Set long type for gather().
    return torch.tensor([action], device=device, dtype=torch.long)


def tensor_to_action(t: torch.Tensor) -> Action:
    assert t.shape == (len(Action),)
    return Action(t.argmax().item())


def state_to_tensor(state: State, device=None) -> torch.Tensor:
    channel_values = [
        # cell (block/empty)
        0,  # dummy
        # hold piece (piece/empty)
        0 if state.hold_piece is None else state.hold_piece,
        # next pieces (piece x 5),
        state.next_pieces.pieces[0],
        state.next_pieces.pieces[1],
        state.next_pieces.pieces[2],
        state.next_pieces.pieces[3],
        state.next_pieces.pieces[4],
        # falling piece (x, y, piece)
        state.falling_piece.pos[0],
        state.falling_piece.pos[1],
        state.falling_piece.piece,
    ]
    # HxWxC
    arr = np.zeros(40 * 10 * 10, dtype=np.float).reshape(40, 10, 10)
    for y in range(40):
        for x in range(10):
            channel_values[0] = 1 if state.playfield.get_cell(
                (x, y)) > 0 else 0
            arr[(y, x)] = channel_values
    # CxHxW
    # arr = arr.transpose(1, 2, 0)
    arr = arr.transpose(2, 0, 1)
    assert arr.shape == (10, 40, 10)
    return torch.as_tensor(arr, device=device, dtype=torch.float)


def best_action(model, state_tensor: torch.Tensor):
    with torch.no_grad():
        # state_tensor => [state_tensor] => [action_tensor] => action_tensor
        # => action
        return tensor_to_action(model(state_tensor.unsqueeze(0))[0])


class Transition(NamedTuple):
    state: State
    action: Action
    next_state: Optional[State]
    reward: int

    def is_end(self):
        return self.next_state is None


class EnvWrapper:
    def __init__(self, device):
        self.env = Environment()
        self.device = device

    def reset(self):
        s = self.env.reset()
        self.num_steps = 0
        self.state_tensor = state_to_tensor(s, self.device)
        self.total_rewards = 0

    def step(self, action: Action):
        self.num_steps += 1
        state, result, done = self.env.step(action)
        reward = 0
        if result.stats_diff.dropped_lines > 0:
            reward += 1
        bonus = 1.2 if result.is_btb else 1
        if result.is_tetris():
            reward += 10 * bonus
        elif result.is_tspin(3):
            reward += 10 * bonus
        elif result.is_tspin(2):
            reward += 10 * bonus
        elif result.is_tspin(1):
            reward += 8 * bonus
        else:
            reward += result.num_cleared_lines
        self.total_rewards += reward
        if not done and self.num_steps > 100:
            done = self.total_rewards / self.num_steps < 0.1
        if done:
            state_tensor = None
        else:
            state_tensor = state_to_tensor(state, self.device)
        t = Transition(self.state_tensor,
                       action_to_tensor(action, self.device),
                       state_tensor, torch.tensor([reward], device=self.device))
        self.state_tensor = state_tensor
        return t


def learn(env: EnvWrapper, model, optimizer, device, writer: SummaryWriter,
          model_factory=None):
    replay_memory_capacity = 50000
    replay_memory_warmup_size = 2000
    eps_end = 0
    eps_start = 0.9
    eps_end = 0.01
    eps_decay = 10000
    num_episodes = 1000
    training_interval = 1
    batch_size = 32
    gamma = 0.999
    target_model_update_interval = 10
    max_steps = 5000

    class FIFOMemory:
        def __init__(self, capacity):
            self.capacity = capacity
            self.memory = []
            self.pos = 0

        def append(self, entry):
            if len(self.memory) < self.capacity:
                self.memory.append(None)
            self.memory[self.pos] = entry
            self.pos = (self.pos + 1) % self.capacity

        def sample(self, size):
            return random.sample(self.memory, size)

        def __len__(self):
            return len(self.memory)

    print('=== Learn ===')
    print('# Parameters')
    param_keys = ['replay_memory_capacity', 'replay_memory_warmup_size',
                  'eps_end', 'eps_start', 'eps_end', 'eps_decay',
                  'num_episodes', 'training_interval', 'batch_size', 'gamma',
                  'target_model_update_interval', 'max_steps']
    for key in param_keys:
        print('{}={}'.format(key, locals()[key]))

    replay_memory = FIFOMemory(replay_memory_capacity)

    print('# Warming up...')
    steps = []
    while len(replay_memory) < replay_memory_warmup_size:
        env.reset()
        for step_id in range(max_steps):
            t = env.step(Action.random())
            replay_memory.append(t)
            if t.is_end():
                break
        steps.append(step_id)

    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.plot(steps)
    plt.show()

    print('# Training...')
    if model_factory is not None:
        target_model = model_factory()
    else:
        target_model = type(model)()

    def update_target_model():
        nonlocal target_model
        target_model.load_state_dict(model.state_dict())

    model.to(device)
    target_model.to(device)
    update_target_model()

    eps_factor = 0

    def select_action(state_tensor: torch.Tensor):
        nonlocal eps_factor
        eps_threshold = eps_end + (eps_start - eps_end) * \
            math.exp(-1. * eps_factor / eps_decay)
        eps_factor += 1
        if np.random.random() > eps_threshold:
            return best_action(model, state_tensor)
        else:
            return Action.random()

    num_total_steps = 0
    steps = []
    rewards = []
    success_counter = 0
    for episode_id in range(num_episodes):
        print('{} ... '.format(episode_id), end='')
        env.reset()
        for step_id in range(max_steps):
            num_total_steps += 1
            t = env.step(select_action(env.state_tensor))
            replay_memory.append(t)

            if num_total_steps % training_interval == 0 or False:
                transitions = replay_memory.sample(batch_size)
                batch = Transition(*zip(*transitions))  # transpose
                non_final_mask = torch.tensor(
                    tuple(map(lambda s: s is not None, batch.next_state)),
                    device=device, dtype=torch.bool)
                non_final_next_states = torch.stack(
                    [s for s in batch.next_state if s is not None])
                state_batch = torch.stack(batch.state)
                action_batch = torch.stack(batch.action)
                reward_batch = torch.cat(batch.reward)

                state_q_values = model(state_batch).gather(1, action_batch)

                next_state_q_values = torch.zeros(batch_size, device=device)
                next_state_best_actions = \
                    model(non_final_next_states).max(1)[1].unsqueeze(1)
                next_state_q_values[non_final_mask] = \
                    target_model(non_final_next_states).gather(
                        1, next_state_best_actions).squeeze().detach()

                expected_q_values = (
                    next_state_q_values * gamma) + reward_batch

                loss = F.smooth_l1_loss(state_q_values,
                                        expected_q_values.unsqueeze(1))

                optimizer.zero_grad()
                loss.backward()
                for param in model.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()

            if num_total_steps % target_model_update_interval == 0:
                update_target_model()

            if t.is_end():
                break

        print('{} ({})'.format(env.num_steps, env.total_rewards))
        steps.append(env.num_steps)
        writer.add_scalar('Episode/Steps', env.num_steps, episode_id)
        writer.add_scalar('Episode/Rewards', env.total_rewards, episode_id)

        if env.num_steps >= max_steps:
            success_counter += 1
        else:
            success_counter = 0

        if success_counter >= 2:
            break

    print()
    plt.plot(steps)
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.show()


def test(env, model, num_episodes=1000):
    pass


def main():
    parser = argparse.ArgumentParser(prog='PROG')
    parser.add_argument('-m', '--model', default='tetris.pt')
    subparsers = parser.add_subparsers(dest='command')
    parser_learn = subparsers.add_parser('learn')
    parser_learn.add_argument('--logdir', default='tmp/logs')
    parser_learn.add_argument('--episodes', type=int, default=500,
                              dest='num_episodes')
    parser_learn.add_argument('--gamma', type=float, default=0.999)
    parser_learn.add_argument('--batch_size', type=int, default=32)
    parser_test = subparsers.add_parser('test')
    parser_test.add_argument('--episodes', type=int, default=1000)

    G = globals()
    args = parser.parse_args(G['ARGS'] if 'ARGS' in G else None)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = EnvWrapper(device)
    model = TetrisModel()
    model.to(device)

    if args.command == 'learn':
        optimizer = optim.RMSprop(model.parameters())
        writer = SummaryWriter(log_dir=args.logdir)
        learn(env, model, optimizer, device, writer)
        writer.close()
        torch.save(model.state_dict(), args.model)
        # test(env, model, num_episodes=args.test_episodes)
    elif args.command == 'test':
        model.load_state_dict(torch.load(args.model))
        # test(env, model, num_episodes=args.episodes)


main()
