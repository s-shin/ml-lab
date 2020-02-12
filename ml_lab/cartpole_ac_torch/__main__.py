import argparse
import time
from collections import namedtuple
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter


class CartPoleModel(nn.Module):
    def __init__(self):
        super(CartPoleModel, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 16)
        # Actor
        self.action_head = nn.Linear(16, 2)
        # Critic
        self.state_value_head = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action_probs = F.softmax(self.action_head(x), dim=-1)
        state_values = self.state_value_head(x)
        return action_probs, state_values[0]


def learn(args):
    MAX_STEPS = 1000
    GAMMA = 0.999
    # Used for escaping division by zero
    MACHINE_EPS = np.finfo(np.float32).eps.item()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CartPoleModel()
    optimizer = optim.RMSprop(model.parameters())
    writer = SummaryWriter(log_dir=args.log_dir)

    env = gym.make('CartPole-v1')

    print('# Learning...')
    for episode_id in range(args.episodes):
        gym_state = env.reset()

        StepRecordEntry = namedtuple(
            'StepRecordEntry', ('action_log_prob', 'state_value', 'reward'))
        step_record = []

        for step_id in range(MAX_STEPS):
            state = torch.tensor(gym_state, device=device, dtype=torch.float)
            action_probs, state_value = model(state)
            m = Categorical(action_probs)
            action = m.sample()
            gym_state, reward, done, _ = env.step(action.item())
            step_record.append(StepRecordEntry(
                m.log_prob(action), state_value, reward))
            if done:
                break

        # Calculate state value function in each step
        expected_state_values = []
        for step_id in range(len(step_record)):
            v = sum([GAMMA ** i * ent.reward
                     for i, ent in enumerate(step_record[step_id:])])
            expected_state_values.append(v)
        # Optimized code:
        # tmp = 0
        # for entry in step_record[::-1]:
        #     tmp = entry.reward + GAMMA * tmp
        #     expected_state_values.insert(0, r)

        expected_state_values = torch.tensor(expected_state_values)
        # Standarize (cf. https://datascience.stackexchange.com/q/20098)
        expected_state_values -= expected_state_values.mean()
        expected_state_values /= (expected_state_values.std() + MACHINE_EPS)

        policy_losses = []
        state_value_losses = []
        for entry, value in zip(step_record, expected_state_values):
            # Subtract critic as baseline.
            advantage = value - entry.state_value
            policy_losses.append(-entry.action_log_prob * advantage)
            state_value_losses.append(
                F.smooth_l1_loss(entry.state_value, value))

        loss = torch.stack(policy_losses).sum() + \
               torch.stack(state_value_losses).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Episode {} => steps: {}, loss: {}'.format(
            episode_id, step_id + 1, loss.item()))

        writer.add_scalar('Episode/Steps', step_id + 1, episode_id)

    writer.close()
    torch.save(model.state_dict(), args.model)


def test(args):
    model = CartPoleModel()
    model.load_state_dict(torch.load(args.model))


def main():
    now_str = time.strftime('%Y%m%d_%H%M%S')
    model_name = 'cartpole_a3c_torch'

    parser = argparse.ArgumentParser(prog='PROG')
    subparsers = parser.add_subparsers(dest='command')

    parser_learn = subparsers.add_parser('learn')
    parser_learn.add_argument(
        '-m', '--model', default='{}-{}.pt'.format(model_name, now_str))
    parser_learn.add_argument(
        '--log_dir', default='tmp/logs/{}'.format(now_str))
    parser_learn.add_argument('--episodes', type=int, default=200)
    parser_learn.set_defaults(func=learn)

    parser_test = subparsers.add_parser('test')
    parser_test.add_argument(
        '-m', '--model', default='{}.pt'.format(model_name))
    parser_test.add_argument('--episodes', type=int, default=100)
    parser_test.set_defaults(func=test)

    G = globals()
    args = parser.parse_args(G['ARGS'] if 'ARGS' in G else None)
    args.func(args)


main()
