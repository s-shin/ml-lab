import argparse
import os
import time
import logging
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import ml_lab.tetris_ai_v3.agent as agent
import ml_lab.tetris_ai_v3.model as M

logger = logging.getLogger(__name__)

MACHINE_EPS = np.finfo(np.float32).eps.item()


def calc_loss(results: List[agent.StepResult], gamma=0.999):
    # Discounted total rewards
    expected_state_values = []
    tmp = 0
    for r in results[::-1]:
        tmp = r.reward + gamma * tmp
        expected_state_values.insert(0, tmp)
    # Standarize
    expected_state_values = torch.tensor(expected_state_values)
    expected_state_values -= expected_state_values.mean()
    expected_state_values /= (expected_state_values.std() + MACHINE_EPS)

    policy_losses = []
    state_value_losses = []
    for r, value in zip(results, expected_state_values):
        advantage = value - r.state_value
        policy_losses.append(-r.action_log_prob * advantage)
        state_value_losses.append(F.smooth_l1_loss(r.state_value, value))

    policy_losses = torch.stack(policy_losses)
    state_value_losses = torch.stack(state_value_losses)
    loss = policy_losses.sum() + state_value_losses.sum()
    return loss


def run(args: Optional[List[str]] = None):
    now_str = time.strftime('%Y%m%d_%H%M%S')

    parser = argparse.ArgumentParser(prog='PROG')
    parser.add_argument('--log_file', default='')
    parser.add_argument('--tb_log_dir',
                        default='tmp/tetris_ai_v3/tb_log/{}'.format(now_str))
    parser.add_argument('-m', '--model', default='tmp/tetris_ai_v3/model.pt')
    parser.add_argument('--learning_episode_interval', default=1, type=int)
    parser.add_argument('--num_episodes', default=5, type=int)
    parser.add_argument('--max_steps', default=500, type=int)

    args = parser.parse_args(args)

    model_file = args.model
    os.makedirs(os.path.dirname(model_file), exist_ok=True)

    log_file = None if len(args.log_file) == 0 else args.log_file
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    format = '%(asctime)s %(levelname)s [%(name)s] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=format,
                        filename=log_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if len(args.tb_log_dir) == 0:
        summary_writer = None
    else:
        summary_writer = SummaryWriter(log_dir=args.tb_log_dir)

    model = M.TetrisModel()
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file))
        logger.info('model state was loaded from {}.'.format(model_file))
    model.to(device)
    # optimizer = optim.RMSprop(model.parameters())
    optimizer = optim.Adam(model.parameters())

    for episode_id in range(args.num_episodes):
        results: List[agent.StepResult] = []

        def on_step_result(_step_i: int, r: agent.StepResult, _score: int):
            results.append(r)
            return True

        num_steps, score, game = agent.run_steps(
            model, device, max_steps=args.max_steps,
            step_result_cb=on_step_result)

        logger.info('Episode {} => steps: {}, score: {}, game:\n{}'.format(
            episode_id, num_steps, score, game))
        if summary_writer is not None:
            summary_writer.add_scalar('Episode/Steps', num_steps, episode_id)
            summary_writer.add_scalar('Episode/Score', score, episode_id)

        if (episode_id + 1) % args.learning_episode_interval == 0:
            logger.info('Learning...')
            optimizer.zero_grad()
            loss = calc_loss(results)
            loss.backward()
            optimizer.step()

            torch.save(model.state_dict(), model_file)
            logger.info('model sate was saved to {}'.format(model_file))
