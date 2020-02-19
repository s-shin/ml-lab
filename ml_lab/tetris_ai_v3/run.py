import argparse
import os
import time
import logging
from typing import List, Optional, NamedTuple
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import ml_lab.tetris_ai_v3.agent as agent
import ml_lab.tetris_ai_v3.model as M

logger = logging.getLogger(__name__)

MACHINE_EPS = np.finfo(np.float32).eps.item()


def calc_loss(results: List[agent.StepResult], gamma=0.99):
    # Discounted total rewards
    rewards = []
    tmp = 0
    for r in results[::-1]:
        tmp = r.reward + gamma * tmp
        rewards.insert(0, tmp)
    # Standarize
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + MACHINE_EPS)

    policy_losses = []
    value_losses = []
    for r, reward in zip(results, rewards):
        advantage = reward - r.state_value.item()  # NOTE: .item() is required!
        policy_losses.append(-r.action_log_prob * advantage)
        value_losses.append(F.smooth_l1_loss(r.state_value, reward))

    policy_losses = torch.stack(policy_losses)
    value_losses = torch.stack(value_losses)
    loss = policy_losses.sum() + value_losses.sum()
    return loss


class Args(NamedTuple):
    log_level: str
    log_file: str
    tb_log_dir: str
    model: str
    optimizer: str
    num_episodes: int
    max_steps: int
    discount_rate: float


def run(args: Optional[List[str]] = None):
    now_str = time.strftime('%Y%m%d_%H%M%S')

    parser = argparse.ArgumentParser(prog='PROG')
    parser.add_argument('--log_level', default='INFO')
    parser.add_argument('--log_file', default='')
    parser.add_argument('--tb_log_dir',
                        default='tmp/tetris_ai_v3/tb_log/{}'.format(now_str))
    parser.add_argument('-m', '--model', default='tmp/tetris_ai_v3/model.pt')
    parser.add_argument('--optimizer', default='adam',
                        choices=['adam', 'rmsprop'])
    parser.add_argument('--num_episodes', default=5, type=int)
    parser.add_argument('--max_steps', default=500, type=int)
    parser.add_argument('--discount_rate', default=0.99, type=float)

    args = parser.parse_args(args)  # type: Args

    model_file = args.model
    os.makedirs(os.path.dirname(model_file), exist_ok=True)

    log_file = None if len(args.log_file) == 0 else args.log_file
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    log_format = '%(asctime)s %(levelname)s [%(name)s] %(message)s'
    logging.basicConfig(level=args.log_level.upper(), format=log_format,
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

    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters())
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters())

    results: List[agent.StepResult] = []

    for episode_id in range(args.num_episodes):
        logger.info('Episode#{}'.format(episode_id))

        def on_step_result(_step_i: int, r: agent.StepResult, _score: int):
            results.append(r)
            return True

        num_steps, score, game = agent.run_steps(
            model, device, max_steps=args.max_steps,
            step_result_cb=on_step_result)

        logger.info('steps: {}, score: {:.3f}, game:\n{}'.format(
            num_steps, score, game))
        if summary_writer is not None:
            summary_writer.add_scalar('Episode/Steps', num_steps, episode_id)
            summary_writer.add_scalar('Episode/Score', score, episode_id)

        logger.info('Learning...')
        loss = calc_loss(results, gamma=args.discount_rate)
        if summary_writer:
            summary_writer.add_scalar('Episode/Loss', loss.item(), episode_id)
        logger.info('loss: {}'.format(loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        results = []

        torch.save(model.state_dict(), model_file)
        logger.info('model sate was saved to {}'.format(model_file))

    logger.info('Finished!')
