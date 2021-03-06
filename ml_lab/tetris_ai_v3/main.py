import argparse
import os
import json
import logging
from typing import List, Optional, NamedTuple, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import ml_lab.tetris_ai_v3.agent as agent
import ml_lab.tetris_ai_v3.model as M

logger = logging.getLogger(__name__)


def setup_logger(level: str = 'INFO', file: Optional[str] = None):
    log_format = '%(asctime)s %(levelname)s [%(name)s] %(message)s'
    logging.basicConfig(level=level.upper(), format=log_format,
                        filename=file)


HYPERPARAMS_FILE = 'hyperparams.json'
RUN_STATE_FILE = 'run_state.json'
MODEL_FILE = 'model.pt'
TB_LOG_DIR = 'tb_log'

DEFAULT_OPTIMIZER = 'adam'
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_MAX_STEPS = 100
DEFAULT_LEARNING_INTERVAL = 7
DEFAULT_CONSTANT_REWARD_FACTOR = 1.0
DEFAULT_DENSITY_REWARD_FACTOR = 1.0
DEFAULT_BONUS_REWARD_FACTOR = 5.0
DEFAULT_REWARD_DISCOUNT_RATE = 0.99


class Hyperparams(NamedTuple):
    optimizer: str = DEFAULT_OPTIMIZER  # adam or rmsprop
    learning_rate: float = DEFAULT_LEARNING_RATE
    max_steps: int = DEFAULT_MAX_STEPS
    learning_interval: int = DEFAULT_LEARNING_INTERVAL
    constant_reward_factor: float = DEFAULT_CONSTANT_REWARD_FACTOR
    density_reward_factor: float = DEFAULT_DENSITY_REWARD_FACTOR
    bonus_reward_factor: float = DEFAULT_BONUS_REWARD_FACTOR
    reward_discount_rate: float = DEFAULT_REWARD_DISCOUNT_RATE


class RunState(NamedTuple):
    last_episode_id: int = -1
    num_learning: int = 0


def load_json(file: str) -> Dict[str, Any]:
    with open(file, 'r') as fp:
        return json.load(fp)


def save_json(file: str, data: Dict[str, Any]):
    with open(file, 'w') as fp:
        json.dump(data, fp)


# --- init ---

class InitArgs(NamedTuple):
    project_dir: str
    force: bool = False
    optimizer: str = DEFAULT_OPTIMIZER
    learning_rate: float = DEFAULT_LEARNING_RATE
    max_steps: int = DEFAULT_MAX_STEPS
    learning_interval: int = DEFAULT_LEARNING_INTERVAL
    constant_reward_factor: float = DEFAULT_CONSTANT_REWARD_FACTOR
    density_reward_factor: float = DEFAULT_DENSITY_REWARD_FACTOR
    bonus_reward_factor: float = DEFAULT_BONUS_REWARD_FACTOR
    reward_discount_rate: float = DEFAULT_REWARD_DISCOUNT_RATE


def init(args: InitArgs):
    os.makedirs(args.project_dir, exist_ok=True)

    hyperparams_file = os.path.join(args.project_dir, HYPERPARAMS_FILE)
    run_state_file = os.path.join(args.project_dir, RUN_STATE_FILE)
    if not args.force:
        for file in [hyperparams_file, run_state_file]:
            if os.path.exists(file):
                logger.fatal('%s already exists', file)
                exit(1)

    logger.info('Initialize %s.', hyperparams_file)
    hyperparams = Hyperparams(
        args.optimizer, args.learning_rate,
        args.max_steps, args.learning_interval,
        args.constant_reward_factor, args.density_reward_factor,
        args.bonus_reward_factor, args.reward_discount_rate,
    )
    save_json(hyperparams_file, hyperparams._asdict())

    logger.info('Initialize %s.', run_state_file)
    run_state = RunState()
    save_json(run_state_file, run_state._asdict())

    logger.info('Done!')


# --- run ---

MACHINE_EPS = np.finfo(np.float32).eps.item()


def calc_loss(results: List[agent.StepResult], gamma=0.99):
    assert len(results) >= 2
    # Calculate discounted total rewards
    rewards = []
    last_idx = len(results) - 1
    tmp = 0 if results[last_idx].is_game_over else results[last_idx].state_value
    for i in range(last_idx - 1, -1, -1):
        r = results[i]
        tmp = r.reward + gamma * tmp
        rewards.insert(0, tmp)
    assert len(rewards) == len(results) - 1
    # Standarize
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + MACHINE_EPS)

    policy_losses = []
    value_losses = []
    for r, reward in zip(results[:-1], rewards):
        advantage = reward - r.state_value.item()  # NOTE: .item() is required!
        policy_losses.append(-r.action_log_prob * advantage)
        value_losses.append(F.smooth_l1_loss(r.state_value, reward))

    policy_losses = torch.stack(policy_losses)
    value_losses = torch.stack(value_losses)
    loss = policy_losses.sum() + value_losses.sum()
    if torch.isnan(loss):
        pass  # for breakpoint
    return loss


class RunArgs(NamedTuple):
    project_dir: str
    num_episodes: int


def run(args: RunArgs):
    hyperparams_file = os.path.join(args.project_dir, HYPERPARAMS_FILE)
    run_state_file = os.path.join(args.project_dir, RUN_STATE_FILE)
    model_file = os.path.join(args.project_dir, MODEL_FILE)
    tb_log_dir = os.path.join(args.project_dir, TB_LOG_DIR)

    hyperparams = Hyperparams(**load_json(hyperparams_file))
    run_state = RunState(**load_json(run_state_file))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    summary_writer = SummaryWriter(log_dir=tb_log_dir)

    model = M.TetrisModel()
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file))
        logger.info('model state was loaded from {}.'.format(model_file))
    model.to(device)

    if hyperparams.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=hyperparams.learning_rate)
    elif hyperparams.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(),
                                  lr=hyperparams.learning_rate)
    else:
        raise Exception('invalid argument: {}'.format(hyperparams.optimizer))

    reward_func = agent.reward_func_factory(
        (agent.DensityRewardFunc(), hyperparams.density_reward_factor),
        (agent.ConstantRewardFunc(), hyperparams.constant_reward_factor),
        (agent.BonusRewardFunc(), hyperparams.bonus_reward_factor),
    )

    results: List[agent.StepResult] = []
    losses: List[float] = []

    episode_id = run_state.last_episode_id
    for ep_i in range(args.num_episodes):
        episode_id += 1
        logger.info('Episode#{} ({})'.format(episode_id, ep_i))

        def learn():
            nonlocal results
            if len(results) <= 1:
                results = []  # discard
                return
            logger.info('Learning...')
            loss = calc_loss(results, gamma=hyperparams.reward_discount_rate)
            if torch.isnan(loss):
                # FIXME
                logger.warning('loss becomes NaN. ignored.')
                results = []
                return
            losses.append(loss.item())
            logger.info('loss: {}'.format(loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            results = []

        def on_step_result(step_id: int, r: agent.StepResult, _score: int):
            results.append(r)
            if (step_id + 1) % hyperparams.learning_interval == 0:
                learn()
            return True

        num_steps, score, game = agent.play(
            model, device, max_steps=hyperparams.max_steps,
            reward_func=reward_func, step_result_cb=on_step_result)

        learn()  # `results` will be cleared

        logger.info('steps: {}, score: {:.3f}, game:\n{}'.format(
            num_steps, score, game))
        if summary_writer is not None:
            summary_writer.add_scalar('Episode/Steps', num_steps, episode_id)
            summary_writer.add_scalar('Episode/Score', score, episode_id)
            summary_writer.add_scalar('Episode/Losses', np.mean(losses), episode_id)
            for i, loss in enumerate(losses):
                summary_writer.add_scalar('Losses', loss,
                                          run_state.num_learning + i)

        torch.save(model.state_dict(), model_file)
        logger.info('model sate was saved to {}'.format(model_file))

        run_state = run_state._replace(
            last_episode_id=episode_id,
            num_learning=run_state.num_learning + len(losses))
        save_json(run_state_file, run_state._asdict())
        logger.info('{} was updated.'.format(run_state_file))

        losses = []

    logger.info('Finished!')


# --- main ---

def main(arg_list: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(prog='PROG')
    parser.add_argument('--log_level', default='INFO')
    sub_parser = parser.add_subparsers()

    DEFAULT_PROJECT_DIR = 'tmp/tetris_ai_v3'

    p = sub_parser.add_parser('init')
    p.add_argument('--project_dir', default=DEFAULT_PROJECT_DIR)
    p.add_argument('--force', action='store_true')
    p.add_argument('--optimizer', default=DEFAULT_OPTIMIZER,
                   choices=['adam', 'rmsprop'])
    p.add_argument('--learning_rate', type=float,
                   default=DEFAULT_LEARNING_RATE)
    p.add_argument('--max_steps', type=int, default=DEFAULT_MAX_STEPS)
    p.add_argument('--learning_interval', type=int,
                   default=DEFAULT_LEARNING_INTERVAL)
    p.add_argument('--constant_reward_factor', type=float,
                   default=DEFAULT_CONSTANT_REWARD_FACTOR)
    p.add_argument('--density_reward_factor', type=float,
                   default=DEFAULT_DENSITY_REWARD_FACTOR)
    p.add_argument('--bonus_reward_factor', type=float,
                   default=DEFAULT_BONUS_REWARD_FACTOR)
    p.add_argument('--reward_discount_rate', type=float,
                   default=DEFAULT_REWARD_DISCOUNT_RATE)
    p.set_defaults(func=init)

    p = sub_parser.add_parser('run')
    p.add_argument('--project_dir', default=DEFAULT_PROJECT_DIR)
    p.add_argument('--num_episodes', default=5, type=int)
    p.set_defaults(func=run)

    args = parser.parse_args(arg_list)
    setup_logger(args.log_level)
    args.func(args)
