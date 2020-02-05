import argparse
import os
import logging
import random
from collections import deque
from typing import List, Deque, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import ml_lab.tetris_ai_v2.player as player
import ml_lab.tetris_ai_v2.model as M

logger = logging.getLogger(__name__)


class StepResultMemory:
    capacity: int
    results: Deque[player.StepResult]

    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.results = deque()

    def __len__(self):
        return len(self.results)

    def append(self, r: player.StepResult):
        if len(self.results) >= self.capacity:
            self.results.popleft()
        self.results.append(r)

    def clear(self):
        self.results = deque()

    def sample(self, n: int) -> List[player.StepResult]:
        return random.sample(self.results, n)


def collect_play_data(model: M.TetrisModel, memory: StepResultMemory,
                      num_simulations=1, max_steps=500, end_score=100):
    def on_step_result(i: int, r: player.StepResult, score):
        memory.append(r)
        return i < max_steps and score < end_score

    for episode_id in range(10):
        r = player.run_single_play(model, num_simulations=num_simulations,
                                   step_result_cb=on_step_result)
        logger.info('Episode {} => is_game_over: {}, score: {}'.format(
            episode_id, r.is_game_over, r.score))


def learn(model: M.TetrisModel, memory: StepResultMemory, batch_size=32):
    optimizer = optim.Adam(model.parameters())
    cross_entropy_loss = nn.CrossEntropyLoss()
    bce_with_logits_loss = nn.BCEWithLogitsLoss()

    logger.info('learn with {} samples'.format(len(memory)))

    if len(memory) < batch_size:
        return

    indices = list(range(len(memory)))
    random.shuffle(indices)
    i = 0
    while i < len(indices):
        target_indices = indices[i:i + batch_size]
        logger.info('{}..{}'.format(i, i + len(target_indices)))
        batch = [memory.results[i] for i in target_indices]

        reward_batch = torch.stack([torch.tensor(r.reward) for r in batch])
        action_batch = torch.stack([torch.tensor(M.fp_to_index(r.dst))
                                    for r in batch])
        state_batch = torch.stack(
            [M.game_state_to_tensor(r.state) for r in batch])
        action_probs_batch, state_value_batch = model(state_batch)

        model.zero_grad()
        loss1 = (cross_entropy_loss(action_probs_batch, action_batch) *
                 reward_batch).mean()
        loss2 = bce_with_logits_loss(state_value_batch, reward_batch.float())
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()

        if len(target_indices) < batch_size:
            break
        i += len(target_indices)


def run(args: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(prog='PROG')
    parser.add_argument('-d', '--basedir', default='tmp/tetris_ai_v2/')
    parser.add_argument('-m', '--model', default='tetris_ai_v2.pt')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--end_score', default=100, type=int)
    parser.add_argument('--num_simulations', default=1, type=int)

    args = parser.parse_args(args)

    os.makedirs(args.basedir, exist_ok=True)
    model_file = os.path.join(args.basedir, args.model)

    format = '%(asctime)s %(levelname)s [%(name)s] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=format)

    model = M.TetrisModel()
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file))

    memory = StepResultMemory()
    collect_play_data(model, memory, num_simulations=args.num_simulations,
                      end_score=args.end_score)
    learn(model, memory, batch_size=args.batch_size)

    torch.save(model.state_dict(), model_file)
