from logging import getLogger
from typing import List, Tuple, Callable
import random
import copy
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import ml_lab.tetris_ai_v3.tetris as tetris
import ml_lab.tetris_ai_v3.model as M

logger = getLogger(__name__)


def decide_action(model: M.TetrisModel, device: torch.device,
                  state: tetris.GameState):
    action_probs_batch, state_value_batch = \
        model(torch.stack([M.game_state_to_tensor(state)]).to(device))
    action_probs = action_probs_batch[0]
    state_value = state_value_batch[0]
    found = state.falling_piece.search_droppable(state.playfield)
    legal_indices = [M.fp_to_index(fp) for fp, _ in found]
    legal_indices_tensor = torch.tensor(legal_indices, device=device)
    legal_action_probs = F.softmax(action_probs[legal_indices_tensor], dim=-1)
    m = Categorical(legal_action_probs)
    action_idx = m.sample()
    return found[action_idx.item()], m.log_prob(action_idx), state_value


class StepResult:
    # state: tetris.GameState
    # dst: tetris.FallingPiece
    action_log_prob: torch.Tensor
    state_value: torch.Tensor
    reward: float

    def __init__(self, action_log_prob: torch.Tensor, state_value: torch.Tensor,
                 reward: float):
        self.action_log_prob = action_log_prob
        self.state_value = state_value
        self.reward = reward


StepResultCallback = Callable[[int, StepResult, int], bool]


def default_step_result_cb(_step_i: int, _r: StepResult, _score: int):
    return True


RewardFunc = Callable[
    [tetris.GameState, tetris.Statistics, tetris.Statistics], float]


class ConstantRewardFunc:
    def __init__(self, r):
        self.r = r

    def __call__(self, _state: tetris.GameState, _prev_stats: tetris.Statistics,
                 _stats: tetris.Statistics) -> float:
        return self.r


class DensityRewardFunc:
    def __init__(self, decay=1):
        self.decay = decay

    def __call__(self, state: tetris.GameState, _prev_stats: tetris.Statistics,
                 _stats: tetris.Statistics) -> float:
        g = state.playfield.grid
        n1 = g.num_non_empty_cells()
        n2 = g.height() - g.top_padding() - g.bottom_padding()
        r = n1 / (n2 * g.width()) * (self.decay ** n2)
        return r


class BonusRewardFunc:
    def __call__(self, _state: tetris.GameState, prev_stats: tetris.Statistics,
                 stats: tetris.Statistics) -> float:
        diff = stats - prev_stats
        r = diff.lines
        if diff.lines > 1:
            if diff.tsm > 0:
                r += 1
            elif diff.tss > 0:
                r += 3
            elif diff.tsd > 0:
                r += 5
            elif diff.tst > 0:
                r += 7
            elif diff.tetris > 0:
                r += 5
            if diff.perfect_clear > 0:
                r += 10
            if diff.btb > 0:
                r += 2
            r += 1.4 ** stats.combos - 1.4
        return r


def reward_func_factory(*func_amp_list: List[Tuple[RewardFunc, float]]):
    def reward_func(state: tetris.GameState, prev_stats: tetris.Statistics,
                    stats: tetris.Statistics) -> float:
        r = 0
        for (fn, amp) in func_amp_list:
            r += fn(state, prev_stats, stats) * amp
        return r

    return reward_func


default_reward_func = reward_func_factory(
    # (DensityRewardFunc(), 1),
    (ConstantRewardFunc(1), 1),
    (BonusRewardFunc(), 5),
)


def run_steps(model: M.TetrisModel, device: torch.device, max_steps=100,
              rand=random.Random(),
              step_result_cb=default_step_result_cb,
              reward_func=default_reward_func) \
        -> Tuple[int, float, tetris.Game]:
    game = tetris.Game.default(rand)
    total_score = 0
    step_id = -1
    prev_stats = tetris.Statistics()
    for step_id in range(max_steps):
        logger.info('step#{}'.format(step_id))
        logger.debug('game:\n{}'.format(game))

        action, action_log_prob, state_value = decide_action(
            model, device, game.state)
        fp, _ = action

        r = game.lock(fp)
        assert r is not None
        reward = reward_func(game.state, prev_stats, game.stats)
        prev_stats = copy.deepcopy(game.stats)
        total_score += reward

        logger.info('state_value: {:.3f}, fp: {} ({:.3f}), reward: {:.3f}'
                    .format(state_value, fp, action_log_prob, reward))

        result = StepResult(
            # copy.deepcopy(game.state),
            # fp,
            action_log_prob,
            state_value,
            reward,
        )

        if not step_result_cb(step_id, result, total_score):
            break

        if game.state.is_game_over:
            break

    return step_id + 1, total_score, game
