from logging import getLogger
from typing import Optional, Tuple, Callable
import random
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
    [int, Optional[tetris.TSpinType], tetris.GameState], float]


def default_reward_func(num_cleared_lines: int,
                        tspin: Optional[tetris.TSpinType],
                        stats: tetris.Statistics) -> float:
    r = num_cleared_lines
    if tspin is tetris.TSpinType.MINI:
        if num_cleared_lines == 1:
            r += 1
    elif tspin is tetris.TSpinType.NORMAL:
        if num_cleared_lines == 1:
            r += 3
        elif num_cleared_lines == 2:
            r += 5
        elif num_cleared_lines == 3:
            r += 6
    elif num_cleared_lines == 4:
        r += 5
    if stats.btb > 0:
        r += 2
    if stats.combos > 0:
        r += 1.4 ** stats.combos - 1.4
    return 1 + r * 5


def run_steps(model: M.TetrisModel, device: torch.device, max_steps=100,
              rand=random.Random(),
              step_result_cb=default_step_result_cb,
              reward_func=default_reward_func) \
        -> Tuple[int, float, tetris.Game]:
    game = tetris.Game.default(rand)
    total_score = 0
    step_id = -1
    for step_id in range(max_steps):
        logger.info('step#{}'.format(step_id))
        logger.debug('game:\n{}'.format(game))

        action, action_log_prob, state_value = decide_action(
            model, device, game.state)
        fp, _ = action

        logger.info('state_value: {:.3f}, fp = {} ({:.3f})'.format(
            state_value, fp, action_log_prob))

        r = game.lock(fp)
        assert r is not None
        reward = reward_func(r[0], r[1], game.stats)
        total_score += reward

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
