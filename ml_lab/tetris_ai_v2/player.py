from logging import getLogger
import copy
from typing import NamedTuple, Optional
import random
import numpy as np
import torch
import ml_lab.tetris_ai_v2.tetris as tetris
import ml_lab.tetris_ai_v2.mcts as mcts
import ml_lab.tetris_ai_v2.model as M

logger = getLogger(__name__)


class MctsValue(NamedTuple):
    state: tetris.GameState
    by_fp: Optional[tetris.FallingPiece]


MctsTree = mcts.Tree[tetris.GameState, MctsValue]
MctsNode = mcts.Node[tetris.GameState, MctsValue]


def simulate(target: MctsNode, model: M.TetrisModel):
    # Select
    leaf = target.select()
    assert leaf.is_leaf()

    # Expand
    v = leaf.value
    found = v.state.falling_piece.search_droppable(v.state.playfield)
    for fp, _ in found:
        g = tetris.Game(copy.deepcopy(v.state))
        g.lock(fp)
        if g.state.is_game_over:
            continue
        leaf.append_child(g.state, MctsValue(g.state, fp))

    # Evaluate
    action_probs, state_value = \
        model(M.game_state_to_tensor(leaf.value.state).unsqueeze(0))
    for node in leaf.children:
        idx = M.fp_to_index(node.value.by_fp)
        prob = action_probs[idx]
        node.params.p = prob

    # Backpropagate
    leaf.backpropagate(state_value)


def select_action(target: MctsNode, tau: int):
    pi = np.zeros(len(target.children), dtype=np.float32)
    action_values = np.zeros(len(target.children), dtype=np.float32)
    for i, node in enumerate(target.children):
        pi[i] = node.params.n ** (1 / max(tau, 1))
        action_values[i] = node.params.q
    sum = np.sum(pi)
    if sum > 0:
        pi /= np.sum(pi)
    logger.debug('{} {}'.format(sum, np.sum(pi)))

    if tau == 0:
        idx = random.choice(np.argwhere(pi == max(pi)))[0]
    else:
        idx = np.where(np.random.multinomial(1, pi) == 1)[0][0]
    action_value = action_values[idx]
    selected_node = target.children[idx]

    return selected_node, action_value


def run_single_play(model: M.TetrisModel, num_simulations=1):
    game = tetris.Game.default()
    mcts_tree = MctsTree(mcts.TreeConfig(lambda x: 1), game.state,
                         MctsValue(game.state, None))
    current_node = mcts_tree.root
    tau = 10
    while not game.state.is_game_over:
        logger.info('simulate %d times', num_simulations)
        for sim_id in range(num_simulations):
            logger.info('simulate#%d', sim_id)
            simulate(current_node, model)
        if current_node.is_leaf():
            # game over?
            break
        node, action_value = select_action(current_node, tau)
        game.lock(node.value.by_fp)
        logger.debug('game:\n%s', game)
        current_node = node
        if tau > 0:
            tau -= 1
