from typings import NamedTuple, List
import torch
import ml_lab.tetris_ai_v2.game as tetris


State = torch.Tensor


def gameToState(game: tetris.Game) -> State:
    return torch.zeros(1)


class NodeValue(NamedTuple):
    state: State
    n: int = 0
    q: float = 0
    w: float = 0
    p: float = 0


class Node:
    value: NodeValue
    children: List['Node'] = []
    is_root: bool

    def __init__(self, puct_c_fn, value=NodeValue(), is_root=False):
        self.puct_c_fn = puct_c_fn
        self.value = value
        self.is_root = is_root

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def select(self):
        node = self
        while not node.is_leaf():
            c = self.puct_c_fn(self.value)
            max_idx = ''
            max_puct = 0
            for idx, child in enumerate(self.children):
                puct = child.value.q + \
                    c * child.value.p * self.value.n**0.5 / (1 + child.value.n)
                if max_puct < puct:
                    max_puct = puct
                    max_idx = idx
            node = self.children[max_idx]
        return node

    def append_child(self, value: NodeValue):
        self.children.append(Node(self.puct_c_fn, value))


class Player:
    mcts: Node

    def __init__(self):
        self.mcts = Node(lambda v: 1, is_root=True)

    def select_action(self, state):
        action = None
        return action


class PlayResult:
    pass


def runSinglePlay(player: Player):
    game = tetris.Game.default()

    while not game.is_game_over:
        action = player.select_action(gameToState(game))
