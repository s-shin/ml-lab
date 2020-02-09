import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ml_lab.tetris_ai_v2.tetris as tetris

INPUT_CHANNELS = 7
PLAYFIELD_SIZE = tetris.DEFAULT_PLAYFIELD_SIZE
NUM_CELLS = PLAYFIELD_SIZE[0] * PLAYFIELD_SIZE[1]
NUM_ROTATION_TYPES = 4
NUM_ACTION_TYPES = NUM_CELLS * NUM_ROTATION_TYPES
# NUM_RESIDUAL_LAYERS = 20
# K = 192
NUM_RESIDUAL_LAYERS = 10
K = 96


def game_state_to_tensor(s: tetris.GameState) -> torch.Tensor:
    a = np.zeros((PLAYFIELD_SIZE[1], PLAYFIELD_SIZE[0], INPUT_CHANNELS),
                 dtype=np.float32)
    for y in range(PLAYFIELD_SIZE[1]):
        for x in range(PLAYFIELD_SIZE[0]):
            a[(y, x)] = (
                # 0: empty, 1: block
                0 if s.playfield.grid.get_cell((x, y)).is_empty() else 1,
                # current piece
                s.falling_piece.piece,
                # next pieces
                s.next_pieces.pieces[0],
                s.next_pieces.pieces[1],
                s.next_pieces.pieces[2],
                s.next_pieces.pieces[3],
                s.next_pieces.pieces[4],
            )
    a = np.transpose(a, (2, 0, 1))
    assert a.shape == (INPUT_CHANNELS, PLAYFIELD_SIZE[1], PLAYFIELD_SIZE[0])
    return torch.as_tensor(a)


def fp_to_index(fp: tetris.FallingPiece) -> int:
    x = fp.pos[0] + int(fp.grid().width() * 0.5)
    y = fp.pos[1] + int(fp.grid().height() * 0.5)
    return (x + 10 * y) * 4 + fp.rotation


class ResidualLayer(nn.Module):
    def __init__(self):
        super(ResidualLayer, self).__init__()
        self.conv1 = nn.Conv2d(K, K, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(K)
        self.conv2 = nn.Conv2d(K, K, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(K)
        self.conv3 = nn.Conv2d(K, K, kernel_size=3, padding=1)

    def forward(self, input):
        x = F.relu(self.norm1(self.conv1(input)))
        x = self.norm2(self.conv2(x))
        x = F.relu(input + x)
        return x


class TetrisModel(nn.Module):
    def __init__(self):
        super(TetrisModel, self).__init__()
        self.conv1 = nn.Conv2d(INPUT_CHANNELS, K, kernel_size=3, padding=1)
        self.res_layers = [ResidualLayer() for i in range(NUM_RESIDUAL_LAYERS)]
        self.action_head = nn.Linear(K * NUM_CELLS, NUM_ACTION_TYPES)
        self.state_value_head = nn.Linear(K * NUM_CELLS, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.conv1(x))
        for layer in self.res_layers:
            x = layer(x)
        x = x.view(batch_size, -1)
        action_probs_batch = F.softmax(self.action_head(x), dim=-1)
        state_value_batch = self.state_value_head(x)
        return action_probs_batch, state_value_batch.squeeze(1)
