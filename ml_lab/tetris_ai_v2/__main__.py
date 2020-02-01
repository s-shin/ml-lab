import logging
import ml_lab.tetris_ai_v2.player as player
from ml_lab.tetris_ai_v2.model import TetrisModel


def main():
    format = '%(asctime)s %(levelname)s <%(name)s.%(funcName)s> %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=format)
    model = TetrisModel()
    player.run_single_play(model)


main()
