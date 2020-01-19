from enum import Enum
import math
import random
import curses


class Action(Enum):
    NONE = 0
    RIGHT = 1
    LEFT = 2


class Game:
    def __init__(self, size=(10, 10), point_config={}, random_seed=None):
        self.size = size
        self.balls = []
        self.gen_counter = 0
        self.rand = random.Random(random_seed)
        self.catcher_x = math.floor(size[0] / 2)
        self.point_config = {
            "ball_max": 5,
            "out": -2,
            **point_config
        }
        self.points = 0
        self.last_points = 0

    def __gen_ball(self):
        self.balls.append({
            "point": self.rand.randrange(self.point_config["ball_max"] + 1),
            "pos": [self.rand.randrange(self.size[0]), self.size[1] - 1],
        })
        self.gen_counter = self.rand.randrange(self.size[1])

    def update(self, action):
        if action == Action.RIGHT:
            self.catcher_x = min(self.size[0] - 1, self.catcher_x + 1)
        elif action == Action.LEFT:
            self.catcher_x = max(0, self.catcher_x - 1)
        remove_idx = 0
        self.last_points = 0
        for i in range(len(self.balls)):
            ball = self.balls[i]
            ball["pos"][1] -= 1
            y = ball["pos"][1]
            if y <= 0:
                remove_idx += 1
                if y == 0 and ball["pos"][0] == self.catcher_x:
                    self.last_points = ball["point"]
                else:
                    self.last_points += self.point_config["out"]
        self.points += self.last_points
        self.balls = self.balls[remove_idx:]
        self.gen_counter -= 1
        if self.gen_counter <= 0:
            self.__gen_ball()

    def __pos_to_index(self, pos):
        return pos[0] + pos[1] * self.size[1]

    def get_state(self):
        state = [self.__pos_to_index([self.catcher_x, 0])]
        for b in self.balls:
            state.append(self.__pos_to_index(b["pos"]))
        return state

    def get_reward(self):
        return self.last_points


def play_on_terminal():
    game = Game()

    def main(stdscr):
        bottom_y = game.size[1] - 1

        while True:
            stdscr.clear()

            for b in game.balls:
                stdscr.addstr(bottom_y - b["pos"][1], b["pos"][0],
                              "{}".format(b["point"]))
            stdscr.addstr(bottom_y, game.catcher_x, "=")

            stdscr.refresh()
            c = stdscr.getch()
            if c == curses.KEY_RIGHT:
                game.update(Action.RIGHT)
            elif c == curses.KEY_LEFT:
                game.update(Action.LEFT)
            elif c == ord('q'):
                break
            else:
                game.update(Action.NONE)

    curses.wrapper(main)
