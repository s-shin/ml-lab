from collections import deque
from enum import IntEnum
import numpy as np
import random
import copy
from typing import Optional, NamedTuple, Tuple

Vector2 = Tuple[int, int]


class Rotation(IntEnum):
    DEG_0 = 0
    DEG_90 = 1
    DEG_180 = 2
    DEG_270 = 3

    def rotate(self, n: int):
        return Rotation.from_int(self + n)

    @classmethod
    def from_int(cls, n: int):
        return Rotation(n % 4)

    @classmethod
    def random(cls):
        return Rotation(random.randint(0, 3))


class Cell(IntEnum):
    Empty = 0
    S = 1
    Z = 2
    L = 3
    J = 4
    I = 5  # noqa: E741
    T = 6
    O = 7  # noqa: E741
    Garbage = 8

    def is_empty(self):
        return self is Cell.Empty

    def __str__(self):
        return ' SZLJITO*'[self]

    def __format__(self, format_spec):
        return str(self)


class Grid:
    @classmethod
    def by_size(cls, size: Vector2):
        g = cls()
        g.cells = np.zeros((size[1], size[0]), dtype='B')
        return g

    @classmethod
    def by_cells(cls, cells, reverse_rows=False):
        g = cls()
        g.cells = np.array(cells, dtype='B')
        if reverse_rows:
            g.cells = np.flipud(g.cells)
        return g

    def __eq__(self, rhs):
        return np.array_equal(self.cells, rhs.cells)

    def width(self):
        return self.cells.shape[1]

    def height(self):
        return self.cells.shape[0]

    def can_get_cell(self, pos: Vector2):
        return 0 <= pos[0] < self.width() and 0 <= pos[1] < self.height()

    def get_cell(self, pos: Vector2) -> Optional[Cell]:
        return Cell(self.cells[pos[1], pos[0]]) \
            if self.can_get_cell(pos) else None

    def set_cell(self, pos: Vector2, cell: Cell):
        self.cells[pos[1], pos[0]] = cell

    def can_put(self, other_pos: Vector2, other_grid: 'Grid') -> bool:
        for og_y in range(other_grid.height()):
            for og_x in range(other_grid.width()):
                if other_grid.get_cell((og_x, og_y)).is_empty():
                    continue
                x = other_pos[0] + og_x
                y = other_pos[1] + og_y
                if x < 0 or x >= self.width() or y < 0 or y >= self.height() \
                        or not self.get_cell((x, y)).is_empty():
                    return False
        return True

    def put(self, other_pos: Vector2, other_grid: 'Grid'):
        for og_y in range(other_grid.height()):
            for og_x in range(other_grid.width()):
                if other_grid.get_cell((og_x, og_y)).is_empty():
                    continue
                x = other_pos[0] + og_x
                y = other_pos[1] + og_y
                assert 0 <= x < self.width()
                assert 0 <= y < self.height()
                assert self.get_cell((x, y)).is_empty()
                self.set_cell((x, y), other_grid.get_cell((og_x, og_y)))

    def drop_filled_rows(self) -> int:
        n = 0
        for y in range(self.height()):
            is_filled = True
            for x in range(self.width()):
                if self.get_cell((x, y)).is_empty():
                    is_filled = False
                    break
            if is_filled:
                self.cells[y] = 0
                n += 1
            elif n > 0:
                # swap two rows
                self.cells[[y-n, y]] = self.cells[[y, y-n]]
        return n


def gen_grids(cells_cw0_list):
    cells = np.flipud(np.array(cells_cw0_list, dtype='B'))
    return [Grid.by_cells(np.rot90(cells, -r)) for r in range(4)]


PIECE_GRIDS = [
    # S
    gen_grids([
        [0, 1, 1],
        [1, 1, 0],
        [0, 0, 0],
    ]),
    # Z
    gen_grids([
        [2, 2, 0],
        [0, 2, 2],
        [0, 0, 0],
    ]),
    # L
    gen_grids([
        [0, 0, 3],
        [3, 3, 3],
        [0, 0, 0],
    ]),
    # J
    gen_grids([
        [4, 0, 0],
        [4, 4, 4],
        [0, 0, 0],
    ]),
    # I
    gen_grids([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 5, 5, 5, 5],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]),
    # T
    gen_grids([
        [0, 6, 0],
        [6, 6, 6],
        [0, 0, 0],
    ]),
    # O
    gen_grids([
        [0, 7, 7],
        [0, 7, 7],
        [0, 0, 0],
    ]),
]


PIECE_INITIAL_POSITIONS = [
    # S
    (3, 18),
    # Z
    (3, 18),
    # L
    (3, 18),
    # J
    (3, 18),
    # I
    (2, 17),
    # T
    (3, 18),
    # O
    (3, 18),
]

SRS_OFFSET_DATA_I = [
    [(0, 0), (-1, 0), (2, 0), (-1, 0), (2, 0)],
    [(-1, 0), (0, 0), (0, 0), (0, 1), (0, -2)],
    [(-1, 1), (1, 1), (-2, 1), (1, 0), (-2, 0)],
    [(0, 1), (0, 1), (0, 1), (0, -1), (0, 2)],
]

SRS_OFFSET_DATA_O = [[(0, 0)], [(0, -1)], [(-1, -1)], [(-1, 0)]]

SRS_OFFSET_DATA_OTHERS = [
    [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
    [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
    [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
    [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
]


class Piece(IntEnum):
    S = Cell.S
    Z = Cell.Z
    L = Cell.L
    J = Cell.J
    I = Cell.I  # noqa: E741
    T = Cell.T
    O = Cell.O  # noqa: E741

    def grid(self, cw=Rotation.DEG_0):
        return PIECE_GRIDS[self.to_index()][cw]

    def initial_pos(self):
        return PIECE_INITIAL_POSITIONS[self.to_index()]

    def to_index(self):
        return self - 1

    @classmethod
    def from_index(cls, i: int):
        return Piece(i + 1)

    def srs_offset_data(self, cw: Rotation):
        if self is Piece.I:
            return SRS_OFFSET_DATA_I[cw]
        if self is Piece.O:
            return SRS_OFFSET_DATA_O[cw]
        return SRS_OFFSET_DATA_OTHERS[cw]

    def __str__(self):
        return 'SZLJITO'[self.to_index()]

    def __format__(self, format_spec):
        return str(self)


class TSpinType(IntEnum):
    NORMAL = 0
    MINI = 1


class FallingPiece:
    piece: Piece
    rotation: Rotation
    pos: Vector2

    def __init__(self, piece: Piece, rotation: Rotation, pos: Vector2):
        self.piece = piece
        self.rotation = rotation
        self.pos = pos

    @classmethod
    def spawn(cls, piece: Piece, playfield: Grid) -> 'FallingPiece':
        g = piece.grid()
        pos = piece.initial_pos()
        if not playfield.can_put(pos, g):
            pos = (pos[0], pos[1] + 1)
        return cls(piece, Rotation.DEG_0, pos)

    def grid(self):
        return self.piece.grid(self.rotation)

    def droppable(self, playfield: Grid) -> int:
        # TODO: optimize more
        g = self.grid()
        dy = 0
        while playfield.can_put((self.pos[0], self.pos[1]-dy), g):
            dy += 1
        return dy - 1

    def drop(self, playfield: Grid, limit=-1) -> bool:
        # TODO: optimize more
        dy = self.droppable(playfield)
        if dy < 0:
            return False
        if limit > 0:
            dy = min(dy, limit)
        self.pos = (self.pos[0], self.pos[1] - dy)
        return True

    def shift(self, playfield: Grid, n: int) -> int:
        if n == 0:
            return 0
        sign = 1 if n > 0 else -1
        g = self.grid()
        x = self.pos[0]
        for i in range(n * sign):
            xx = x + sign
            if not playfield.can_put((xx, self.pos[1]), g):
                break
            x = xx
        self.pos = (x, self.pos[1])
        return i - 1

    def lock(self, playfield: Grid) \
            -> Optional[Tuple[int, Optional[TSpinType]]]:
        if self.droppable(playfield) > 0:
            return None
        g = self.grid()
        if not playfield.can_put(self.pos, g):
            return None
        tspin = self.check_tspin_type(playfield)
        playfield.put(self.pos, g)
        n = playfield.drop_filled_rows()
        return (n, tspin)

    def check_tspin_type(self, playfield: Grid) -> Optional[TSpinType]:
        if self.piece is not Piece.T:
            return None
        num_corner_blocks = 0
        for dy in [0, 2]:
            for dx in [0, 2]:
                pf_pos = (self.pos[0] + dx, self.pos[1] + dy)
                if not playfield.can_get_cell(pf_pos) or \
                        not playfield.get_cell(pf_pos).is_empty():
                    num_corner_blocks += 1
        if num_corner_blocks >= 3:
            behind = [(0, -1), (-1, 0), (0, 1), (1, 0)][self.rotation]
            # (x + 1, y + 1) is center of T block.
            pf_pos = (self.pos[0] + 1 + behind[0], self.pos[1] + 1 + behind[1])
            if not playfield.can_get_cell(pf_pos) or \
                    not playfield.get_cell(pf_pos).is_empty():
                if num_corner_blocks == 4:
                    return TSpinType.NORMAL  # triple variants
                return TSpinType.MINI
            return TSpinType.NORMAL
        return None

    def rotate(self, playfield: Grid, cw: bool) -> Optional['FallingPiece']:
        current_cw = self.rotation
        next_cw = self.rotation.rotate(1 if cw else -1)
        next_grid = self.piece.grid(next_cw)
        offsets1 = self.piece.srs_offset_data(current_cw)
        offsets2 = self.piece.srs_offset_data(next_cw)
        for i in range(len(offsets1)):
            (x, y) = self.pos
            x += offsets1[i][0] - offsets2[i][0]
            y += offsets1[i][1] - offsets2[i][1]
            if not playfield.can_put((x, y), next_grid):
                continue
            self.pos = (x, y)
            self.rotation = next_cw
            return True
        return False


class NextPieces:
    def __init__(self, auto_gen=True, visible=5):
        self.pieces = deque()
        self.auto_gen = auto_gen
        if auto_gen:
            self.generate()
        self.visible = visible

    def generate(self, lt=8):
        if len(self.pieces) >= lt:
            return
        ps = [Piece.from_index(i) for i in range(7)]
        random.shuffle(ps)
        self.pieces.extend(ps)

    def pop(self) -> Optional[Piece]:
        if self.auto_gen:
            self.generate()
        if len(self.pieces) == 0:
            return None
        return self.pieces.popleft()

    def __str__(self):
        s = ''
        for i, p in enumerate(self.pieces):
            if i >= self.visible:
                break
            s += str(p)
        return s

    def __format__(self, format_spec):
        return str(self)


class RotateAction:
    def __init__(self, cw=True, rotation=Rotation.DEG_0):
        self.cw = cw
        self.rotation = rotation

    @classmethod
    def random(cls):
        return cls(random.choice([True, False]), Rotation.random())


class ShiftAction:
    MIN = -9
    MAX = 9

    def __init__(self, n=0):
        self.n = n

    def is_shifted_to_right(self):
        return self.n >= 0

    def is_shifted_to_left(self):
        return self.n < 0

    @classmethod
    def random(cls, range=(-9, 9)):
        return cls(random.randint(*range))


class DropHoldAction(IntEnum):
    FIRM_DROP = 0
    HARD_DROP = 1
    HOLD = 2

    @classmethod
    def random(cls):
        return cls(random.randint(0, 2))


class Action:
    def __init__(self, rotate=RotateAction(), shift=ShiftAction(),
                 drop_hold=DropHoldAction.FIRM_DROP):
        self.rotate = rotate
        self.shift = shift
        self.drop_hold = drop_hold

    @classmethod
    def random(cls, shift_range=(ShiftAction.MIN, ShiftAction.MAX)):
        return cls(
            RotateAction.random(),
            ShiftAction.random(range=shift_range),
            DropHoldAction.random(),
        )


class Statistics:
    lines = 0
    tetris = 0
    combos = 0
    max_combos = 0
    tst = 0
    tsd = 0
    tss = 0
    tsm = 0
    tsz = 0
    btb = 0
    max_btb = 0


class State:
    def __init__(self, playfield: Grid, next_pieces: NextPieces,
                 falling_piece: FallingPiece, hold_piece: Optional[Piece]):
        self.playfield = playfield
        self.next_pieces = next_pieces
        self.falling_piece = falling_piece
        self.hold_piece = hold_piece
        self.stats = Statistics()
        self.is_game_over = False

    def __str__(self):
        stats_lines = []
        stats_lines.append('==============')
        stats_lines.append('TETRIS  COMBOS')
        combos_str = '{}/{}'.format(self.stats.combos, self.stats.max_combos)
        stats_lines.append('{:6}  {:>6}'.format(self.stats.tetris, combos_str))
        stats_lines.append('TST___  TSD___')
        stats_lines.append('{:6}  {:6}'.format(self.stats.tst, self.stats.tsd))
        stats_lines.append('TSS___  TSM___')
        stats_lines.append('{:6}  {:6}'.format(self.stats.tss, self.stats.tsm))
        stats_lines.append('BTB___  LINES_')
        btb_str = '{}/{}'.format(self.stats.btb, self.stats.max_btb)
        stats_lines.append('{:>6}  {:6}'.format(btb_str, self.stats.lines))
        stats_lines.append('==============')
        lines = []
        lines.append('[{}]      {:5}'.format(
            self.hold_piece or ' ', self.next_pieces))
        lines.append('--+----------+  {}'.format(stats_lines[0]))
        fp = self.falling_piece
        fp_grid = fp.grid()
        for i in range(20):
            y = 19 - i
            row = []
            for x in range(10):
                cell = fp_grid.get_cell((x - fp.pos[0], y - fp.pos[1]))
                if cell is not None and not cell.is_empty():
                    row.append(str(cell))
                    continue
                row.append(str(self.playfield.get_cell((x, y))))
            stats_line = stats_lines[i+1] if i + 1 < len(stats_lines) else ''
            lines.append('{:02}|{}|  {}'.format(y, ''.join(row), stats_line))
        lines.append('--+----------+')
        lines.append('##|0123456789|')
        return '\n'.join(lines)

    def __format__(self, format_spec):
        return str(self)


class StepResult(NamedTuple):
    num_cleared_lines: int = 0
    tspin: Optional[TSpinType] = None
    num_combos: int = 0
    btb: bool = False
    game_over: bool = False

    def is_tetris(self):
        return self.num_cleared_lines == 4

    def is_tspin(self, n):
        return self.tspin is TSpinType.NORMAL and self.num_cleared_lines == n

    def is_tspin_mini(self, n):
        return self.tspin is TSpinType.MINI and self.num_cleared_lines == n


class Environment:
    will_get_btb = False
    can_hold = False

    def reset(self) -> State:
        playfield = Grid.by_size((10, 40))
        next_pieces = NextPieces()
        falling_piece = FallingPiece.spawn(next_pieces.pop(), playfield)
        assert falling_piece is not None
        self.state = State(playfield, next_pieces, falling_piece, None)
        return copy.deepcopy(self.state)

    def step(self, action: Action) -> Tuple[State, StepResult, bool]:
        s = self.state

        if s.is_game_over:
            return copy.deepcopy(s), StepResult(), True

        fp = s.falling_piece

        # Rotate
        for _ in range(action.rotate.rotation):
            if not fp.rotate(s.playfield, action.rotate.cw):
                break

        # Shift
        fp.shift(s.playfield, action.shift.n)

        # Drop/Hold
        num_cleared_lines = 0
        tspin = None

        if action.drop_hold is DropHoldAction.FIRM_DROP:
            r = fp.drop(s.playfield)
            assert r
        elif action.drop_hold is DropHoldAction.HARD_DROP:
            fp.drop(s.playfield)
            r = fp.lock(s.playfield)
            assert r is not None
            (num_cleared_lines, tspin) = r
            if num_cleared_lines > 0:
                s.stats.lines += num_cleared_lines
                s.stats.combos += 1
                s.stats.max_combos = max(s.stats.combos, s.stats.max_combos)
                if num_cleared_lines == 4:
                    s.stats.tetris += 1
                if tspin is TSpinType.NORMAL:
                    if num_cleared_lines == 3:
                        s.stats.tst += 1
                    elif num_cleared_lines == 2:
                        s.stats.tsd += 1
                    elif num_cleared_lines == 1:
                        s.stats.tss += 1
                elif tspin is TSpinType.MINI:
                    if num_cleared_lines == 1:
                        s.stats.tsm += 1
                    elif num_cleared_lines == 0:
                        s.stats.tsz += 1
                if num_cleared_lines == 4 or tspin is not None:
                    if self.will_get_btb:
                        s.stats.btb += 1
                        s.stats.max_btb = max(s.stats.btb, s.stats.max_btb)
                    else:
                        self.will_get_btb = True
            else:
                s.stats.combos = 0
                s.stats.btb = 0
                self.will_get_btb = False
            fp = FallingPiece.spawn(s.next_pieces.pop(), s.playfield)
            self.can_hold = True
        elif action.drop_hold is DropHoldAction.HOLD:
            if self.can_hold:
                if s.hold_piece is not None:
                    fp = FallingPiece.spawn(s.hold_piece, s.playfield)
                s.hold_piece = fp.piece
                self.can_hold = False

        s.falling_piece = fp
        s.is_game_over = not s.playfield.can_put(fp.pos, fp.grid())

        result = StepResult(num_cleared_lines, tspin, s.stats.combos,
                            s.stats.btb > 0, s.is_game_over)
        return copy.deepcopy(s), result, s.is_game_over
