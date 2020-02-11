from collections import deque
import itertools
from enum import IntEnum
import copy
import random
import operator
from typing import Optional, Tuple, List, NamedTuple, Deque
import numpy as np
from bitarray import bitarray

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
    cells: np.ndarray
    bit_cells: bitarray

    @classmethod
    def by_size(cls, size: Vector2):
        g = cls()
        g.cells = np.zeros((size[1], size[0]), dtype='B')
        g.__sync()
        return g

    @classmethod
    def by_cells(cls, cells, reverse_rows=False):
        g = cls()
        g.cells = np.array(cells, dtype='B')
        if reverse_rows:
            g.cells = np.flipud(g.cells)
        g.__sync()
        return g

    def __init__(self):
        self.bit_cells = bitarray()

    def __eq__(self, rhs):
        return np.array_equal(self.cells, rhs.cells)

    def __hash__(self):
        # NOTE: Zobrist hash can be used?
        return hash(self.cells.tostring())

    def __sync(self):
        self.bit_cells = bitarray(self.cells.reshape(self.cells.size).tolist())

    def __bit_index(self, pos: Vector2):
        return pos[1] * self.width() + pos[0]

    def __crop_bit_cells(self, pos: Vector2, size: Vector2) \
            -> Optional[bitarray]:
        if pos[0] < 0 or pos[0] + size[0] >= self.width() \
                or pos[1] < 0 or pos[1] + size[1] >= self.height():
            return None
        r = bitarray()
        for y in range(size[1]):
            i = self.__bit_index((pos[0], pos[1] + y))
            r += self.bit_cells[i:i + size[0]]
        return r

    def width(self) -> int:
        return self.cells.shape[1]

    def height(self) -> int:
        return self.cells.shape[0]

    def size(self) -> Vector2:
        return self.width(), self.height()

    def is_empty(self) -> bool:
        return not self.bit_cells.any()

    def can_get_cell(self, pos: Vector2) -> bool:
        return 0 <= pos[0] < self.width() and 0 <= pos[1] < self.height()

    def get_cell(self, pos: Vector2) -> Optional[Cell]:
        return Cell(self.cells[pos[1], pos[0]]) \
            if self.can_get_cell(pos) else None

    def set_cell(self, pos: Vector2, cell: Cell):
        self.cells[pos[1], pos[0]] = cell
        self.bit_cells[self.__bit_index(pos)] = cell

    def can_put(self, other_pos: Vector2, other_grid: 'Grid') -> bool:
        bits = self.__crop_bit_cells(other_pos, other_grid.size())
        if bits is None:
            return self.can_put_slow(other_pos, other_grid)
        return not (bits & other_grid.bit_cells).any()

    def can_put_slow(self, other_pos: Vector2, other_grid: 'Grid') -> bool:
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

    def is_row_filled(self, y: int) -> bool:
        for x in range(self.width()):
            if self.get_cell((x, y)).is_empty():
                return False
        return True

    def is_col_filled(self, x: int) -> bool:
        for y in range(self.height()):
            if self.get_cell((x, y)).is_empty():
                return False
        return True

    def is_row_empty(self, y: int) -> bool:
        for x in range(self.width()):
            if not self.get_cell((x, y)).is_empty():
                return False
        return True

    def is_col_empty(self, x: int) -> bool:
        for y in range(self.height()):
            if not self.get_cell((x, y)).is_empty():
                return False
        return True

    def bottom_padding(self) -> int:
        h = self.height()
        i = -1
        for i in range(h):
            y = i
            if not self.is_row_empty(y):
                return i
        return i

    def top_padding(self) -> int:
        h = self.height()
        i = -1
        for i in range(h):
            y = h - i - 1
            if not self.is_row_empty(y):
                return i
        return i

    def left_padding(self) -> int:
        w = self.width()
        i = -1
        for i in range(w):
            x = i
            if not self.is_col_empty(x):
                return i
        return i

    def right_padding(self) -> int:
        w = self.width()
        i = -1
        for i in range(w):
            x = w - i - 1
            if not self.is_col_empty(x):
                return i
        return i

    def drop_filled_rows(self) -> int:
        n = 0
        for y in range(self.height()):
            if self.is_row_filled(y):
                self.cells[y] = 0
                n += 1
            elif n > 0:
                # swap two rows
                self.cells[[y - n, y]] = self.cells[[y, y - n]]
        self.__sync()
        return n

    def non_empty_rows_density(self) -> float:
        num_non_empty_rows = \
            self.height() - self.top_padding() - self.bottom_padding()
        if num_non_empty_rows == 0:
            return 1
        return self.bit_cells.count() / (self.width() * num_non_empty_rows)


def gen_grids(cells_cw0_list):
    cells = np.flipud(np.array(cells_cw0_list, dtype='B'))
    return [Grid.by_cells(np.rot90(cells, r)) for r in range(4)]


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


PIECES: Tuple[Piece] = tuple([Piece.from_index(i) for i in range(7)])


class TSpinType(IntEnum):
    NORMAL = 0
    MINI = 1


DEFAULT_PLAYFIELD_SIZE: Vector2 = (10, 40)
DEFAULT_PLAYFIELD_VISIBLE_HEIGHT: int = 20


class Playfield(NamedTuple):
    grid: Grid
    visible_height: int

    @classmethod
    def default(cls) -> 'Playfield':
        return cls(Grid.by_size(DEFAULT_PLAYFIELD_SIZE),
                   DEFAULT_PLAYFIELD_VISIBLE_HEIGHT)


class MoveType(IntEnum):
    HORIZONTAL = 0
    VERTICAL = 1
    ROTATION = 2


class Move(NamedTuple):
    type: MoveType
    n: int


class MovePath(NamedTuple):
    path: List[Move] = []

    def join(self, p: 'MovePath'):
        if len(self.path) == 0:
            return p
        if len(p.path) == 0:
            return self
        m1 = self.path[-1]
        m2 = p.path[0]
        # NOTE: Rotations cannot be merged.
        if m1.type != m2.type or m1.type == MoveType.ROTATION:
            return MovePath(self.path + p.path)
        if m1.n + m2.n == 0:
            # canceled
            return MovePath(self.path[:-1] + p.path[1:])
        # merge
        return MovePath(
            self.path[:-1] + [Move(m1.type, m1.n + m2.n)] + p.path[1:])


class FallingPiece:
    piece: Piece
    rotation: Rotation
    pos: Vector2

    def __init__(self, piece: Piece, rotation: Rotation, pos: Vector2):
        self.piece = piece
        self.rotation = rotation
        self.pos = pos

    @classmethod
    def spawn(cls, piece: Piece, pf: Playfield) -> 'FallingPiece':
        g = piece.grid()
        pos = piece.initial_pos()
        if not pf.grid.can_put(pos, g):
            pos = (pos[0], pos[1] + 1)
        return cls(piece, Rotation.DEG_0, pos)

    def clone(self) -> 'FallingPiece':
        return copy.deepcopy(self)

    def move(self, dx=0, dy=0) -> 'FallingPiece':
        self.pos = (self.pos[0] + dx, self.pos[1] + dy)
        return self

    def __str__(self):
        return '{}:{}:{},{}'.format(
            self.piece, self.rotation, self.pos[0], self.pos[1])

    def __iter__(self):
        yield self.piece
        yield self.rotation
        yield self.pos

    def __hash__(self):
        return hash(tuple(self))

    def __eq__(self, rhs: 'FallingPiece'):
        return tuple(self) == tuple(rhs)

    def grid(self):
        return self.piece.grid(self.rotation)

    def can_put_onto(self, pf: Playfield) -> bool:
        return pf.grid.can_put(self.pos, self.grid())

    def droppable(self, pf: Playfield, n=-1) -> int:
        # NOTE: maybe can optimize more
        g = self.grid()
        for i in range(1, n + 2 if n > 0 else pf.grid.height()):
            y = self.pos[1] - i
            if not pf.grid.can_put((self.pos[0], y), g):
                break
        return i - 1

    def drop(self, pf: Playfield, limit=-1) -> int:
        dy = self.droppable(pf, limit)
        if dy < 0:
            return dy
        self.pos = (self.pos[0], self.pos[1] - dy)
        return dy

    def liftable(self, pf: Playfield, n=-1) -> int:
        g = self.grid()
        for i in range(1, n + 2 if n > 0 else pf.grid.height()):
            y = self.pos[1] + i
            if not pf.grid.can_put((self.pos[0], y), g):
                break
        return i - 1

    def lift(self, pf: Playfield, limit=-1) -> int:
        dy = self.liftable(pf, limit)
        if dy < 0:
            return dy
        self.pos = (self.pos[0], self.pos[1] + dy)
        return dy

    def shift(self, pf: Playfield, n: int) -> int:
        if n == 0:
            return 0
        sign = 1 if n > 0 else -1
        g = self.grid()
        x = self.pos[0]
        for i in range(n * sign):
            xx = x + sign
            if not pf.grid.can_put((xx, self.pos[1]), g):
                break
            x = xx
        self.pos = (x, self.pos[1])
        return i - 1

    def lock(self, pf: Playfield) \
            -> Optional[Tuple[int, Optional[TSpinType]]]:
        if self.droppable(pf, 1) > 0:
            return None
        g = self.grid()
        if not pf.grid.can_put(self.pos, g):
            return None
        tspin = self.check_tspin_type(pf)
        pf.grid.put(self.pos, g)
        n = pf.grid.drop_filled_rows()
        return (n, tspin)

    def check_tspin_type(self, pf: Playfield) -> Optional[TSpinType]:
        """FIXME: The logic for mini may be incorrect"""
        if self.piece is not Piece.T:
            return None
        num_corner_blocks = 0
        for dy in [0, 2]:
            for dx in [0, 2]:
                pf_pos = (self.pos[0] + dx, self.pos[1] + dy)
                if not pf.grid.can_get_cell(pf_pos) or \
                        not pf.grid.get_cell(pf_pos).is_empty():
                    num_corner_blocks += 1
        if num_corner_blocks >= 3:
            behind = [(0, -1), (-1, 0), (0, 1), (1, 0)][self.rotation]
            # (x + 1, y + 1) is center of T block.
            pf_pos = (self.pos[0] + 1 + behind[0], self.pos[1] + 1 + behind[1])
            if not pf.grid.can_get_cell(pf_pos) or \
                    not pf.grid.get_cell(pf_pos).is_empty():
                if num_corner_blocks == 4:
                    return TSpinType.NORMAL  # triple variants
                return TSpinType.MINI
            return TSpinType.NORMAL
        return None

    def rotate(self, pf: Playfield, cw: bool) -> bool:
        """SRS by true rotation method"""
        current_cw = self.rotation
        next_cw = self.rotation.rotate(1 if cw else -1)
        next_grid = self.piece.grid(next_cw)
        offsets1 = self.piece.srs_offset_data(current_cw)
        offsets2 = self.piece.srs_offset_data(next_cw)
        for i in range(len(offsets1)):
            (x, y) = self.pos
            x += offsets1[i][0] - offsets2[i][0]
            y += offsets1[i][1] - offsets2[i][1]
            if not pf.grid.can_put((x, y), next_grid):
                continue
            self.pos = (x, y)
            self.rotation = next_cw
            return True
        return False

    def search_move_path(self, pf: Playfield, dst: Vector2, rotation: Rotation,
                         trace=False) -> Optional[MovePath]:
        """
        Currently, by brute-force search.
        Thus, the performance is not good and sometimes non best path will be
        returned.
        """
        checked = set()

        def trace_print(depth, s):
            if trace:
                print('{}{}'.format(' ' * depth * 2, s))

        def search(fp, depth=0) -> Optional[MovePath]:
            nonlocal checked
            trace_print(depth, 'fp = {}'.format(fp))

            if fp in checked:
                trace_print(depth, '=> checked')
                return None
            checked.add(fp)

            # assert fp.can_put_onto(pf)

            if fp.pos == dst:
                trace_print(depth, '=> achieved!')
                return MovePath()

            # TODO: should be able to control search priorities by option?

            trace_print(depth, '| y - 1')
            next_fp = fp.clone().move(dy=-1)
            if next_fp.can_put_onto(pf):
                p = search(next_fp, depth + 1)
                if p is not None:
                    return MovePath([Move(MoveType.VERTICAL, -1)]).join(p)

            trace_print(depth, '| x + 1')
            next_fp = fp.clone().move(dx=1)
            if next_fp.can_put_onto(pf):
                p = search(next_fp, depth + 1)
                if p is not None:
                    return MovePath([Move(MoveType.HORIZONTAL, 1)]).join(p)

            trace_print(depth, '| x - 1')
            next_fp = fp.clone().move(dx=-1)
            if next_fp.can_put_onto(pf):
                p = search(next_fp, depth + 1)
                if p is not None:
                    return MovePath([Move(MoveType.HORIZONTAL, -1)]).join(p)

            trace_print(depth, '| cw')
            next_fp = fp.clone()
            if next_fp.rotate(pf, True):
                p = search(next_fp, depth + 1)
                if p is not None:
                    return MovePath([Move(MoveType.ROTATION, 1)]).join(p)

            trace_print(depth, '| ccw')
            next_fp = fp.clone()
            if next_fp.rotate(pf, False):
                p = search(next_fp, depth + 1)
                if p is not None:
                    return MovePath([Move(MoveType.ROTATION, -1)]).join(p)

            return None

        return search(self)

    def search_droppable(self, pf: Playfield) \
            -> List[Tuple['FallingPiece', MovePath]]:
        candidates = []
        yend = min(pf.grid.height() - pf.grid.top_padding(), pf.visible_height)
        for y in range(-1, yend):
            for x in range(-1, 8):
                for r in range(4):
                    fp = FallingPiece(self.piece, Rotation(r), (x, y))
                    if fp.droppable(pf) > 0:
                        continue
                    if pf.grid.can_put(fp.pos, fp.grid()):
                        candidates.append(fp)
        r = []
        for fp in candidates:
            path = self.search_move_path(pf, fp.pos, fp.rotation)
            if path is not None:
                r.append((fp, path))
        return r


DEFAULT_NUM_VISIBLE_NEXT_PIECES = 5


class NextPieces:
    pieces: Deque[Piece]
    auto_gen: bool
    visible_num: int
    rand: random.Random

    def __init__(self, auto_gen=True,
                 visible_num=DEFAULT_NUM_VISIBLE_NEXT_PIECES,
                 rand=random.Random(None)):
        self.pieces = deque()
        self.auto_gen = auto_gen
        self.visible_num = visible_num
        self.rand = rand
        if self.auto_gen:
            self.generate()

    def generate(self, lt=8):
        if len(self.pieces) >= lt:
            return
        ps = [Piece.from_index(i) for i in range(7)]
        self.rand.shuffle(ps)
        self.pieces.extend(ps)

    def pop(self) -> Optional[Piece]:
        if self.auto_gen:
            self.generate()
        if len(self.pieces) == 0:
            return None
        return self.pieces.popleft()

    def fix(self):
        self.pieces = deque(itertools.islice(self.pieces, 0, self.visible_num))
        self.auto_gen = False

    def __str__(self):
        s = ''
        for i, p in enumerate(self.pieces):
            if i >= self.visible_num:
                break
            s += str(p)
        return s

    def __format__(self, format_spec):
        return str(self)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, rhs):
        return str(self) == str(rhs)


class GameState:
    playfield: Playfield
    next_pieces: NextPieces
    falling_piece: Optional[FallingPiece]
    hold_piece: Optional[Piece]
    is_game_over: bool = False
    can_hold: bool = False
    is_in_btb: bool = False

    def __init__(self, playfield, next_pieces, falling_piece, hold_piece):
        self.playfield = playfield
        self.next_pieces = next_pieces
        self.falling_piece = falling_piece
        self.hold_piece = hold_piece

    @classmethod
    def default(cls, rand: random.Random):
        playfield = Playfield.default()
        next_pieces = NextPieces(rand=rand)
        falling_piece = FallingPiece.spawn(next_pieces.pop(), playfield)
        assert falling_piece is not None
        return cls(playfield, next_pieces, falling_piece, None)

    def __iter__(self):
        yield self.playfield
        yield self.next_pieces
        yield self.falling_piece
        yield self.hold_piece
        yield self.is_game_over
        yield self.can_hold
        yield self.is_in_btb

    def __hash__(self):
        return hash(tuple(self))

    def __eq__(self, rhs):
        return tuple(self) == tuple(rhs)

    def __str__(self):
        s = self
        lines = []
        fp = s.falling_piece
        lines.append('[{}]   ({}){:5}'.format(
            s.hold_piece or ' ', fp.piece, s.next_pieces))
        lines.append('--+----------+')
        fp_grid = fp.grid()
        for i in range(s.playfield.visible_height):
            y = s.playfield.visible_height - 1 - i
            row = []
            for x in range(10):
                cell = fp_grid.get_cell((x - fp.pos[0], y - fp.pos[1]))
                if cell is not None and not cell.is_empty():
                    row.append(str(cell))
                    continue
                row.append(str(s.playfield.grid.get_cell((x, y))))
            lines.append('{:02}|{}|'.format(y, ''.join(row)))
        lines.append('--+----------+')
        lines.append('##|0123456789|')
        return '\n'.join(lines)


class Statistics:
    lines = 0
    tetris = 0
    combos = 0  # can be reset
    max_combos = 0
    tst = 0
    tsd = 0
    tss = 0
    tsm = 0
    tsz = 0
    btb = 0  # can be reset
    max_btb = 0
    dropped_lines = 0
    dropped_pieces = 0
    hold = 0
    perfect_clear = 0

    def to_tuple(self):
        return (self.lines, self.tetris, self.combos, self.max_combos,
                self.tst, self.tsd, self.tss, self.tsm, self.tsz, self.btb,
                self.max_btb, self.dropped_lines, self.dropped_pieces,
                self.hold, self.perfect_clear)

    @classmethod
    def from_tuple(cls, t):
        s = cls()
        (s.lines, s.tetris, s.combos, s.max_combos, s.tst, s.tsd,
         s.tss, s.tsm, s.tsz, s.btb, s.max_btb, s.dropped_lines,
         s.dropped_pieces, s.hold, s.perfect_clear) = t
        return s

    def __sub__(self, rhs: 'Statistics'):
        return Statistics.from_tuple(
            map(operator.sub, self.to_tuple(), rhs.to_tuple()))

    def __str__(self):
        s = self
        combos_str = '{}/{}'.format(s.combos, s.max_combos)
        btb_str = '{}/{}'.format(s.btb, s.max_btb)
        lines = [
            '==============',
            'TETRIS  COMBOS',
            '{:6}  {:>6}'.format(s.tetris, combos_str),
            'TST___  TSD___',
            '{:6}  {:6}'.format(s.tst, s.tsd),
            'TSS___  TSM___',
            '{:6}  {:6}'.format(s.tss, s.tsm),
            'BTB___  LINES_',
            '{:>6}  {:6}'.format(btb_str, s.lines),
            'PC____',
            '{:>6}'.format(s.perfect_clear),
            '==============',
        ]
        return '\n'.join(lines)


class Game:
    state: GameState
    stats: Statistics

    @classmethod
    def default(cls, rand: random.Random):
        return cls(GameState.default(rand))

    def __init__(self, state: GameState):
        self.state = state
        self.stats = Statistics()

    def __str__(self):
        state_lines = str(self.state).split('\n')
        stats_lines = str(self.stats).split('\n')
        lines = state_lines[:1]
        for i in range(len(stats_lines)):
            lines.append('{}  {}'.format(state_lines[1 + i], stats_lines[i]))
        lines.extend(state_lines[1 + len(stats_lines):])
        return '\n'.join(lines)

    def rotate(self, is_cw):
        assert self.state.falling_piece is not None
        self.state.falling_piece.rotate(self.state.playfield, is_cw)

    def shift(self, n):
        assert self.state.falling_piece is not None
        self.state.falling_piece.shift(self.state.playfield, n)

    def drop(self, n=-1):
        assert self.state.falling_piece is not None
        n = self.state.falling_piece.drop(self.state.playfield, n)
        if n > 0:
            self.stats.dropped_lines += n

    def hard_drop(self) -> Tuple[int, Optional[TSpinType]]:
        assert self.state.falling_piece is not None
        s = self.state
        n = s.falling_piece.drop(s.playfield)
        if n > 0:
            self.stats.dropped_lines += n
        self.lock()

    def lock(self, fp: Optional[FallingPiece] = None):
        assert self.state.falling_piece is not None
        s = self.state
        stats = self.stats
        if fp is not None:
            s.falling_piece = fp
        if s.falling_piece.pos[1] >= s.playfield.visible_height:
            s.is_game_over = True
            return None
        r = s.falling_piece.lock(s.playfield)
        assert r is not None
        (num_cleared_lines, tspin) = r
        if num_cleared_lines > 0:
            stats.lines += num_cleared_lines
            stats.combos += 1
            stats.max_combos = max(stats.combos, stats.max_combos)
            if num_cleared_lines == 4:
                stats.tetris += 1
            if tspin is TSpinType.NORMAL:
                if num_cleared_lines == 3:
                    stats.tst += 1
                elif num_cleared_lines == 2:
                    stats.tsd += 1
                elif num_cleared_lines == 1:
                    stats.tss += 1
            elif tspin is TSpinType.MINI:
                if num_cleared_lines == 1:
                    stats.tsm += 1
                elif num_cleared_lines == 0:
                    stats.tsz += 1
            if num_cleared_lines == 4 or tspin is not None:
                if s.is_in_btb:
                    stats.btb += 1
                    stats.max_btb = max(stats.btb, stats.max_btb)
                else:
                    s.is_in_btb = True
            if s.playfield.grid.is_empty():
                stats.perfect_clear += 1
        else:
            stats.combos = 0
            stats.btb = 0
            s.is_in_btb = False
        next = s.next_pieces.pop()
        if next is None:
            s.falling_piece = None
            s.is_game_over = True
        else:
            fp = FallingPiece.spawn(next, s.playfield)
            s.falling_piece = fp
            s.is_game_over = not fp.can_put_onto(s.playfield)
        s.can_hold = True
        return r

    def hold(self):
        assert self.state.falling_piece is not None
        s = self.state
        if s.can_hold:
            piece_to_be_held = s.falling_piece.piece
            s.falling_piece = FallingPiece.spawn(
                s.next_pieces.pop()
                if s.hold_piece is None else s.hold_piece,
                s.playfield)
            s.hold_piece = piece_to_be_held
            s.can_hold = False
            self.stats.hold += 1
