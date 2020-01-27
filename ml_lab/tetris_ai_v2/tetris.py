from collections import deque
from enum import IntEnum
import numpy as np
import random
import operator
from typing import Optional, Tuple, List

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
        for i in range(h):
            y = i
            if not self.is_row_empty(y):
                return i
        return i

    def top_padding(self) -> int:
        h = self.height()
        for i in range(h):
            y = h - i - 1
            if not self.is_row_empty(y):
                return i
        return i

    def left_padding(self) -> int:
        w = self.width()
        for i in range(w):
            x = i
            if not self.is_col_empty(x):
                return i
        return i

    def right_padding(self) -> int:
        w = self.width()
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


class Playfield:
    grid: Grid
    visible_height: int

    def __init__(self, grid: Grid, visible_height: int):
        self.grid = grid
        self.visible_height = visible_height

    @classmethod
    def default(cls) -> 'Playfield':
        return cls(Grid.by_size((10, 40)), 20)


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

    def __str__(self):
        return '<FallingPiece {} {} ({}, {})>'.format(
            self.piece, self.rotation, self.pos[0], self.pos[1])

    def grid(self):
        return self.piece.grid(self.rotation)

    def droppable(self, pf: Playfield) -> int:
        # TODO: optimize more
        g = self.grid()
        dy = 0
        while pf.grid.can_put((self.pos[0], self.pos[1]-dy), g):
            dy += 1
        return dy - 1

    def drop(self, pf: Playfield, limit=-1) -> int:
        # TODO: optimize more
        dy = self.droppable(pf)
        if dy < 0:
            return dy
        if limit > 0:
            dy = min(dy, limit)
        self.pos = (self.pos[0], self.pos[1] - dy)
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
        if self.droppable(pf) > 0:
            return None
        g = self.grid()
        if not pf.grid.can_put(self.pos, g):
            return None
        tspin = self.check_tspin_type(pf)
        pf.grid.put(self.pos, g)
        n = pf.grid.drop_filled_rows()
        return (n, tspin)

    def check_tspin_type(self, pf: Playfield) -> Optional[TSpinType]:
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

    def rotate(self, pf: Playfield, cw: bool) -> Optional['FallingPiece']:
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

    def has_ceiling(self, pf: Playfield, limit=None) -> bool:
        grid = self.grid()
        if limit is None:
            limit = pf.visible_height - self.pos[1]
        for y in range(self.pos[1], self.pos[1] + limit):
            start = self.pos[0] + grid.left_padding()
            stop = self.pos[0] + grid.width() - grid.right_padding()
            for x in range(start, stop):
                if not pf.grid.get_cell((x, y)).is_empty():
                    return True
        return False

    def escape_from(self, pf: Playfield):
        pass

    def search_droppable(self, pf: Playfield) -> List[Vector2]:
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
        for fp in candidates:
            pass
        return candidates


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
    dropped_lines = 0
    dropped_pieces = 0
    hold = 0

    def to_tuple(self):
        return (self.lines, self.tetris, self.combos, self.max_combos,
                self.tst, self.tsd, self.tss, self.tsm, self.tsz, self.btb,
                self.max_btb, self.dropped_lines, self.dropped_pieces,
                self.hold)

    @classmethod
    def from_tuple(cls, t):
        s = cls()
        (s.lines, s.tetris, s.combos, s.max_combos, s.tst, s.tsd,
         s.tss, s.tsm, s.tsz, s.btb, s.max_btb, s.dropped_lines,
         s.dropped_pieces, s.hold) = t
        return s

    def __sub__(self, rhs: 'Statistics'):
        return Statistics.from_tuple(
            map(operator.sub, self.to_tuple(), rhs.to_tuple()))


class Game:
    @classmethod
    def default(cls):
        playfield = Playfield.default()
        next_pieces = NextPieces()
        falling_piece = FallingPiece.spawn(next_pieces.pop(), playfield)
        assert falling_piece is not None
        return cls(playfield, next_pieces, falling_piece, None)

    def __init__(self, playfield: Playfield, next_pieces: NextPieces,
                 falling_piece: FallingPiece, hold_piece: Optional[Piece]):
        self.playfield = playfield
        self.next_pieces = next_pieces
        self.falling_piece = falling_piece
        self.hold_piece = hold_piece
        self.stats = Statistics()
        self.is_game_over = False
        self.can_hold = False
        self.is_in_btb = False

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
        for i in range(self.playfield.visible_height):
            y = 19 - i
            row = []
            for x in range(10):
                cell = fp_grid.get_cell((x - fp.pos[0], y - fp.pos[1]))
                if cell is not None and not cell.is_empty():
                    row.append(str(cell))
                    continue
                row.append(str(self.playfield.grid.get_cell((x, y))))
            stats_line = stats_lines[i+1] if i + 1 < len(stats_lines) else ''
            lines.append('{:02}|{}|  {}'.format(y, ''.join(row), stats_line))
        lines.append('--+----------+')
        lines.append('##|0123456789|')
        return '\n'.join(lines)

    def __format__(self, format_spec):
        return str(self)

    def rotate(self, is_cw):
        self.falling_piece.rotate(self.playfield, is_cw)

    def shift(self, n):
        self.falling_piece.shift(self.playfield, n)

    def drop(self, n=-1):
        n = self.falling_piece.drop(self.playfield, n)
        if n > 0:
            self.stats.dropped_lines += n

    def hard_drop(self):
        fp = self.falling_piece
        playfield = self.playfield
        stats = self.stats
        n = fp.drop(playfield)
        if n > 0:
            stats.dropped_lines += n
        r = fp.lock(playfield)
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
                if self.is_in_btb:
                    stats.btb += 1
                    stats.max_btb = max(stats.btb, stats.max_btb)
                else:
                    self.is_in_btb = True
        else:
            stats.combos = 0
            stats.btb = 0
            self.is_in_btb = False
        fp = FallingPiece.spawn(self.next_pieces.pop(), playfield)
        self.falling_piece = fp
        self.is_game_over = not playfield.can_put(fp.pos, fp.grid())
        self.can_hold = True
        return r  # TODO

    def hold(self):
        if self.can_hold:
            piece_to_be_held = self.falling_piece.piece
            self.falling_piece = FallingPiece.spawn(
                self.next_pieces.pop()
                if self.hold_piece is None else self.hold_piece,
                self.playfield)
            self.hold_piece = piece_to_be_held
            self.can_hold = False
