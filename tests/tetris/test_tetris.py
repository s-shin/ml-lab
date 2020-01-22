import unittest
import ml_lab.tetris as tetris


class TestGrid(unittest.TestCase):
    def setUp(self):
        self.g1 = tetris.Grid.by_size((6, 5))
        self.g2 = tetris.Grid.by_cells([
            [0, 1, 1, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
        ], reverse_rows=True)

    def test_size(self):
        g1, g2 = self.g1, self.g2
        self.assertEqual(6, g1.width())
        self.assertEqual(5, g1.height())
        self.assertEqual(4, g2.width())
        self.assertEqual(3, g2.height())

    def test_get_cell(self):
        g2 = self.g2
        self.assertTrue(g2.can_get_cell((0, 0)))
        self.assertFalse(g2.can_get_cell((-1, 0)))
        self.assertFalse(g2.can_get_cell((0, -1)))
        self.assertTrue(g2.can_get_cell((3, 2)))
        self.assertFalse(g2.can_get_cell((4, 2)))
        self.assertFalse(g2.can_get_cell((3, 3)))
        self.assertEqual(0, g2.get_cell((0, 0)))
        self.assertEqual(0, g2.get_cell((1, 0)))
        self.assertEqual(0, g2.get_cell((2, 0)))
        self.assertEqual(0, g2.get_cell((3, 0)))
        self.assertEqual(None, g2.get_cell((4, 0)))
        self.assertEqual(0, g2.get_cell((0, 2)))
        self.assertEqual(1, g2.get_cell((1, 2)))
        self.assertEqual(1, g2.get_cell((2, 2)))
        self.assertEqual(0, g2.get_cell((3, 2)))

    def test_put(self):
        g1, g2 = self.g1, self.g2
        self.assertTrue(g1.can_put((0, 0), g2))
        self.assertFalse(g1.can_put((-1, 0), g2))
        self.assertTrue(g1.can_put((0, -1), g2))
        self.assertFalse(g1.can_put((0, -2), g2))

        g1.put((0, -1), g2)
        self.assertEqual(1, g1.get_cell((0, 0)))
        self.assertEqual(1, g1.get_cell((1, 0)))
        self.assertEqual(1, g1.get_cell((1, 1)))
        self.assertEqual(1, g1.get_cell((2, 1)))

    def test_drop_filled_rows(self):
        g1 = self.g1
        for y in range(g1.height()):
            g1.set_cell((0, y), y+1)
        for x in range(g1.width()):
            g1.set_cell((x, 1), 1)
            g1.set_cell((x, 3), 1)
        self.assertEqual(2, g1.drop_filled_rows())
        self.assertEqual(tetris.Grid.by_cells([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [5, 0, 0, 0, 0, 0],
            [3, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
        ], reverse_rows=True), g1)


class TestFallingPiece(unittest.TestCase):
    def setUp(self):
        self.pf = tetris.Grid.by_size((10, 40))

    def test_basic(self):
        pf = self.pf
        fp = tetris.FallingPiece.spawn(tetris.Piece.T, pf)
        self.assertEqual(18, fp.pos[1])
        self.assertEqual(19, fp.droppable(pf))
        self.assertTrue(fp.drop(pf))
        self.assertEqual(-1, fp.pos[1])
        fp.shift(pf, -10)
        self.assertEqual(0, fp.pos[0])
        fp.shift(pf, 1)
        self.assertEqual(1, fp.pos[0])
        fp.shift(pf, 10)
        self.assertEqual(7, fp.pos[0])
        fp.shift(pf, -10)
        self.assertEqual(0, fp.pos[0])
        fp.lock(pf)
        self.assertFalse(pf.get_cell((0, 0)).is_empty())
        self.assertFalse(pf.get_cell((1, 0)).is_empty())
        self.assertFalse(pf.get_cell((1, 1)).is_empty())
        self.assertFalse(pf.get_cell((2, 0)).is_empty())


class TestEnvironment(unittest.TestCase):
    def test_basic(self):
        env = tetris.Environment()
        env.reset()
        for _ in range(100):
            state, result, done = env.step(tetris.Action.random())
        # print(state)
        # print(result)
        # print(done)


if __name__ == '__main__':
    unittest.main()
