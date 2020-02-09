import weakref
from typing import NamedTuple, List, Dict, TypeVar, Generic, Callable, Optional


class NodeParams:
    n: int = 0
    q: float = 0
    w: float = 0
    p: float = 0


PuctConstantFunc = Callable[[NodeParams], float]


class TreeConfig(NamedTuple):
    puct_c_fn: PuctConstantFunc


IdT = TypeVar('IdT')
ValueT = TypeVar('ValueT')


class Tree(Generic[IdT, ValueT]):
    config: TreeConfig
    nodes: Dict[IdT, weakref.ProxyType]
    root: 'Node[IdT, ValueT]'

    def __init__(self, config: TreeConfig, id: IdT, value: ValueT):
        self.config = config
        self.nodes = {}
        self.root = Node(weakref.proxy(self), None, id, value)

    def append_node(self, node: 'Node[IdT, ValueT]'):
        """For Node class."""
        self.nodes[node.id] = weakref.proxy(node)

    def calc_puct_c(self, params: NodeParams) -> float:
        """For Node class."""
        return self.config.puct_c_fn(params)


class Node(Generic[IdT, ValueT]):
    tree: weakref.ProxyType
    parent: Optional[weakref.ProxyType]
    children: List['Node']
    id: IdT
    value: ValueT
    params: NodeParams

    def __init__(self, tree: weakref.ProxyType, parent: Optional['Node'],
                 id: IdT, value: ValueT):
        self.tree = tree
        self.parent = parent
        self.children = []
        self.id = id
        self.value = value
        self.params = NodeParams()
        tree.append_node(self)

    def is_root(self) -> bool:
        return self.parent is None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def select(self) -> 'Node[IdT, ValueT]':
        """Select best leaf node."""
        node = self
        while not node.is_leaf():
            c = self.tree.calc_puct_c(node.params)
            max_idx = 0
            max_puct = 0
            for idx, child in enumerate(node.children):
                puct = child.params.q + \
                       c * child.params.p * node.params.n ** 0.5 / \
                       (1 + child.params.n)
                if max_puct < puct:
                    max_puct = puct
                    max_idx = idx
            node = node.children[max_idx]
        return node

    def append_child(self, id: IdT, value: ValueT):
        self.children.append(Node(self.tree, self, id, value))

    def backpropagate(self, v: float):
        p = self.params
        p.n += 1
        p.w += v
        p.q = p.w / p.n
        if self.parent is not None:
            self.parent.backpropagate(v)
