from collections import defaultdict, deque
import sys
from typing import List, Optional
import heapq
import math


class UnionFind():
    def __init__(self, n):
        self.n = n
        self.parents = [-1] * n

    def find(self, x):
        if self.parents[x] < 0:
            return x
        else:
            self.parents[x] = self.find(self.parents[x])
            return self.parents[x]

    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)

        if x == y:
            return

        if self.parents[x] > self.parents[y]:
            x, y = y, x

        self.parents[x] += self.parents[y]
        self.parents[y] = x

    def size(self, x):
        return -self.parents[self.find(x)]

    def members(self, x):
        root = self.find(x)
        return [i for i in range(self.n) if self.find(i) == root]

    def roots(self):
        return [i for i, x in enumerate(self.parents) if x < 0]

    def all_group_members(self):
        d = {root: [] for root in self.roots()}
        for i in range(self.n):
            d[self.find(i)].append(i)
        return d

    def __str__(self):
        return '\n'.join('{}: {}'.format(r, self.members(r)) for r in self.roots())

class Game:
    def __init__(self, N: int, C: List[List[int]]):
        self.N = N
        self.init_C = C
        self.C = [[C[i][j] for j in range(N)] for i in range(N)]
        self.fixed = [[False]*N for _ in range(N)]
        self.spaces = set()
        for i in range(N):
            for j in range(N):
                if C[i][j] == 0:
                    self.spaces.add((i, j))
        self.moves = []
        self.connects = []

    def swap(self, i, j, i_, j_):
        assert min(self.C[i][j], self.C[i_][j_])==0
        assert self.fixed[i][j]+self.fixed[i_][j_]==0
        if self.C[i][j]!=self.C[i_][j_]:
            if self.C[i][j] == 0:
                self.spaces.remove((i, j))
                self.spaces.add((i_, j_))
            else:
                self.spaces.remove((i_, j_))
                self.spaces.add((i, j))
            self.C[i][j], self.C[i_][j_] = self.C[i_][j_], self.C[i][j]
            self.moves.append((i, j, i_, j_))
        # self.print()

    def connect(self, i, j, i_, j_):
        assert (i==i_) or (j==j_)
        assert min(self.C[i][j], self.C[i_][j_])!=0
        if i==i_:
            assert sum([self.C[i][x] for x in range(min(j, j_)+1, max(j, j_))])==0
            assert sum([self.fixed[i][x] for x in range(min(j, j_)+1, max(j, j_))])==0
            for x in range(min(j, j_)+1, max(j, j_)):
                self.spaces.remove((i, x))
                self.fixed[i][x] = True
            self.fixed[i][j] = True
            self.fixed[i_][j_] = True
            self.connects.append((i, j, i_, j_))
        else:
            assert sum([self.C[x][j] for x in range(min(i, i_)+1, max(i, i_))])==0
            assert sum([self.fixed[x][j] for x in range(min(i, i_)+1, max(i, i_))])==0
            for x in range(min(i, i_)+1, max(i, i_)):
                self.spaces.remove((x, j))
                self.fixed[x][j] = True
            self.fixed[i][j] = True
            self.fixed[i_][j_] = True
            self.connects.append((i, j, i_, j_))
        print(self.fixed[3][8], i, j, i_, j_)

    def print(self):
        if len(self.moves)<=200:
            print(len(self.moves))
            for move in self.moves:
                print(*move)
            if len(self.moves)+len(self.connects)<=200:
                print(len(self.connects))
                for connect in self.connects:
                    print(*connect)
            else:
                print(200-len(self.moves))
                for connect in self.connects[:200-len(self.moves)]:
                    print(*connect)
        else:
            print(200)
            for move in self.moves[:200]:
                print(*move)
            print(0)

    def score(self):
        N = self.N
        uf = UnionFind(N*N)
        for i, j, i_, j_ in self.moves:
            uf.union(i*N+j, i_*N+j_)
        ret = 0
        for g in uf.all_group_members().values():
            d = defaultdict(int)
            cnt = 0
            for x in g:
                i, j = x//N, x%N
                if self.C[i][j]!=0:
                    d[self.C[i][j]] += 1
                    cnt += 1
            for v in d.values():
                ret += (v*(v-1)) - (v*(cnt-v))
        return ret//2

class Node:
    # def __init__(self, pos: int, parent: Optional[Node], children: List[Node], game: Game):
    def __init__(self, pos: int, parent: Optional[object], children: List[object], game: Game):
        self.pos = pos
        self.parent = parent
        self.children = children
        self.game = game
        self.n_one_children = None

    @property
    def v(self):
        return self.game.C[self.pos//self.game.N][self.pos%self.game.N]

    def get_n_one_children(self):
        if self.n_one_children is not None:
            return self.n_one_children
        ret = self.v!=0
        for child in self.children:
            ret += child.get_n_one_children()
        self.n_one_children = ret
        return self.n_one_children

    def swap_with_parent(self):
        assert self.parent is not None
        assert min(self.v, self.parent.v)==0
        assert max(self.v, self.parent.v)!=0
        N = self.game.N
        self.game.swap(self.pos//N, self.pos%N, self.parent.pos//N, self.parent.pos%N)
        self.get_n_one_children()
        self.n_one_children += 1-2*(self.v==0)

    def satisfy(self):
        if self.v==0:
            for child in self.children:
                if (self.parent is not None) and (self.pos-self.parent.pos==child.pos-self.pos):
                    continue
                if child.get_n_one_children()!=0:
                    q = [child]
                    cur = child
                    while cur.v==0:
                        for node in cur.children[::-1]:
                            if node.get_n_one_children()!=0:
                                cur = node
                                q.append(cur)
                                break
                    for node in q[::-1]:
                        node.swap_with_parent()
                    break
        for child in self.children:
            if child.get_n_one_children()!=0:
                child.satisfy()

    def connect(self):
        for child in self.children:
            child.connect()
        if self.parent is not None and self.v!=0:
            self.connect_to_parent()

    def connect_to_parent(self):
        if self.v==0:
            return
        p = self.parent
        while p is not None:
            if p.v!=0:
                assert p.v!=0 and self.v!=0
                self.game.connect(self.pos//N, self.pos%N, p.pos//N, p.pos%N)
                return
            p = p.parent
        raise ValueError("all parent is 0")

    def treemap(self, C: List[List[int]], v):
        C[self.pos//self.game.N][self.pos%self.game.N] = v
        for child in self.children:
            child.treemap(C, v+1)
        if v==1:
            for x in C:
                print(*x, sep="")


def solve_(game: Game, target: int):
    # targetを連結に
    # 後で実装
    N = game.N
    fixed = [[game.fixed[i][j] or game.C[i][j]==target for j in range(N)] for i in range(N)]
    spaces = set(game.spaces)
    uf_reachble = UnionFind(N*N)
    for i in range(N):
        for j in range(N):
            for i_, j_ in [(i+1, j), (i, j+1), (i-1, j), (i, j-1)]:
                if not (0<=i_<N and 0<=j_<N):
                    continue
                if fixed[i_][j_]:
                    continue
                uf_reachble.union(i*N+j, i_*N+j_)
    # treeの構成
    fixed = [[game.fixed[i][j] or (game.C[i][j]!=target and game.C[i][j]!=0) for j in range(N)] for i in range(N)]
    roots = []
    appeared = [[False]*N for _ in range(N)]
    g = [[] for _ in range(N*N*4)]
    for i in range(N):
        for j in range(N):
            if appeared[i][j] or game.C[i][j]!=target:
                continue
            # まず既にsatisfyなところから木を作る
            q = deque([(i, j, None)])
            while q:
                i_, j_, p = q.popleft()
                if appeared[i_][j_]:
                    continue
                appeared[i_][j_] = True
                # 入った時の処理
                node = Node(N*i_+j_, p, [], game)
                if p is None:
                    roots.append(node)
                else:
                    node.parent.children.append(node)
                # 探索先を追加
                for i__, j__ in [(i_+1, j_), (i_, j_+1), (i_-1, j_), (i_, j_-1)]:
                    if not (0<=i__<N and 0<=j__<N):
                        continue
                    if fixed[i__][j__]:
                        continue
                    if p is not None and (p.pos//N)-i_==i_-i__ and (p.pos%N)-j_==j_-j__:
                        if not appeared[i__][j__]:
                            q.append((i__, j__, node))
                for i__, j__ in [(i_+1, j_), (i_, j_+1), (i_-1, j_), (i_, j_-1)]:
                    if not (0<=i__<N and 0<=j__<N):
                        continue
                    if fixed[i__][j__]:
                        continue
                    if not appeared[i__][j__]:
                        q.append((i__, j__, node))
    for root in roots:
        # targetを配置なおす
        root.satisfy()
        # 接続する
        root.connect()
        # root.treemap([[0]*N for _ in range(N)], 1)


def solve(N: int, K: int, C: List[int]):
    game = Game(N, C)
    for target in range(1, K+1):
        solve_(game, target)
    game2 = Game(N, C)
    for i in range(N):
        for j in range(N-1):
            if C[i][j]!=0 and C[i][j]==C[i][j+1]:
                game2.connect(i, j, i, j+1)
    for i in range(N-1):
        for j in range(N):
            if C[i][j]!=0 and C[i][j]==C[i+1][j]:
                game2.connect(i, j, i+1, j)
    if game.score()>game2.score():
        game.print()
        game2.print()
    else:
        game2.print()
        game.print()



input = sys.stdin.readline
N, K = map(int, input().split())
C = [list(map(int, input().rstrip())) for _ in range(N)]
solve(N, K, C)
