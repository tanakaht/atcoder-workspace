import math
import random
from time import time

TS = time()
TL = 1.8
DIR = ["F", "R", "B", "L"]

F = list(map(int, input().split()))


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


# [a, b)
def randint(a, b):
    return int(random.random()*(b-a))+a


# 連結前提
class State:
    def __init__(self, P=None, cnt=0):
        self.P = [p for p in P] if P is not None else [None]*100
        self.cnt = cnt

    @classmethod
    def get_initial_state(cls, **kwargs):
        return State(None)

    def add(self, i, f):
        cnt = 0
        for i_ in range(100):
            cnt += self.P[i_] is None
            if cnt==i:
                self.P[i_] = f
                self.cnt += 1
                return
        raise ValueError

    def move(self, dir):
        if dir=="F":
            P = [None]*100
            for j in range(10):
                fs_j = [p for p in self.P[j:100:10] if p is not None]
                for i, f in enumerate(fs_j):
                    P[10*i+j] = f
            self.P = P
        elif dir=="R":
            P = [None]*100
            for i in range(10):
                fs_i = [p for p in self.P[10*i:10*i+10] if p is not None][::-1]
                for j, f in enumerate(fs_i):
                    P[10*i+9-j] = f
            self.P = P
        elif dir=="B":
            P = [None]*100
            for j in range(10):
                fs_j = [p for p in self.P[j:100:10] if p is not None][::-1]
                for i, f in enumerate(fs_j):
                    P[90-10*i+j] = f
            self.P = P
        elif dir=="L":
            P = [None]*100
            for i in range(10):
                fs_i = [p for p in self.P[10*i:10*i+10] if p is not None]
                for j, f in enumerate(fs_i):
                    P[10*i+j] = f
            self.P = P

    def get_neighbors(self, mode="dir", n_neighbor=10):
        if mode=="dir":
            ret = []
            for dir in DIR:
                state = self.copy()
                state.move(dir)
                ret.append((dir, state))
            return ret
        elif mode=="add":
            ret = []
            for i in random.sample(range(1, 101-self.cnt), min(n_neighbor, 100-self.cnt)):
                state = self.copy()
                state.add(i, F[self.cnt])
                ret.append((i, state))
            return ret

    def get_score(self, **params):
        return self.get_raw_score()

    # TODO: bfsでよかったわ
    def get_raw_score(self, **params):
        # 生スコア(n*nの和だけ)
        score = 0
        appeared = [False]*100
        for start_node in range(100):
            if appeared[start_node] or self.P[start_node] is None:
                continue
            cnt = 0
            q = [start_node]
            while q:
                i = q.pop()
                cnt += 1
                for i_ in [i+1, i-1, i+10, i-10]:
                    if not 0<=i_<100:
                        continue
                    if abs((i%10)-(i_%10))==9:
                        continue
                    if (not appeared[i_]) and self.P[i]==self.P[i_]:
                        appeared[i_] = True
                        q.append(i_)
            score += cnt**2
        return score

    def get_raw_connect_score(self, **params):
        # 生スコア(n*nの和だけ)+閉塞したクラスタの大きさ
        uf = UnionFind(100)
        for i in range(10):
            for j in range(9):
                if self.P[10*i+j] is not None and self.P[10*i+j]==self.P[10*i+j+1]:
                    uf.union(10*i+j, 10*i+j+1)
        for i in range(9):
            for j in range(10):
                if self.P[10*i+j] is not None and self.P[10*i+j]==self.P[10*(i+1)+j]:
                    uf.union(10*i+j, 10*(i+1)+j)
        uf2 = UnionFind(101)
        for i in range(100):
            if self.P[i] is None:
                uf2.union(100, i)
        for i in range(10):
            for j in range(9):
                if self.P[10*i+j] is None or self.P[10*i+j+1] or self.P[10*i+j]==self.P[10*i+j+1]:
                    uf2.union(10*i+j, 10*i+j+1)
        for i in range(9):
            for j in range(10):
                if self.P[10*i+j] is None or self.P[10*(i+1)+j] or self.P[10*i+j]==self.P[10*(i+1)+j]:
                    uf2.union(10*i+j, 10*(i+1)+j)
        score = 0
        for i_root in uf.roots():
            if self.P[i] is not None:
                if uf2.find(i_root) == uf2.find(100):
                    score += uf.size(i_root)**2
                else:
                    score -= uf.size(i_root)**2*0.5
        return score


    def print(self):
        raise NotImplementedError

    def copy(self):
        return State(P=self.P, cnt=self.cnt)


class StateNode:
    def __init__(self, state, is_moveturn):
        self.state = state
        self.is_moveturn = is_moveturn
        self.children = []
        self.score_ = None

    def add_child(self, node):
        self.children.append(node)

    def score(self):
        if self.score_ is not None:
            return self.score_
        if not self.children:
            score = self.state.get_score()
        elif self.is_moveturn:
            score = max([child.score() for child in self.children])
        else:
            score = sum([child.score() for child in self.children])/len(self.children)
        self.score_ = score
        return self.score_


class StateTree:
    def __init__(self, root_state, vervose=False, TL_=0.018, n_neighbor=10, depth=2):
        depth = min(depth, 100-root_state.cnt)
        self.root_node = StateNode(root_state, is_moveturn=True)
        self.n_neighbor = 10
        # tree構築
        self.child_nodes = [(dir, StateNode(state, is_moveturn=False)) for dir, state in root_state.get_neighbors(mode="dir")]
        [self.root_node.add_child(node) for dir, node in self.child_nodes]
        q = [node for dir, node in self.child_nodes]
        for _ in range(depth):
            q_ = []
            for node in q:
                state = node.state
                for i, state_ in state.get_neighbors(mode="add", n_neighbor=n_neighbor):
                    node_ = StateNode(state_, is_moveturn=True)
                    node.add_child(node_)
                    for dir, state__ in state_.get_neighbors(mode="dir"):
                        node__ = StateNode(state__, is_moveturn=False)
                        node_.add_child(node__)
                        q_.append(node__)
            q = q_

    def get_best_move(self):
        best_score, best_dir = -math.inf, None
        for dir, node in self.child_nodes:
            score = node.score()
            if best_score < score:
                best_score, best_dir = score, dir
        return best_dir

    def score(self):
        return self.root_node.score()


state = State()
for turn in range(100):
    i = int(input())
    state.add(i, F[turn])
    tree = StateTree(state, n_neighbor=3)
    dir = tree.get_best_move()
    state.move(dir)
    print(dir)

# init_state = State.get_initial_state()
# opt = Optimizer()
# best_state = opt.climbing(init_state)
# best_state = opt.simanneal(init_state)
# best_state.print()
# opt.visualize()
