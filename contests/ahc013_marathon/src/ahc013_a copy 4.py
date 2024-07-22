from collections import defaultdict, deque
from re import X
from select import select
from shutil import move
from sqlite3 import connect
import sys
from time import time
from typing import List, Optional
import heapq
import math
import random

DIR = [(-1, 0), (0, 1), (1, 0), (0, -1)]
TS = time()
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
    def __init__(self, N: int, C: List[List[int]], K: int):
        self.N = N
        self.K = K
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

    def copy(self):
        ret = Game(self.N, self.init_C, self.K)
        for i, j, i_, j_ in self.moves:
            ret.swap(i, j, i_, j_)
        for i, j, i_, j_ in self.connects:
            ret.connect(i, j, i_, j_)
        return ret

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

    def get_ans(self):
        ret = ""
        moves, connects = self.filter_ans()
        ret += f"{len(moves)}\n"
        for move in moves:
            ret += f"{move[0]} {move[1]} {move[2]} {move[3]}\n"
        ret += f"{len(connects)}\n"
        for connect in connects:
            ret += f"{connect[0]} {connect[1]} {connect[2]} {connect[3]}\n"
        return ret

    def print(self):
        print(self.get_ans())

    def score(self):
        moves, connects = self.filter_ans()
        N = self.N
        uf = UnionFind(N*N)
        for i, j, i_, j_ in connects:
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

    def score_(self, target):
        N = self.N
        connects = []
        joints = []
        for i in range(N):
            cur = None
            cur_joint = None
            for j in range(N):
                if self.C[i][j]==target:
                    if cur is not None:
                        connects.append((cur[0], cur[1], i, j))
                    cur = (i, j)
                    if cur_joint is not None:
                        joints.append((cur_joint[0], cur_joint[1], i, j))
                    cur_joint = (i, j)
                elif self.C[i][j]==0:
                    if cur_joint is not None:
                        joints.append((cur_joint[0], cur_joint[1], i, j))
                    cur_joint = (i, j)
                else:
                    cur = None
                    cur_joint = None
        for j in range(N):
            cur = None
            cur_joint = None
            for i in range(N):
                if self.C[i][j]==target:
                    if cur is not None:
                        connects.append((cur[0], cur[1], i, j))
                    cur = (i, j)
                    if cur_joint is not None:
                        joints.append((cur_joint[0], cur_joint[1], i, j))
                    cur_joint = (i, j)
                elif self.C[i][j]==0:
                    if cur_joint is not None:
                        joints.append((cur_joint[0], cur_joint[1], i, j))
                    cur_joint = (i, j)
                else:
                    cur = None
                    cur_joint = None
        uf = UnionFind(N*N)
        for i, j, i_, j_ in connects:
            uf.union(i*N+j, i_*N+j_)
        score = 0
        for g in uf.all_group_members().values():
            d = defaultdict(int)
            cnt = 0
            for x in g:
                i, j = x//N, x%N
                if self.C[i][j]!=0:
                    d[self.C[i][j]] += 1
                    cnt += 1
            for v in d.values():
                score += (v*(v-1)) - (v*(cnt-v))
        score = score//2
        uf_joint = UnionFind(N*N)
        for i, j, i_, j_ in joints:
            uf_joint.union(i*N+j, i_*N+j_)
        score_joint = 0
        for g in uf_joint.all_group_members().values():
            d = defaultdict(int)
            cnt = 0
            for x in g:
                i, j = x//N, x%N
                if self.C[i][j]!=0:
                    d[self.C[i][j]] += 1
                    cnt += 1
            for v in d.values():
                score_joint += (v*(v-1)) - (v*(cnt-v))
        score_joint = score_joint//2
        ret = score*100+score_joint*100000
        return ret

    def filter_ans(self):
        if len(self.moves)+len(self.connects)<=self.K*100:
            return self.moves, self.connects
        N = self.N
        last_touch = [[-1]*N for _ in range(N)]
        for idx, (i, j, i_, j_) in enumerate(self.moves):
            last_touch[i][j] = idx
            last_touch[i_][j_] = idx
        dependance = [-1]*len(self.connects)
        for idx, (i, j, i_, j_) in enumerate(self.connects):
            if i==i_:
                for x in range(min(j, j_), max(j, j_)+1):
                    dependance[idx] = max(dependance[idx], last_touch[i][x])
            else:
                for x in range(min(i, i_), max(i, i_)+1):
                    dependance[idx] = max(dependance[idx], last_touch[x][j])
        actions = []
        cidx = 0
        for idx, (i, j, i_, j_) in enumerate(self.moves):
            while cidx<len(self.connects) and dependance[cidx]<idx:
                actions.append((1, self.connects[cidx]))
                cidx += 1
            actions.append((0, self.moves[idx]))
        ret = ([], [])
        for idx, arg in actions[:self.K*100]:
            ret[idx].append(arg)
        return ret


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

    def get_root(self):
        if self.parent is None:
            return self
        return self.parent.get_root()

    def connect(self):
        if self.parent is not None and self.v!=0:
            self.connect_to_parent()
        for child in self.children:
            child.connect()

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

    def filter_0(self):
        for child in list(self.children):
            if child.get_n_one_children()==0:
                self.children.remove(child)
            child.filter_0()

    def change_parent(self, p):
        assert (p==self.parent) or (p is None) or (p in self.children)
        if p==self.parent:
            return
        elif p is None:
            self.parent.change_parent(self)
            self.children.append(self.parent)
            self.parent = p
        elif p in self.children:
            if self.parent is None:
                self.parent = p
                self.children.remove(p)
            else:
                self.parent.change_parent(self)
                self.children.append(self.parent)
                self.parent = p
                self.children.remove(p)
        else:
            raise ValueError

    def get_all_children(self):
        ret = [self]
        for child in self.children:
            ret += child.get_all_children()
        return ret

def solve_tree(game: Game, target: int, root_N=0):
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
    # グラフ構築
    g = [[] for _ in range(N*N*4)]
    for i in range(N):
        for j in range(N):
            if fixed[i][j]:
                continue
            if game.C[i][j]==0:
                for dir in range(4):
                    for dir_ in range(4):
                        if dir==dir_:
                            continue
                        if (dir+2)%4==dir_:
                            g[i*N*4+j*4+dir].append((i*N*4+j*4+dir_, 0.001))
                        else:
                            g[i*N*4+j*4+dir].append((i*N*4+j*4+dir_, 100))
            else:
                for dir in range(4):
                    for dir_ in range(4):
                        if dir==dir_:
                            continue
                        if (dir+2)%4==dir_:
                            g[i*N*4+j*4+dir].append((i*N*4+j*4+dir_, 0))
                        else:
                            g[i*N*4+j*4+dir].append((i*N*4+j*4+dir_, 1))
            for dir in range(4):
                i_, j_ = i+DIR[dir][0], j+DIR[dir][1]
                if not (0<=i_<N and 0<=j_<N):
                    continue
                if fixed[i_][j_]:
                    continue
                g[i*N*4+j*4+dir].append((i_*N*4+j_*4+((dir+2)%4), 0))
    roots = []
    appeared = [[False]*N for _ in range(N)]
    appeared_dir = [False]*(N*N*4)
    dist_dir = [math.inf]*(N*N*4)
    from_dir = [None]*(N*N*4)
    cnt = 0
    for i in range(N):
        for j in range(N):
            if appeared[i][j] or game.C[i][j]!=target:
                continue
            cnt += 1
            if cnt!=root_N:
                continue
            cnt = 0
            # dijkstraして、きた方向を記録
            q = [(0, i*N*4+j*4+dir, -1) for dir in range(4)]
            nodes = []
            while q:
                d, u, p = heapq.heappop(q)
                if appeared_dir[u]:
                    continue
                if game.C[u//(N*4)][(u%(N*4))//4]==target:
                    appeared[u//(N*4)][(u%(N*4))//4] = True
                appeared_dir[u] = True
                dist_dir[u] = d
                from_dir[u] = p
                nodes.append(u)
                for v, c in g[u]:
                    d_ = d+c
                    if dist_dir[v] > d_:
                        dist_dir[v] = d_
                        heapq.heappush(q, (d_, v, u))
            node_dict = {None: None}
            for u in nodes:
                if u//4 in node_dict:
                    continue
                if from_dir[u]!=-1:
                    p = node_dict[from_dir[u]//4]
                else:
                    p = None
                node = Node(u//4, p, [], game)
                node_dict[u//4] = node
                if p is not None:
                    p.children.append(node)
            roots.append(node_dict[i*N+j])
    for root in sorted(roots, key=lambda x: -x.get_n_one_children()):
        # targetを配置なおす
        root.satisfy()
        # 接続する
        root.connect()
        # root.treemap([[0]*N for _ in range(N)], 1)


def solve_ad(game: Game):
    N = game.N
    connects = []
    for target in range(1, game.K+1):
        for i in range(N):
            cur = None
            for j in range(N):
                if game.C[i][j]==target:
                    if cur is not None:
                        connects.append((cur[0], cur[1], i, j))
                    cur = (i, j)
                else:
                    cur = None
        for j in range(N):
            cur = None
            for i in range(N):
                if game.C[i][j]==target:
                    if cur is not None:
                        connects.append((cur[0], cur[1], i, j))
                    cur = (i, j)
                else:
                    cur = None
    for i, j, i_, j_ in connects:
        game.connect(i, j, i_, j_)


def solve_tree_sa(game: Game, target: int, root_N=0):
    N = game.N
    targets = set()
    for i in range(N):
        for j in range(N):
            if game.C[i][j]==target:
                targets.add((i, j))
    # 100*N^2*
    connects = []
    joints = []
    for i in range(N):
        cur = None
        cur_joint = None
        for j in range(N):
            if self.C[i][j]==target:
                if cur is not None:
                    connects.append((cur[0], cur[1], i, j))
                cur = (i, j)
                if cur_joint is not None:
                    joints.append((cur_joint[0], cur_joint[1], i, j))
                cur_joint = (i, j)
            elif self.C[i][j]==0:
                if cur_joint is not None:
                    joints.append((cur_joint[0], cur_joint[1], i, j))
                cur_joint = (i, j)
            else:
                cur = None
                cur_joint = None

    return

def make_union(N, C, K, target):
    games = []
    game = Game(N, C, K)
    # 現状連結なものをfixedにする
    appeared = [[False]*N for _ in range(N)]
    roots = []
    for i in range(N):
        for j in range(N):
            if appeared[i][j] or game.C[i][j]!=target:
                continue
            root = Node(i*N+j, None, [], game)
            q = deque([root])
            appeared[i][j] = True
            roots.append(root)
            while q:
                u = q.popleft()
                i_, j_ = u.pos//N, u.pos%N
                for dir in range(4):
                    i__, j__ = i_+DIR[dir][0], j_+DIR[dir][1]
                    if not (0<=i__<N and 0<=j__<N):
                        continue
                    if (not appeared[i__][j__]) and (game.C[i__][j__]==target or game.C[i__][j__]==0):
                        v = Node(i__*N+j__, u, [], game)
                        u.children.append(v)
                        q.append(v)
                        appeared[i__][j__] = True
    fixed = [[False]*N for _ in range(N)]
    for root in roots:
        root.filter_0()
        for u in root.get_all_children():
            fixed[u.pos//N][u.pos%N] = True
    roots = sorted(roots, key=lambda x: -x.get_n_one_children())
    # root同士を繋げる(貪欲)
    games = [game]
    while len(roots)>1:
        if time()-TS<2.5:
            break
        game = games[-1].copy()
        games.append(game)
        root = roots[0]
        root_nodes = set([(node.pos//N, node.pos%N) for node in root.get_all_children()])
        # 各マスの近くのspaceを事前に求めておく
        nearest_space = [[[] for _ in range(N)] for _ in range(N)]
        for i, j in game.spaces:
            if fixed[i][j]:
                continue
            q = deque([(0, i, j)])
            appeared = set([(i, j)])
            while q:
                d, i_, j_ = q.popleft()
                nearest_space[i_][j_].append((d, i, j))
                for dir in range(4):
                    i__, j__ = i_+DIR[dir][0], j_+DIR[dir][1]
                    if not (0<=i__<N and 0<=j__<N):
                        continue
                    if (not fixed[i__][j__]) and ((i__, j__) not in appeared) and d<10:
                        q.append((d+1, i__, j__))
                        appeared.add((i__, j__))
        for i in range(N):
            for j in range(N):
                nearest_space[i][j] = sorted(nearest_space[i][j])
        # rootからspaceを繋げていくdijkstra的な
        q = [(0, i, j) for i, j in root_nodes]
        dists = [[(math.inf, set()) for _ in range(N)] for _ in range(N)]
        appeared = [[False]*N for _ in range(N)]
        for i, j in root_nodes:
            dists[i][j] = (0, set())
        found = False
        p, connect_point = None, None
        if time()-TS<2.0:
            break
        while q and not found:
            d, i_, j_ = heapq.heappop(q)
            if appeared[i_][j_]:
                continue
            appeared[i_][j_] = True
            for dir in range(4):
                i__, j__ = i_+DIR[dir][0], j_+DIR[dir][1]
                if not (0<=i__<N and 0<=j__<N):
                    continue
                if appeared[i__][j__] or (i__, j__) in root_nodes:
                    continue
                if fixed[i__][j__]:
                    found = True
                    connect_point = (i__, j__)
                    p = (i_, j_)
                    break
                else:
                    used = dists[i_][j_][1]
                    for d_space, i_space, j_space in nearest_space[i__][j__]:
                        if (i_space, j_space) in used:
                            continue
                        if d+d_space<dists[i__][j__][0]:
                            used_ = set(used)
                            used_.add((i_space, j_space))
                            dists[i__][j__] = (d+d_space, used_)
                            heapq.heappush(q, (d+d_space, i__, j__))
        # 最大集合から行けるところがなければroot取り出す
        if not found:
            roots.pop(0)
            break
        pre_node = None
        for r in list(roots):
            for node in r.get_all_children():
                if node.pos//N==connect_point[0] and node.pos%N==connect_point[1]:
                    pre_node = node
                    roots.remove(r)
                    break
            if pre_node is not None:
                break
        # 経路復元して連結にさせる
        while p not in root_nodes:
            i, j = p[0], p[1]
            d, used = dists[i][j]
            for dir in range(4):
                i_, j_ = i+DIR[dir][0], j+DIR[dir][1]
                if not (0<=i_<N and 0<=j_<N):
                    continue
                # 来た経路か判定
                d_, used_ = dists[i_][j_]
                if not (len(used)-len(used_)==1 and len(used-used_)==1):
                    continue
                space = list(used-used_)[0]
                for c, i_space, j_space in nearest_space[i][j]:
                    if i_space==space[0] and j_space==space[1]:
                        break
                if d-c==d_:
                    new_p = (i_, j_)
                    space = space
                    break
            # spaceをpまで持ってくる
            q = deque([(0, space[0], space[1], None, None)])
            appeared = set([(space[0], space[1])])
            rireki = defaultdict(list)
            while q:
                d, i_, j_, i_pre, j_pre = q.popleft()
                rireki[i_, j_] = rireki[i_pre, j_pre]+[(i_, j_)]
                if i_==i and j_==j:
                    break
                for dir in range(4):
                    i__, j__ = i_+DIR[dir][0], j_+DIR[dir][1]
                    if not (0<=i__<N and 0<=j__<N):
                        continue
                    if (not fixed[i__][j__]) and ((i__, j__) not in appeared) and d<15 and game.C[i__][j__]!=0:
                        q.append((d+1, i__, j__, i_, j_))
                        appeared.add((i__, j__))
            cur = space
            for i_, j_ in rireki[i, j]:
                game.swap(i_, j_, cur[0], cur[1])
                cur = (i_, j_)
            # pをfixedにする
            fixed[i][j] = True
            # 途中をnodeにして処理
            pre_node.change_parent(None)
            node = Node(i*N+j, None, [pre_node], game)
            pre_node.parent = node
            pre_node = node
            p = new_p
        # pと繋げる
        found = False
        for node_ in root.get_all_children():
            if not (node_.pos//N==p[0] and node_.pos%N==p[1]):
                continue
            node.change_parent(None)
            node_.children.append(node)
            node.parent = node_
            found = True
            break
        if K*100-len(game.moves)<=120:
            break
    if len(games)>4:
        games = games[:-1:len(games)//4]+[games[-1]]
    return games

def move_not_target(game, target):
    not_target = set()
    for i in range(N):
        for j in range(N):
            if game.C[i][j] != target and game.C[i][j] != 0:
                not_target.add((i, j))
    cur_score = game.score_(target)
    cnt = 0
    for i, j in list(not_target):
        for diff in DIR:
            i_, j_ = i+diff[0], j+diff[1]
            if not (0<=i_<N and 0<=j_<N):
                continue
            if not (i_, j_) in game.spaces:
                continue
            game.swap(i, j, i_, j_)
            new_score=game.score_(target)
            if cur_score<new_score:
                not_target.remove((i, j))
                not_target.add((i_, j_))
                cur_score = new_score
                cnt += 1
                break
            else:
                game.swap(i_, j_, i, j)
                game.moves.pop()
                game.moves.pop()

def select_target(N, K, C):
    # 現状連結なものをfixedにする
    scores = [100]*(K+1)
    for target in range(1, K+1):
        appeared = [[False]*N for _ in range(N)]
        uf = UnionFind(N*N)
        roots = []
        for i in range(N):
            for j in range(N):
                if appeared[i][j] or C[i][j]!=target:
                    continue
                q = deque([(i, j)])
                appeared[i][j] = True
                roots.append((i, j))
                while q:
                    i_, j_ = q.popleft()
                    for dir in range(4):
                        i__, j__ = i_+DIR[dir][0], j_+DIR[dir][1]
                        if not (0<=i__<N and 0<=j__<N):
                            continue
                        if (not appeared[i__][j__]) and (C[i__][j__]==target or C[i__][j__]==0):
                            q.append((i__, j__))
                            uf.union(i*N+j, i__*N+j__)
                            appeared[i__][j__] = True
        scores[target] = len(roots)
    return sorted(enumerate(scores), key=lambda x: x[1])[0][0]

def solve(N: int, K: int, C: List[int]):
    anss = []
    # target選定
    target = select_target(N, K, C)
    # targetを連結にさせる
    games = make_union(N, C, K, target)
    for game in games:
        game_ = game.copy()
        for i in range(K):
            solve_tree(game, (target+i-1)%K+1, root_N=1)
        anss.append((game.score(), game.get_ans()))
        # target以外を動かす
        game = game_
        move_not_target(game, target)
        for i in range(K):
            solve_tree(game, (target+i-1)%K+1, root_N=1)
        anss.append((game.score(), game.get_ans()))
    game = Game(N, C, K)
    solve_ad(game)
    anss.append((game.score(), game.get_ans()))
    # print(time()-TS)
    for score, ans in sorted(anss)[::-1]:
        print(ans)
        break


input = sys.stdin.readline
N, K = map(int, input().split())
C = [list(map(int, input().rstrip())) for _ in range(N)]
solve(N, K, C)
