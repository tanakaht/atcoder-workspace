from time import time
TS = time()
import sys
import math
from typing import List, Optional, Tuple, Union, Set, Dict
from collections import defaultdict, deque
import heapq
from random import random
#from line_profiler import LineProfiler
# from __future__ import annotations
dir_inv = {
    "U": "D",
    "D": "U",
    "R": "L",
    "L": "R",
}
DIR = {
    "U": (-1, 0),
    "D": (1, 0),
    "R": (0, 1),
    "L": (0, -1),
}
# 左方向を1、上方向を2、右方向を4、下方向を8としたビットマスクを用いて以下のように表現される。
DIR2 = [[0, -1], [-1, 0], [0, 1], [1, 0]]
DIR2_s = ["L", "U", "R", "D"]

def dist(p1: Tuple[int, int], p2: Tuple[int, int]):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1-x2)+abs(y1-y2)

# 入力取得
N, T = map(int, input().split()) # T=2*N^3
S = [list(map(lambda x: int(x, 16), input())) for _ in range(N)]

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


class Board:
    def __init__(self, S: List[List[int]], move_hist=None, targets=None):
        self.N = len(S)
        # 左方向を1、上方向を2、右方向を4、下方向を8としたビットマスクを用いて以下のように表現される。
        self.S = [[i for i in s] for s in S]
        for i in range(len(S)):
            for j in range(len(S[0])):
                if self.S[i][j]==0:
                    self.p0 = (i, j)
                    break
        N = self.N
        # moveの履歴
        self.move_hist = [] if move_hist is None else [x for x in move_hist]
        self.targets = [[None]*self.N for _ in range(self.N)] if targets is None else targets

    def __str__(self) -> str:
        # 1６以上入れるとバグる可能性あるので","を入れている
        return "\n".join([",".join([hex(x)[2:] for x in s]) for s in self.S])

    def copy(self):
        return Board(self.S, move_hist=self.move_hist)

    def swap(self, p1: Tuple[int, int], p2: Tuple[int, int])->None:
        N = self.N
        x1, y1 = p1
        x2, y2 = p2
        assert 0<=x1<len(self.S) and 0<=x2<len(self.S) and 0<=y1<len(self.S[0]) and 0<=y2<len(self.S[0])
        self.S[x1][y1], self.S[x2][y2] = self.S[x2][y2], self.S[x1][y1]
        self.targets[x1][y1], self.targets[x2][y2] = self.targets[x2][y2], self.targets[x1][y1]
        if p1==self.p0:
            self.p0 = p2
        elif p2==self.p0:
            self.p0 = p1

    def move(self, S: Union[str, List[str]], replace=True):
        if not replace:
            ret = Board(self.S, move_hist=self.move_hist)
            ret.move(S)
            return ret
        for s in S:
            x, y = self.p0
            if s=="R":
                p2 = (x, y+1)
            elif s=="D":
                p2 = (x+1, y)
            elif s=="L":
                p2 = (x, y-1)
            elif s=="U":
                p2 = (x-1, y)
            else:
                raise ValueError
            self.swap(self.p0, p2)
            if self.move_hist and self.move_hist[-1]==dir_inv[s]:
                self.move_hist.pop()
            else:
                self.move_hist.append(s)

    def unmove(self, S: Union[str, List[str]], replace=True):
        return self.move("".join([dir_inv[s] for s in S[::-1]]), replace=replace)

    def move_to(self, target_p: Tuple[int, int], cantuse: Optional[set]=None, ignore_v: Optional[int]=None)->List[str]:
        cantuse = set() if cantuse is None else cantuse
        px, py = target_p
        q = deque([self.p0])
        froms = [[None]*self.N for _ in range(self.N)]
        appeared = [[False]*self.N for _ in range(self.N)]
        appeared[self.p0[0]][self.p0[1]] = True

        q = [(0, self.p0, None)]
        dists = [[math.inf]*N for _ in range(N)]
        dists[self.p0[0]][self.p0[1]] = 0
        appeared = [[False]*N for _ in range(N)]
        while q:
            d, p, fr = heapq.heappop(q)
            x, y = p
            if appeared[p[0]][p[1]]:
                continue
            appeared[p[0]][p[1]] = True
            froms[x][y] = fr
            if p==target_p:
                break
            for dir_s in "UDLR":
                x_ = x+DIR[dir_s][0]
                y_ = y+DIR[dir_s][1]
                if (not (0<=x_<self.N and 0<=y_<self.N)) or ((x_, y_) in cantuse):
                    continue
                p_ = (x_, y_)
                d_ = d+1
                if self.targets[x_][y_] is not None and self.S[x_][y_]!=ignore_v:
                    xt, yt = self.targets[x_][y_]
                    d_pre = abs(xt-x_)+abs(yt-y_)
                    d_new = abs(xt-x)+abs(yt-y)
                    if 0<d_pre<d_new:
                        d_ += 21
                if dists[x_][y_] > d_:
                    dists[x_][y_] = d_
                    heapq.heappush(q, (d_, (x_, y_), dir_s))
        cur_x, cur_y = (px, py)
        ret = []
        while froms[cur_x][cur_y] is not None:
            dir_s = froms[cur_x][cur_y]
            ret.append(dir_s)
            cur_x -= DIR[dir_s][0]
            cur_y -= DIR[dir_s][1]
        if (cur_x, cur_y)!=self.p0:
            raise ValueError("cant touch")
        ret = ret[::-1]
        for dir_s in ret:
            self.move(dir_s)
        return ret

    def move_tile(self, p: Tuple[int, int], dir: str, cantuse: Optional[set]=None, ignore_v: Optional[int]=None)->List[str]:
        """pのタイルをdirに動かす
        """
        ret = []
        cantuse = set() if cantuse is None else cantuse
        target_x, target_y = p
        moveto_x, moveto_y = target_x+1, target_y+1
        cantuse.add((target_x, target_y))
        ret += self.move_to((target_x+DIR[dir][0], target_y+DIR[dir][1]), cantuse=cantuse, ignore_v=ignore_v)
        cantuse.remove((target_x, target_y))
        ret.append(dir_inv[dir])
        self.move(dir_inv[dir])
        return ret

    def find(self, v: int, fr:Optional[Tuple[int, int]]=None, cantuse: Optional[set]=None):
        fr = self.p0 if fr is None else fr
        cantuse = set() if cantuse is None else cantuse
        q = deque([fr])
        appeared = [[False]*self.N for _ in range(self.N)]
        appeared[self.p0[0]][self.p0[1]] = True
        while q:
            x, y = q.popleft()
            if self.S[x][y]==v:
                return (x, y)
            for dir_s in "UDLR":
                x_ = x+DIR[dir_s][0]
                y_ = y+DIR[dir_s][1]
                if (not (0<=x_<self.N and 0<=y_<self.N)) or (x_, y_) in cantuse or appeared[x_][y_]:
                    continue
                q.append((x_, y_))
                appeared[x_][y_] = True
        raise ValueError("not found")

def solve(B: Board, B_ideal: Board)->List[str]:
    """BをB_idealにする移動列を返す
    1. B_idealの穴を(0, 0)にやる
    2. 右、下、右、下と揃えていく
    3. 最後に出来上がっていなければ同一の模様が出るまで拡張、模様を別々に見て、同一の模様を入れ替えたものをidealとして同様に解いていく(再帰するだけ)
    4. 1の逆操作を追加する
    """
    B = B.copy()
    B_ideal = B_ideal.copy()
    # 1
    S1 = "U"*B_ideal.p0[0]+"L"*B_ideal.p0[1] # step1のための移動列
    B_ideal.move(S1)
    ## targetsを確認
    v2ps = defaultdict(set)
    for x in range(N-1, -1, -1):
        for y in range(N-1, -1, -1):
            v = B_ideal.S[x][y]
            v2ps[v].add((x, y))
    targets = [[None]*N for _ in range(N)]
    for x in range(N-1, -1, -1):
        for y in range(N-1, -1, -1):
            v = B.S[x][y]
            if v==0:
                continue
            min_dist, p = math.inf, None
            for x_, y_ in v2ps[v]:
                d_ = abs(x-x_)+abs(y-y_)
                if min_dist>d_:
                    min_dist = d_
                    p = (x_, y_)
            targets[x][y] = p
            v2ps[v].remove(p)
    B.targets = targets

    # 2
    for i in range(N-1, 2, -1):
        if i%2==0:
            solve_R(B, B_ideal, i, inv=False)
            solve_D(B, B_ideal, i, inv=False)
        else:
            solve_D(B, B_ideal, i, inv=True)
            solve_R(B, B_ideal, i, inv=True)
        v2ps = defaultdict(set)
        for x in range(N-1, -1, -1):
            for y in range(N-1, -1, -1):
                v = B_ideal.S[x][y]
                v2ps[v].add((x, y))
        targets = [[None]*N for _ in range(N)]
        for x in range(N-1, -1, -1):
            for y in range(N-1, -1, -1):
                v = B.S[x][y]
                if v==0:
                    continue
                min_dist, p = math.inf, None
                for x_, y_ in v2ps[v]:
                    d_ = abs(x-x_)+abs(y-y_)
                    if min_dist>d_:
                        min_dist = d_
                        p = (x_, y_)
                targets[x][y] = p
                v2ps[v].remove(p)
        B.targets = targets

    solve_R(B, B_ideal, 2, inv=False)
    # 3
    try:
        moves = solve_6(B, B_ideal)
        B.move(moves)
    except ValueError:
        # 全部1対1にして、同じ模様一箇所swapしてもう一回とく
        v2ps = [[] for _ in range(16)]
        idx = 17
        pair = None
        for x, y in [(x, y) for x in range(N) for y in range(N) if (x>2 or y>=2)]:
            v = B_ideal.S[x][y]
            if len(v2ps[v])!=0 and pair is None:
                pair = ((x, y), v2ps[v][0])
            v2ps[v].append((x, y))
            B_ideal.S[x][y], B.S[x][y] = idx, idx
            B.S[x][y] = idx
            idx += 1
        p1, p2 = pair
        B_ideal.swap(p1, p2)
        B_ideal.unmove(S1)
        B.unmove(B.move_hist)
        ret = solve(B, B_ideal)
        return ret
    # 4
    B.unmove(S1)
    return B.move_hist

def solve_R(B: Board, B_ideal: Board, i: int, inv=False):
    """solveの補助関数(2をやる(i列目))
    Bは破壊的
    """
    ret = []
    y = i
    if not inv:
        xs = list(range(i-1))
        cantuse = set([(x_, y_) for x_ in range(N) for y_ in range(N) if max(x_, y_)>i])
    else:
        xs = list(range(i-1, 1, -1))
        cantuse = set([(x_, y_) for x_ in range(N) for y_ in range(N) if max(x_, y_)>i or x_==i])
    for x in xs:
        target_x, target_y = B.find(B_ideal.S[x][y], fr=(x, y), cantuse=cantuse)
        while target_x<x:
            B.move_tile((target_x, target_y), "D", cantuse=cantuse)
            target_x += 1
        while target_x>x:
            B.move_tile((target_x, target_y), "U", cantuse=cantuse)
            target_x -= 1
        while target_y<y:
            B.move_tile((target_x, target_y), "R", cantuse=cantuse)
            target_y += 1
        cantuse.add((x, y))
    # ラスト2個揃える
    if B.p0[1]==y:
        B.move("L")
    # 退避させる
    if not inv:
        if (B_ideal.S[i][y]!=B_ideal.S[i-1][y]) and B.S[i][y]==B_ideal.S[i-1][y]:
            B.move_tile((i, y), "U", cantuse=cantuse)
        if (B_ideal.S[i][y]!=B_ideal.S[i-1][y]) and B.S[i-1][y]==B_ideal.S[i-1][y]:
            B.move_tile((i-1, y), "L", cantuse=cantuse)
        if (B_ideal.S[i][y]!=B_ideal.S[i-1][y]) and B.S[i][y-1]==B_ideal.S[i-1][y]:
            B.move_tile((i, y-1), "U", cantuse=cantuse)
        if (B_ideal.S[i][y]!=B_ideal.S[i-1][y]) and B.S[i-1][y-1]==B_ideal.S[i-1][y]:
            B.move_tile((i-1, y-1), "U", cantuse=cantuse)
        ignore_v = B_ideal.S[i-1][y]
        vxy = [(B_ideal.S[i][y], i-1, y), (B_ideal.S[i-1][y], i-1, y-1)]
    else:
        if (B_ideal.S[0][y]!=B_ideal.S[1][y]) and B.S[0][y]==B_ideal.S[1][y]:
            B.move_tile((0, y), "D", cantuse=cantuse)
        if (B_ideal.S[0][y]!=B_ideal.S[1][y]) and B.S[1][y]==B_ideal.S[1][y]:
            B.move_tile((1, y), "L", cantuse=cantuse)
        if (B_ideal.S[0][y]!=B_ideal.S[1][y]) and B.S[0][y-1]==B_ideal.S[1][y]:
            B.move_tile((0, y-1), "D", cantuse=cantuse)
        if (B_ideal.S[0][y]!=B_ideal.S[1][y]) and B.S[1][y-1]==B_ideal.S[1][y]:
            B.move_tile((1, y-1), "D", cantuse=cantuse)
        ignore_v = B_ideal.S[1][y]
        vxy = [(B_ideal.S[0][y], 1, y), (B_ideal.S[1][y], 1, y-1)]
    for v, x_, y_ in vxy:
        # 既に揃っていたら抜ける
        if ((not inv) and B.p0==(i, y)) or (inv and B.p0==(0, y)):
            ret.append("L")
            B.move("L")
        if (not inv) and B_ideal.S[i][y]==B.S[i][y] and B_ideal.S[i-1][y]==B.S[i-1][y]:
            break
        if inv and B_ideal.S[0][y]==B.S[0][y] and B_ideal.S[1][y]==B.S[1][y]:
            break
        # わざわざ角から取らない
        if not inv:
            if (x_, y_)==(i-1, y-1):
                cantuse.add((i, y))
                target_x, target_y = B.find(v, fr=(x_, y_), cantuse=cantuse)
                cantuse.remove((i, y))
            else:
                target_x, target_y = B.find(v, fr=(x_, y_), cantuse=cantuse)
        else:
            if (x_, y_)==(1, y-1):
                cantuse.add((0, y))
                target_x, target_y = B.find(v, fr=(x_, y_), cantuse=cantuse)
                cantuse.remove((0, y))
            else:
                target_x, target_y = B.find(v, fr=(x_, y_), cantuse=cantuse)
        while target_x<x_:
            B.move_tile((target_x, target_y), "D", cantuse=cantuse, ignore_v=ignore_v)
            target_x += 1
        while target_x>x_:
            B.move_tile((target_x, target_y), "U", cantuse=cantuse, ignore_v=ignore_v)
            target_x -= 1
        while target_y<y_:
            B.move_tile((target_x, target_y), "R", cantuse=cantuse, ignore_v=ignore_v)
            target_y += 1
        cantuse.add((x_, y_))
    if (not inv) and (not (B_ideal.S[i][y]==B.S[i][y] and B_ideal.S[i-1][y]==B.S[i-1][y])):
        B.move_to((i, i), cantuse=cantuse)
        B.move("U")
        B.move("L")
    if (inv) and (not (B_ideal.S[0][y]==B.S[0][y] and B_ideal.S[1][y]==B.S[1][y])):
        B.move_to((0, i), cantuse=cantuse)
        B.move("D")
        B.move("L")

def solve_D(B: Board, B_ideal: Board, i: int, inv=False):
    """solveの補助関数(2をやる(i行目))
    Bは破壊的
    """
    """solveの補助関数(2をやる(i列目))
    Bは破壊的
    """
    ret = []
    x = i
    if not inv:
        ys = list(range(i-1, 1, -1))
        cantuse = set([(x_, y_) for x_ in range(N) for y_ in range(N) if max(x_, y_)>i or y_==i])
    else:
        ys = list(range(i-1))
        cantuse = set([(x_, y_) for x_ in range(N) for y_ in range(N) if max(x_, y_)>i])
    for y in ys:
        target_x, target_y = B.find(B_ideal.S[x][y], fr=(x, y), cantuse=cantuse)
        while target_y<y:
            B.move_tile((target_x, target_y), "R", cantuse=cantuse)
            target_y += 1
        while target_y>y:
            B.move_tile((target_x, target_y), "L", cantuse=cantuse)
            target_y -= 1
        while target_x<x:
            B.move_tile((target_x, target_y), "D", cantuse=cantuse)
            target_x += 1
        cantuse.add((x, y))
    # ラスト2個揃える
    if B.p0[0]==x:
        ret.append("U")
        B.move("U")
    if not inv:
        if (B_ideal.S[x][0]!=B_ideal.S[x][1]) and B.S[x][0]==B_ideal.S[x][1]:
            B.move_tile((x, 0), "R", cantuse=cantuse)
        if (B_ideal.S[x][0]!=B_ideal.S[x][1]) and B.S[x][1]==B_ideal.S[x][1]:
            B.move_tile((x, 1), "U", cantuse=cantuse)
        if (B_ideal.S[x][0]!=B_ideal.S[x][1]) and B.S[x-1][0]==B_ideal.S[x][1]:
            B.move_tile((x-1, 0), "R", cantuse=cantuse)
        if (B_ideal.S[x][0]!=B_ideal.S[x][1]) and B.S[x-1][1]==B_ideal.S[x][1]:
            B.move_tile((x-1, 1), "R", cantuse=cantuse)
        ignore_v = B_ideal.S[x][1]
        vxy = [(B_ideal.S[x][0], x, 1), (B_ideal.S[x][1], x-1, 1)]
    else:
        # 退避
        if (B_ideal.S[x][i]!=B_ideal.S[x][i-1]) and B.S[x][i]==B_ideal.S[x][i-1]:
            B.move_tile((x, i), "L", cantuse=cantuse)
        if (B_ideal.S[x][i]!=B_ideal.S[x][i-1]) and B.S[x][i-1]==B_ideal.S[x][i-1]:
            B.move_tile((x, i-1), "U", cantuse=cantuse)
        if (B_ideal.S[x][i]!=B_ideal.S[x][i-1]) and B.S[x-1][i]==B_ideal.S[x][i-1]:
            B.move_tile((x-1, i), "L", cantuse=cantuse)
        if (B_ideal.S[x][i]!=B_ideal.S[x][i-1]) and B.S[x-1][i-1]==B_ideal.S[x][i-1]:
            B.move_tile((x-1, i-1), "L", cantuse=cantuse)
        ignore_v = B_ideal.S[x][i-1]
        vxy = [(B_ideal.S[x][i], x, i-1), (B_ideal.S[x][i-1], x-1, i-1)]
    for v, x_, y_ in vxy:
        # 既に揃っていたら抜ける
        if ((not inv) and B.p0==(x, 0)) or (inv and B.p0==(x, i)):
            ret.append("U")
            B.move("U")
        if (not inv) and B_ideal.S[x][0]==B.S[x][0] and B_ideal.S[x][1]==B.S[x][1]:
            break
        if (inv) and B_ideal.S[x][i-1]==B.S[x][i-1] and B_ideal.S[x][i]==B.S[x][i]:
            break
        if not inv:
            if (x_, y_)==(x-1, 1):
                # 角から取らない
                cantuse.add((x, 0))
                target_x, target_y = B.find(v, fr=(x_, y_), cantuse=cantuse)
                cantuse.remove((x, 0))
            else:
                target_x, target_y = B.find(v, fr=(x_, y_), cantuse=cantuse)
        else:
            if (x_, y_)==(x-1, i-1):
                # 角から取らない
                cantuse.add((x, i))
                target_x, target_y = B.find(v, fr=(x_, y_), cantuse=cantuse)
                cantuse.remove((x, i))
            else:
                target_x, target_y = B.find(v, fr=(x_, y_), cantuse=cantuse)
        while target_y<y_:
            B.move_tile((target_x, target_y), "R", cantuse=cantuse, ignore_v=ignore_v)
            target_y += 1
        while target_y>y_:
            B.move_tile((target_x, target_y), "L", cantuse=cantuse, ignore_v=ignore_v)
            target_y -= 1
        while target_x<x_:
            B.move_tile((target_x, target_y), "D", cantuse=cantuse, ignore_v=ignore_v)
            target_x += 1
        cantuse.add((x_, y_))
    if (not inv) and (not (B_ideal.S[x][0]==B.S[x][0] and B_ideal.S[x][1]==B.S[x][1])):
        B.move_to((i, 0), cantuse=cantuse)
        B.move("R")
        B.move("U")
    if (inv) and (not (B_ideal.S[x][i]==B.S[x][i] and B_ideal.S[x][i-1]==B.S[x][i-1])):
        B.move_to((i, i), cantuse=cantuse)
        B.move("L")
        B.move("U")
    return ret

def solve_6(B: Board, B_ideal: Board)->List[str]:
    """最後はbfsでとく
    破壊的ではない!
    """
    B = Board([[B.S[i][j] for j in range(2)] for i in range(3)])
    B_ideal = Board([[B_ideal.S[i][j] for j in range(2)] for i in range(3)])
    t = str(B_ideal)
    if str(B)==str(B_ideal):
        return []
    appeared = set([str(B)])
    q = deque([[]])
    while q:
        s = q.popleft()
        B.move(s)
        for dir_s in "UDLR":
            try:
                B.move(dir_s)
                strb = str(B)
                if str(B)==t:
                    return s+[dir_s]
                if strb not in appeared:
                    q.append(s+[dir_s])
                    appeared.add(strb)
                B.unmove(dir_s)
            except AssertionError:
                pass
        B.unmove(s)
    raise ValueError("cant solve")

def main():
    v_cnt_S = [0]*16
    for x in range(N):
        for y in range(N):
            v_cnt_S[S[x][y]] += 1
    ans = None
    ans_hoken = None
    cnts = 0
    while time()-TS<2.75:
        cnts += 1
        cnt = 0
        S_ideal = [[0]*N for _ in range(N)]
        score = 2*(N*N-1)
        v_cnt_S_ideal = [0]*16
        v_cnt_S_ideal[0] = N*N
        uf = UnionFind(N*N)
        max_cnt=1000//36*(N*N)
        # 4の位置はデフォのところにさせる
        ps_4 = []
        for x in range(N):
            for y in range(N):
                if S[x][y] == 15:
                    x_ = x + (x==0) - (x==N-1)
                    y_ = y + (y==0) - (y==N-1)
                    ps_4.append((x_, y_))
        if False:
            for x_, y_ in ps_4:
                for dir in range(4):
                    if not ((S_ideal[x_][y_]>>dir) & 1):
                        p1 = (x_, y_)
                        p2 = (p1[0]+DIR2[dir][0], p1[1]+DIR2[dir][1])
                        v1_pre = S_ideal[p1[0]][p1[1]]
                        v2_pre = S_ideal[p2[0]][p2[1]]
                        v1 = v1_pre^(1<<dir)
                        v2 = v2_pre^(1<<((dir+2)%4))
                        uf.union(p1[0]*N+p1[1], p2[0]*N+p2[1])
                        S_ideal[p1[0]][p1[1]] = v1
                        S_ideal[p2[0]][p2[1]] = v2
                        v_cnt_S_ideal[v1_pre] -= 1
                        v_cnt_S_ideal[v2_pre] -= 1
                        v_cnt_S_ideal[v1] += 1
                        v_cnt_S_ideal[v2] += 1
            score = sum([abs(v_cnt_S[i]-v_cnt_S_ideal[i]) for i in range(16)]) #差分だけみる
        while cnt<max_cnt:
            cnt += 1
            # 最終盤面の探索
            p1 = (int(random()*N), int(random()*N))
            dir = int(random()*4)
            p2 = (p1[0]+DIR2[dir][0], p1[1]+DIR2[dir][1])
            if not (0<=p2[0]<N and 0<=p2[1]<N):
                continue
            v1_pre = S_ideal[p1[0]][p1[1]]
            v2_pre = S_ideal[p2[0]][p2[1]]
            v1 = v1_pre^(1<<dir)
            v2 = v2_pre^(1<<((dir+2)%4))
            # score = sum([abs(v_cnt_S[i]-v_cnt_S_ideal[i]) for i in range(16)]) #差分だけみる
            score_ = score
            if v_cnt_S[v1_pre]<v_cnt_S_ideal[v1_pre]:
                score_ -= 1
            else:
                score_ += 1
            v_cnt_S_ideal[v1_pre] -= 1
            if v_cnt_S[v2_pre]<v_cnt_S_ideal[v2_pre]:
                score_ -= 1
            else:
                score_ += 1
            v_cnt_S_ideal[v2_pre] -= 1
            if v_cnt_S[v1]>v_cnt_S_ideal[v1]:
                score_ -= 1
            else:
                score_ += 1
            v_cnt_S_ideal[v1] += 1
            if v_cnt_S[v2]>v_cnt_S_ideal[v2]:
                score_ -= 1
            else:
                score_ += 1
            v_cnt_S_ideal[v2] += 1
            # score_ = sum([abs(v_cnt_S[i]-v_cnt_S_ideal[i]) for i in range(16)])
            if score >= score_ and (v1_pre>v1 or uf.find(p1[0]*N+p1[1])!=uf.find(p2[0]*N+p2[1])):
                score = score_
                if v1_pre<v1:
                    uf.union(p1[0]*N+p1[1], p2[0]*N+p2[1])
                else:
                    uf = UnionFind(N*N)
                    for x in range(N):
                        for y in range(N):
                            v = S_ideal[x][y]
                            if v&4:
                                uf.union(x*N+y, x*N+y+1)
                            if v&8:
                                uf.union(x*N+y, x*N+y+N)
                    # uf.disjoint(p1[0]*N+p1[1], p2[0]*N+p2[1])
                S_ideal[p1[0]][p1[1]] = v1
                S_ideal[p2[0]][p2[1]] = v2
            else:
                v_cnt_S_ideal[v1_pre] += 1
                v_cnt_S_ideal[v2_pre] += 1
                v_cnt_S_ideal[v1] -= 1
                v_cnt_S_ideal[v2] -= 1
            if score==0:
                B = Board(S)
                B_ideal = Board(S_ideal)
                try:
                    for moves in solve(Board(S), B_ideal):
                        B.move(moves)
                except Exception:
                    continue
                tmpans = "".join(B.move_hist)
                if ans is None or len(ans)>len(tmpans):
                    ans = tmpans
                break
        if ans_hoken is None and score==2:
            # 見つかんなければとりあえず作っとく
            for x, y in [(x, y) for x in range(N) for y in range(N)]:
                v = S_ideal[x][y]
                if v_cnt_S[v]<v_cnt_S_ideal[v]:
                    for v_ in range(16):
                        if v_cnt_S[v_]>v_cnt_S_ideal[v_]:
                            break
                    S_ideal[x][y] = v_
                    v_cnt_S_ideal[v] -= 1
                    v_cnt_S_ideal[v_] += 1
                    break
            for x, y in [(x, y) for x in range(N-1, -1, -1) for y in range(N-1, -1, -1)]:
                v = S_ideal[x][y]
                if v_cnt_S[v]<v_cnt_S_ideal[v]:
                    for v_ in range(16):
                        if v_cnt_S[v_]>v_cnt_S_ideal[v_]:
                            break
                    S_ideal[x][y] = v_
                    v_cnt_S_ideal[v] -= 1
                    v_cnt_S_ideal[v_] += 1
                    break
            B = Board(S)
            B_ideal = Board(S_ideal)
            for moves in solve(Board(S), B_ideal):
                B.move(moves)
            ans_hoken = "".join(B.move_hist)

    if ans is not None:
        #if len(ans)>T:
        #    raise ValueError
        print(ans[:T])
    elif ans_hoken is not None:
        #raise ValueError
        print(ans_hoken[:T])
    else:
        #raise ValueError
        print("")

main()
sys.exit(0)
#prof = LineProfiler()
#prof.add_function(main)
#prof.runcall(main)
#prof.print_stats()
