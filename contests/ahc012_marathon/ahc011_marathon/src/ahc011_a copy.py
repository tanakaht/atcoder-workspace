import sys
import math
from time import time
from typing import List, Optional, Tuple, Union, Set, Dict
from collections import defaultdict, deque
import random
# from __future__ import annotations
TIMELIMIT = 2.8
TS = time()
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


class BIT:
    def __init__(self,len_A):
        self.N = len_A + 10
        self.bit = [0]*(len_A+10)

    # sum(A0 ~ Ai)
    # O(log N)
    def query(self,i):
        res = 0
        idx = i+1
        while idx:
            res += self.bit[idx]
            idx -= idx&(-idx)
        return res

    def get(self, i):
        return  self.query(i)-self.query(i-1)

    # Ai += x
    # O(log N)
    def update(self,i,x):
        idx = i+1
        while idx < self.N:
            self.bit[idx] += x
            idx += idx&(-idx)

    # min_i satisfying {sum(A0 ~ Ai) >= w} (Ai >= 0)
    # O(log N)
    def lower_left(self,w):
        if (w < 0):
            return -1
        x = 0
        k = 1<<(self.N.bit_length()-1)
        while k > 0:
            if x+k < self.N and self.bit[x+k] < w:
                w -= self.bit[x+k]
                x += k
            k //= 2
        return x

class Board:
    def __init__(self, S: List[List[int]], p_dict: Optional[Dict[int, Tuple[int, int]]]=None, move_hist=None):
        self.N = len(S)
        # 左方向を1、上方向を2、右方向を4、下方向を8としたビットマスクを用いて以下のように表現される。
        self.S = [[i for i in s] for s in S]
        for i in range(len(S)):
            for j in range(len(S[0])):
                if self.S[i][j]==0:
                    self.p0 = (i, j)
                    break
        N = self.N
        # 初期位置で0~N^2-1まで番号付け、それぞれがどこいったのか記録
        self.p_dict = {i*N+j: (i, j) for i in range(N) for j in range(N)} if p_dict is None else {k: v for k, v in p_dict.items()}
        # moveの履歴
        self.move_hist = [] if move_hist is None else [x for x in move_hist]

    def __str__(self) -> str:
        # 1６以上入れるとバグる可能性あるので","を入れている
        return "\n".join([",".join([hex(x)[2:] for x in s]) for s in self.S])

    def copy(self):
        return Board(self.S, p_dict=self.p_dict, move_hist=self.move_hist)

    def score(self)->int:
        res = self.metric()
        uf = res["uf"]
        n_loop = res["n_loop"]
        S = 0
        for u in uf.roots():
            if n_loop[u]==0:
                S = max(S, uf.size(u))
        return S # TODO: まあよし

    def score_p(self, p: Tuple[int, int])->int:
        ret = 1
        x, y = p
        v = self.S[x][y]
        for dir in range(4):
            x_, y_ = x+DIR2[dir][0], y+DIR2[dir][1]
            if 0<=x_<self.N and 0<=y_<self.N:
                v_ = self.S[x_][y_]
                ret += (((v>>dir)&1)^((v_>>((dir+2)%4))&1))*10
            else:
                ret += ((v>>dir)&1)*100
        return ret

    def metric(self)->dict:
        uf = UnionFind(self.N*self.N)
        n_loop = defaultdict(int)
        # たて
        for i in range(self.N-1):
            for j in range(self.N):
                if (self.S[i][j]>>3)&1 and (self.S[i+1][j]>>1)&1:
                    r1, r2 = uf.find(i*self.N+j), uf.find((i+1)*self.N+j)
                    if r1==r2:
                        n_loop[r1] += 1
                    else:
                        uf.union(i*self.N+j, (i+1)*self.N+j)
                        n_loop[uf.find(i*self.N+j)] = n_loop[r1] + n_loop[r2]
        # 横
        for i in range(self.N):
            for j in range(self.N-1):
                if (self.S[i][j]>>2)&1 and (self.S[i][j+1]>>0)&1:
                    r1, r2 = uf.find(i*self.N+j), uf.find(i*self.N+j+1)
                    if r1==r2:
                        n_loop[r1] += 1
                    else:
                        uf.union(i*self.N+j, i*self.N+j+1)
                        n_loop[uf.find(i*self.N+j)] = n_loop[r1] + n_loop[r2]
        ret = {
            "n_loop": n_loop,
            "uf": uf
        }
        return ret

    def swap(self, p1: Tuple[int, int], p2: Tuple[int, int])->None:
        N = self.N
        x1, y1 = p1
        x2, y2 = p2
        assert 0<=x1<len(self.S) and 0<=x2<len(self.S) and 0<=y1<len(self.S[0]) and 0<=y2<len(self.S[0])
        s1, s2 = self.S[x1][y1], self.S[x2][y2]
        self.S[x1][y1] = s2
        self.S[x2][y2] = s1
        if p1==self.p0:
            self.p0 = p2
        elif p2==self.p0:
            self.p0 = p1
        self.p_dict[N*x1+y1], self.p_dict[N*x2+y2] = self.p_dict[N*x2+y2], self.p_dict[N*x1+y1]

    def move(self, S: Union[str, List[str]], replace=True):
        if not replace:
            ret = Board(self.S, p_dict=self.p_dict, move_hist=self.move_hist)
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

    def dist(self, B_ideal):
        N = self.N
        ret = 0
        for i in range(self.N*self.N):
            if self.p0!=self.p_dict[i]:
                ret += (dist(self.p_dict[i], B_ideal.p_dict[i])*pow(10, min(B_ideal.p_dict[i])))**2
        return ret

    def move_to(self, p: Tuple[int, int], cantuse: Optional[set]=None)->List[str]:
        cantuse = set() if cantuse is None else cantuse
        px, py = p
        q = deque([self.p0])
        fr = [[None]*self.N for _ in range(self.N)]
        appeared = [[False]*self.N for _ in range(self.N)]
        appeared[self.p0[0]][self.p0[1]] = True
        while q:
            x, y = q.popleft()
            if x==px and y==py:
                break
            for dir_s in "UDLR":
                x_ = x+DIR[dir_s][0]
                y_ = y+DIR[dir_s][1]
                if (not (0<=x_<self.N and 0<=y_<self.N)) or (x_, y_) in cantuse or appeared[x_][y_]:
                    continue
                fr[x_][y_] = dir_s
                q.append((x_, y_))
                appeared[x_][y_] = True
        cur_x, cur_y = (px, py)
        ret = []
        while fr[cur_x][cur_y] is not None:
            dir_s = fr[cur_x][cur_y]
            ret.append(dir_s)
            cur_x -= DIR[dir_s][0]
            cur_y -= DIR[dir_s][1]
        if (cur_x, cur_y)!=self.p0:
            raise ValueError("cant touch")
        ret = ret[::-1]
        for dir_s in ret:
            self.move(dir_s)
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

    # 2
    stepwise_moves = []
    for i in range(N-1, 2, -1):
        moves = solve_R(B, B_ideal, i)
        stepwise_moves.append(moves)
        moves = solve_D(B, B_ideal, i)
        stepwise_moves.append(moves)
    moves = solve_R(B, B_ideal, 2)
    stepwise_moves.append(moves)
    # 3
    try:
        moves = solve_6(B, B_ideal)
        B.move(moves)
        stepwise_moves.append(moves)
    except ValueError:
        # 皮一枚ずつ拡張して同じ模様を見つけつつ、idxをふりなおす
        num2ps = defaultdict(list)
        idx = 17
        found = False
        for aug_idx in range(2, N):
            for x in range(aug_idx+1):
                if not found:
                    num = B_ideal.S[x][aug_idx]
                    num2ps[num].append((x, aug_idx))
                    if len(num2ps[num])>=2:
                        found = True
                B.S[x][aug_idx] = idx
                B_ideal.S[x][aug_idx] = idx
                idx += 1
            if aug_idx!=2:
                for y in range(aug_idx):
                    if not found:
                        num = B_ideal.S[aug_idx][y]
                        num2ps[num].append((aug_idx, y))
                        if len(num2ps[num])>=2:
                            found = True
                    B.S[aug_idx][y] = idx
                    B_ideal.S[aug_idx][y] = idx
                    idx += 1
        # 同じ模様のidxを入れ替え
        p1, p2 = sorted(num2ps.values(), key=lambda x: len(x))[-1]
        B_ideal.S[p1[0]][p1[1]], B_ideal.S[p2[0]][p2[1]] = B_ideal.S[p2[0]][p2[1]], B_ideal.S[p1[0]][p1[1]]
        # 皮一枚揃えるためのmovesを消す
        B.unmove(stepwise_moves.pop())
        for i in range(3, aug_idx+1):
            B.unmove(stepwise_moves.pop())
            B.unmove(stepwise_moves.pop())
        # 再帰でこの先を揃えてもらう
        for moves in solve(B, B_ideal):
            stepwise_moves.append(moves)
            B.move(moves)
    # 4
    stepwise_moves.append([dir_inv[s] for s in S1[::-1]])
    B.unmove(S1)
    return ["".join(moves) for moves in stepwise_moves]

def solve_R(B: Board, B_ideal: Board, i: int)->List[str]:
    """solveの補助関数(2をやる(i列目))
    Bは破壊的
    """
    ret = []
    y = i
    cantuse = set([(x_, y_) for x_ in range(N) for y_ in range(N) if max(x_, y_)>i])
    for x in range(i-1):
        target_x, target_y = B.find(B_ideal.S[x][y], fr=(x, y), cantuse=cantuse)
        while target_x<x:
            cantuse.add((target_x, target_y))
            ret += B.move_to((target_x+1, target_y), cantuse=cantuse)
            cantuse.remove((target_x, target_y))
            target_x += 1
            ret.append("U")
            B.move("U")
        while target_x>x:
            cantuse.add((target_x, target_y))
            ret += B.move_to((target_x-1, target_y), cantuse=cantuse)
            cantuse.remove((target_x, target_y))
            target_x -= 1
            ret.append("D")
            B.move("D")
        while target_y<y:
            cantuse.add((target_x, target_y))
            ret += B.move_to((target_x, target_y+1), cantuse=cantuse)
            cantuse.remove((target_x, target_y))
            target_y += 1
            ret.append("L")
            B.move("L")
        cantuse.add((x, y))
    # ラスト2個揃える
    if B.p0[1]==y:
        ret.append("L")
        B.move("L")
    # 退避させる
    if (B_ideal.S[i][y]!=B_ideal.S[i-1][y]) and B.S[i][y]==B_ideal.S[i-1][y]:
        cantuse.add((i, y))
        ret += B.move_to((i-1, y), cantuse=cantuse)
        cantuse.remove((i, y))
        ret.append("D")
        B.move("D")
    if (B_ideal.S[i][y]!=B_ideal.S[i-1][y]) and B.S[i-1][y]==B_ideal.S[i-1][y]:
        cantuse.add((i-1, y))
        ret += B.move_to((i-1, y-1), cantuse=cantuse)
        cantuse.remove((i-1, y))
        ret.append("R")
        B.move("R")
    if (B_ideal.S[i][y]!=B_ideal.S[i-1][y]) and B.S[i][y-1]==B_ideal.S[i-1][y]:
        cantuse.add((i, y-1))
        ret += B.move_to((i-1, y-1), cantuse=cantuse)
        cantuse.remove((i, y-1))
        ret.append("D")
        B.move("D")
    if (B_ideal.S[i][y]!=B_ideal.S[i-1][y]) and B.S[i-1][y-1]==B_ideal.S[i-1][y]:
        cantuse.add((i-1, y-1))
        ret += B.move_to((i-2, y-1), cantuse=cantuse)
        cantuse.remove((i-1, y-1))
        ret.append("D")
        B.move("D")
    for v, x_, y_ in [(B_ideal.S[i][y], i-1, y), (B_ideal.S[i-1][y], i-1, y-1)]:
        # 既に揃っていたら抜ける
        if B.p0==(i, y):
            ret.append("L")
            B.move("L")
        if B_ideal.S[i][y]==B.S[i][y] and B_ideal.S[i-1][y]==B.S[i-1][y]:
            break
        if (x_, y_)==(i-1, y-1):
            cantuse.add((i, y))
        target_x, target_y = B.find(v, fr=(x_, y_), cantuse=cantuse)
        if (x_, y_)==(i-1, y-1):
            cantuse.remove((i, y))
        while target_x<x_:
            cantuse.add((target_x, target_y))
            ret += B.move_to((target_x+1, target_y), cantuse=cantuse)
            cantuse.remove((target_x, target_y))
            target_x += 1
            ret.append("U")
            B.move("U")
        while target_x>x_:
            cantuse.add((target_x, target_y))
            ret += B.move_to((target_x-1, target_y), cantuse=cantuse)
            cantuse.remove((target_x, target_y))
            target_x -= 1
            ret.append("D")
            B.move("D")
        while target_y<y_:
            cantuse.add((target_x, target_y))
            ret += B.move_to((target_x, target_y+1), cantuse=cantuse)
            cantuse.remove((target_x, target_y))
            target_y += 1
            ret.append("L")
            B.move("L")
        cantuse.add((x_, y_))
    if not (B_ideal.S[i][y]==B.S[i][y] and B_ideal.S[i-1][y]==B.S[i-1][y]):
        ret += B.move_to((i, i), cantuse=cantuse)
        ret.append("U")
        B.move("U")
        ret.append("L")
        B.move("L")
    return ret

def solve_D(B: Board, B_ideal: Board, i: int)->List[str]:
    """solveの補助関数(2をやる(i行目))
    Bは破壊的
    """
    """solveの補助関数(2をやる(i列目))
    Bは破壊的
    """
    ret = []
    x = i
    cantuse = set([(x_, y_) for x_ in range(N) for y_ in range(N) if max(x_, y_)>i or y_==i])
    for y in range(i-2):
        target_x, target_y = B.find(B_ideal.S[x][y], fr=(x, y), cantuse=cantuse)
        while target_y<y:
            cantuse.add((target_x, target_y))
            ret += B.move_to((target_x, target_y+1), cantuse=cantuse)
            cantuse.remove((target_x, target_y))
            target_y += 1
            ret.append("L")
            B.move("L")
        while target_y>y:
            cantuse.add((target_x, target_y))
            ret += B.move_to((target_x, target_y-1), cantuse=cantuse)
            cantuse.remove((target_x, target_y))
            target_y -= 1
            ret.append("R")
            B.move("R")
        while target_x<x:
            cantuse.add((target_x, target_y))
            ret += B.move_to((target_x+1, target_y), cantuse=cantuse)
            cantuse.remove((target_x, target_y))
            target_x += 1
            ret.append("U")
            B.move("U")
        cantuse.add((x, y))
    # ラスト2個揃える
    if B.p0[0]==x:
        ret.append("U")
        B.move("U")
    if (B_ideal.S[x][i-1]!=B_ideal.S[x][i-2]) and B.S[x][i-1]==B_ideal.S[x][i-2]:
        cantuse.add((x, i-1))
        ret += B.move_to((x, i-2), cantuse=cantuse)
        cantuse.remove((x, i-1))
        ret.append("R")
        B.move("R")
    if (B_ideal.S[x][i-1]!=B_ideal.S[x][i-2]) and B.S[x][i-2]==B_ideal.S[x][i-2]:
        cantuse.add((x, i-2))
        ret += B.move_to((x-1, i-2), cantuse=cantuse)
        cantuse.remove((x, i-2))
        ret.append("D")
        B.move("D")
    if (B_ideal.S[x][i-1]!=B_ideal.S[x][i-2]) and B.S[x-1][i-1]==B_ideal.S[x][i-2]:
        cantuse.add((x-1, i-1))
        ret += B.move_to((x-1, i-2), cantuse=cantuse)
        cantuse.remove((x-1, i-1))
        ret.append("R")
        B.move("R")
    if (B_ideal.S[x][i-1]!=B_ideal.S[x][i-2]) and B.S[x-1][i-2]==B_ideal.S[x][i-2]:
        cantuse.add((x-1, i-2))
        ret += B.move_to((x-1, i-3), cantuse=cantuse)
        cantuse.remove((x-1, i-2))
        ret.append("R")
        B.move("R")
    for v, x_, y_ in [(B_ideal.S[x][i-1], x, i-2), (B_ideal.S[x][i-2], x-1, i-2)]:
        # 既に揃っていたら抜ける
        if B.p0==(x, i-1):
            ret.append("U")
            B.move("U")
        if B_ideal.S[x][i-1]==B.S[x][i-1] and B_ideal.S[x][i-2]==B.S[x][i-2]:
            break
        if (x_, y_)==(x-1, i-2):
            cantuse.add((x, i-1))
        target_x, target_y = B.find(v, fr=(x_, y_), cantuse=cantuse)
        if (x_, y_)==(x-1, i-2):
            cantuse.remove((x, i-1))
        while target_y<y_:
            cantuse.add((target_x, target_y))
            ret += B.move_to((target_x, target_y+1), cantuse=cantuse)
            cantuse.remove((target_x, target_y))
            target_y += 1
            ret.append("L")
            B.move("L")
        while target_y>y_:
            cantuse.add((target_x, target_y))
            ret += B.move_to((target_x, target_y-1), cantuse=cantuse)
            cantuse.remove((target_x, target_y))
            target_y -= 1
            ret.append("R")
            B.move("R")
        while target_x<x_:
            cantuse.add((target_x, target_y))
            ret += B.move_to((target_x+1, target_y), cantuse=cantuse)
            cantuse.remove((target_x, target_y))
            target_x += 1
            ret.append("U")
            B.move("U")
        cantuse.add((x_, y_))
    if not (B_ideal.S[x][i-1]==B.S[x][i-1] and B_ideal.S[x][i-2]==B.S[x][i-2]):
        ret += B.move_to((i, i-1), cantuse=cantuse)
        ret.append("L")
        B.move("L")
        ret.append("U")
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

while time()-TS<1.8:
    B_ideal = Board(S)
    scores_bit = BIT(N*N)
    for x in range(N):
        for y in range(N):
            v = B_ideal.score_p((x, y))
            scores_bit.update(x*N+y, v)
            print(v, scores_bit.get(x*N+y))
    ans = ""
    cnt = 0
    start_temp = 1000
    end_temp = 0.1
    step = 100000
    temp_c = (end_temp/start_temp)**(1/step)
    temp = start_temp
    while temp>end_temp and time()-TS<1.8:
        cnt += 1
        # 最終盤面の探索
        score = scores_bit.query(N*N)
        idx1 = scores_bit.lower_left(random.randint(1, score))
        idx2 = scores_bit.lower_left(random.randint(1, score))
        p1 = (idx1//N, idx1%N)
        p2 = (idx2//N, idx2%N)
        # min_i satisfying {sum(A0 ~ Ai) >= w} (Ai >= 0)
        # O(log N)
        score_points = set()
        for x, y in [p1, p2]:
            for dir in range(4):
                x_, y_ = x+DIR2[dir][0], y+DIR2[dir][1]
                if 0<=x_<N and 0<=y_<N:
                    score_points.add((x_, y_))
            score_points.add((x, y))
        B_ideal.swap(p1, p2)
        score_ = score
        tmp_score_d = {}
        for x, y in score_points:
            v = B_ideal.score_p((x, y))
            v_pre = scores_bit.get(x*N+y)
            score_ += v - v_pre
            tmp_score_d[x*N+y] = v-v_pre
        # if score > score_:
        if random.randint(0, 1) < math.exp((score-score_)/temp):
            print(cnt)
            score = score_
            for i, x in tmp_score_d.items():
                scores_bit.update(i, x)
            print(1, score, scores_bit.query(N*N))
        else:
            B_ideal.swap(p1, p2)


        if score==0:
            B = Board(S)
            for moves in solve(Board(S), B_ideal):
                B.move(moves)
            tmpans = "".join(B.move_hist)
            if len(ans)==0 or len(ans)>len(tmpans):
                ans = tmpans
            break
    print(cnt)
    if ans!="":
        print(ans[:T])
        sys.exit(0)


ret = []
for s in "u2r2d3ru2ld2ru3ld3l2u3r2d2l2dru3rd3l2u2r3dl3dru2r2d2":
    if pre is not None:
        if s not in "udlr":
            ret.append(pre*int(s))
            pre = None
        else:
            ret.append(pre)
            pre = s
    else:
        pre = s
