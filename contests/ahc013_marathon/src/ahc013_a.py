from collections import defaultdict, deque
from re import X
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


def diff(C1: List[List[int]], C2: List[List[int]]):
    N = len(C1)
    moves = []
    for p in range(N*N):
        i, j = p//N, p%N
        if C1[i][j]==C2[i][j]:
            continue
        if C1[i][j]==0:
            # 周辺の-1だけを
            pass

    return len(moves)

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




def solve(N: int, K: int, C: List[int]):
    anss = []
    # target選定
    target = 1
    # targetを連結にさせる
    # beam_search
    seed = [(0, Game(N, C, K))]
    beam_haba = 100
    for _ in range(K*100):
        new_seed = []
        for _, game in seed:
            for i, j in game.spaces:
                for dir in range(4):
                    i_, j_ = i+DIR[dir][0], j+DIR[dir][1]
                    if (not (0<=i_<N and 0<=j_<N)) or ((i_, j_) in game.spaces):
                        continue
                    game_ = game.copy()
                    game_.swap(i_, j_, i, j)
                    new_seed.append((game_.score_(target), game_))
        for _, game in seed:
            solve_ad(game)
            anss.append((game.score(), game.get_ans()))
        seed = sorted(new_seed, key=lambda x: -x[0])[:beam_haba]
        anss = sorted(anss, key=lambda x: -x[0])[:10]
    game = Game(N, C, K)
    solve_ad(game)
    anss.append((game.score(), game.get_ans()))
    # print(time()-TS)
    for score, ans in sorted(anss)[::-1]:
        print(ans)


input = sys.stdin.readline
N, K = map(int, input().split())
C = [list(map(int, input().rstrip())) for _ in range(N)]
solve(N, K, C)
