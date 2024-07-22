from asyncore import write
import math
import random
from statistics import mean, variance
from time import time
import sys
from timeit import repeat
from collections import deque

TS = time()
TL = 0.8*50

# [a, b)
def randint(a, b):
    return int(random.random()*(b-a))+a

def argmin(X):
    ret, v = None, math.inf
    for i, x in enumerate(X):
        if x<v:
            ret, v = i, x
    return ret

def argmax(X):
    ret, v = None, -math.inf
    for i, x in enumerate(X):
        if x>v:
            ret, v = i, x
    return ret

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

N, K, L = map(int, input().split())
AB = [list(map(int, input().split())) for _ in range(K)]
C = [list(map(lambda x: int(x)-1, input().split())) for _ in range(N)]
G = [[] for k in range(K)]
for x in range(N):
    for y in range(N):
        k = C[x][y]
        if k==-1:
            continue
        for x_, y_ in [(x+1, y), (x, y+1)]:
            if 0<=x_<N and 0<=y_<N and C[x_][y_]!=-1 and C[x_][y_]!=C[x][y]:
                k_ = C[x_][y_]
                G[k].append(k_)
                G[k_].append(k)


# 連結前提
class State:
    def __init__(self, ans):
        self.belongs = [set() for _ in range(L)]
        self.Ps = [0]*L
        self.Qs = [0]*L
        self.ans = ans
        for k, v in enumerate(ans):
            self.belongs[v].add(k)
            a, b = AB[k]
            self.Ps[v] += a
            self.Qs[v] += b

    @classmethod
    def get_initial_state(cls):
        uf = UnionFind(K)
        for _ in range(K-L):
            best = (math.inf, None, None)
            for k in range(K):
                for k_ in G[k]:
                    if uf.find(k)!=uf.find(k_):
                        cnt = uf.size(k)+uf.size(k_)
                        if cnt < best[0]:
                            best = (cnt, k, k_)
                        if best[0]<=10:
                            break
                if best[0]<=10:
                    break
            uf.union(best[1], best[2])
        ans = [-1]*K
        for v, ks in enumerate(uf.all_group_members().values()):
            for k in ks:
                ans[k] = v
        return State(ans)

    def can_move(self, *params):
        if params[0]==0:
            _, k, _ = params
            v = self.ans[k]
            # kを抜いても連結か
            q = [k_ for k_ in G[k] if k_ in self.belongs[v]][:1]
            if len(q)==0:
                return False
            appeared = set()
            cnt = 0
            while q:
                k_ = q.pop()
                if k_ in appeared:
                    continue
                cnt += 1
                appeared.add(k_)
                for k__ in G[k_]:
                    if k__ not in appeared and self.ans[k__]==v and k__!=k:
                        q.append(k__)
            return cnt==len(self.belongs[v])-1
        else:
            # 実際違うが2回判定
            flg, k1, v1, k2, v2 = params
            for params_ in [(flg, k1, v1, k2, v2), (flg, k2, v2, k1, v1)]:
                assert self.ans[k1]==v2 and self.ans[k2]==v1
                _, k1, v1, k2, v2 = params_
                # kを抜いても連結か
                q = [k_ for k_ in G[k1] if k_ in self.belongs[v2]][:1]
                if len(q)==0:
                    return False
                appeared = set()
                cnt = 0
                while q:
                    k_ = q.pop()
                    if k_ in appeared:
                        continue
                    cnt += 1
                    appeared.add(k_)
                    for k__ in G[k_]:
                        if k__ not in appeared and (self.ans[k__]==v2 or k__==k2) and k__!=k1:
                            q.append(k__)
                if not cnt==len(self.belongs[v2])-1+1:
                    return False
            return True

    def move(self, *params):
        kvs = []
        if params[0]==0:
            kvs = [(params[1], params[2])]
        else:
            kvs = [(params[1], params[2]), (params[3], params[4])]
        for k, v in kvs:
            a, b = AB[k]
            # 連結チェック, しない？
            # if not self.can_move(*params):
            #     return False
            # 前の所属から削除
            v_before = self.ans[k]
            self.belongs[v_before].remove(k)
            self.Ps[v_before] -= a
            self.Qs[v_before] -= b
            # 新しい所属に追加
            self.ans[k] = v
            self.belongs[v].add(k)
            self.Ps[v] += a
            self.Qs[v] += b
        return True

    def unmove(self, *args):
        raise NotImplementedError

    def get_neighbor(self):
        if random.random()<=0.5:
            flg = random.random()
            # 最大を削る
            if flg<=0.4:
                v = argmax(self.Ps) if random.random()>=0.5 else argmax(self.Qs)
                for k in random.sample(tuple(self.belongs[v]), len(self.belongs[v])):
                    shuuhen = set([self.ans[k]]+[self.ans[k_] for k_ in G[k]])
                    shuuhen.remove(self.ans[k])
                    if len(shuuhen)>=1:
                        return (0, k, random.choice(list(shuuhen)))
            # 最小タス
            elif flg<=0.8:
                v = argmin(self.Ps) if random.random()>=0.5 else argmin(self.Qs)
                for k_ in random.sample(tuple(self.belongs[v]), len(self.belongs[v])):
                    for k in G[k_]:
                        if self.ans[k]!=v:
                            return (0, k, v)
            else:
                v = randint(0, L)
                for k in random.sample(tuple(self.belongs[v]), len(self.belongs[v])):
                    shuuhen = set([self.ans[k]]+[self.ans[k_] for k_ in G[k]])
                    shuuhen.remove(self.ans[k])
                    if len(shuuhen)>=1:
                        return (0, k, random.choice(list(shuuhen)))
        else:
            v1 = randint(0, L)
            for k1 in random.sample(tuple(self.belongs[v1]), len(self.belongs[v1])):
                shuuhen = set([self.ans[k1]]+[self.ans[k_] for k_ in G[k1]])
                shuuhen.remove(self.ans[k1])
                for v2 in shuuhen:
                    for k2 in random.sample(tuple(self.belongs[v2]), len(self.belongs[v2])):
                        shuuhen2 = set([self.ans[k_] for k_ in G[k2]])
                        if v1 in shuuhen2:
                            return (1, k1, v2, k2, v1)
        return self.get_neighbor()

    # TODO: 実装次第で修正
    def get_neighbor_with_score(self):
        params = self.get_neighbor()
        if params[0]==0:
            k, v = params[1:]
            pre_v = self.ans[k]
            self.move(0, k, v)
            score = self.get_score()
            self.move(0, k, pre_v)
            return params, score
        else:
            k1, v1, k2, v2 = params[1:]
            pre_v1 = self.ans[k1]
            pre_v2 = self.ans[k2]
            self.move(1, k1, v1, k2, v2)
            score = self.get_score()
            self.move(1, k1, pre_v1, k2, pre_v2)
            return params, score


    def get_score(self, **params):
        return round(10**6 * min(min(self.Ps)/max(self.Ps), min(self.Qs)/max(self.Qs)))
        pvar, qvar = variance(self.Ps), variance(self.Qs)
        return -(pvar+qvar*2500)

    def print(self):
        print(*[x+1 for x in self.ans], sep="\n")
        # print(*[x+1 if x!=-1 else 1 for x in self.ans], sep="\n")

    def copy(self):
        return State([x for x in self.ans])


class Optimizer:
    def __init__(self, verbose=False):
        self.history = []

    def climbing(self, initial_state, TL_=TL, save_history=False):
        state = initial_state.copy()
        score = state.get_score()
        cnt = 0
        if save_history:
            self.history.append((cnt, score, state))
        TS_ = time()
        while time()-TS_<TL_:
            cnt += 1
            params, new_score = state.get_neighbor_with_score()
            if new_score >= score and state.can_move(*params):
                state.move(*params)
                score = new_score
                if save_history:
                    self.history.append((cnt, score, state.copy()))
        return state

    def simanneal(self, initial_state, TL_=TL, save_history=False):
        TS_ = TS
        state = initial_state.copy()
        score = state.get_score()
        start_temp = 100000
        end_temp = 0.1
        cnt = 0
        if save_history:
            self.history.append((cnt, score, state))
        while True:
            cur_time =time()
            if cur_time-TS_>=TL_:
                break
            cnt += 1
            params, new_score = state.get_neighbor_with_score()
            temp = start_temp + (end_temp-start_temp)*(cur_time-TS)/TL
            if (new_score>=score or math.exp((new_score-score)/temp) > random.random()) and state.can_move(*params):
                state.move(*params)
                score = new_score
                self.history.append((cnt, score, state.copy()))
        return state

    def beam_search(self, TL=1.8, beam_width=10):
        raise NotImplementedError

    def visualize(self):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import matplotlib.cm as cm
        fig = plt.figure()
        ims = []
        cmap = list(cm.get_cmap("tab20").colors)+[(0, 0, 0)]
        for cnt, score, state in self.history:
            ans = state.ans + [-1]
            M = [[cmap[ans[C[x][y]]] for y in range(N)] for x in range(N)]
            im = plt.imshow(M)
            ims.append([im])
        ani = animation.ArtistAnimation(fig, ims, interval=100)
        ani.save("./results/vis.gif", writer="pillow")
        # plt.show()

init_state = State.get_initial_state()
opt = Optimizer()
# best_state = opt.climbing(init_state)
best_state = opt.simanneal(init_state)
best_state.print()
# opt.visualize()
