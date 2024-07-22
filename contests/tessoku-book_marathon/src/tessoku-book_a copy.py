from asyncore import write
import math
import random
from statistics import mean, variance
from time import time
import sys
from timeit import repeat
from collections import deque

TS = time()

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
        if ans is None:
            self.initialize()
            return
        self.belongs = [set() for _ in range(L)]
        self.Ps = [0]*L
        self.Qs = [0]*L
        self.ans = ans
        for k, v in enumerate(ans):
            self.belongs[v].add(k)
            a, b = AB[k]
            self.Ps[v] += a
            self.Qs[v] += b

    def initialize(self):
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
        self.__init__(ans)

    def change_belong(self, k, v):
        a, b = AB[k]
        # 連結チェック, しない？
        # if not self.can_remove(k):
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

    def can_remove(self, k):
        v = self.ans[k]
        # kを抜いても連結か
        q = [k_ for k_ in G[k] if k_ in self.belongs[v]][:1]
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

    def get_neighbors(self, **params):
        ret = []
        for k in range(K):
            shuuhen = set([self.ans[k]]+[self.ans[k_] for k_ in G[k]])
            shuuhen.remove(self.ans[k])
            if len(shuuhen)==0:
                continue
            for v in shuuhen:
                ret.append((k, v))
        return ret

    def get_neighbor(self, **params):
        # 最大を削る
        flg = random.random()
        if flg<=0.4:
            v = argmax(self.Ps) if random.random()>=0.5 else argmax(self.Qs)
            for k in random.sample(tuple(self.belongs[v]), len(self.belongs[v])):
                shuuhen = set([self.ans[k]]+[self.ans[k_] for k_ in G[k]])
                shuuhen.remove(self.ans[k])
                if len(shuuhen)>=1 and self.can_remove(k):
                    return (k, random.choice(list(shuuhen)))
        # 最小タス
        elif flg<=0.8:
            v = argmin(self.Ps) if random.random()>=0.5 else argmin(self.Qs)
            for k_ in random.sample(tuple(self.belongs[v]), len(self.belongs[v])):
                for k in G[k_]:
                    if self.ans[k]!=v and self.can_remove(k):
                        return (k, v)
        else:
            v = randint(0, L)
            for k in random.sample(tuple(self.belongs[v]), len(self.belongs[v])):
                shuuhen = set([self.ans[k]]+[self.ans[k_] for k_ in G[k]])
                shuuhen.remove(self.ans[k])
                if len(shuuhen)>=1 and self.can_remove(k):
                    return (k, random.choice(list(shuuhen)))
        return self.get_neighbor(**params)

    def get_score(self, **params):
        return round(10**6 * min(min(self.Ps)/max(self.Ps), min(self.Qs)/max(self.Qs)))

    def get_score_(self, **params):
        # pmean, qmean = mean(self.Ps), mean(self.Qs)
        pvar, qvar = variance(self.Ps), variance(self.Qs)
        return -(pvar+qvar)

    def print(self):
        print(*[x+1 for x in self.ans], sep="\n")
        # print(*[x+1 if x!=-1 else 1 for x in self.ans], sep="\n")

    def copy(self):
        return State([x for x in self.ans])

# 近傍とscoreを色々変えて試す用
class StateWrapper(State):
    def __init__(self, state: State, get_neighbors, get_score, get_neigbors_with_score=None):
        self.state = state
        self._get_neighbors = get_neighbors
        self._get_score = get_score
        self._get_neighbors_with_score = get_neigbors_with_score

    def get_neighbors(self, **params):
        return self._get_neighbors(**params)

    def get_score(self, **params):
        return self._get_score(**params)

    def get_neighbors_with_score(self, **params):
        if self._get_neighbors_with_score is not None:
            return self._get_neighbors_with_score(**params)
        ret = []
        for state in self.get_neighbors(**params):
            score = state.get_score(**params)
            ret.append((score, state))
        return ret


class Optimizer:
    def __init__(self, initial_states):
        self.initial_states = initial_states
        self.history = []

    def climbing(self, TL=0.8):
        state = self.initial_states[0].copy()
        score = state.get_score()
        cnt = 0
        self.history.append((cnt, score, state))
        TS_ = time()
        while time()-TS_<TL:
            cnt += 1
            k, v = state.get_neighbor()
            v_before = state.ans[k]
            state.change_belong(k, v)
            new_score = state.get_score()
            # print(new_score>score, score, new_score)
            if new_score >= score:
                cnt += 1
                # print(cnt, new_score)
                score = new_score
                self.history.append((cnt, score, state.copy()))
            else:
                state.change_belong(k, v_before)
        return state

    def simanneal(self, TL=0.8):
        TS = time()
        state = self.initial_states[0].copy()
        score = state.get_score()
        start_temp = 100000
        end_temp = 0.1
        cnt = 0
        self.history.append((cnt, score, state))
        while True:
            cur_time =time()
            if cur_time-TS>=TL:
                break
            cnt += 1
            k, v = state.get_neighbor()
            v_before = state.ans[k]
            state.change_belong(k, v)
            new_score = state.get_score()
            temp = start_temp + (end_temp-start_temp)*(cur_time-TS)/TL
            if math.exp((new_score-score)/temp) > random.random():
                score = new_score
                self.history.append((cnt, score, state.copy()))
            else:
                state.change_belong(k, v_before)
        return state

    def beam_search(self, TL=1.8, beam_width=10):
        return

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

state = State(None)
opt = Optimizer([state])
best_state = opt.climbing()
# best_state = opt.simanneal()
best_state.print()
# opt.visualize()
