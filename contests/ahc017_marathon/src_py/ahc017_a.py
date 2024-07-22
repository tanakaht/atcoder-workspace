import math
import random
from time import time
import sys
import heapq
from collections import defaultdict
import itertools
input = sys.stdin.readline

TS = time()
TL = 30 # 3600*3

# [a, b)
def randint(a, b):
    return int(random.random()*(b-a))+a

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @classmethod
    def mean(self, p1, p2):
        return Point((p1.x+p2.x)/2, (p1.y+p2.y)/2)

    def dist(self, point):
        return (self.x-point.x)*(self.x-point.x)+(self.y-point.y)*(self.y-point.y)

    def min_dist(self, points):
        if len(points)==0:
            return 100000000
        return min([self.dist(point) for point in points])


N, M, D, K = map(int, input().split())
UVW = []
for _ in range(M):
    u, v, w = map(int, input().split())
    UVW.append((u-1, v-1, w))
XY = [list(map(int, input().split())) for _ in range(N)]

G = [[] for _ in range(N)]
for i, (u, v, w) in enumerate(UVW):
    G[u].append((v, w, i))
    G[v].append((u, w, i))

depends_info = [None]*M
for edge_id, (u, v, w) in enumerate(UVW):
    start_node = u
    end_node = v
    q = [(0, start_node)]
    dists = defaultdict(lambda: 10000000000)
    dists[start_node] = 0
    appeared = defaultdict(lambda: False)
    while q:
        d, u = heapq.heappop(q)
        if u==end_node:
            break
        if appeared[u]:
            continue
        appeared[u] = True
        for v, c, i in G[u]:
            if i==edge_id:
                continue
            d_ = d+c
            if dists[v] > d_:
                dists[v] = d_
                heapq.heappush(q, (d_, v))
    # 経路復元
    depend = set()
    cur = end_node
    while cur!=start_node:
        for v, c, i in G[cur]:
            if dists[v]+c==dists[cur]:
                cur = v
                depend.add(i)
                break
    depends_info[edge_id] = (depend, dists[end_node])

depends_each = [set(depends_info[i][0]) for i in range(M)]
depends_pair = set()
for i in range(M):
    for j in depends_each[i]:
        depends_each[j].add(i)
        depends_pair.add((i, j))
        depends_pair.add((j, i))
points = [Point(x, y) for x, y in XY]
edge_center = [Point.mean(points[u], points[v]) for i, (u, v, w) in enumerate(UVW)]
edge_point = [(points[u], points[v]) for i, (u, v, w) in enumerate(UVW)]

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

# 連結前提
class State:
    def __init__(self, removed_edges):
        self.removed_edges = [set(x) for x in removed_edges]

    @classmethod
    def get_initial_state(cls, **kwargs):
        ret = cls.get_initial_state3()
        if ret is None:
            ret = cls.get_initial_state4()
        if ret is None:
            raise ValueError
            ret = cls.get_initial_state1()
        return ret

    @classmethod
    def get_initial_state1(cls, **kwargs):
        """
        各日連結
        """
        # 連結前提でできるだけ同じnodeから選ばないように初期解とる
        for max_cnt_for_same_node in range(1, 11):
            for K_ in range(math.ceil(M/D), K+1):
                removed = set()
                removed_edges = []
                for d in range(D):
                    daily_removed_edges = []
                    node_cnt = [0]*N
                    uf = UnionFind(N)
                    for i in removed_edges:
                        for u, v, w in UVW:
                            uf.union(u, v)
                    for i, (u, v, w) in random.sample(list(enumerate(UVW)), M):
                        if i in removed:
                            continue
                        if max(node_cnt[u], node_cnt[v])>=max_cnt_for_same_node:
                            continue
                        if len(daily_removed_edges)==K_:
                            break
                        if uf.find(u)==uf.find(v):
                            daily_removed_edges.append(i)
                            removed.add(i)
                            node_cnt[u] += 1
                            node_cnt[v] += 1
                        else:
                            uf.union(u, v)
                    removed_edges.append(daily_removed_edges)
                if len(removed)==M:
                    ret = State(removed_edges)
                    return ret

    @classmethod
    def get_initial_state2(cls, **kwargs):
        """
        各日連結かつできるだけ分散、かつ代替の最短経路を工事しない
        """
        for K_ in range(math.ceil(M/D), K+1):
            for max_cnt_for_same_node in range(1, 11):
                for _ in range(20):
                    removed = set()
                    removed_edges = []
                    for d in range(D):
                        daily_removed_edges = []
                        depend = set()
                        node_cnt = [0]*N
                        uf = UnionFind(N)
                        for i in removed_edges:
                            for u, v, w in UVW:
                                uf.union(u, v)
                        for i, (u, v, w) in random.sample(list(enumerate(UVW)), M):
                            if len(daily_removed_edges)==K_:
                                break
                            if i in removed:
                                uf.union(u, v)
                            elif max(node_cnt[u], node_cnt[v])>=max_cnt_for_same_node:
                                uf.union(u, v)
                            elif i in depend:
                                uf.union(u, v)
                            elif uf.find(u)==uf.find(v):
                                daily_removed_edges.append(i)
                                removed.add(i)
                                depend |= depends_info[i][0]
                                node_cnt[u] += 1
                                node_cnt[v] += 1
                            else:
                                uf.union(u, v)
                        removed_edges.append(daily_removed_edges)
                    if len(removed)==M:
                        ret = State(removed_edges)
                        return ret

    @classmethod
    def get_initial_state3(cls, **kwargs):
        for i in range(100):
            removed_edges = [set() for _ in range(D)]
            depends = [set() for _ in range(D)]
            node_removed_edge_cnt = [[0]*N for _ in range(D)]
            for i, (u, v, w) in random.sample(list(enumerate(UVW)), M):
                best_d, best_metrics = None, (math.inf, math.inf)
                for d in range(D):
                    if len(removed_edges[d])==K:
                        continue
                    depend_cnt = sum([((i, j) in depends_pair) for j in removed_edges[d]])
                    if depend_cnt!=0:
                        continue
                    node_edge_cnt = max(node_removed_edge_cnt[d][u], node_removed_edge_cnt[d][v])
                    metrics = (1+node_edge_cnt-2*(node_edge_cnt==1), len(removed_edges[d]))
                    if (best_metrics[0]>metrics[0]) or (best_metrics[0]==metrics[0] and best_metrics[1]>metrics[1]):
                        best_d, best_metrics = d, metrics
                if best_d is None:
                    break
                removed_edges[best_d].add(i)
                depends[best_d] |= depends_info[i][0]
                node_removed_edge_cnt[best_d][u] += 1
                node_removed_edge_cnt[best_d][v] += 1
            if sum([len(x) for x in removed_edges])==M:
                ret = State(removed_edges)
                return ret

    @classmethod
    def get_initial_state4(cls, **kwargs):
        removed_edges = [set() for _ in range(D)]
        depends = [set() for _ in range(D)]
        node_removed_edge_cnt = [[0]*N for _ in range(D)]
        for i, (u, v, w) in random.sample(list(enumerate(UVW)), M):
            best_d, best_metrics = None, (math.inf, math.inf, math.inf)
            for d in range(D):
                if len(removed_edges[d])==K:
                    continue
                depend_cnt = sum([((i, j) in depends_pair) for j in removed_edges[d]])
                node_edge_cnt = max(node_removed_edge_cnt[d][u], node_removed_edge_cnt[d][v])
                metrics = (depend_cnt, 1+node_edge_cnt-2*(node_edge_cnt==1), len(removed_edges[d]))
                # metrics = (depend_cnt, len(removed_edges[d]), max(node_removed_edge_cnt[d][u], node_removed_edge_cnt[d][v]))
                if best_metrics>metrics:
                    best_d, best_metrics = d, metrics
            if best_d is None:
                break
            removed_edges[best_d].add(i)
            depends[best_d] |= depends_info[i][0]
            node_removed_edge_cnt[best_d][u] += 1
            node_removed_edge_cnt[best_d][v] += 1
        if sum([len(x) for x in removed_edges])==M:
            ret = State(removed_edges)
            return ret

    def move(self, params):
        if params[0]=="swap":
            _, i, x, j, y = params
            self.removed_edges[i].remove(x)
            self.removed_edges[j].remove(y)
            self.removed_edges[i].add(y)
            self.removed_edges[j].add(x)
        elif params[0]=="move":
            _, i, x, j = params
            self.removed_edges[i].remove(x)
            self.removed_edges[j].add(x)
        else:
            raise NotImplementedError

    def is_connected(self, removed_edges):
        if not isinstance(removed_edges, set):
            removed_edges = set(removed_edges)
        uf = UnionFind(N)
        for i, (u, v, w) in enumerate(UVW):
            if i in removed_edges:
                continue
            uf.union(u, v)
        return len(uf.roots)==1

    def unmove(self, params):
        if params[0]=="swap":
            _, i, x, j, y = params
            self.move(("swap", i, y, j, x))
        elif params[0]=="move":
            _, i, x, j = params
            self.move(("move", j, x, i))

    def get_neighbor(self, *args):
        # swap, move
        percentages_ = [80, 20]
        percentages = [percentages_[0]]
        for x in percentages_[1:]:
            percentages.append(percentages[-1]+x)
        flg = randint(0, percentages[-1])
        # swap
        if flg<=percentages[0]:
            for _ in range(100):
                i, j = randint(0, D), randint(0, D)
                if i==j or min(len(self.removed_edges[i]), len(self.removed_edges[j]))==0:
                    continue
                x, y = random.choice(list(self.removed_edges[i])), random.choice(list(self.removed_edges[j]))
                return ("swap", i, x, j, y)
        elif flg<=percentages[1]:
            for _ in range(100):
                i, j = randint(0, D), randint(0, D)
                if i==j or len(self.removed_edges[i])==0 or len(self.removed_edges[j])==K:
                    continue
                if len(self.removed_edges[i])<len(self.removed_edges[j]) and randint(0, 100)>=20:
                    continue
                x = random.choice(list(self.removed_edges[i]))
                return ("move", i, x, j)

    def get_neighbor_with_score_p1(self, *args):
        params = self.get_neighbor()
        return params, self.get_simple_score_diff(params)

    def get_neighbor_with_score(self, *args):
        params = self.get_neighbor()
        if params[0]=="swap":
            _, i, x, j, y = params
            # sampling_point = set([UVW[x][0], UVW[x][1], UVW[y][1], UVW[y][1]]+[randint(0, N) for _ in range(16)])
            sampling_point = set([randint(0, N) for _ in range(20)])
            # sampling_point = set(range(N))
            pre_score_i = self.get_score_with_sampleing_point(self.removed_edges[i], sampling_point)
            pre_score_j = self.get_score_with_sampleing_point(self.removed_edges[j], sampling_point)
            aft_score_i = self.get_score_with_sampleing_point((self.removed_edges[i]-set([x]))|set([y]), sampling_point)
            aft_score_j = self.get_score_with_sampleing_point((self.removed_edges[j]-set([y]))|set([x]), sampling_point)
            score_diff = aft_score_i+aft_score_j-(pre_score_i+pre_score_j)
        elif params[0]=="move":
            _, i, x, j = params
            # sampling_point = set([UVW[x][0], UVW[x][1]]+[randint(0, N) for _ in range(18)])
            sampling_point = set([randint(0, N) for _ in range(20)])
            # sampling_point = set(range(N))
            pre_score_i = self.get_score_with_sampleing_point(self.removed_edges[i], sampling_point)
            pre_score_j = self.get_score_with_sampleing_point(self.removed_edges[j], sampling_point)
            aft_score_i = self.get_score_with_sampleing_point((self.removed_edges[i]-set([x])), sampling_point)
            aft_score_j = self.get_score_with_sampleing_point(self.removed_edges[j]|set([x]), sampling_point)
            score_diff = aft_score_i+aft_score_j-(pre_score_i+pre_score_j)
        else:
            raise NotImplementedError
        return params, score_diff
        pre_score_i = self.get_daily_score(params[1])
        pre_score_j = self.get_daily_score(params[3])
        self.move(params)
        aft_score_i = self.get_daily_score(params[1])
        aft_score_j = self.get_daily_score(params[3])
        self.unmove(params)
        score_diff = (aft_score_i+aft_score_j)-(pre_score_i+pre_score_j)
        return params, score_diff

    def get_simple_score_diff___(self, params):
        if params[0]=="swap":
            _, i, x, j, y = params
            pre_depend_cnt_i = sum([(x, f) in depends_pair for f in self.removed_edges[i] if f!=x])
            pre_dist_i = edge_center[x].min_dist([edge_center[f] for f in self.removed_edges[i] if f!=x])
            pre_depend_cnt_j = sum([(y, f) in depends_pair for f in self.removed_edges[j] if f!=y])
            pre_dist_j = edge_center[y].min_dist([edge_center[f] for f in self.removed_edges[j] if f!=y])
            pre_score = 1000000*(pre_depend_cnt_i+pre_depend_cnt_j)-min(pre_dist_i, pre_dist_j)

            aft_depend_cnt_i = sum([(y, f) in depends_pair for f in self.removed_edges[i] if f!=x])
            aft_dist_i = edge_center[y].min_dist([edge_center[f] for f in self.removed_edges[i] if f!=x])
            aft_depend_cnt_j = sum([(x, f) in depends_pair for f in self.removed_edges[j] if f!=y])
            aft_dist_j = edge_center[x].min_dist([edge_center[f] for f in self.removed_edges[j] if f!=y])
            aft_score = 1000000*(aft_depend_cnt_i+aft_depend_cnt_j)-min(aft_dist_i, aft_dist_j)
            diff = aft_score-pre_score
            return diff
        elif params[0]=="move":
            _, i, x, j = params
            pre_depend_cnt_i = sum([(x, f) in depends_pair for f in self.removed_edges[i] if f!=x])
            pre_dist_i = edge_center[x].min_dist([edge_center[f] for f in self.removed_edges[i] if f!=x])
            pre_score = 1000000*(pre_depend_cnt_i)-pre_dist_i
            aft_depend_cnt_j = sum([(x, f) in depends_pair for f in self.removed_edges[j]])
            aft_dist_j = edge_center[x].min_dist([edge_center[f] for f in self.removed_edges[j]])
            aft_score = 1000000*(aft_depend_cnt_j)-aft_dist_j
            diff = aft_score-pre_score
            return diff
        else:
            raise NotImplementedError

    def get_simple_score_diff(self, params):
        if params[0]=="swap":
            _, i, x, j, y = params
            edge_point_i = [point for f in self.removed_edges[i] if f!=x for point in  edge_point[f]]
            edge_point_j = [point for f in self.removed_edges[j] if f!=y for point in  edge_point[f]]
            pre_depend_cnt_i = sum([(x, f) in depends_pair for f in self.removed_edges[i] if f!=x])
            pre_dist_i = min(edge_point[x][0].min_dist(edge_point_i), edge_point[x][1].min_dist(edge_point_i))
            pre_depend_cnt_j = sum([(y, f) in depends_pair for f in self.removed_edges[j] if f!=y])
            pre_dist_j = min(edge_point[y][0].min_dist(edge_point_j), edge_point[y][1].min_dist(edge_point_j))
            pre_score = 1000000*(pre_depend_cnt_i+pre_depend_cnt_j)-min(pre_dist_i, pre_dist_j)

            aft_depend_cnt_i = sum([(y, f) in depends_pair for f in self.removed_edges[i] if f!=x])
            aft_dist_i = min(edge_point[y][0].min_dist(edge_point_i), edge_point[y][1].min_dist(edge_point_i))
            aft_depend_cnt_j = sum([(x, f) in depends_pair for f in self.removed_edges[j] if f!=y])
            aft_dist_j = min(edge_point[x][0].min_dist(edge_point_j), edge_point[x][1].min_dist(edge_point_j))
            aft_score = 1000000*(aft_depend_cnt_i+aft_depend_cnt_j)-min(aft_dist_i, aft_dist_j)
            diff = aft_score-pre_score
            return diff
        elif params[0]=="move":
            _, i, x, j = params
            edge_point_i = [point for f in self.removed_edges[i] if f!=x for point in  edge_point[f]]
            edge_point_j = [point for f in self.removed_edges[j] for point in  edge_point[f]]
            pre_depend_cnt_i = sum([(x, f) in depends_pair for f in self.removed_edges[i] if f!=x])
            pre_dist_i = min(edge_point[x][0].min_dist(edge_point_i), edge_point[x][1].min_dist(edge_point_i))
            pre_score = 1000000*(pre_depend_cnt_i)-pre_dist_i
            aft_depend_cnt_j = sum([(x, f) in depends_pair for f in self.removed_edges[j]])
            aft_dist_j = min(edge_point[x][0].min_dist(edge_point_j), edge_point[x][1].min_dist(edge_point_j))
            aft_score = 1000000*(aft_depend_cnt_j)-aft_dist_j
            diff = aft_score-pre_score
            return diff
        else:
            raise NotImplementedError

    def get_score_with_sampleing_point(self, removed_edge, sampleing_point):
        score = 0
        for start_node in sampleing_point:
            q = [(0, start_node)]
            dists = [10000000000]*N
            dists[start_node] = 0
            appeared = [False]*N
            while q:
                d, u = heapq.heappop(q)
                if appeared[u]:
                    continue
                appeared[u] = True
                for v, c, i in G[u]:
                    if i in removed_edge:
                        continue
                    d_ = d+c
                    if dists[v] > d_:
                        dists[v] = d_
                        heapq.heappush(q, (d_, v))
            score += sum(dists)
        return score

    def get_simple_score(self, removed_edges, target_uv_limit):
        if not isinstance(removed_edges, set):
            removed_edges = set(removed_edges)
        # if not self.is_connected(removed_edges):
          #   return 100000000000000
        score = 0
        for start_node, end_node, limit in target_uv_limit:
            q = [(0, start_node)]
            dists = defaultdict(lambda: 10000000000)
            dists[start_node] = 0
            appeared = defaultdict(lambda: False)
            cnt = 0
            # 近所の　上位Xに入らなければアウト
            while q:
                d, u = heapq.heappop(q)
                if u==end_node:# or d>limit:
                    break
                if appeared[u]:
                    continue
                appeared[u] = True
                for v, c, i in G[u]:
                    if i in removed_edges:
                        continue
                    d_ = d+c
                    if dists[v] > d_:
                        dists[v] = d_
                        heapq.heappush(q, (d_, v))
            score += dists[end_node]
        return score

    def get_daily_score(self, day: int):
        daily_removed_edge = self.removed_edges[day]
        score = 0
        for start_node in range(N):
            q = [(0, start_node)]
            dists = [10000000000]*N
            dists[start_node] = 0
            appeared = [False]*N
            while q:
                d, u = heapq.heappop(q)
                if appeared[u]:
                    continue
                appeared[u] = True
                for v, c, i in G[u]:
                    if i in daily_removed_edge:
                        continue
                    d_ = d+c
                    if dists[v] > d_:
                        dists[v] = d_
                        heapq.heappush(q, (d_, v))
            score += sum(dists)
        return score

    def get_score(self, **params):
        return round(sum([self.get_daily_score(day) for day in range(D)])/N/(N-1)/D*1000)

    def print(self):
        ans = [None]*M
        for d, daily_removed_edge in enumerate(self.removed_edges):
            for i in daily_removed_edge:
                ans[i] = d+1
        print(*ans)

    def copy(self):
        return State(self.removed_edges)


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
            if new_score >= score:
                state.move(params)
                score = new_score
                if save_history:
                    self.history.append((cnt, score, state.copy()))
        return state

    def simanneal(self, initial_state, TL_=TL):
        TS = time()
        state = initial_state.copy()
        score = 0
        start_temp = 100000
        end_temp = 0.1
        cnt = 0
        cnt2 = 0
        last_log = time()
        last_log_score = 0
        # print("score: ", state.get_score()-489885287, int((time()-TS)*100))
        while True:
            cur_time =time()
            if cur_time-TS>=TL_:
                break
            cnt += 1
            params, score_diff = state.get_neighbor_with_score()
            new_score = score+score_diff
            temp = start_temp + (end_temp-start_temp)*(cur_time-TS)/TL
            if (new_score<=score):# or math.exp((score-new_score)/temp) > random.random()):
                state.move(params)
                score = new_score
                # print(int((time()-TS)*100), score_diff, params[0])
                # print(int((time()-TS)*100), score_diff, score, new_score, params)
                # state.print()
                cnt2 += 1
            #if time()-last_log>180:
            #    # state.print()
            #    print("score: ", state.get_score()-489885287, int((time()-TS)*100), cnt2/cnt, cnt2, cnt)
            #    last_log = time()
            #    last_log_score = score
        print(cnt, cnt2)
        self.cnt = cnt
        return state

    def simanneal_p1(self, initial_state, TL_=TL):
        TS = time()
        state = initial_state.copy()
        score = 0
        start_temp = 100000
        end_temp = 0.1
        cnt = 0
        while True:
            cur_time =time()
            if cur_time-TS>=TL_:
                break
            cnt += 1
            params, score_diff = state.get_neighbor_with_score_p1()
            new_score = score+score_diff
            temp = start_temp + (end_temp-start_temp)*(cur_time-TS)/TL
            if (new_score<=score or math.exp((score-new_score)/temp) > random.random()):
                state.move(params)
                score = new_score
                # state.print()
        # print(cnt)
        self.cnt = cnt
        return state

    def beam_search(self, TL=1.8, beam_width=10):
        raise NotImplementedError

    def visualize(self):
        raise NotImplementedError

init_state = State.get_initial_state()
# init_state.print()
opt = Optimizer()
# best_state = opt.climbing(init_state)
best_state = opt.simanneal_p1(init_state, TL_=1)
best_state = opt.simanneal(best_state, TL_=TL-(time()-TS))
best_state.print()
# print(len(opt.history), opt.cnt, [x[1] for x in opt.history])
# opt.visualize()
