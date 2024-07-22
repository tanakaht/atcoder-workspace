import math
import heapq
from ntpath import join
import sys
from collections import defaultdict, deque
from typing import List, Optional, Tuple, Union, Set
from time import time
import random


ts = time()
char2dir = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1),
            "u": (-1, 0), "d": (1, 0), "l": (0, -1), "r": (0, 1)}
dir2char = {(-1, 0): "U", (1, 0): "D", (0, -1): "L", (0, 1): "R"}
# logfile = open("../dev/log", "w")

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

# https://smijake3.hatenablog.com/entry/2019/08/24/164727 の拾い物を利用
# TODO: 30*30*4*300=10^6にしては遅い気がする?
def get_joints(g: List[Set[int]], available: List[bool]) -> Set[int]:
    # availableなものだけでグラフ再構成
    newid2oldid = {}
    oldid2newid = {}
    cnt = 0
    for i in range(len(available)):
        if available[i]:
            newid2oldid[cnt] = i
            oldid2newid[i] = cnt
            cnt += 1
    G = [[] for _ in range(cnt)]
    N = len(G)
    for i in range(N):
        for j_old in g[newid2oldid[i]]:
            j = oldid2newid[j_old]
            G[i].append(j)
            G[j].append(i)
    P = [0]*N
    G0 = [[] for i in range(N)]
    V = []
    lb = [0]*N
    def dfs(v, p):
        P[v] = p
        V.append(v)
        lb[v] = len(V)
        for w in G[v]:
            if w == p:
                continue
            if lb[w]:
                if lb[v] < lb[w]:
                    G0[v].append(w)
                continue
            dfs(w, v)
    dfs(0, -1)
    B = []
    ap = [0]*N
    used = [0]*N
    first = 1
    used[0] = 1
    for u in V:
        if not used[u]:
            p = P[u]
            B.append((u, p) if u < p else (p, u))
            if len(G[u]) > 1:
                ap[u] = 1
            if len(G[p]) > 1:
                ap[p] = 1
        cycle = 0
        for v in G0[u]:
            w = v
            while w != u and not used[w]:
                used[w] = 1
                w = P[w]
            if w == u:
                cycle = 1
        if cycle:
            if not first:
                ap[u] = 1
            first = 0
    joints = set([newid2oldid[v] for v in range(N) if ap[v]])
    return joints

class Room:
    def __init__(self):
        self.R: List[List[str]] = [["."]*30 for _ in range(30)]

    def __getitem__(self, i: int) -> List[str]:
        return self.R[i]

    def __eq__(self, other):
        if type(other) != Room:
            return False
        for x in range(30):
             for y in range(30):
                 if self[x][y] != other[x][y]:
                     return False
        return True

    def close(self, x: int, y: int) -> None:
        self.R[x][y] = "#"

    # 900くらい
    def dist(self, x: int, y: int) -> List[List[int]]:
        ret = [[math.inf]*30 for _ in range(30)]
        ret[x][y] = 0
        appeared = [[False]*30 for _ in range(30)]
        appeared[x][y] = True
        q = deque([(x, y)])
        while q:
            x, y = q.popleft()
            for x_, y_ in [(x+1, y), (x, y+1), (x-1, y), (x, y-1)]:
                if not (0<=x_<30 and 0<= y_<30):
                    continue
                if self.R[x_][y_] == "#":
                    continue
                if not appeared[x_][y_]:
                    ret[x_][y_] = ret[x][y]+1
                    q.append((x_, y_))
                    appeared[x_][y_] = True
        return ret

def Rnogamene() -> Room:
    ret = Room()
    ret.R = [list(s) for s in """
..............................
..............................
..............................
..............................
..............................
..............................
..............................
..............................
..............................
..............................
..............................
..............................
..............................
..............................
..............................
..............................
..............................
..............................
..............................
..............................
..............................
..............................
..............................
..............................
..............................
..............................
..............................
..............................
..............................
..............................""".split("\n")[1:]]
    return ret



def R0(game) -> Room:
    ret = Room()
    n, m = len(game.humans)//2, len(game.humans)-len(game.humans)//2
    ret.R = [list(s) for s in f"""
...{n}.......................{m}..
..###########################.
..............................
..............................
..............................
..............................
..............................
..............................
..............................
..............................
..............................
..............................
..............................
..............................
..............................
..............................
..............................
..............................
..............................
..............................
..............................
..............................
..............................
..............................
..............................
..............................
..............................
..............................
..............................
..............................""".split("\n")[1:]]
    return ret

def R2(game) -> Room:
    ret = Room()
    ret.R = [list(s) for s in """
....#.....................#...
..###########################.
..............................
##############...#############
..............#.#.............
..............................
##############...#############
..............#.#.............
..............................
##############...#############
..............#.#.............
..............................
##############...#############
..............#.#.............
..............................
##############...#############
..............#.#.............
..............................
##############...#############
..............#.#.............
..............................
##############...#############
..............#.#.............
..............................
##############...#############
..............#.#.............
..............................
##############...#############
..............................
..............###.............""".split("\n")[1:]]
    points = [15, 10, 20, 5, 25, 12, 17, 22, 7, 15]
    for i in range(len(game.humans)):
        x, y = points[i], points[i]
        if ret[x][15] == ".":
            ret[x][15] = "1"
        else:
            ret[x][15] = str(int(ret[x][15])+1)
    return ret

def R4(game) -> Room:
    ret = Room()
    ret.R = [list(s) for s in """
..............#.#.............
..............###.............
..............................
##############...#############
..............#.#.............
..............................
##############...#############
..............#.#.............
..............................
##############...#############
..............#.#.............
..............................
##############...#############
..............#.#.............
..............................
##############...#############
..............#.#.............
..............................
##############...#############
..............#.#.............
..............................
##############...#############
..............#.#.............
..............................
##############...#############
..............#.#.............
..............................
##############...#############
..............................
..............###.............""".split("\n")[1:]]
    points = [15, 10, 20, 5, 25, 12, 17, 22, 7, 15]
    for i in range(len(game.humans)):
        x, y = points[i], points[i]
        if ret[x][15] == ".":
            ret[x][15] = "1"
        else:
            ret[x][15] = str(int(ret[x][15])+1)
    return ret


class Operation:
    def __init__(self, op: Optional[str], arg: Optional[Tuple[int, int]], is_target_xy: bool=False):
        self.op = op
        if arg is not None:
            self.x, self.y = arg
            if is_target_xy and self.op in "udlr":
                dx, dy = char2dir[self.op]
                self.x -= dx
                self.y -= dy
        else:
            self.x, self.y = None, None

    def target(self) -> Optional[Tuple[int, int]]:
        if self.op in "udlrUDLR":
            dx, dy = char2dir[self.op]
            return (self.x+dx, self.y+dy)
        else:
            return (self.x, self.y)

    def is_done(self, human, R: Room) -> bool:
        x, y = self.target()
        if self.op in "udlrUDLR":
            return R[x][y] == "#"
        elif self.op == "to":
            return human.x==x and human.y==y
        else:
            return True

class Task:
    def __init__(self, ops: List[Operation]):
        self.ops = ops
        self.i = 0
        self.done = False

    def cur_op(self, human, R: Room) -> Operation:
        while self.i<len(self.ops):
            if self.ops[self.i].is_done(human, R):
                self.i += 1
            else:
                return self.ops[self.i]
        if self.i==len(self.ops):
            self.done = True
            return Operation(None, None)

class Living:
    def __init__(self, R: Room, x: int, y: int, id_: int):
        self.R = R
        self.x = x
        self.y = y
        self.id_ = id_

    def move(self, c: str) -> None:
        if c==".":
            return
        xd, yd = char2dir[c]
        x_, y_ = self.x+xd, self.y+yd
        if c in "UDLR":
            self.x, self.y = x_, y_
        elif c in "udrl":
            self.R.close(x_, y_)

class Pet(Living):
    def __init__(self, R: Room, x: int, y: int, pet_type: int, id_: int):
        super(Pet, self).__init__(R, x, y, id_)
        self.pet_type = pet_type
        self.is_valid = True

class Human(Living):
    def __init__(self, R: Room, x: int, y: int, id_: int):
        super(Human, self).__init__(R, x, y, id_)
        self.task = Task([])
        self.cut_in_op = None
        self.is_free = False

    def next_cmd(self) -> str:
        if self.cut_in_op is not None:
            op = self.cut_in_op
        else:
            op = self.task.cur_op(self, self.R)
        return self.op2cmd(op)

    def op2cmd(self, op: Operation) -> str:
        if op.op is None:
            return "."
        if not (self.x==op.x and self.y==op.y):
            dists = self.R.dist(op.x, op.y)
            for d in "UDLR":
                dx, dy = char2dir[d]
                x_, y_ = self.x+dx, self.y+dy
                if 0<=x_<30 and 0<=y_<30 and dists[self.x][self.y] > dists[x_][y_]:
                    return d
            return "." # Errorだけど...
        else:
            if op.op == "to":
                return "."
            else:
                return op.op

class Game:
    def __init__(self, R: Room, pets: List[Pet], humans: List[Human], basepoint: Tuple[int]=(15, 15)):
        self.turn = 0
        self.R = R
        self.pets = pets
        self.humans = humans
        self.basepoint = basepoint # 最終的に人はこのますにいける
        self.pre_ideal_R = None
        self.humans_tasks = [[(None, None)] for _ in range(len(humans))]
        self.stage = 4-4*(sum([pet.pet_type==4 for pet in self.pets])!=0)
        self.init_assigned = False

    def can_close(self, x, y, include_human=True):
        if self.R[x][y] == "#":
            return False
        for pet in self.pets:
            if abs(x-pet.x)+abs(y-pet.y) <= 1:
                return False
        if include_human:
            for human in self.humans:
                if abs(x-human.x)+abs(y-human.y) == 0:
                    return False
        return True

    def set_basepoint(self, x: int, y: int) -> None:
        if x==self.basepoint[0] and y==self.basepoint[1]:
            return
        self.basepoint = (x, y)

    # 関節点以外はまとめて、basepointを根としたツリーを作って関節点ごとに(x, y, 空間の減少量, ペットの閉じ込め量, 人の閉じ込め量)を返す
    def joints_info(self, room: Room) -> List[Tuple[int, int, int, int, int]]:
        g = [set() for _ in range(30*30)]
        for x in range(30):
            for y in range(30):
                if room[x][y]=="#":
                    continue
                for d in "DR":
                    dx, dy = char2dir[d]
                    x_, y_ = x+dx, y+dy
                    if 0<=x_<30 and 0<=y_<30 and room[x_][y_]!="#":
                        g[x*30+y].add(x_*30+y_)
                        g[x_*30+y_].add(x*30+y)
        # basepointからいける点だけ扱う
        available = [False]*(30*30)
        q = [self.basepoint[0]*30+self.basepoint[1]]
        while q:
            u = q.pop()
            for v in g[u]:
                if not available[v]:
                    q.append(v)
                    available[v] = True
        # 関節点求める
        joints = get_joints(g, available)
        for i in list(joints):
            x, y = i//30, i%30
            # 通り道は省く
            if y==15:
                joints.remove(i)
        if len(joints) == 0:
            return []
        # まとめる
        uf = UnionFind(30*30+1)
        for i in range(900):
            if not available[i]:
                uf.union(30*30, i)
            elif i in joints:
                pass
            else:
                for j in g[i]:
                    if j not in joints:
                        uf.union(i, j)
        # グラフ再構築
        roots = set(uf.roots())
        roots.remove(uf.find(30*30))
        root2id = {k: v for v, k in enumerate(roots)}
        g2 = [set() for _ in range(len(root2id))]
        for i in joints:
            i_id = root2id[uf.find(i)]
            for j in g[i]:
                j_id = root2id[uf.find(j)]
                g2[i_id].add(j_id)
                g2[j_id].add(i_id)
        # ツリー構築
        children = [[] for _ in range(len(root2id))]
        parents = [None]*len(root2id)
        root_id = root2id[uf.find(self.basepoint[0]*30+self.basepoint[1])]
        q = [(root_id, None)]
        dfs_ord = []
        while len(q) > 0:
            u, p = q.pop()
            dfs_ord.append(u)
            parents[u] = p
            for v in g2[u]:
                if v != p:
                    q.append((v, u))
                    children[u].append(v)
        # 情報集計
        id2info = {i: [0, set(), 0] for i in range(len(root2id))}
        for root, mem in uf.all_group_members().items():
            try:
                id2info[root][0] = len(mem)
            except KeyError:
                pass
        for pet in self.pets:
            i = pet.x*30+pet.y
            try:
                id2info[root2id[uf.find(i)]][1].add(pet.id_)
            except KeyError:
                pass
        for human in self.humans:
            i = human.x*30+human.y
            try:
                id2info[root2id[uf.find(i)]][2] += 1
            except KeyError:
                pass
        # 木dp
        dp = [None]*len(root2id)
        def _dfs(u):
            if dp[u] is not None:
                return dp[u]
            ret = [x for x in id2info[u]]
            for v in children[u]:
                res = _dfs(v)
                ret[0] += res[0]
                ret[1] = ret[1] | res[1]
                ret[2] += res[2]
            dp[u] = ret
            return ret
        dp2 = [None]*len(root2id)
        def _dfs2(u):
            if dp2[u] is not None:
                return dp[u]
            ret = _dfs(u)
            ret[1] = ret[1] - id2info[u][1]
            for v in children[u]:
                ret[1] = ret[1] - id2info[v][1]
            dp2[u] = ret
            return ret
        for u in dfs_ord[::-1]:
            _dfs(u)
        ret = []
        joint2id = {i: root2id[uf.find(i)] for i in joints}
        id2joint = {k: v for v, k in joint2id.items()}
        jointids = set([joint2id[i] for i in joints])
        dfs_ord_joints = []
        for i_id in dfs_ord[::-1]:
            if i_id in jointids:
                i = id2joint[i_id]
                i_id = root2id[uf.find(i)]
                ret.append((i//30, i%30, *_dfs2(i_id)))
        return ret

    def satisfy(self, room: Room, include_decimal: bool=True) -> bool:
        for x in range(30):
            for y in range(30):
                if room[x][y]=="#" and self.R[x][y]!="#":
                    return False
                elif include_decimal and room[x][y].isdecimal():
                    if int(room[x][y])>sum([(x==human.x and y==human.y) for human in self.humans]):
                        return False
        return True

    def create_ideal_R(self, joints_info) -> Room:
        if self.stage == 0:
            return R0(self)
        elif self.stage == 1:
            cnt = sum([pet.pet_type==4 for pet in self.pets])
            for pet in self.pets:
                cnt -= (pet.pet_type==4 and pet.x==0 and 5<=pet.y<=25)
            if cnt==0 and self.can_close(0, 4) and self.can_close(0, 26):
                ret = R0(self)
                ret[0][4] = "#"
                ret[0][26] = "#"
                return ret
            else:
                return R0(self)
        elif self.stage == 2:
            return R2(self)
        elif self.stage == 3: #
            ret = R2(self)
            for x in range(30):
                for y in range(30):
                    if self.R[x][y] == "#":
                        ret[x][y] = self.R[x][y]
            closed_pet = set()
            petid2xy = {}
            for i, (x_, y_, n_close_room, s_close_pet, n_close_hum) in enumerate(joints_info):
                s_close_pet_diff = s_close_pet - closed_pet
                if len(s_close_pet_diff) > 0:
                    ret[x_][y_] = "#"
                    for pet_id in (closed_pet & s_close_pet):
                        x, y = petid2xy[pet_id]
                        ret[x][y] = "."
                    closed_pet = closed_pet | s_close_pet
                    for pet_id in s_close_pet:
                        petid2xy[pet_id] = (x_, y_)
            return ret
        elif self.stage == 4:
            return R4(self)
        elif self.stage == 5:
            ret = R4(self)
            for x in range(30):
                for y in range(30):
                    if self.R[x][y] == "#":
                        ret[x][y] = self.R[x][y]
            closed_pet = set()
            petid2xy = {}
            for i, (x_, y_, n_close_room, s_close_pet, n_close_hum) in enumerate(joints_info):
                s_close_pet_diff = s_close_pet - closed_pet
                if len(s_close_pet_diff) > 0:
                    ret[x_][y_] = "#"
                    for pet_id in (closed_pet & s_close_pet):
                        x, y = petid2xy[pet_id]
                        ret[x][y] = "."
                    closed_pet = closed_pet | s_close_pet
                    for pet_id in s_close_pet:
                        petid2xy[pet_id] = (x_, y_)
            return ret

    def update_stage(self):
        if self.stage==0:
            room = R0(self)
            if self.satisfy(room):
                self.stage += 1
                self.init_assigned = False
                for human in self.humans:
                    human.is_free=False
        if self.stage == 1:
            room = R0(self)
            room[0][4] = "#"
            room[0][26] = "#"
            if self.satisfy(room):
                self.stage += 1
                self.init_assigned = False
                for human in self.humans:
                    human.is_free=False
        if self.stage == 2:
            if self.satisfy(R2(self), include_decimal=False):
                self.stage += 1
                self.init_assigned = False
                for human in self.humans:
                    human.is_free=False
        if self.stage == 3:
            pass
        if self.stage == 4:
            if self.satisfy(R4(self), include_decimal=False):
                self.stage += 1
                self.init_assigned = False
                for human in self.humans:
                    human.is_free=False
        if self.stage == 5:
            pass

    def get_targets(self, ideal_R: Room) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        close_target = []
        to_target = []
        for x in range(30):
            for y in range(30):
                if ideal_R[x][y] == "#" and self.R[x][y] != "#":
                    close_target.append((x, y))
                elif ideal_R[x][y].isdecimal():
                    for _ in range(int(ideal_R[x][y])):
                        to_target.append((x, y))
        return close_target, to_target

    def enum_turn(self, human: Human, dists: List[List[Optional[List[List[int]]]]], close_targets: Tuple[int, int], to_target: Optional[Tuple[int, int]]) -> Tuple[int, List[Operation]]:
        cur_x, cur_y = human.x, human.y
        n_turn = 0
        ops = []
        for i in range(len(close_targets)):
            x, y = close_targets[i]
            x_, y_ = close_targets[min(i+1, len(close_targets)-1)]
            best_d, best_d_, best_absxy_, best_op = math.inf, math.inf, math.inf, None
            for d in "udlr":
                op = Operation(d, (x, y), is_target_xy=True)
                if not (0<=op.x<30 and 0<= op.y<30):
                    continue
                for d_ in "udlr":
                    op_ = Operation(d_, (x_, y_), is_target_xy=True)
                    if not (0<=op_.x<30 and 0<= op_.y<30):
                        continue
                    cur_d = dists[cur_x][cur_y][op.x][op.y]
                    cur_d_ = cur_d + dists[op.x][op.y][op_.x][op_.y]
                    if cur_d_ < best_d_ or (cur_d_==best_d_ and abs(op.x-op.y)<best_absxy_):
                        best_d = cur_d
                        best_d_ = cur_d_
                        best_absxy_ = abs(op.x-op.y)
                        best_op = op
            n_turn += best_d + 1
            ops.append(best_op)
            cur_x, cur_y = best_op.x, best_op.y
        if to_target is not None:
            op = Operation("to", to_target)
            n_turn += dists[cur_x][cur_y][op.x][op.y]
            ops.append(op)
        return n_turn, ops

    # 一手でできる改善があればアサイン
    def cut_in_assign(self, ideal_R: Room, joints_info: List[Tuple]):
        assgined = [False]*len(joints_info)
        for human in self.humans:
            x, y = human.x, human.y
            for i, (x_, y_, n_close_room, s_close_pet, n_close_hum) in enumerate(joints_info):
                if (not assgined[i]) and n_close_hum==0 and len(s_close_pet) > 0 and (abs(x-x_)+abs(y-y_)==1) and self.can_close(x_, y_):
                    assgined[i] = True
#                     print(x_, y_, s_close_pet, file=sys.stderr)
                    for d in "udlr":
                        dx, dy = char2dir[d]
                        if x+dx==x_ and y+dy==y_:
                            human.cut_in_op = Operation(d, (x_, y_), is_target_xy=True)
                    break
        return

    def assign(self, ideal_R: Room) -> None:
        m = len(self.humans)
        available = [[d!=math.inf for d in line] for line in ideal_R.dist(self.basepoint[0], self.basepoint[1])]
        close_target, to_target = self.get_targets(ideal_R)
        tasks = [[] for _ in range(len(self.humans))]
        dists = [[None]*30 for _ in range(30)]
        tmp_xy = [(human.x, human.y) for human in self.humans] + to_target
        for x, y in close_target:
            for d in "udlr":
                op = Operation(d, (x, y), is_target_xy=True)
                x_, y_ = op.x, op.y
                if (0<=x_<30 and 0<= y_<30):
                    tmp_xy.append((x_, y_))
        for x, y in tmp_xy:
            if dists[x][y] is None:
                dists[x][y] = ideal_R.dist(x, y)
        humans_close_target = [[] for _ in range(len(self.humans))]
        humans_to_target = [None for _ in range(len(self.humans))]
        # 初期解, 貪欲
        """
        is_done = [False]*len(close_target)
        cur_dist = [dists[human.x][human.y] for human in self.humans]
        # ランダムアサイン
        while sum(is_done) < len(close_target):
            break
            hum_id = random.randint(0, len(self.humans)-1)
            task_id = random.choice([x for x in range(len(is_done)) if not is_done[x]])
            humans_close_target[hum_id].append(close_target[task_id])
            is_done[task_id] = True
        hum_turn = [0]*len(self.humans)
        # 最短一個を貪欲assignZ
        while sum(is_done) < len(close_target):
            best_d, best_pair = math.inf, None
            for hum_id in range(len(self.humans)):
                for task_id in range(len(close_target)):
                    if is_done[task_id]:
                        continue
                    for d in "udlr":
                        op = Operation(d, close_target[task_id], is_target_xy=True)
                        if (0<=op.x<30 and 0<= op.y<30) and cur_dist[hum_id][op.x][op.y]+hum_turn[hum_id] < best_d:
                            best_d = cur_dist[hum_id][op.x][op.y]+hum_turn[hum_id]+1
                            best_pair = (hum_id, task_id, op)
            hum_id, task_id, op = best_pair
            humans_close_target[hum_id].append(close_target[task_id])
            is_done[task_id] = True
            cur_dist[hum_id] = dists[op.x][op.y]
            hum_turn[hum_id] = best_d
        """
        # x//6でグループ分け
        groups = defaultdict(list)
        for x, y in close_target:
            groups[x//6+100*(y>15)].append((x, y))
        for k in groups.keys():
            if k>=100:
                groups[k] = sorted(groups[k], key=lambda x: (x[0], -x[1]))
            else:
                groups[k] = sorted(groups[k], key=lambda x: (x[0], x[1]))
            groups[k] = groups[k][:len(groups[k])//2][::-1] + groups[k][len(groups[k])//2:]
        is_done = {k: False for k in groups.keys()}
        cur_dist = [dists[human.x][human.y] for human in self.humans]
        hum_turn = [0]*len(self.humans)
        # グループごとに最短一個を貪欲assign
        while sum(is_done.values()) < len(groups):
            best_d, best_pair = math.inf, None
            for hum_id in range(len(self.humans)):
                for g_id in groups.keys():
                    if is_done[g_id]:
                        continue
                    targets = groups[g_id]
                    if len(targets) ==1:
                        op = Operation("to", (targets[0][0], targets[0][1]))
                        op_ = Operation("to", (targets[0][0], targets[0][1]))
                    else:
                        op = Operation("to", ((targets[0][0]+targets[1][0])//2, (targets[0][1]+targets[1][1])//2))
                        op_ = Operation("to", ((targets[-1][0]+targets[-2][0])//2, (targets[-1][1]+targets[-2][1])//2))
                    # 2グループ入ってないとバグるので
                    if not available[op.x][op.y]:
                        for dir in "udlr":
                            tmpop = Operation(dir, (x, y), is_target_xy=True)
                            if (0<=tmpop.x<30 and 0<= tmpop.y<30) and available[tmpop.x][tmpop.y]:
                                op = tmpop
                    if not available[op_.x][op_.y]:
                        for dir in "udlr":
                            tmpop_ = Operation(dir, (x, y), is_target_xy=True)
                            if (0<=tmpop_.x<30 and 0<= tmpop_.y<30) and available[tmpop_.x][tmpop_.y]:
                                op_ = tmpop_
                    if cur_dist[hum_id][op.x][op.y]+hum_turn[hum_id] < best_d:
                        best_d = cur_dist[hum_id][op.x][op.y]+hum_turn[hum_id]+1
                        best_pair = (hum_id, g_id, op, op_)
            hum_id, g_id, op, op_ = best_pair
            humans_close_target[hum_id] += groups[g_id]
            is_done[g_id] = True
            cur_dist[hum_id] = dists[op_.x][op_.y]
            hum_turn[hum_id] = best_d+2*len(groups[g_id])
        #  TODO: 山登りで改善
        if self.stage==0:
            m = len(self.humans)
            cur_turn = [self.enum_turn(human, dists, close_target, None)[0] for human, close_target, to_target in zip(self.humans, humans_close_target, humans_to_target)]
            for _ in range(5000):
                # 変更先を決定
                i, j = random.randint(0, m-1), random.randint(0, m-1)
                while max(len(humans_close_target[i]), len(humans_close_target[j]))==0:
                    i, j = random.randint(0, m-1), random.randint(0, m-1)
                ti, tj = random.randint(-1, len(humans_close_target[i])-1), random.randint(-1, len(humans_close_target[j])-1) # 基本swapする。-1だったら適当にinsert
                ti, tj = -1, -1
                while ti==-1 and tj==-1:
                    ti, tj = random.randint(-1, len(humans_close_target[i])-1), random.randint(-1, len(humans_close_target[j])-1) # 基本swapする。-1だったら適当にinsert
                # 評価
                targets_i = [x for x in humans_close_target[i]]
                targets_j = [x for x in humans_close_target[j]]
                if i==j:
                    while ti==-1 or tj==-1:
                        ti, tj = random.randint(-1, len(humans_close_target[i])-1), random.randint(-1, len(humans_close_target[j])-1) # 基本swapする。-1だったら適当にinsert
                    targets_i[ti], targets_i[tj] = targets_i[tj], targets_i[ti]
                    targets_j[ti], targets_j[tj] = targets_j[tj], targets_j[ti]
                elif ti==-1:
                    targets_i.insert(random.randint(0, len(targets_i)), targets_j.pop(tj))
                elif tj==-1:
                    targets_j.insert(random.randint(0, len(targets_j)), targets_i.pop(ti))
                else:
                    targets_i[ti], targets_j[tj] = targets_j[tj], targets_i[ti]
                ni = self.enum_turn(self.humans[i], dists, targets_i, None)[0]
                nj = self.enum_turn(self.humans[j], dists, targets_j, None)[0]
                # 入れ替え
                if max(cur_turn[i], cur_turn[j]) >= max(ni, nj):
                    humans_close_target[i] = targets_i
                    humans_close_target[j] = targets_j
                    cur_turn[i] = ni
                    cur_turn[j] = nj
        # toをassign
        if True:
            for i in range(len(self.humans)):
                if len(humans_close_target[i])==0:
                    cur_dist[i] = dists[self.humans[i].x][self.humans[i].y]
                    continue
                x, y = humans_close_target[i][-1]
                for d in "udlr":
                    op = Operation(d, (x, y), is_target_xy=True)
                    if (0<=op.x<30 and 0<= op.y<30) and available[op.x][op.y]:
                        cur_dist[i] = dists[op.x][op.y]
            is_done = [False]*len(to_target)
            while sum(is_done) < len(to_target):
                best_d, best_pair = math.inf, None
                for hum_id in range(len(self.humans)):
                    for task_id in range(len(to_target)):
                        if is_done[task_id]:
                            continue
                        op = Operation("to", to_target[task_id], is_target_xy=True)
                        if (0<=op.x<30 and 0<= op.y<30) and cur_dist[hum_id][op.x][op.y]+hum_turn[hum_id] < best_d:
                            best_d = cur_dist[hum_id][op.x][op.y]+hum_turn[hum_id]+1
                            best_pair = (hum_id, task_id, op)
                hum_id, task_id, op = best_pair
                humans_to_target[hum_id] = to_target[task_id]
                is_done[task_id] = True
                hum_turn[hum_id] = math.inf
        # assign
        for human, ct, tt in zip(self.humans, humans_close_target, humans_to_target):
            if self.stage == 0:
                ops = [Operation("d", (x, y), is_target_xy=True) for x, y in ct]
                if tt is not None:
                    ops.append(Operation("to", tt))
                human.task = Task(ops)
            else:
                ops = []
                for x, y in ct:
                    if x%6==0 and x!=0:
                        d = "u"
                    elif x%6==3:
                        d = "d"
                    elif y==15:
                        d = "d" if x > 15 else "u"
                    else:
                        d = "l" if y>15 else "r"
                    ops.append(Operation(d, (x, y), is_target_xy=True))
                if tt is not None:
                    ops.append(Operation("to", tt))
                human.task = Task(ops)
            # print(f"{task}", file=logfile)

    def assign2(self, ideal_R: Room, hima_only=False) -> None:
        if not hima_only:
            for human in self.humans:
                human.task = Task([])
        m = len(self.humans)
        available = [[d!=math.inf for d in line] for line in ideal_R.dist(self.basepoint[0], self.basepoint[1])]
        close_target, to_target = self.get_targets(ideal_R)
        tasks = [[] for _ in range(len(self.humans))]
        dists = [self.R.dist(human.x, human.y) for human in self.humans]
        is_assigned = [(hima_only and not human.is_free) for human in self.humans]
        is_assigned_close = [False]*len(close_target)
        is_assigned_to = [False]*len(to_target)
        assign_pair = []
        for i, human in enumerate(self.humans):
            if hima_only and not human.is_free:
                for j, target in enumerate(close_target):
                    for op in human.task.ops:
                        if target[0]==op.target()[0] and target[1]==op.target()[1]:
                            is_assigned_close[j] = True
                continue
            for j, target in enumerate(close_target):
                for d in "udlr":
                    op = Operation(d, close_target[j], is_target_xy=True)
                    if (0<=op.x<30 and 0<= op.y<30) and available[op.x][op.y]:
                        assign_pair.append((dists[i][op.x][op.y], i, j, op))
        for d, i, j, op in sorted(assign_pair, key=lambda x: x[0]):
            if is_assigned[i] or is_assigned_close[j]:
                continue
            # assign
            self.humans[i].task = Task([op])
            is_assigned[i] = True
            is_assigned_close[j] = True
        assign_pair = []
        for i, human in enumerate(self.humans):
            if is_assigned[i]:
                continue
            for j, target in enumerate(to_target):
                op = Operation("to", to_target[j], is_target_xy=True)
                assign_pair.append((dists[i][op.x][op.y], i, j, op))
        for d, i, j, op in sorted(assign_pair, key=lambda x: x[0]):
            if is_assigned[i] or is_assigned_to[j]:
                continue
            # assign
            self.humans[i].task = Task([op])
            is_assigned[i] = True
            is_assigned_to[j] = True

    def hima_assign(self, joints_info: List[Tuple]):
        for human in self.humans:
            if human.task.done:
                human.is_free = True
        if sum([human.is_free for human in self.humans])==0:
            return
        self.stage += 1
        ideal_R = self.create_ideal_R(joints_info)
        self.stage -= 1
        self.assign2(ideal_R, hima_only=True)

    def step(self):
        # print(f"{self.turn} {ts-time()}", file=logfile)
        joints_info = self.joints_info(self.R)
        # 理想盤面
        ideal_R = self.create_ideal_R(joints_info)
        if not self.init_assigned:
            # assign
            if self.stage%2==1:
                self.assign2(ideal_R)
            else:
                self.assign(ideal_R)
                self.init_assigned = True
        else:
            if self.stage >= 2 and False:
                self.hima_assign(joints_info)
        # 割り込みタスクを割り当て
        self.cut_in_assign(ideal_R, joints_info)
#         # print([human.task.ops for human in self.humans], file=sys.stderr)
        # 人行動
        self.human_move()
        # ペット行動
        self.pet_move()
        # 情報アップデート
        self.turn += 1
        self.pre_ideal_R = ideal_R
        for human in self.humans:
            human.cut_in_op = None
        self.update_stage()

    def human_move(self) -> None:
        human_cmds = [human.next_cmd() for human in self.humans]
        # udlrができなければ.に変更する
        for i in range(len(self.humans)):
            # op = self.humans[i].task.cur_op(self.humans[i], self.R)
            #print(i, op.x, op.y, op.op, file=sys.stderr)
            if human_cmds[i] in "udlr":
                x, y = self.humans[i].x, self.humans[i].y
                dx, dy = char2dir[human_cmds[i]]
                if not self.can_close(x+dx, y+dy):
                    human_cmds[i] = "."
        # 一旦UDLRで動く
        for i in range(len(self.humans)):
            # op = self.humans[i].task.cur_op(self.humans[i], self.R)
            #print(i, op.x, op.y, op.op, file=sys.stderr)
            if human_cmds[i] in "UDLR":
                x, y = self.humans[i].x, self.humans[i].y
                dx, dy = char2dir[human_cmds[i]]
                if self.R[x+dx][y+dy]=="#":
                    human_cmds[i] = "."
                self.humans[i].move(human_cmds[i])
        # udlrが問題あれば.に変更してmove
        for i, (human, cmd) in enumerate(zip(self.humans, human_cmds)):
            if cmd in "udlr":
                xd, yd = char2dir[cmd]
                if self.can_close(human.x+xd, human.y+yd):
                    human.move(cmd)
                else:
                    human_cmds[i] = "."
        print("".join(human_cmds))

    def pet_move(self):
        # petがaction
        pet_cmds = input()
        for pet, cmds in zip(self.pets, pet_cmds.split(" ")):
            for cmd in cmds:
                assert cmd in ".udlrUDLR", f"{pet_cmds} {cmds} {cmd}"
                pet.move(cmd)

R = Room()
N = int(input())
pets = []
for i in range(N):
    x, y, t = map(int, input().split())
    x -= 1
    y -= 1
    pets.append(Pet(R, x, y, t, i))

M = int(input())
humans = []
for i in range(M):
    x, y = map(int, input().split())
    x -= 1
    y -= 1
    humans.append(Human(R, x, y, i))
game = Game(R, pets, humans)

for turn in range(300):
    game.step()
print(time()-ts, file=sys.stderr)
