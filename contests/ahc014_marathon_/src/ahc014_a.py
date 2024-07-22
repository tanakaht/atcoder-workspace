import random
import sys
import math
from time import time

N, M = map(int, input().split())
XY = [list(map(int, input().split())) for _ in range(M)]
DIR = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]


class Game:
    def __init__(self, N: int, XY=None):
        self.N = N
        self.M = [[False]*N for _ in range(N)]
        self.lines = set()
        self.rects = []
        if XY is not None:
            for x, y in XY:
                self.M[x][y] = True

    def all_candidate(self):
        N = self.N
        ret = []
        for sx in range(N):
            for sy in range(N):
                for rect in self.candidate(sx, sy):
                    ret.append(rect)
        return ret

    # x, yから弾けるrectの候補
    def candidate(self, sx: int, sy: int):
        if self.M[sx][sy]:
            return []
        N = self.N
        ret = []
        for dir in range(8):
            tmp = [sx, sy]
            for dir_ in range(4):
                dx, dy = DIR[(dir+dir_*2)%8]
                x, y = tmp[-2]+dx, tmp[-1]+dy
                while 0<=x<N and 0<=y<N and (not self.M[x][y]) and (not (x==sx and y==sy)):
                    x += dx
                    y += dy
                if not (0<=x<N and 0<=y<N):
                    break
                if not self.can_line(tmp[-2], tmp[-1], x, y):
                    break
                tmp.append(x)
                tmp.append(y)
            if len(tmp)==10 and x==sx and y==sy:
                ret.append(tmp[:8])
        return ret

    def can_line(self, x1, y1, x2, y2):
        if (x1, y1)>=(x2, y2):
            x1, y1, x2, y2 = x2, y2, x1, y1
        line_length = max(abs(x2-x1), abs(y2-y1))
        x, y =  x1, y1
        dx, dy = ((x2-x1)//line_length, (y2-y1)//line_length)
        for i in range(line_length):
            if (x, y, x+dx, y+dy) in self.lines:
                return False
            x += dx
            y += dy
        return True

    # チェックはしない
    def draw_line(self, x1, y1, x2, y2):
        if (x1, y1)>=(x2, y2):
            x1, y1, x2, y2 = x2, y2, x1, y1
        line_length = max(abs(x2-x1), abs(y2-y1))
        x, y =  x1, y1
        dx, dy = ((x2-x1)//line_length, (y2-y1)//line_length)
        for i in range(line_length):
            self.lines.add((x, y, x+dx, y+dy))
            self.lines.add((x+dx, y+dy, x, y))
            x += dx
            y += dy

    # チェックはしない
    def draw_rect(self, points):
        # points: x0, y0, x1, y1, x2, y2, x3, y3
        for i in range(4):
            x1, y1 = points[2*i:2*i+2]
            x2, y2 = points[(2*i+2)%8:(2*i+2)%8+2]
            self.draw_line(x1, y1, x2, y2)
        assert self.M[points[0]][points[1]]==False
        self.M[points[0]][points[1]] = True
        self.rects.append(points)

    def remove_line(self, x1, y1, x2, y2):
        if (x1, y1)>=(x2, y2):
            x1, y1, x2, y2 = x2, y2, x1, y1
        line_length = max(abs(x2-x1), abs(y2-y1))
        x, y =  x1, y1
        dx, dy = ((x2-x1)//line_length, (y2-y1)//line_length)
        for i in range(line_length):
            self.lines.remove((x, y, x+dx, y+dy))
            self.lines.remove((x+dx, y+dy, x, y))
            x += dx
            y += dy

    def remove_rect(self, idx):
        assert 0<=idx<len(self.rects)
        points = self.rects[idx]
        remove_rect_idxs = set([idx])
        self.M[points[0]][points[1]] = False
        for j in range(4):
            x1, y1 = points[2*j:2*j+2]
            x2, y2 = points[(2*j+2)%8:(2*j+2)%8+2]
            self.remove_line(x1, y1, x2, y2)
        for i in range(idx+1, len(self.rects)):
            is_ok = True
            points = self.rects[i]
            for j in range(4):
                is_ok = is_ok and (self.M[points[2*j]][points[2*j+1]])
            if is_ok:
                continue
            remove_rect_idxs.add(i)
            self.M[points[0]][points[1]] = False
            for j in range(4):
                x1, y1 = points[2*j:2*j+2]
                x2, y2 = points[(2*j+2)%8:(2*j+2)%8+2]
                self.remove_line(x1, y1, x2, y2)
        self.rects = [self.rects[i] for i in range(len(self.rects)) if i not in remove_rect_idxs]

    def copy(self):
        N = self.N
        ret = Game(self.N)
        ret.M = [[self.M[x][y] for y in range(N)] for x in range(N)]
        ret.lines = list(self.lines)
        ret.rects = list(self.rects)
        return ret

    def raw_score(self):
        N = self.N
        c = (N-1)/2
        w_sum = 0
        S = 0
        for x in range(N):
            for y in range(N):
                w = (x-c)*(x-c)+(y-c)*(y-c)+1
                S += w
                w_sum += self.M[x][y]*w
        score = round(10**6*N*N/M*w_sum/S)
        return score

    def score(self):
        N = self.N
        score = 0
        # 生のスコア
        c = (N-1)/2
        w_sum = 0
        S = 0
        for x in range(N):
            for y in range(N):
                w = (x-c)*(x-c)+(y-c)*(y-c)+1
                S += w
                w_sum += self.M[x][y]*w
        raw_score = round(10**6*N*N/M*w_sum/S)

        # 有効な点の数
        available_cnt = 0
        for x in range(N):
            for y in range(N):
                if not self.M[x][y]:
                    continue
                for dir in range(8):
                    available_cnt += (x, y, x+DIR[dir][0], y+DIR[dir][1]) not in self.lines

        # 線の長さ
        line_length = len(self.lines)
        score = raw_score + available_cnt*10**5 - line_length*10
        return score



class State:
    def __init__(self, game: Game):
        self.game = game

    def get_neighbors(self, **params):
        ret = [(0, x) for x in self.game.all_candidate()]
        ret += [(1, i) for i in range(len(self.game.rects))]
        return ret

    def get_score(self, move) -> float:
        if move is None:
            return self.game.score()
        if move[0]==0:
            self.game.draw_rect(move[1])
            score = self.game.score()
            self.game.remove_rect(len(self.game.rects)-1)
        elif move[0] == 1:
            game_ = self.game.copy()
            game_.remove_rect(move[1])
            score = game_.score()
        return score

    def move(self, move):
        if move[0]==0:
            game.draw_rect(move[1])
        elif move[0] == 1:
            game.remove_rect(move[1])


class Optimizer:
    def __init__(self, initial_state):
        self.initial_state = initial_state
        self.history = []

    def climbing(self, TL1=18, TL2=28):
        state = self.initial_state
        score = state.get_score(None)
        cnt = 0
        self.history.append((cnt, score, state))
        TS = time()
        while time()-TS<TL1:
            cnt += 1
            move = random.choice(state.get_neighbors())
            new_score = state.get_score(move)
            if new_score >= score:
                state.move(move)
                score = new_score
                self.history.append((cnt, score, state))
        while time()-TS<TL2:
            cnt += 1
            neigbors = [x for x in state.get_neighbors() if x[0]==0]
            move = random.choice(neigbors)
            new_score = state.get_score(move)
            if new_score >= score:
                state.move(move)
                score = new_score
                self.history.append((cnt, score, state))
        return state

    def simanneal(self, TL=1.8):
        state = self.initial_states[0]
        score = state.get_score()
        best_state, best_score = state, score
        cnt = 0
        self.history.append((cnt, score, state))
        TS = time()
        while time()-TS<TL:
            cnt += 1
            new_state = state.get_neighbors()
            new_score = new_state.get_score()
            if new_score >= score:
                state, score = new_state, new_score
                self.history.append((cnt, score, state))
        return state

    def beam_search(self, TL=1.8, beam_width=10):
        return

game = Game(N, XY)
state = State(game)
opt = Optimizer(state)
fin_state = opt.climbing()

print(len(fin_state.game.rects))
for rect in fin_state.game.rects:
    print(*rect)

"""
candidate = game.all_candidate()
while candidate:
    candidate = sorted(candidate, key=lambda x: max(abs(x[0]-x[2]), abs(x[1]-x[3]))+max(abs(x[2]-x[4]), abs(x[3]-x[5]))+max(abs(x[4]-x[6]), abs(x[5]-x[7]))+max(abs(x[6]-x[0]), abs(x[7]-x[1])))
    points = candidate[0]
    # points = random.choice(candidate)
    game.draw_rect(points)
    if random.random()>0.925:
        game.remove_rect(random.randint(0, len(game.rects)-1))
    candidate = game.all_candidate()

print(len(game.rects))
for rect in game.rects:
    print(*rect)
"""
