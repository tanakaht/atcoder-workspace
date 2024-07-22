import math
import random
from time import time

TS = time()
TL = 1.8

N = int(input())
A = [list(map(int, reversed(input().split()))) for _ in range(N)]
anss = [[] for _ in range(N)]
for i in range(N):
    for op in "PRRRQLLLPRRQLLPRQ":
        anss[i].append(op)
for i in range(1, N):
    anss[i].append("B")
M = [[None]*N for _ in range(N)]
for i in range(N):
    for j in range(3, -1, -1):
        M[i][j] = A[i].pop()
cur = (0, 1)
def move(cur, to):
    for _ in range(max(0, cur[0]-to[0])):
        anss[0].append("U")
    for _ in range(max(0, to[0]-cur[0])):
        anss[0].append("D")
    for _ in range(max(0, cur[1]-to[1])):
        anss[0].append("L")
    for _ in range(max(0, to[1]-cur[1])):
        anss[0].append("R")


def dist(cur, to):
    return abs(cur[0]-to[0])+abs(cur[1]-to[1])

done = [[] for _ in range(N)]

cnt = 0
while cnt<N*N:
    best_p = None
    best_d = 10**9
    # 運んでいいの見つけたら運ぶ
    for i in range(N):
        for j in range(N):
            if M[i][j] is None:
                continue
            v = M[i][j]
            idx = v//N
            if len(done[idx])==0 and v%N==0:
                if best_d>dist(cur, (i, j)):
                    best_p = (i, j)
                    best_d = dist(cur, (i, j))
            else:
                cand = [x for x in range(idx*N, (idx+1)*N) if x not in done[idx]]
                if min(cand)==v:
                    if best_d>dist(cur, (i, j)):
                        best_p = (i, j)
                        best_d = dist(cur, (i, j))
    if best_p is not None:
        v = M[best_p[0]][best_p[1]]
        idx = v//N
        move(cur, best_p)
        anss[0].append("P")
        move(best_p, (idx, N-1))
        anss[0].append("Q")
        done[idx].append(v)
        cur = (idx, N-1)
        M[best_p[0]][best_p[1]] = None
        if best_p[1]==0 and A[best_p[0]]:
            M[best_p[0]][0] = A[best_p[0]].pop()
        cnt += 1
        continue
    # 空いてるところに適当に運ぶ
    best_i = None
    best_space = None
    best_d = 10**9
    spaces = []
    for i in range(N):
        for j in range(N):
            if M[i][j] is None:
                spaces.append((i, j))
    for i in range(N):
        if A[i]:
            for (x, y) in spaces:
                d = dist(cur, (i, 0))+dist((i, 0), (x, y))
                if best_d>d:
                    best_i = i
                    best_space = (x, y)
                    best_d = d
    if best_i is not None:
        v = M[best_i][0]
        move(cur, (best_i, 0))
        anss[0].append("P")
        move((best_i, 0), best_space)
        anss[0].append("Q")
        cur = (best_space[0], best_space[1])
        M[best_space[0]][best_space[1]] = v
        M[best_i][0] = A[best_i].pop()
        continue
    # 諦めて搬出
    best_p = None
    best_c = 10**9
    # 運んでいいの見つけたら運ぶ
    for i in range(N):
        for j in range(N):
            if M[i][j] is None:
                continue
            v = M[i][j]
            idx = v//N
            if done[idx].len()==0 and v%N==0:
                if best_d>dist(cur, (i, j)):
                    best_p = (i, j)
                    best_d = dist(cur, (i, j))
            else:
                cand = [i for i in range(idx*N, (idx+1)*N) if i not in done[idx]]
                c = dist(cur, (i, j))+(v-min(cand))*1000
                if best_c>c:
                    best_p = (i, j)
                    best_c = c
    if best_p is not None:
        v = M[best_p[0]][best_p[1]]
        idx = v//N
        move(cur, best_p)
        anss[0].append("P")
        move(best_p, (idx, N-1))
        anss[0].append("Q")
        done[idx].append(v)
        cur = (idx, N-1)
        M[cur[0]][cur[1]] = None
        cnt += 1
        continue
    else:
        raise ValueError

for ans in anss:
    print("".join(ans))
