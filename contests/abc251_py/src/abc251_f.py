import sys
import math
from collections import deque

N, M = map(int, input().split())
AB = [list(map(int, input().split())) for _ in range(M)]
g = [[] for _ in range(N)]
for a, b in AB:
    a -= 1
    b -= 1
    g[a].append(b)
    g[b].append(a)


appeared, withdrawed = [False]*N, [False]*N
start_node = 0
q = [(~start_node, None), (start_node, None)]
cur = None
while q:
    u, fr = q.pop()
    if u >= 0:
        if appeared[u]:
            continue
        appeared[u] = True
        # 入った時の処理
        if fr is not None:
            print(fr+1, u+1)
        # 探索先を追加
        for v in g[u][::-1]:
            if appeared[v]:
                continue
            q.append((~v, u))
            q.append((v, u))
    else:
        if withdrawed[~u]:
            continue
        withdrawed[~u] = True


appeared, withdrawed = [False]*N, [False]*N
start_node = 0
q = deque([(start_node, None), (~start_node, None)])
while q:
    u, fr = q.popleft()
    if u >= 0:
        if appeared[u]:
            continue
        appeared[u] = True
        # 入った時の処理
        if fr is not None:
            print(fr+1, u+1)
        # 探索先を追加
        for v in g[u][::-1]:
            if appeared[v]:
                continue
            q.append((v, u))
            q.append((~v, u))
    else:
        if withdrawed[~u]:
            continue
        withdrawed[~u] = True
        # 出た時の処理
        cur = ~u
