import sys
import math


def articulation_detection(graph, start=0):
    n = len(graph)
    order = [-1] * n
    cnt = -1
    articulation_points = []
    def dfs(u, prev):
        nonlocal cnt
        cnt += 1
        low_pt = order[u] = cnt
        f_cnt = art_flag = 0
        for v in graph[u]:
            if v == prev:
                continue
            if order[v] == -1:
                v_low_pt = dfs(v, u)
                art_flag |= v_low_pt >= order[u]
                low_pt = min(v_low_pt, low_pt)
                f_cnt += 1
            else:
                low_pt = min(low_pt, order[v])
        if len(graph[u]) > 1 and ((prev != -1 and art_flag) or (prev == -1 and f_cnt > 1)):
            articulation_points.append(u)
        return low_pt
    dfs(start, -1)
    return articulation_points


T = int(input())
for caseid in range(1, T+1):
    N, L = map(int, input().split())
    K = []
    for _ in range(L):
        input()
        K.append(list(map(int, input().split())) )
    g = [[] for _ in range(N+L)]
    for i, S in enumerate(K):
        for s in S:
            s -= 1
            g[s].append(i+N)
            g[i+N].append(s)
    ans = 0
    for x in articulation_detection(g):
        ans += x>=N
    print(f"Case #{caseid}: {ans}")
