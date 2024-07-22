import sys, re
sys.setrecursionlimit(10**9)

N, M = map(int, input().split())
uv = [list(map(int, input().split())) for _ in range(M)]

g = [[] for _ in range(N)]
for u, v in uv:
    g[u-1].append(v-1)
    g[v-1].append(u-1)

passed = [False]*N
passed[0] = True
cnt = 0

appeared, withdrawed = [False]*N, [False]*N
t = 0
start_node = 0
q = [~start_node, start_node]
while q:
    u = q.pop()
    if cnt >= 1_000_000:
        break
    if u >= 0:
        if appeared[u]:
            continue
        cnt += 1
        appeared[u] = True
        # 入った時の処理
        # 探索先を追加
        for v in g[u][::-1]:
            if appeared[v]:
                continue
            q.append(~v)
            q.append(v)
    else:
        appeared[~u] = False
print(min(cnt, 1_000_000))
