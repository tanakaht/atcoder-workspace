from collections import deque
import sys
import math
from collections import deque

N, M = map(int, input().split())
UV = [list(map(int, input().split())) for _ in range(M)]
g = [[] for _ in range(N)]
for a, b in UV:
    a -= 1
    b -= 1
    g[a].append(b)
    g[b].append(a)

dp = [math.inf]*((1<<N)*N)
appeared = [False]*((1<<N)*N)
dp[0] = 0
appeared[0] = True
q = deque([])
for i in range(N):
    x = (1<<i)*N + i
    dp[(1<<i)*N+i] = 1
    q.append((1, x))
    appeared[x] = True
while q:
    d, x = q.popleft()
    bit, u = x//N, x%N
    for v in g[u]:
        bit_ = bit^(1<<v)
        x_ = bit_*N + v
        if not appeared[x_]:
            q.append((d+1, x_))
            dp[x_] = d+1
            appeared[x_] = True
ans = 0
for bit in range(1<<N):
    ans += min(dp[bit*N:bit*N+N])
print(ans)
