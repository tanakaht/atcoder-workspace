import sys
import math

N = int(input())
C = [0]+list(map(int, input().split()))

dp = [0]*(N+1)
cur = 0
for i in range(N+1):
    for j in range(1, 10):
        cur = max(cur, dp[i])
        dp[i] = cur
        if i+C[j]<=N:
            dp[i+C[j]] = max(dp[i+C[j]], dp[i]+1, cur)
if dp[-1]<20:
    dp = [0]*(N+1)
    for i in range(N+1):
        for j in range(1, 10):
            if i+C[j]<=N:
                dp[i+C[j]] = max(dp[i+C[j]], 10*dp[i]+j)
    print(max(dp))
    sys.exit()

ans = []
c2v = {x: -1 for x in range(10)}
for x, c in enumerate(C):
    c2v[c] = x
i = N
while dp[i] != 0:
    maxcv = 0
    j = i-1
    nexti = j
    while j>=0 and dp[i]<=dp[j]+1:
        try:
            if maxcv < c2v[i-j]:
                nexti = j
                maxcv = c2v[i-j]
        except Exception:
            pass
        j -= 1
    ans.append(maxcv)
    i = nexti
print("".join(map(str, ans)))
