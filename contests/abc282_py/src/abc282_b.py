import sys
import math

N, M = map(int, input().split())
S = [input() for _ in range(N)]
ans = 0
for i in range(N):
    for j in range(i+1, N):
        cnt = 0
        for m in range(M):
            cnt += (S[i][m]=="o") or (S[j][m]=="o")
        ans += cnt==M
print(ans)
