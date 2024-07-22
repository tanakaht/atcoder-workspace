import sys
import math

N = int(input())
A = list(map(int, input().split()))
ans = 0
# ながさ20以上は全部OK
if N>=25:
    ans += (N-24)*(N-23)//2
# ながさ19以下を全探索
for l in range(N-2):
    for r in range(l+2, min(l+24, N)):
        found = False
        for m in range(l+1, r):
            s = set()
            for i in range(l, m):
                s.add(A[i]-A[m])
            for i in range(m+1, r+1):
                if A[m]-A[i] in s:
                    found = True
                    break
            if found:
                break
        ans += found
print(ans)
