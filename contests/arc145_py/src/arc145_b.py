import sys
import math

N, A, B = map(int, input().split())
if A<=B:
    ans = max(0, N-A+1)
    print(ans)
else:
    winperA = B
    if N>=A:
        ans = winperA*(N//A-1)+max(0, min(N%A+1, B))
    else:
        ans = 0
    print(ans)
