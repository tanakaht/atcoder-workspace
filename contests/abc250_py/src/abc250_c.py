import sys
import math

N, Q = map(int, input().split())
X = [int(input())for _ in range(Q)]
A = list(range(1, N+1))
d = {i: i-1 for i in range(N+1)}
for x in X:
    i = d[x]
    j = i+1 if (i+1)!=N else i-1
    A[i], A[j] = A[j], A[i]
    d[A[i]] = i
    d[A[j]] = j
print(*A)
