import sys
import math

N, K = map(int, input().split())
A = list(map(int, input().split()))
Ak = [a for a in A]
for bi in bin(K)[3:]:
    Ak = [Ak[i]+Ak[(i+Ak[i])%N] for i in range(N)]
    if bi=="1":
        Ak = [A[i]+Ak[(i+A[i])%N] for i in range(N)]
print(Ak[0])
