import sys
import math

N, Q = map(int, input().split())
S = input()
for _ in range(Q):
    L, R = map(int, input().split())
    L -= 1
    print(L, R, S[L:R])
