import sys
import math

N, M, X, T, D = map(int, input().split())
print(T-max(0, X-M)*D)

