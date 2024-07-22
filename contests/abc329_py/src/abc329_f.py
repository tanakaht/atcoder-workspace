import sys
import math

N, Q = map(int, input().split())
C = list(map(lambda x: set([int(x)]), input().split()))
queries = [list(map(int, input().split())) for _ in range(Q)]
for a, b in queries:
    a -= 1
    b -= 1
    if len(C[a]) > len(C[b]):
        C[a], C[b] = C[b], C[a]
    for x in C[a]:
        C[b].add(x)
    C[a] = set()
    print(len(C[b]))
