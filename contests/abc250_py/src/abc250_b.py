import sys
import math

N, A, B = map(int, input().split())
S = [[None]*N*B for _ in range(N*A)]
for i in range(N*A):
    for j in range(N*B):
        if ((i//A)+(j//B))%2:
            S[i][j] = "#"
        else:
            S[i][j] = "."

for i in range(N*A):
    print(*S[i], sep="")
