import sys
from collections import defaultdict

T = int(input())
for caseid in range(1, T+1):
    R, C = map(int, input().split())
    M = [list(input()) for x in range(R)]
    cantuse = [[M[i][j]=="#" for j in range(C)] for i in range(R)]
    def bfs(i, j):
        ad_usable = []
        for i_, j_ in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
            if not (0<=i_<R and 0<=j_<C):
                continue
            if not cantuse[i_][j_]:
                ad_usable.append((i_, j_))
        if len(ad_usable)<=1:
            cantuse[i][j] = True
            for i_, j_ in ad_usable:
                bfs(i_, j_)
    for i in range(R):
        for j in range(C):
            if M[i][j] == "#":
                continue
            else:
                bfs(i, j)
    M2 = [[None]*C for _ in range(R)]
    ans = "Possible"
    for i in range(R):
        for j in range(C):
            if M[i][j] == "#":
                M2[i][j] = "#"
            else:
                if cantuse[i][j] and M[i][j]=="^":
                    ans="Impossible"
                elif cantuse[i][j] and M[i][j]==".":
                    M2[i][j] = "."
                else:
                    M2[i][j] = "^"
    print(f'Case #{caseid}: {ans}')
    if ans=="Possible":
        for x in M2:
            print("".join(x))
