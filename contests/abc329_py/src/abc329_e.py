import sys
import math

N, M = map(int, input().split())
S = input()
T = input()
if S[0] != T[0]:
    print("No")
    sys.exit()
ptns = set([T])
for i in range(N-M):
    new_ptns = set()
    for ptn in ptns:
        if ptn[0] == S[i]:
            new_ptns.add(ptn[1:] + "#")
            for x in range(1, M):
                if ptn[x]=="#":
                    new_ptns.add(ptn[1:x] + T[x:] + "#")
        if T[0] == S[i]:
            new_ptns.add(T[1:] + "#")
    ptns = new_ptns
for ptn in ptns:
    if T==S[-M:]:
        print("Yes")
        sys.exit()
    for x in range(M):
        if ptn[:x]+T[x:] == S[-M:] and ptn[x]=="#":
            print("Yes")
            sys.exit()
print("No")
