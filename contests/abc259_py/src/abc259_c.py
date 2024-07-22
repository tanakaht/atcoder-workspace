import sys
import math

S = input()
T = input()
si = 0
for t in T:
    if si==len(S) or S[si]!=t:
        if si>=2 and S[si-1]==S[si-2]==t:
            continue
        else:
            print("No")
            sys.exit()
    else:
        si += 1
if si==len(S):
    print("Yes")
else:
    print("No")
