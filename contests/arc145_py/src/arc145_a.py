import sys
import math

N = int(input())
S = list(input())

if S[0]=="A" and S[-1]=="B":
    print("No")
else:
    if N==2 and S[0]!=S[1]:
        print("No")
    else:
        print("Yes")
