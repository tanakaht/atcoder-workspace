import sys
from itertools import product

T = int(input())
C_ton = []
C_tu = []
for x in product([".", "-"], repeat=10):
    x = "".join(x)
    if x.startswith("."):
        C_ton.append(x)
    else:
        C_tu.append(x)


for caseid in range(1, T+1):
    N = int(input())
    C = input()
    print(f'Case #{caseid}:')
    if C.startswith("."):
        for x in C_tu[:N-1]:
            print(x)
    else:
        for x in C_ton[:N-1]:
            print(x)
