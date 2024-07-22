import sys
import math

A_ = [list(map(int, input().split())) for _ in range(3)]
A = []
for i in range(3):
    for j in range(3):
        A.append(A_[i][j])
banmen = [0]*(pow(3, 9)) # 0: 未定, 1: 先手(takahashi)勝ち, 2: 後手(aoki)勝ち
pow3 = [1]
for i in range(1, 10):
    pow3.append(pow3[-1]*3)
def is_win(idx):
    # 縦横斜めの判定
    for i in range(3):
        if idx//(pow3[i])%3==idx//(pow3[i+3])%3==idx//(pow3[i+6])%3==1:
            return 1
        if idx//(pow3[i])%3==idx//(pow3[i+3])%3==idx//(pow3[i+6])%3==2:
            return 2
        if idx//(pow3[3*i])%3==idx//(pow3[3*i+1])%3==idx//(pow3[3*i+2])%3==1:
            return 1
        if idx//(pow3[3*i])%3==idx//(pow3[3*i+1])%3==idx//(pow3[3*i+2])%3==2:
            return 2
    if idx//(pow3[0])%3==idx//(pow3[4])%3==idx//(pow3[8])%3==1:
        return 1
    if idx//(pow3[0])%3==idx//(pow3[4])%3==idx//(pow3[8])%3==2:
        return 2
    if idx//(pow3[2])%3==idx//(pow3[4])%3==idx//(pow3[6])%3==1:
        return 1
    if idx//(pow3[2])%3==idx//(pow3[4])%3==idx//(pow3[6])%3==2:
        return 2
    # 数値判定
    takahashi = 0
    aoki = 0
    for i in range(9):
        if idx//(pow3[i])%3==1:
            takahashi += A[i]
        elif idx//(pow3[i])%3==2:
            aoki += A[i]
        else:
            return 0
    if takahashi>aoki:
        return 1
    else:
        return 2

def solve(idx, turn):
    if banmen[idx]!=0:
        return banmen[idx]
    if is_win(idx)!=0:
        banmen[idx] = is_win(idx)
        return banmen[idx]
    if turn==1:
        for i in range(9):
            if idx//(pow3[i])%3==0:
                if solve(idx+pow3[i], 2)==1:
                    banmen[idx] = 1
                    return 1
        banmen[idx] = 2
        return 2
    else:
        for i in range(9):
            if idx//(pow3[i])%3==0:
                if solve(idx+2*pow3[i], 1)==2:
                    banmen[idx] = 2
                    return 2
        banmen[idx] = 1
        return 1

if solve(0, 1) == 1:
    print("Takahashi")
else:
    print("Aoki")
