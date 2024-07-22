import sys
import math

S = input()
MOD = 998244353
ans = 0
cur = (0, 0, 0, 0, 0) # 最後の演算子以降の数値, 最後の演算子以降にlがある場合の数値のsum 今にかかるmulの値, 今までのaddのsum, 最後の演算子以降にlがある場合のパターン数
appear_plus = False
for s in S:
    if s == '+':
        appear_plus = True
        cur = (0, 0, 0, (cur[3]+cur[0]*cur[2]+cur[1])%MOD, 0)
    elif s == '*':
        cur = (0, 0, (cur[0]*cur[2]+cur[1])%MOD, cur[3], 0)
    else:
        cur = (cur[0] * 10 + int(s), cur[1]*10+(cur[4]+1)*int(s), cur[2], cur[3], cur[4]+1)
        # rがsな場合を集計
        # xxx+yyy*zzz
        # xxx+yyy
        ans = (ans + cur[1] + cur[0]*cur[2] + cur[0]*(cur[2]+(cur[2]==0 and appear_plus))+cur[3]) % MOD
    print(cur, s, ans)
print(ans)
