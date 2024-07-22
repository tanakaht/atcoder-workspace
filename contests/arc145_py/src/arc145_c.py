import sys
import math

N = int(input())
MOD = 998244353
# 再利用する時あらかじめN以下の計算しとく
kaizyo = [1]
kaizyo_inv = [1]
tmp = 1
for i in range(1, 2*N+10):
    tmp = (tmp*i) % MOD
    kaizyo.append(tmp)
    kaizyo_inv.append(pow(tmp, MOD - 2, MOD))


def comb(n, r):
    if n < r or n < 0:
        return 0
    elif n == r or r==0:
        return 1
    else:
        return (((kaizyo[n] * kaizyo_inv[r])%MOD) * kaizyo_inv[n - r])%MOD

# ペア位置の組み合わせ(型乱数)
cn = (comb(2*N, N) - comb(2*N, N-1))%MOD

# ペアないの入れ替え
ans = (cn*pow(2, N, MOD))%MOD

# ペアの順列
ans = (ans*kaizyo[N])%MOD
print(ans)
