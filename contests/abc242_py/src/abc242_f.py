import sys
import math

N, M, B, W = map(int, input().split())
MOD = 998244353

# 再利用する時あらかじめN以下の計算しとく
kaizyo = [1]
kaizyo_inv = [1]
tmp = 1
for i in range(1, 10000):
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

ans = 0
brptns = [[0]*(M+1) for _ in range(N+1)]
for bh in range(1, N+1):
    for bw in range(1, M+1):
        brptn = comb(bh*bw, B)
        for h_ in range(1, bh+1):
            for w_ in range(1, bw+1):
                brptn = (brptn - brptns[h_][w_]*comb(bw, w_)*comb(bh, h_))%MOD
        brptn = max(0, brptn)
        if brptn != 0:
            pass
            # print(bh, bw, brptn)
        brptns[bh][bw] = brptn
        ans = (ans+brptn*comb((N-bh)*(M-bw), W)*comb(N, bh)*comb(M, bw))%MOD
print(ans)
