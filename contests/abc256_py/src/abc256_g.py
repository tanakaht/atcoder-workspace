N, D = map(int, input().split())
MOD = 998244353
# 再利用する時あらかじめN以下の計算しとく
kaizyo = [1]
kaizyo_inv = [1]
tmp = 1
for i in range(1, D+2):
    tmp = (tmp*i) % MOD
    kaizyo.append(tmp)
    kaizyo_inv.append(pow(tmp, MOD - 2, MOD))
def comb(n, r):
    if n < r or n < 0 or r<0:
        return 0
    elif n == r or r==0:
        return 1
    else:
        return (((kaizyo[n] * kaizyo_inv[r])%MOD) * kaizyo_inv[n - r])%MOD

def mat_mul(a, b) :
    I, J, K = len(a), len(b[0]), len(b)
    c = [[0] * J for _ in range(I)]
    for i in range(I) :
        for j in range(J) :
            for k in range(K) :
                c[i][j] += a[i][k] * b[k][j]
            c[i][j] %= MOD
    return c


def mat_pow(x, n):
    y = [[0] * len(x) for _ in range(len(x))]

    for i in range(len(x)):
        y[i][i] = 1

    while n > 0:
        if n & 1:
            y = mat_mul(x, y)
        x = mat_mul(x, x)
        n >>= 1

    return y

ans = 2 # 0個と全置きの分
# wcnt=返上におく　wの　数
for wcnt in range(1, D+1):
    p0, p1, p2 = comb(D-1, wcnt), comb(D-1, wcnt-1), comb(D-1, wcnt-2)
    X = [
        [p0, p1],
        [p1, p2]
    ]
    XX = mat_pow(X, N-1)
    tmpans = (XX[0][0]*p0+XX[1][0]*p1+XX[0][1]*p1+XX[1][1]*p2)
    ans = (ans+tmpans)%MOD
print(ans)
