import math
def main():
    T = int(input())
    Xs = [int(input()) for _ in range(T)]
    thres = 10**10
    def sqrtfloor(x):
        if x<thres:
            return int(math.sqrt(x))
        ng, ok = thres, 0
        while abs(ok - ng) > 1:
            mid = (ok + ng) // 2
            if mid*mid<=x:
                ok = mid
            else:
                ng = mid
        return ok
    # N = int(math.sqrt(int(math.sqrt(max(Xs)))))+2
    N = int(max(Xs)**(0.25))+2
    cumsum = [0]*N
    cumsum[1] = 1
    for i in range(2, N):
        cumsum[i] = cumsum[i-1]+cumsum[int(math.sqrt(i))]

    def solve(x):
        i = 1
        ret = 0
        x_sqrtfloor = sqrtfloor(x)
        while i<=x_sqrtfloor:
            i_sqrtfloor = int(math.sqrt(i))
            i_ = min(i+2*i_sqrtfloor, x_sqrtfloor)
            ret += cumsum[i_sqrtfloor] * (i_-i+1)
            i = i_+1
        return ret

    for x in Xs:
        ans = solve(x)
        print(ans)

main()
