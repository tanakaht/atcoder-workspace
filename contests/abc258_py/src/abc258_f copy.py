drdc = [(-1, 0), (0, -1), (0, 1), (1, 0)]
B = 1  # this will be replaced


def around(x):
    lo = (x // B) * B
    hi = lo + B
    return lo, hi


def distOnCrossMain(r1, c1, r2, c2):
    #assert r1 % B == 0 and c1 % B == 0
    #assert r2 % B == 0 and c2 % B == 0
    return abs(r1 - r2) + abs(c1 - c2)


def distToCrossMain(r, c):
    #assert r % B == 0 or c % B == 0
    if r % B == 0:
        for cc in around(c):
            yield (r, cc), abs(c - cc)
    if c % B == 0:
        for rr in around(r):
            yield (rr, c), abs(r - rr)


inf = float("inf")
def distOnMain(r1, c1, r2, c2):
    #assert r1 % B == 0 or c1 % B == 0
    #assert r2 % B == 0 or c2 % B == 0
    ret = inf
    if r1 == r2 and r1 % B == r2 % B == 0:
        ret = min(ret, abs(c1 - c2))
    elif c1 == c2 and c1 % B == c2 % B == 0:
        ret = min(ret, abs(r1 - r2))
    elif r1 // B == r2 // B or c1 // B == c2 // B:
        for (a, b), d1 in distToCrossMain(r1, c1):
            for (c, d), d2 in distToCrossMain(r2, c2):
                ret = min(ret, d1 + d2 + distOnCrossMain(a, b, c, d))
    else:
        ret = abs(r1 - r2) + abs(c1 - c2)
    return ret


def distToMain(r, c):
    r1, r2 = around(r)
    c1, c2 = around(c)

    yield (r, c1), abs(c1 - c) * K
    yield (r, c2), abs(c2 - c) * K
    yield (r1, c), abs(r1 - r) * K
    yield (r2, c), abs(r2 - r) * K
