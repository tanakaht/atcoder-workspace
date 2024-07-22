import sys
from time import time
import random

debug = False
ts = time()
T = [list(map(int, input())) for _ in range(30)]
rotate = [[0]*30 for _ in range(30)]

connection = [
    [3, -1, -1, 0],
    [-1, -1, 3, 2],
    [-1, 2, 1, -1],
    [1, 0, -1, -1],
    [3, 2, 1, 0],
    [1, 0, 3, 2],
    [-1, 3, -1, 1],
    [2, -1, 0, -1],
]

def get_output():
    return "".join(["".join(map(str, x)) for x in rotate])

def printans():
    print(get_output())

dir2move = [(-1, 0), (0, 1), (1, 0), (0, -1)]

def cal_tile(t, r):
    if t<4:
        return (t+r)%4
    else:
        return t^(r%2)

def cur_tile(i, j):
    return cal_tile(T[i][j], rotate[i][j])

def judge():
    # なんかバグってる
    return 0
    print(f"judge start: {time()-ts}")
    appeared = [[[False]*4 for _ in range(30)] for _ in range(30)]
    loops = [0, 0]
    for i in range(30):
        for j in range(30):
            for initial_d in range(4):
                if appeared[i][j][initial_d]:
                    continue
                t = cur_tile(i, j)
                appeared[i][j][initial_d] = True
                appeared[i][j][connection[t][initial_d]] = True
                if connection[t][initial_d]==-1:
                    continue
                dx, dy = dir2move[connection[t][initial_d]]
                cur = (i+dx, j+dy, (connection[t][initial_d]+2)%4)
                end = (i, j, initial_d)
                cnt = 1
                while 0<=cur[0]<30 and 0<=cur[1]<30:
                    x, y, d = cur
                    if  appeared[x][y][d]:
                        break
                    appeared[i][j][d] = True
                    d_ = connection[t][d]
                    if connection[t][d] == -1:
                        break
                    appeared[i][j][d_] = True
                    t = cur_tile(x, y)
                    cnt += 1
                    dx, dy = dir2move[connection[t][d]]
                    cur = (x+dx, y+dy, (connection[t][d]+2)%4)
                    if cur==end:
                        loops.append(cnt)
                        loops = sorted(loops)[::-1]
                        loops.pop()
                        break
    print(f"judge end: {time()-ts}")
    return loops[0]*loops[1]


# タイルi, jスタートでどれだけ長く行けるか
# return: looplen, pathlen
def loop_path_len(i, j):
    rets = [loop_path_len_(i, j, d) for d in range(4)]
    looplen, pathlen, appeared1, appeared2 = 0, 0, set(), set()
    for d in range(4):
        if rets[d][0]==1:
            if looplen < rets[d][1]:
                looplen = rets[d][1]
                appeared1 = rets[d][2]
        else:
            d_ = connection[cur_tile(i, j)][d]
            if pathlen < rets[d][1]+rets[d_][1]:
                pathlen = rets[d][1]+rets[d_][1]
                appeared2 = rets[d][2] + rets[d_][2]
    return looplen, pathlen, appeared1, appeared2

# return: flg, len (flg==1ならloop)
def loop_path_len_(i, j, initial_d):
    t = cur_tile(i, j)
    appeared = set()
    appeared.add((i, j))
    if connection[t][initial_d]==-1:
        return (0, 0, appeared)
    dx, dy = dir2move[connection[t][initial_d]]
    cur = (i+dx, j+dy, (connection[t][initial_d]+2)%4)
    end = (i, j, initial_d)
    cnt = 1
    while 0<=cur[0]<30 and 0<=cur[1]<30:
        x, y, d = cur
        appeared.add((x, y))
        t = cur_tile(x, y)
        if connection[t][d] == -1:
            return (0, cnt, appeared)
        cnt += 1
        dx, dy = dir2move[connection[t][d]]
        cur = (x+dx, y+dy, (connection[t][d]+2)%4)
        if cur==end:
            return (1, cnt, appeared)
    return (0, cnt, appeared)

def time2acloop(t):
    if t<=1.4:
        return 50
    elif t<=1.7:
        return 30
    else:
        return 0

# rotateを決定
stage = 0
fixed_stage = False
midans=[]
while time()-ts<1.8:
    if stage==0:
        i, j = random.randint(0, 29), random.randint(0, 29)
        t = T[i][j]
        cur_loop_path_len = loop_path_len(i, j)
        if cur_loop_path_len[0]>=4 and stage!=0:
            continue
        if t<4:
            r = random.randint(1, 3)
        else:
            r = 1
        rotate[i][j] = (rotate[i][j]+r)%4
        new_loop_path_len = loop_path_len(i, j)
        # TODO: 書き換えの戦略
        # 最初はただただ長くする
        if stage==0:
            if max(cur_loop_path_len)>max(new_loop_path_len):
                rotate[i][j] = (rotate[i][j]-r)%4
            else:
                if debug:
                    midans.append(get_output())
    # ループを優先する
    else:
        apps = list(sts_sets.pop())
        dists = []
        for idx1, p1 in enumerate(apps):
            for p2 in apps[idx1+1:]:
                dists.append((abs(p1[0]-p2[0])+abs(p1[1]-p2[1]), p1, p2))
        dists = sorted(dists)
        # 確率的にする
        _, p1, p2 = dists[0]


        # 一定以上のループは受け入れ
        if new_loop_path_len[0]>=time2acloop(time()-ts):
            if debug:
                midans.append(get_output())
        # 損失の程度によっても受け入れ
        elif False:
            if debug:
                midans.append(get_output())
        # その他は拒否
        else:
            rotate[i][j] = (rotate[i][j]-r)%4
    """
    elif stage==1:
        # TODO: 焼きなまし的に受け入れ、looplength, pathlength考慮
        if cur_loop_path_len[0]>new_loop_path_len[0] or new_loop_path_len[0]<50:
            rotate[i][j] = (rotate[i][j]-r)%4
        else:
            if debug:
                midans.append(get_output())
    elif stage==2:
        # TODO: 焼きなまし的に受け入れ、looplength, pathlength考慮
        if cur_loop_path_len[0]>new_loop_path_len[0] or new_loop_path_len[0]<30:
            rotate[i][j] = (rotate[i][j]-r)%4
        else:
            if debug:
                midans.append(get_output())
    elif stage==3:
        # TODO: 焼きなまし的に受け入れ、looplength, pathlength考慮
        if cur_loop_path_len[0]>new_loop_path_len[0]:
            rotate[i][j] = (rotate[i][j]-r)%4
        else:
            if debug:
                midans.append(get_output())
    """
    # stage更新
    if stage==0 and time()-ts>=0.4:
        stage += 1
        app_tmp = set()
        sts_sets = []
        for i in range(30):
            for j in range(30):
                if (i, j) in app_tmp:
                    continue
                lpl = loop_path_len(i, j)
                app_tmp += lpl[3]
                sts_sets.append(lpl[3])
        sts_sets = sorted(sts_sets, key=lambda x: len(x))

    """
    elif stage==1 and time()-ts>=1.4 and (not fixed_stage):
        if judge()==0:
            stage += 1
        else:
            fixed_stage = True
    elif stage==2 and time()-ts>=1.6 and (not fixed_stage):
        if judge()==0:
            stage += 1
        else:
            fixed_stage = True
    """

if debug:
    print(*midans, sep="\n")
printans()
