import math
import random
from time import time
import sys
import heapq
input = sys.stdin.readline

logfile = open("./log.txt", "w")
def logprint(*args):
    print(*args, file=logfile)
TS = time()
TL = 4.0
# [a, b)
def randint(a, b, n=1):
    if n==1:
        return int(random.random()*(b-a))+a
    else:
        return (int(random.random()*(b-a))+a for _ in range(n))

class Worker:
    def __init__(self, worker_id, v_init, Lmax, *job_types) -> None:
        self.worker_id = worker_id
        self.v_init = v_init
        self.Lmax = Lmax
        self.job_types = set(job_types)
        self.v = v_init
        # self.jobs = []
        # 移動完了までの時間
        self.move_cnt = 0
        self.acts = [None]*1000

    def clear(self):
        self.v = self.v_init
        self.acts = [None]*1000

class Job:
    def __init__(self, job_id, job_type, Ntask, v, rewards, depends) -> None:
        self.job_id = job_id
        self.job_type = job_type
        self.Ntask = Ntask
        self.rest_task = Ntask
        self.v = v
        self.depends = depends
        self.depend_cnt = len(depends)
        self.rewards = [0]*1000
        pre_t, pre_y = rewards[0], rewards[1]
        self.end_turn = math.inf
        it = iter(rewards[2:])
        for t, y in zip(it, it):
            t -= 1
            for i in range(pre_t, t):
                self.rewards[i] = (y-pre_y)*(i-pre_t)/(t-pre_t)+pre_y
            pre_t, pre_y = t, y
        self.max_rewards = max(self.rewards)

    def clear(self):
        self.depend_cnt = len(self.depends)
        self.end_turn = math.inf
        self.rest_task = self.Ntask


Tmax = int(input())
Nv, Ne = map(int, input().split())
UVD = [list(map(int, input().split())) for _ in range(Ne)]
Nworker = int(input())
workers = []
for worker_id in range(Nworker):
    v_init, Lmax, Njobtypes, *job_types = map(int, input().split())
    v_init -= 1
    worker = Worker(worker_id, v_init, Lmax, *job_types)
    workers.append(worker)
Njob = int(input())
jobs = []
for job_id in range(Njob):
    _, job_type, Ntask, vjob = map(int, input().split())
    vjob -= 1
    rewards = list(map(int, input().split()))[1:]
    depends = list(map(lambda x: int(x)-1, input().split()))[1:]
    job = Job(job_id, job_type, Ntask, vjob, rewards, depends)
    jobs.append(job)

G = [[] for _ in range(Nv)]
for u, v, d in UVD:
    u -= 1
    v -= 1
    G[u].append((v, d))
    G[v].append((u, d))

target_v = set([job.v for job in jobs]+[worker.v for worker in workers])
dists = [[math.inf]*Nv for _ in range(Nv)]
for start_node in range(Nv):
    if start_node not in target_v:
        continue
    q = [(0, start_node)]
    dists_ = dists[start_node]
    dists_[start_node] = 0
    appeared = [False]*Nv
    while q:
        d, u = heapq.heappop(q)
        if appeared[u]:
            continue
        appeared[u] = True
        for v, c in G[u]:
            d_ = d+c
            if dists_[v] > d_:
                dists_[v] = d_
                heapq.heappush(q, (d_, v))


class State:
    def __init__(self, workers, jobs, dists):
        self.workers = workers
        self.jobs = jobs
        self.dists = dists
        self.job_allocations = [[] for _ in range(len(self.workers))]

    @classmethod
    def get_initial_state(cls, **kwargs):
        ret = State(workers, jobs, dists)
        # 貪欲割り当て, それぞれ適当なscore最大化するjobを一つずつ割り当てる
        rest_jobs = set(range(len(jobs)))
        for turn in range(Tmax):
            for worker_id in range(Nworker):
                worker = ret.workers[worker_id]
                if worker.acts[turn] is not None:
                    continue
                best_job_id, best_score = None, -math.inf
                for job_id in rest_jobs:
                    job = jobs[job_id]
                    # 実行可能
                    if job.job_type not in worker.job_types:
                        continue
                    # 依存解決ずみ
                    is_ok = True
                    for depend_job_id in job.depends:
                        if not (jobs[depend_job_id].end_turn<=turn):
                            is_ok = False
                            break
                    if not is_ok:
                        continue
                    # スコア計算
                    move_turn = dists[worker.v][job.v]
                    exe_turn = math.ceil(job.rest_task/worker.Lmax)
                    start_turn = min(Tmax, turn+move_turn)
                    end_turn = min(turn+move_turn+exe_turn, Tmax)
                    n_task = min(job.rest_task, (end_turn-start_turn)*worker.Lmax)
                    if n_task<=0:
                        continue
                    rewards = (job.rewards[start_turn], job.rewards[end_turn-1])
                    if min(rewards)==0:
                        continue
                    # 移動ターンが少ない
                    # 収穫時期が良い
                    # rewardsが多い
                    score = (rewards[0]+rewards[1])/2*n_task/(move_turn+exe_turn) # /((move_turn+1)**4)
                    # 更新
                    if score > best_score:
                        best_score = score
                        best_job_id = job_id
                if best_job_id is None:
                    continue
                job_id = best_job_id
                job = ret.jobs[job_id]
                cur_worker_v = worker.v
                # 移動
                for turn_ in range(turn, turn+dists[cur_worker_v][job.v]):
                    worker.acts[turn_] = ("move", job.v)
                worker.v = job.v
                # job実行
                end_turn = min(Tmax, turn+dists[cur_worker_v][job.v]+math.ceil(job.rest_task/worker.Lmax))
                for turn_ in range(turn+dists[cur_worker_v][job.v], end_turn):
                    n = min(worker.Lmax, job.rest_task)
                    worker.acts[turn_] = ("execute", job_id, n)
                    job.rest_task -= n
                job.end_turn = end_turn
                rest_jobs.remove(job_id)
                ret.job_allocations[worker_id].append(job_id)
        # やりきれなかったものも適当に割り当てる
        # TODO: この区間も大事なので、移動距離を考慮しつつ、やる
        # 余分なものは省いてしまう。
        # turnが1.5Tmaxになるまでなるまでうける
        worker_last_move_turns = [ret.jobs[job_allocation[-1]].end_turn for job_allocation in ret.job_allocations]
        worker_vs = [worker.v for worker in workers]
        rest_job_allocation_cnt = [0]*len(workers)
        for _ in range(100):
            for worker_id, worker in enumerate(workers):
                if worker_last_move_turns[worker_id] > Tmax*1.2:
                    continue
                best_job_id = None
                best_job_metrics = (math.inf, math.inf, -math.inf) # move_turn, end_turn, reward
                for job_id in list(rest_jobs):
                    job = jobs[job_id]
                    if jobs[job_id].job_type not in worker.job_types:
                        continue
                    # 依存解決ずみ
                    is_ok = True
                    for depend_job_id in job.depends:
                        if not (jobs[depend_job_id].end_turn<=Tmax):
                            is_ok = False
                            break
                    if not is_ok:
                        continue
                    move_turn = dists[worker_vs[worker_id]][job.v]
                    exe_turn = math.ceil(job.rest_task/worker.Lmax)
                    metrics = (move_turn, move_turn+exe_turn, job.max_rewards)
                    if (metrics[0]<best_job_metrics[0]) or (metrics[0]==best_job_metrics[0] and metrics[2]>best_job_metrics[2]):
                        best_job_id = job_id
                        best_job_metrics = metrics
                if best_job_id is None:
                    continue
                ret.job_allocations[worker_id].append(best_job_id)
                worker_last_move_turns[worker_id] += best_job_metrics[1]
                worker_vs[worker_id] = jobs[best_job_id].v
        # 特別効率高いものをつける
        for job_id in sorted(list(rest_jobs), key=lambda x: jobs[x].max_rewards)[:len(workers)*3]:
            best_worker_id, best_dist = None, math.inf
            for worker_id, worker in enumerate(workers):
                if jobs[job_id].job_type in worker.job_types:
                    if dists[worker_vs[worker_id]][jobs[job_id].v]<best_dist:
                        best_worker_id = worker_id
                        best_dist = dists[worker_vs[worker_id]][jobs[job_id].v]
            if best_worker_id is None:
                continue
            ret.job_allocations[best_worker_id].append(job_id)
        return ret

    # 2opt (distで事前に改善を比較)
    # worker内move (distで事前に改善比較)
    # 期間swap (少なくとも片方は実行範囲)
    # 期間move (少なくとも片方は実行範囲)
    # 一点swap (少なくとも片方は実行範囲)
    # 一点move (少なくとも片方は実行範囲)
    def move(self, params):
        # jobのswap ("swap", i, j, x, y)
        # worker:iのx番目のjobとworker: jのy番目を入れ替え
        if params[0]=="swap":
            i, j, x, y = params[1:]
            job_id_x = self.job_allocations[i][x]
            job_id_y = self.job_allocations[j][y]
            self.job_allocations[i][x], self.job_allocations[j][y] = job_id_y, job_id_x
        # jobの移動 ("move", i, j, x, y)
        # worker:iのx番目のjobをworker: jのy番目に挿入
        elif params[0]=="move":
            i, j, x, y = params[1:]
            job_id = self.job_allocations[i].pop(x)
            self.job_allocations[j].insert(y, job_id)
        # 休憩挿入 ("ins_rest", i, x)
        elif params[0]=="ins_rest":
            i, x = params[1:]
            self.job_allocations[i].insert(x, -1)
        # 休憩pop ("pop_rest", i, x)
        elif params[0]=="pop_rest":
            i, x = params[1:]
            job_id = self.job_allocations[i].pop(x)
            assert job_id==-1
        elif params[0]=="swap_duration":
            i, j, x1, x2, y1, y2 = params[1:]
            job_allocation_i = self.job_allocations[i][:x1] + self.job_allocations[j][y1:y2] + self.job_allocations[i][x2:]
            job_allocation_j = self.job_allocations[j][:y1] + self.job_allocations[i][x1:x2] + self.job_allocations[j][y2:]
            self.job_allocations[i] = job_allocation_i
            self.job_allocations[j] = job_allocation_j
        elif params[0]=="move_duration":
            i, j, x1, x2, y = params[1:]
            job_allocation_i = self.job_allocations[i][:x1] + self.job_allocations[i][x2:]
            job_allocation_j = self.job_allocations[j][:y] + self.job_allocations[i][x1:x2] + self.job_allocations[j][y:]
            self.job_allocations[i] = job_allocation_i
            self.job_allocations[j] = job_allocation_j
        elif params[0]=="2opt":
            i, x, y = params[1:]
            job_allocation_i = self.job_allocations[i][:x] + self.job_allocations[i][x:y][::-1] + self.job_allocations[i][y:]
            self.job_allocations[i] = job_allocation_i
        elif params[0]=="single_opt":
            i, x, y = params[1:]
            job_id = self.job_allocations[i].pop(x)
            self.job_allocations[i].insert(y-(x<y), job_id)
        elif params[0]=="duration_opt":
            i, x1, x2, y = params[1:]
            job_allocation_i = self.job_allocations[i][:x1] + self.job_allocations[i][x2:]
            job_allocation_i = job_allocation_i[:y-(x1<y)*(x2-x1)] + self.job_allocations[i][x1:x2] + job_allocation_i[y-(x1<y)*(x2-x1):]
            self.job_allocations[i] = job_allocation_i
        else:
            pass

    def unmove(self, params):
        if params[0]=="swap":
            self.move(params)
        # jobの移動 ("move", i, j, x, y)
        # worker:iのx番目のjobをworker: jのy番目に挿入
        elif params[0]=="move":
            i, j, x, y = params[1:]
            self.move(("move", j, i, y, x))
        # 休憩挿入 ("ins_rest", i, x)
        elif params[0]=="ins_rest":
            i, x = params[1:]
            self.move(("pop_rest", i, x))
        # 休憩pop ("pop_rest", i, x)
        elif params[0]=="pop_rest":
            i, x = params[1:]
            self.move(("ins_rest", i, x))
        elif params[0]=="swap_duration":
            i, j, x1, x2, y1, y2 = params[1:]
            self.move(("swap_duration", i, j, x1, x1+(y2-y1), y1, y1+(x2-x1)))
        elif params[0]=="move_duration":
            i, j, x1, x2, y = params[1:]
            self.move(("move_duration", j, i, y, y+(x2-x1), x1))
        elif params[0]=="2opt":
            i, x, y = params[1:]
            self.move(("2opt", i, x, y))
        elif params[0]=="single_opt":
            i, x, y = params[1:]
            self.move(("single_opt", i, y-(x<y), x+(y<x)))
        elif params[0]=="duration_opt":
            i, x1, x2, y = params[1:]
            self.move(("duration_opt", i, y-(x1<y)*(x2-x1), y-(x1<y)*(x2-x1)+(x2-x1), x1+(y<x1)*(x2-x1)))
        else:
            pass

    # 2opt (distで事前に改善を比較)
    # worker内move (distで事前に改善比較)
    # 期間swap (少なくとも片方は実行範囲)
    # 期間move (少なくとも片方は実行範囲)
    # 一点swap (少なくとも片方は実行範囲)
    # 一点move (少なくとも片方は実行範囲)
    # rest関連
    def get_neighbor(self, *args):
        # swap, move, ins_rest, pop_rest, swap_duration, move_duration, 2opt, single_opt, duration_opt
        # percentages = [70, 90, 92, 100, 120, 140, 240, 280, 300]
        percentages_ = [20, 20, 20, 20, 20, 20, 20, 20, 20]
        percentages = [percentages_[0]]
        for x in percentages_[1:]:
            percentages.append(percentages[-1]+x)
        flg = randint(0, percentages[-1])
        # swap
        if flg<=percentages[0]:
            for _ in range(100):
                i, j = randint(0, len(self.workers)), randint(0, len(self.workers))
                if min(len(self.job_allocations[i]), len(self.job_allocations[j]))==0:
                    continue
                x, y = randint(0, len(self.job_allocations[i])), randint(0, len(self.job_allocations[j]))
                if i==j and x==y:
                    continue
                # xとyが入れ替え可能かチェック
                jobid_x, jobid_y = self.job_allocations[i][x], self.job_allocations[j][y]
                if jobid_x==-1 and jobid_y==-1:
                    continue
                if jobid_x!=-1 and self.jobs[jobid_x].job_type not in self.workers[j].job_types:
                    continue
                if jobid_y!=-1 and self.jobs[jobid_y].job_type not in self.workers[i].job_types:
                    continue
                return ("swap", i, j, x, y)
            return ("none", )
        # move
        elif flg<=percentages[1]:
            for _ in range(100):
                i, j = randint(0, len(self.workers)), randint(0, len(self.workers))
                x, y = randint(0, len(self.job_allocations[i])), randint(0, len(self.job_allocations[j])+(i!=j))
                if len(self.job_allocations[i])==0:
                    continue
                if i==j and x==y:
                    continue
                # xとyが入れ替え可能かチェック
                jobid_x = self.job_allocations[i][x]
                if jobid_x!=-1 and self.jobs[jobid_x].job_type in self.workers[j].job_types:
                    continue
                return ("move", i, j, x, y)
            return ("none", )
        # ins_rest
        elif flg<=percentages[2]:
            i = randint(0, len(self.workers))
            x = randint(0, len(self.job_allocations[i])+1)
            return ("ins_rest", i, x)
        # pop_rest
        elif flg<=percentages[3]:
            total_rest = sum([sum([x==-1 for x in job_allocation]) for job_allocation in self.job_allocations])
            if total_rest==0:
                return ("none", )
            target_rest = randint(0, total_rest)
            rest_cnt = 0
            for i in range(len(self.workers)):
                for x, job_id in enumerate(self.job_allocations[i]):
                    if job_id==-1:
                        if target_rest==rest_cnt:
                            return ("pop_rest", i, x)
                        rest_cnt += 1
        # swap, move, ins_rest, pop_rest, swap_duration, move_duration, 2opt, single_opt, duration_opt
        # swap_duration
        elif flg<=percentages[4]:
            for _ in range(100):
                i, j = randint(0, len(self.workers)), randint(0, len(self.workers))
                x1, x2 = randint(0, len(self.job_allocations[i]), n=2)
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = randint(0, len(self.job_allocations[j]), n=2)
                y1, y2 = min(y1, y2), max(y1, y2)
                worker_i, worker_j = self.workers[i], self.workers[j]
                flg = True
                for x in range(x1, x2):
                    flg = flg and (self.jobs[self.job_allocations[i][x]].job_type in worker_j.job_types)
                    if not flg:
                        break
                for y in range(y1, y2):
                    flg = flg and (self.jobs[self.job_allocations[j][y]].job_type in worker_i.job_types)
                    if not flg:
                        break
                if not flg:
                    continue
                return ("swap_duration", i, j, x1, x2, y1, y2)
            return ("none", )
        # move_duration
        elif flg<=percentages[5]:
            for _ in range(100):
                i, j = randint(0, len(self.workers)), randint(0, len(self.workers))
                x1, x2 = randint(0, len(self.job_allocations[i]), n=2)
                y = randint(0, len(self.job_allocations[j]))
                worker_i, worker_j = self.workers[i], self.workers[j]
                flg = True
                for x in range(x1, x2):
                    flg = flg and (self.jobs[self.job_allocations[i][x]].job_type in worker_j.job_types)
                    if not flg:
                        break
                if not flg:
                    continue
                return ("move_duration", i, j, x1, x2, y)
            return ("none", )
            pass
        # 2opt
        elif flg<=percentages[6]:
            for _ in range(100):
                i = randint(0, len(self.workers))
                x1, x2 = randint(0, len(self.job_allocations[i]), n=2)
                x1, x2 = min(x1, x2), max(x1, x2)
                job_id1, job_id2 = self.job_allocations[i][x1], self.job_allocations[i][x2]
                job_id_pre = self.job_allocations[i][x1-1] if x1!=0 else None
                job_id_aft = self.job_allocations[i][x2+1] if x2+1<len(self.job_allocations[i]) else None
                if -1 in [job_id1, job_id2, job_id_pre, job_id_aft]:
                    continue
                x1_v, x2_v = self.jobs[job_id1].v, self.jobs[job_id2].v
                pre_v = self.jobs[job_id_pre].v if job_id_pre is not None else self.workers[i].v_init
                # TODO: 違うけどまあよし
                aft_v = self.jobs[job_id_aft].v if job_id_aft is not None else self.workers[i].v_init
                if self.dists[pre_v][x2_v]+self.dists[x1_v][aft_v]<self.dists[pre_v][x1_v]+self.dists[x2_v][aft_v]:
                    return ("2opt", i, x1, x2)
            return ("none", )
        # single_opt
        elif flg<=percentages[7]:
            for _ in range(100):
                i = randint(0, len(self.workers))
                x1, x2 = randint(0, len(self.job_allocations[i]), n=2)
                job_id1 = self.job_allocations[i][x1]
                job_id_pre_x1 = self.job_allocations[i][x1-1] if x1!=0 else None
                job_id_aft_x1 = self.job_allocations[i][x1+1] if x1+1<len(self.job_allocations[i]) else None
                job_id_pre_x2 = self.job_allocations[i][x2-1] if x2!=0 else None
                job_id_aft_x2 = self.job_allocations[i][x2] if x2+1<len(self.job_allocations[i]) else None
                if -1 in [job_id1, job_id_pre_x1, job_id_aft_x1, job_id_pre_x2, job_id_aft_x2]:
                    continue
                x1_v = self.jobs[job_id1].v
                pre_x1_v = self.jobs[job_id_pre_x1].v if job_id_pre_x1 is not None else self.workers[i].v_init
                # TODO: 違うけどまあよし
                aft_x1_v = self.jobs[job_id_aft_x1].v if job_id_aft_x1 is not None else self.workers[i].v_init
                pre_x2_v = self.jobs[job_id_pre_x2].v if job_id_pre_x2 is not None else self.workers[i].v_init
                # TODO: 違うけどまあよし
                aft_x2_v = self.jobs[job_id_aft_x2].v if job_id_aft_x2 is not None else self.workers[i].v_init
                if self.dists[pre_x2_v][x1_v]+self.dists[x1_v][aft_x2_v]<self.dists[pre_x1_v][x1_v]+self.dists[x1_v][aft_x1_v]:
                    return ("single-opt", i, x1, x2)
            return ("none", )
        # duration_opt
        elif flg<=percentages[8]:
            return ("none", )
            pass


    # TODO: 実装次第で修正
    def get_neighbor_with_score(self, *args):
        params = self.get_neighbor()
        if params[0]=="none":
            return params, -math.inf
        prev_allocations = [[x for x in job_allocation] for job_allocation in self.job_allocations]
        self.move(params)
        score = self.get_score()
        self.job_allocations = prev_allocations
        return params, score
        self.unmove(params)
        for i, job_allocation in enumerate(self.job_allocations):
            assert job_allocation==prev_allocations[i], f"{params}"
        return params, score

    def get_score(self, **params):
        for worker in self.workers:
            worker.clear()
        for job in self.jobs:
            job.clear()
        job_idx = [0]*len(self.workers)
        score = 0
        for turn in range(Tmax):
            for worker_id in range(Nworker):
                worker = self.workers[worker_id]
                if (worker.acts[turn] is not None) or (len(self.job_allocations[worker_id])<=job_idx[worker_id]):
                    continue
                job_id = self.job_allocations[worker_id][job_idx[worker_id]]
                # -1は休憩
                if job_id == -1:
                    job_idx[worker_id] += 1
                    continue
                job = self.jobs[job_id]
                # 実行可能check
                if job.job_type not in worker.job_types:
                    return 0
                # job依存解決まで待機
                is_ok = True
                for depend_job_id in job.depends:
                    if not (jobs[depend_job_id].end_turn<=turn):
                        is_ok = False
                        break
                if not is_ok:
                    continue
                cur_worker_v = worker.v
                move_turn = dists[worker.v][job.v]
                exe_turn = math.ceil(job.rest_task/worker.Lmax)
                start_turn = min(Tmax, turn+move_turn)
                end_turn = min(turn+move_turn+exe_turn, Tmax)
                # 移動
                for turn_ in range(turn, start_turn):
                    worker.acts[turn_] = ("move", job.v)
                worker.v = job.v
                # 実行可能チェック
                rewards = (job.rewards[min(start_turn, Tmax-1)], job.rewards[end_turn-1])
                if job.rest_task<=0 or min(rewards)==0:
                    # 不可なら休憩
                    continue
                # job実行
                for turn_ in range(start_turn, end_turn):
                    n = min(worker.Lmax, job.rest_task)
                    assert n>0
                    worker.acts[turn_] = ("execute", job_id, n)
                    job.rest_task -= n
                    score += job.rewards[turn_]*n
                job.end_turn = end_turn
                job_idx[worker_id] += 1
        return score

    def print(self):
        # self.get_score()
        for i in range(Tmax):
            for worker in self.workers:
                act = worker.acts[i]
                if act is None:
                    print("stay")
                else:
                    if act[0]=="stay":
                        print(*act)
                    elif act[0]=="move":
                        print(act[0], act[1]+1)
                    elif act[0] == "execute":
                        print(act[0], act[1]+1, act[2])
                    else:
                        raise ValueError


    def copy(self):
        ret = State(workers, jobs, dists)
        ret.job_allocations = [list(x) for x in self.job_allocations]
        return ret


class Optimizer:
    def __init__(self, verbose=False):
        self.history = []

    def climbing(self, initial_state, TL_=TL, save_history=False):
        state = initial_state.copy()
        score = state.get_score()
        cnt = 0
        if save_history:
            self.history.append((cnt, score, state))
        TS_ = time()
        while time()-TS_<TL_:
            cnt += 1
            params, new_score = state.get_neighbor_with_score()
            if new_score >= score:
                state.move(params)
                score = new_score
                if save_history:
                    self.history.append((cnt, score, state.copy()))
        return state

    def simanneal(self, initial_state, TL_=TL, save_history=False):
        TS = time()
        state = initial_state.copy()
        score = state.get_score()
        best_state, best_score = state.copy(), score
        start_temp = 100000
        end_temp = 0.1
        cnt = 0
        if save_history:
            self.history.append((cnt, score, state))
        while True:
            cur_time =time()
            if cur_time-TS>=TL_:
                break
            cnt += 1
            params, new_score = state.get_neighbor_with_score()
            if params[0]=="none":
                continue
            temp = start_temp + (end_temp-start_temp)*(cur_time-TS)/TL_
            if (new_score>=score or math.exp((new_score-score)/temp) > random.random()):
                state.move(params)
                score = new_score
                if save_history:
                    self.history.append((cnt, score, state.copy()))
            if score > best_score:
                best_state, best_score = state.copy(), score
        return best_state

    def beam_search(self, TL=1.8, beam_width=10):
        raise NotImplementedError

    def visualize(self):
        raise NotImplementedError

init_state = State.get_initial_state()
# init_state.get_score()
# init_state.print()
opt = Optimizer()
# best_state = opt.climbing(init_state)
best_state = opt.simanneal(init_state, TL_=TL-(time()-TS))
best_state.get_score()
best_state.print()
# print(init_state.get_score(), best_state.get_score())
# print(best_state.get_score(), init_state.get_score())
# opt.visualize()
