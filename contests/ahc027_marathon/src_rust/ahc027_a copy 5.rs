#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused)]
use std::collections::{HashSet, HashMap, VecDeque};
use std::process::exit;
use std::{vec, cmp};
use itertools::Itertools;
use num::{range, ToPrimitive};
use num_integer::Roots;
use proconio::{input, marker::Chars};
use rand_core::block;
use std::collections::BinaryHeap;
use std::f64::consts::PI;
use std::fmt::Display;
use std::cmp::{Reverse, Ordering};
#[allow(unused_imports)]
use rand::prelude::*;
use std::time::Instant;
use ndarray::prelude::*;
use num::BigUint;


#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Input {
    n: usize,
    neighbors: Vec<Vec<usize>>,
    d: Vec<usize>,
    mindistmap: Vec<Vec<Vec<usize>>>, // (h, w)への最短経路で、(h_, w_)から次に向かう座標を複数
    mindist: Vec<Vec<usize>>, // (h, w)への最短経路で、(h_, w_)への距離
    cluster_points: HashSet<usize>, // dが一定以上の点の集合(入力生成の都合、クラスタっぽくなってる)
    cluster_neighbors: Vec<Vec<usize>>, // neighborsのうち、cluster_pointsに含まれる点のみ
}

impl Input {
    fn read_input() -> Self {
        input! {
            n: usize,
            h_: [Chars; n-1],
            v_: [Chars; n],
            d: [usize; n*n],
        }
        let mut d_ = d.clone();
        d_.sort();
        d_.reverse();
        let cluster_threshold = cmp::min(100, d_[n]);
        let mut cluster_points = HashSet::new();
        for idx in 0..n*n{
            if d[idx]>=100{
                cluster_points.insert(idx);
            }
        }
        let mut neighbors = vec![vec![]; n*n];
        let mut cluster_neighbors = vec![vec![]; n*n];
        for h in 0..n-1{
            for w in 0..n{
                if h_[h][w]=='0'{
                    let idx = h*n+w;
                    let idx_ = (h+1)*n+w;
                    neighbors[idx].push(idx_);
                    neighbors[idx_].push(idx);
                    if cluster_points.contains(&idx){
                        cluster_neighbors[idx_].push(idx);
                    }
                    if cluster_points.contains(&idx_){
                        cluster_neighbors[idx].push(idx_);
                    }
                }
            }
        }
        for h in 0..n{
            for w in 0..n-1{
                if v_[h][w]=='0'{
                    let idx = h*n+w;
                    let idx_ = h*n+w+1;
                    neighbors[idx].push(idx_);
                    neighbors[idx_].push(idx);
                    if cluster_points.contains(&idx){
                        cluster_neighbors[idx_].push(idx);
                    }
                    if cluster_points.contains(&idx_){
                        cluster_neighbors[idx].push(idx_);
                    }
                }
            }
        }
        let mut mindistmap = vec![vec![vec![]; n*n]; n*n];
        let mut mindist = vec![vec![usize::MAX; n*n]; n*n];
        for h in 0..n{
            for w in 0..n{
                let idx = h*n+w;
                let mut q = VecDeque::new();
                let mut appeared = vec![false; n*n];
                mindist[idx][idx] = 0;
                q.push_front((idx, 0));
                while !q.is_empty(){
                    let (idx_, d) = q.pop_front().unwrap();
                    if appeared[idx_]{
                        continue;
                    }
                    appeared[idx_] = true;
                    for &nidx in neighbors[idx_].iter(){
                        if mindist[idx][nidx]>=d+1{
                            mindistmap[h*n+w][nidx].push(idx_);
                            mindist[h*n+w][nidx] = d+1;
                            q.push_back((nidx, d+1));
                            mindist[idx][nidx]=d+1;
                        }
                    }
                }
            }
        }
        Self { n, neighbors, d, mindistmap, mindist, cluster_points, cluster_neighbors }
    }
}

struct Neighbor{
    fr: usize,
    to: usize,
    new_pathes: Vec<usize>,
    // old_pathes: Vec<usize>,
    neighbortype: usize,
}

#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct State{
    pathes: Vec<usize>,
    cnts: Vec<usize>,
}

impl State{
    fn init_state(input: &Input) -> Self{
        let mut pathes =  vec![];
        let mut rng = rand::thread_rng();
        let mut cur = 0;
        pathes.push(0);
        for loop_i in 0..(30-input.n/2){
        // for loop_i in 0..15{
                // ランダムに未訪問の点を選んで向かう。途中でできるだけ未訪問の点を選ぶ
            let mut unvisited = HashSet::new();
            for h in 0..input.n{
                for w in 0..input.n{
                    unvisited.insert(h*input.n+w);
                }
            }
            unvisited.remove(&cur);
            while unvisited.len()>0{
                let mut to = unvisited.iter().choose(&mut rng).unwrap().clone();
                let mut q = VecDeque::new();
                q.push_front(cur);
                let mut appeared = vec![false; input.n*input.n];
                while !q.is_empty(){
                    to = q.pop_front().unwrap();
                    if unvisited.contains(&to){
                        break;
                    }
                    // if appeared[to.0][to.1]{
                    //     continue;
                    // }
                    for &idx in input.neighbors[to].iter(){
                        if !appeared[idx]{
                            q.push_back(idx);
                            appeared[idx] = true;
                        }
                    }
                }
                let mut idxs = input.neighbors[cur].clone();
                if loop_i%2==0{
                // if loop_i>(30-input.n/2)/2{
                    idxs.reverse();
                }
                // idxs.shuffle(&mut rng);
                // idxs = [&idxs[loop_i%idxs.len()..], &idxs[..loop_i%idxs.len()]].concat();
                for &idx in idxs.iter(){
                    if unvisited.contains(&idx){
                        to = idx;
                        break;
                    }
                }
                while cur!=to{
                    let next = input.mindistmap[to][cur].choose(&mut rng).unwrap().clone();
                    // eprintln!("cur: {:?}, to: {:?}, next: {:?}", cur, to, next);
                    unvisited.remove(&next);
                    pathes.push(next);
                    cur = next;
                }
            }
            let to = rng.gen::<usize>()%(input.n*input.n);
            while cur!=to{
                let next = input.mindistmap[to][cur].choose(&mut rng).unwrap().clone();
                // eprintln!("cur: {:?}, to: {:?}, next: {:?}", cur, to, next);
                unvisited.remove(&next);
                pathes.push(next);
                cur = next;
            }
            // eprintln!("{:?}", pathes);
            //panic!();
        }
        let to = 0;
        while cur!=to{
            let next = input.mindistmap[to][cur].choose(&mut rng).unwrap().clone();
            // eprintln!("cur: {:?}, to: {:?}, next: {:?}", cur, to, next);
            pathes.push(next);
            cur = next;
        }

        // eprintln!("{:?}", pathes);
        // panic!();
        // let mut visited = vec![vec![false; input.n]; input.n];
        // let mut q = VecDeque::new();
        // q.push_front((1, 0, 0));
        // while !q.is_empty(){
        //     let (is_go, h, w) = q.pop_front().unwrap();
        //     if is_go==1 && visited[h][w]{
        //         continue;
        //     }
        //     if is_go==0 && visited[h][w]{
        //         let (h_, w_) = pathes.last().unwrap();
        //         if !(h_==&h && w_==&w){
        //             pathes.push((h, w));
        //         }
        //         continue;
        //     }
        //     pathes.push((h, w));
        //     visited[h][w] = true;
        //     for &(nh, nw) in input.neighbors[h][w].iter(){
        //         if !visited[nh][nw]{
        //             q.push_front((0, h, w));
        //             q.push_front((1, nh, nw));
        //         }
        //     }
        // }
        // pathes.append(&mut pathes[1..].to_vec().clone());
        let mut cnts = vec![0; input.n*input.n];
        for &idx in pathes.iter(){
            cnts[idx] += 1;
        }
        Self {pathes, cnts}
    }

    fn init_state2(input: &Input) -> Self{
        // ちょっと貪欲っぽくやる
        let mut pathes =  vec![];
        let mut rng = rand::thread_rng();
        let mut cur = 0;
        pathes.push(0);
        let mut last_visited = vec![0; input.n*input.n];
        let mut unvisited = HashSet::new();
        for i in 0..input.n*input.n{
            unvisited.insert(i);
        }
        while !unvisited.is_empty(){
            let mut turn = pathes.len()+1000;
            // 基本、近隣の汚れたマスに向かう
            // 低確立で最も汚れたマスに向かう
            let mut to = cur;
            let mut max_d = 0;
            let mut idxs = HashSet::new();
            idxs.insert(cur);
            for _ in 0..5{
                let mut idxs_ = HashSet::new();;
                for &idx in idxs.iter(){
                    idxs_.insert(idx);
                    for &idx_ in input.neighbors[idx].iter(){
                        idxs_.insert(idx_);
                    }
                }
                idxs = idxs_;
            }
            for &idx in idxs.iter(){
                if (turn-last_visited[idx])*input.d[idx]>max_d{
                    to = idx;
                    max_d = (turn-last_visited[idx])*input.d[idx];
                }
            }
            if rng.gen_bool(0.00005){
                let mut max_d = 0;
                for idx in 0..input.n*input.n{
                    if (turn-last_visited[idx])*input.d[idx]>max_d{
                        to = idx;
                        max_d = (turn-last_visited[idx])*input.d[idx];
                    }
                }
            }
            while cur!=to{
                let mut next = input.mindistmap[to][cur][0];
                for &idx in input.mindistmap[to][cur].iter(){
                    if last_visited[next]>last_visited[idx]{
                        next = idx;
                    }
                }
                // let next = *input.mindistmap[to][cur].choose(&mut rng).unwrap();
                unvisited.remove(&next);
                last_visited[next] = turn;
                turn += 1;
                pathes.push(next);
                cur = next;
            }
        }
        while cur!=0{
            let next = *input.mindistmap[0][cur].choose(&mut rng).unwrap();
            unvisited.remove(&next);
            last_visited[next] = pathes.len();
            pathes.push(next);
            cur = next;

        }
        let mut cnts = vec![0; input.n*input.n];
        for &idx in pathes.iter(){
            cnts[idx] += 1;
        }
        Self {pathes, cnts}
    }


    fn update(&mut self, params: Neighbor){
        // pathesのfrからtoまでのpathをparams.pathesに更新する
        for &idx in self.pathes[params.fr..params.to].iter(){
            self.cnts[idx] -= 1;
        }
        self.pathes = [&self.pathes[..params.fr], &params.new_pathes, &self.pathes[params.to..]].concat();
        for &idx in params.new_pathes.iter(){
            self.cnts[idx] += 1;
        }
    }


    fn get_neighbor(&mut self, input: &Input) -> Neighbor{
        let mut rng = rand::thread_rng();
        let mut flg = rng.gen::<usize>()%110;
        // 近傍0: 何もしない
        // 近傍1: (x, y)->(x+1, y)のところで(x, y)->(x, y+1)->(x+1, y+1)->(x+1, y) と寄り道する
        // 近傍2: (x, y)->(x+1, y)->(x,y)と寄り道する
        // 近傍3: (x, y)->(x_, y_)のところで(mx, my)に寄り道する
        // 近傍4: (x, y)->(x_, y_)のところで最短経路でいかせる
        // 近傍5: (x, y)->..(x+1, y)のところを省略する
        // 近傍6: (x, x)->..(x, x)のところを逆順にする
        // 近傍7: クラスタへ寄り道
        // 近傍8: パスのswap
        // 近傍9: (x, x)->..(x, x)のところを逆順にする, 前後同じところは省く
        // 近傍10: (x, x)->..(x, x)のループを移動
        // 近傍10: (x, x)->..(x, x)のループをコピー
        //                            0   1   2   3   4   5   6   7   8   9, 10, 11
        const FLG_IDX: [usize; 12] = [0, 20, 20, 30, 60, 90, 90, 90, 95, 100, 110, 110];
        // const FLG_IDX: [usize; 10] = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
        // const FLG_IDX: [usize; 10] = [0, 20, 20, 30, 40, 70, 80, 80, 90, 100, 110];
        if flg<FLG_IDX[1]{
            // 近傍1: (x, y)->(x+1, y)のところで(x, y)->(x, y+1)->(x+1, y+1)->(x+1, y) と寄り道する
            let fr = rng.gen::<usize>()%(self.pathes.len()-1);
            let to = fr+1;
            let idx = self.pathes[fr];
            let idx_ = self.pathes[to];
            // ちょいきもい
            let mut neighbor_pairs = vec![];
            // TODO: 事前計算
            for &nidx in input.neighbors[idx].iter(){
                if nidx==idx_{
                    continue;
                }
                for &nidx_ in input.neighbors[idx_].iter(){
                    if nidx_==idx{
                        continue;
                    }
                    neighbor_pairs.push((nidx, nidx_));
                }
            }
            neighbor_pairs.shuffle(&mut rng);
            for (nidx, nidx_) in neighbor_pairs.iter(){
                if input.neighbors[*nidx].contains(nidx_){
                    return Neighbor{fr, to, new_pathes: vec![idx, *nidx, *nidx_], neighbortype: 1};
                }
            }
        } else if flg<FLG_IDX[2]{
            // 近傍2: (x, y)->(x+1, y)->(x,y)と寄り道する
            loop{
                let fr = 1+rng.gen::<usize>()%(self.pathes.len()-2);
                let to = fr+1;
                let idx = self.pathes[fr];
                let idx_ = self.pathes[to];
                let idx__ = self.pathes[fr-1];
                if input.neighbors[idx].len()<=2{
                    continue;
                }
                loop {
                    let nidx = *input.neighbors[idx].choose(&mut rng).unwrap();
                    if nidx==idx_ || nidx==idx__{
                        continue;
                    }
                    return Neighbor{fr, to, new_pathes: vec![idx, nidx, idx], neighbortype: 2};
                }
            }
        }else if flg<FLG_IDX[3]{
            // 近傍3: (x, y)->(x_, y_)のところで(mx, my)に寄り道する
            let fr = rng.gen::<usize>()%(self.pathes.len()-3);
            // let to = fr+1+rng.gen::<usize>()%(self.pathes.len()-(fr+1)-1);
            let to = cmp::min(self.pathes.len()-1, fr+1+rng.gen::<usize>()%100);
            let idx = self.pathes[fr];
            let idx_ = self.pathes[to];
            let midx = rng.gen::<usize>()%(input.n*input.n);
            let mut new_pathes = vec![];
            let mut cur = idx;
            while cur!=midx{
                let next = *input.mindistmap[midx][cur].choose(&mut rng).unwrap();
                new_pathes.push(next);
                cur = next;
            }
            while cur!=idx_{
                let next = *input.mindistmap[idx_][cur].choose(&mut rng).unwrap();
                new_pathes.push(next);
                cur = next;
            }
            return Neighbor{fr: fr+1, to: to+1, new_pathes, neighbortype: 3};
        } else if flg<FLG_IDX[4]{
            // 近傍4: (x, y)->(x_, y_)のところで最短経路でいかせる
            let fr = rng.gen::<usize>()%(self.pathes.len()-3);
            let to = cmp::min(self.pathes.len()-1, fr+1+rng.gen::<usize>()%100);
            let idx = self.pathes[fr];
            let idx_ = self.pathes[to];
            let mut new_pathes = vec![];
            let mut cur = idx;
            while cur!=idx_{
                let next = *input.mindistmap[idx_][cur].choose(&mut rng).unwrap();
                new_pathes.push(next);
                cur = next;
            }
            return Neighbor{fr: fr+1, to: to+1, new_pathes, neighbortype: 4};
        } else if flg<FLG_IDX[5]{
            // 近傍5: (x, y)->..(x+1, y)のところを省略する
            let fr = rng.gen::<usize>()%(self.pathes.len()-2);
            let idx = self.pathes[fr];
            // let mut idxs = (fr+2..self.pathes.len()).collect::<Vec<_>>();
            // idxs.shuffle(&mut rng);
            // for to in fr+2..cmp::min(self.pathes.len(), fr+100){
            for to in fr+2..self.pathes.len(){
                let idx_ = self.pathes[to];
                if input.neighbors[idx].contains(&idx_){
                    return Neighbor{fr: fr+1, to, new_pathes: vec![], neighbortype: 5};
                }
            }
        } else if flg<FLG_IDX[6]{
            // 近傍6: (x, x)->..(x, x)のところを逆順にする
            let fr = rng.gen::<usize>()%(self.pathes.len()-2);
            let idx = self.pathes[fr];
            if self.cnts[idx]<=1{
                return Neighbor { fr: 0, to: 0, new_pathes: vec![], neighbortype: 0};
                // continue;
            }
            // let mut idxs = (fr+2..self.pathes.len()).collect::<Vec<_>>();
            // idxs.shuffle(&mut rng);
            for to in fr+2..self.pathes.len(){
                let idx_ = self.pathes[to];
                if idx==idx_{
                    let mut new_pathes = self.pathes[fr+1..to].to_vec().clone();
                    new_pathes.reverse();
                    return Neighbor{fr: fr+1, to, new_pathes, neighbortype: 6};
                }
            }
        } else if flg<FLG_IDX[7]{
            // 近傍7: クラスタへ寄り道
            let fr = rng.gen::<usize>()%(self.pathes.len()-3);
            // let to = fr+1+rng.gen::<usize>()%(self.pathes.len()-(fr+1)-1);
            let to = cmp::min(self.pathes.len()-1, fr+1+rng.gen::<usize>()%100);
            let idx = self.pathes[fr];
            let idx_ = self.pathes[to];
            let midx = *input.cluster_points.iter().choose(&mut rng).unwrap();
            let mut appeared = HashSet::new();
            let mut new_pathes = vec![];
            let mut cur = idx;
            appeared.insert(cur);
            while cur!=midx{
                let next = *input.mindistmap[midx][cur].choose(&mut rng).unwrap();
                new_pathes.push(next);
                cur = next;
                appeared.insert(cur);
            }
            loop{
                let mut found = false;
                // TODO: 2マス先まで見る?
                for &next in input.cluster_neighbors[cur].iter(){
                    if !appeared.contains(&next){
                        new_pathes.push(next);
                        cur = next;
                        appeared.insert(cur);
                        found = true;
                        break;
                    }
                }
                if !found{
                    break;
                }
            }
            while cur!=idx_{
                let next = *input.mindistmap[idx_][cur].choose(&mut rng).unwrap();
                new_pathes.push(next);
                cur = next;
            }
            return Neighbor{fr: fr+1, to: to+1, new_pathes, neighbortype: 7};
        } else if flg<FLG_IDX[8]{
            // 近傍8: パスのswap
            let fr = rng.gen::<usize>()%(self.pathes.len()-2);
            let idx = self.pathes[fr];
            if self.cnts[idx]<=1{
                return Neighbor { fr: 0, to: 0, new_pathes: vec![], neighbortype: 0};
                // continue;
            }
            let mut candidates = HashMap::new();
            for to in fr+2..cmp::min(self.pathes.len(), fr+100){
                let idx_ = self.pathes[to];
                if self.cnts[idx_]>=2{
                    candidates.insert(idx_, to);
                }
            }
            for fr2 in 0..self.pathes.len()-2{
                if fr==fr2||self.pathes[fr2]!=idx{
                    continue;
                }
                let idx2 = self.pathes[fr2];
                for to2 in fr2+2..cmp::min(self.pathes.len(), fr2+100){
                    let idx2_ = self.pathes[to2];
                    if let Some(&to) = candidates.get(&idx2_){
                        if cmp::max(fr, fr2)<=cmp::min(to, to2){
                            continue;
                        }
                        let (fr_min, to_min) = (cmp::min(fr, fr2), cmp::min(to, to2));
                        let (fr_max, to_max) = (cmp::max(fr, fr2), cmp::max(to, to2));
                        if to_min-fr_min==to_max-fr_max && self.pathes[fr_min..to_min]==self.pathes[fr_max..to_max]{
                            continue;
                        }
                        let mut new_pathes = [self.pathes[fr_max..to_max].to_vec(), self.pathes[to_min..fr_max].to_vec(), self.pathes[fr_min..to_min].to_vec()].concat();
                        return Neighbor{fr: fr_min, to: to_max, new_pathes, neighbortype: 8}
                    }
                }
            }
        } else if flg<FLG_IDX[9]{
            // 近傍9: (x, x)->..(x, x)のところを逆順にする, 前後同じところは省く
            let fr = rng.gen::<usize>()%(self.pathes.len()-2);
            let idx = self.pathes[fr];
            if self.cnts[idx]<=1{
                return Neighbor { fr: 0, to: 0, new_pathes: vec![], neighbortype: 0};
                // continue;
            }
            for to in fr+2..self.pathes.len(){
                let idx_ = self.pathes[to];
                if idx==idx_{
                    let mut fr_b = 0;
                    while fr_b<fr && self.pathes[fr-fr_b-1]==self.pathes[to-fr_b-1] && rng.gen_bool(0.95){
                        fr_b += 1;
                    }
                    let mut to_f = 0;
                    while fr+to_f+2<to-fr_b && to+to_f<self.pathes.len()-1 && self.pathes[to+to_f+1]==self.pathes[fr+to_f+1] && rng.gen_bool(0.9){
                        to_f += 1;
                    }
                    let mut new_pathes = self.pathes[fr+to_f..to-fr_b+1].to_vec().clone();
                    new_pathes.reverse();
                    return Neighbor{fr: fr-fr_b, to: to+to_f+1, new_pathes, neighbortype: 9};
                }
            }
        } else if flg<FLG_IDX[10]{
            // 近傍10: (x, x)->..(x, x)のループを移動
            let fr = rng.gen::<usize>()%(self.pathes.len()-2);
            let idx = self.pathes[fr];
            if self.cnts[idx]<=2{
                return Neighbor { fr: 0, to: 0, new_pathes: vec![], neighbortype: 0};
                // continue;
            }
            let mut turns = vec![];
            for (t, &idx_) in self.pathes.iter().enumerate(){
                if idx==idx_{
                    turns.push(t);
                }
            }
            let mut fr_idx = (0..turns.len()-1).choose(&mut rng).unwrap();
            let mut fr2_idx = (0..turns.len()-1).choose(&mut rng).unwrap();
            while fr_idx==fr2_idx{
                fr2_idx = (0..turns.len()-1).choose(&mut rng).unwrap();
            }
            if fr_idx>fr2_idx{
                std::mem::swap(&mut fr_idx, &mut fr2_idx);
            }
            let (fr, to) = (turns[fr_idx], turns[fr_idx+1]);
            let (fr2, to2) = (turns[fr2_idx], turns[fr2_idx+1]);
            let new_pathes = [self.pathes[fr2..to2].to_vec(), self.pathes[to..fr2].to_vec(), self.pathes[fr..to].to_vec()].concat();
            if fr<fr2{
                return Neighbor{fr, to: fr2, new_pathes: [self.pathes[to..fr2].to_vec(), self.pathes[fr..to].to_vec()].concat(), neighbortype: 10};
            } else {
                return Neighbor{fr: fr2, to, new_pathes: [self.pathes[fr..to].to_vec(), self.pathes[fr2..fr].to_vec()].concat(), neighbortype: 10};

            }
        } else if flg<FLG_IDX[11]{
            // 近傍11: (x, x)->..(x, x)のループをコピー
            let fr = rng.gen::<usize>()%(self.pathes.len()-2);
            let idx = self.pathes[fr];
            if self.cnts[idx]<=2{
                return Neighbor { fr: 0, to: 0, new_pathes: vec![], neighbortype: 0};
                // continue;
            }
            let mut turns = vec![];
            for (t, &idx_) in self.pathes.iter().enumerate(){
                if idx==idx_{
                    turns.push(t);
                }
            }
            let mut fr_idx = (0..turns.len()-1).choose(&mut rng).unwrap();
            let mut fr2_idx = (0..turns.len()-1).choose(&mut rng).unwrap();
            while fr_idx==fr2_idx{
                fr2_idx = (0..turns.len()-1).choose(&mut rng).unwrap();
            }
            if fr_idx>fr2_idx{
                std::mem::swap(&mut fr_idx, &mut fr2_idx);
            }
            let (fr, to) = (turns[fr_idx], turns[fr_idx+1]);
            let (fr2, to2) = (turns[fr2_idx], turns[fr2_idx+1]);
            let new_pathes = [self.pathes[fr2..to2].to_vec(), self.pathes[to..fr2].to_vec(), self.pathes[fr..to].to_vec()].concat();
            return Neighbor{fr: fr2, to: fr2, new_pathes: self.pathes[fr..to].to_vec(), neighbortype: 10};
        }
        Neighbor { fr: 0, to: 0, new_pathes: vec![], neighbortype: 0}
    }

    fn get_score_from_neighbor(&self, input: &Input, neighbor: &Neighbor)->f64{
        let l = self.pathes.len()+neighbor.new_pathes.len()+neighbor.fr-neighbor.to-1;
        let s1 = neighbor.fr;
        let s2 = neighbor.fr+neighbor.new_pathes.len();
        let mut last_visit = vec![usize::MAX; input.n*input.n];
        let mut s = 0;
        for (i, &idx) in self.pathes[..neighbor.fr].iter().enumerate(){
            last_visit[idx] = i;
        }
        for (i, &idx) in neighbor.new_pathes.iter().enumerate(){
            last_visit[idx] = i+s1;
        }
        for (i, &idx) in self.pathes[neighbor.to..].iter().enumerate(){
            last_visit[idx] = i+s2;
        }
        if last_visit.iter().max().unwrap()==&usize::MAX{
            return f64::MAX;
        }
        for (i, &idx) in self.pathes[..neighbor.fr].iter().enumerate(){
            let lv = last_visit[idx];
            s += input.d[idx]*(i+l-lv)*(i+l-lv-1)/2;
            last_visit[idx] = i+l;
        }
        for (i, &idx) in neighbor.new_pathes.iter().enumerate(){
            let lv = last_visit[idx];
            s += input.d[idx]*(i+s1+l-lv)*(i+s1+l-lv-1)/2;
            last_visit[idx] = i+s1+l;
        }
        for (i, &idx) in self.pathes[neighbor.to..].iter().enumerate(){
            let lv = last_visit[idx];
            s += input.d[idx]*(i+s2+l-lv)*(i+s2+l-lv-1)/2;
            last_visit[idx] = i+s2+l;
        }
        (s as f64)/(l as f64)
    }

    fn get_score(&self, input: &Input) -> f64{
        let l = self.pathes.len()-1;
        let mut last_visit = vec![usize::MAX; input.n*input.n];
        let mut s = 0;
        for (i, &idx) in self.pathes.iter().enumerate(){
            last_visit[idx] = i;
        }
        if last_visit.iter().max().unwrap()==&usize::MAX{
            return f64::MAX;
        }
        for (i, &idx) in self.pathes.iter().enumerate(){
            let lv = last_visit[idx];
            s += input.d[idx]*(i+l-lv)*(i+l-lv-1)/2;
            last_visit[idx] = i+l;
        }
        (s as f64)/(l as f64)
    }

    fn print(&mut self, input: &Input){
        let mut ans = "".to_string();
        for i in 0..self.pathes.len()-1{
            let (x, y) = (self.pathes[i]/input.n, self.pathes[i]%input.n);
            let (x_, y_) = (self.pathes[i+1]/input.n, self.pathes[i+1]%input.n);
            if x+1==x_{
                ans += "D";
            } else if x==x_+1{
                ans += "U";
            } else if y+1==y_{
                ans += "R";
            } else if y==y_+1{
                ans += "L";
            } else {
                panic!("cant move fr({} {}) to({} {})", x, y, x_, y_);
            }
        }
        println!("{}", ans);
    }
}


fn simanneal(input: &Input, init_state: State, timer: &Instant, limit: f64) -> State{
    let mut state = init_state.clone();
    let mut best_state = init_state.clone();
    let mut best_score = best_state.get_score(input);
    let mut cur_score = best_score;
    let mut all_iter = 0;
    let mut accepted_cnt = 0;
    let mut rng = rand::thread_rng();
    let start_temp: f64 = 1000.0;
    let end_temp: f64 = 0.1;
    let mut temp = start_temp;
    let mut neighbor_cnt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let mut neighbor_improve = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let mut elasped_time = timer.elapsed().as_secs_f64();
    eprintln!("simanneal start at {}", elasped_time);
    loop {
        if all_iter%100==0{
            elasped_time = timer.elapsed().as_secs_f64();
            if elasped_time >= limit{
                break;
            }
        }
        all_iter += 1;
        let neighbor = state.get_neighbor(input);
        if neighbor.neighbortype==0{
            continue;
        }
        let new_score = state.get_score_from_neighbor(input, &neighbor);
        // state.update(&neighbor);
        // let new_score = state.get_score(input);
        // eprintln!("{} {}", new_score, state.get_score(input));
        let score_diff = cur_score-new_score;
        // start_temp = (start_temp*(all_iter-1)+score_diff)/all_iter;
        temp = start_temp + (end_temp-start_temp)*elasped_time/limit;
        if score_diff>=0.0 || rng.gen_bool((score_diff/temp).exp()){
            // state.print(input);
            accepted_cnt += 1;
            if accepted_cnt%100000==0{
                eprintln!("{} {} {} {:?} {:?}", cur_score, new_score, all_iter, state.pathes.len(), neighbor_improve);
            }
            // eprintln!("neighbor_cnt  : {:?}", neighbor_cnt);
            // eprintln!("neighbor_improvement  : {:?}", neighbor_improve);
            if elasped_time>=1.0{
                neighbor_cnt[neighbor.neighbortype] += 1;
                neighbor_improve[neighbor.neighbortype] += score_diff as usize;
            }
            state.update(neighbor);
            cur_score = new_score;
            if new_score<best_score{
                best_state = state.clone();
                best_score = new_score;
            }
        // } else {
        //    state.undo(&neighbor);
        }
    }
    eprintln!("===== simanneal =====");
    eprintln!("all iter   : {}", all_iter);
    eprintln!("accepted   : {}", accepted_cnt);
    eprintln!("score  : {}", best_state.get_score(input));
    eprintln!("neighbor_cnt  : {:?}", neighbor_cnt);
    eprintln!("neighbor_improvement  : {:?}", neighbor_improve);
    eprintln!("");
    best_state
}

fn main() {
    let timer = Instant::now();
    let input = Input::read_input();
    eprintln!("et: {}", timer.elapsed().as_secs_f64());
    solve(&input, &timer, 1.85);
}

fn solve(input: &Input, timer:&Instant, tl: f64){
    let mut init_state = State::init_state(input);
    // let et = timer.elapsed().as_secs_f64();
    // let mut stetes = vec![];
    // stetes.push(simanneal(input, init_state.clone(), timer, et+0.2));
    // stetes.push(simanneal(input, init_state.clone(), timer, et+0.4));
    // stetes.push(simanneal(input, init_state.clone(), timer, et+0.6));
    // let mut best_state = stetes[0].clone();
    // for state in stetes.iter(){
    //     if state.get_score(input)<best_state.get_score(input){
    //         best_state = state.clone();
    //     }
    // }
    // best_state = simanneal(input, best_state, timer, tl);
    let mut best_state = simanneal(input, init_state, timer, tl);
    best_state.print(input);
    // best_state = simanneal(input, best_state, timer, tl+10.0);
    // best_state.print(input);
}
