#![allow(non_snake_case)]
#![allow(unused_variables)]
use std::collections::{HashSet, HashMap, VecDeque};
use std::{vec, cmp};
use itertools::Itertools;
use proconio::{input, marker::Chars};
#[allow(unused_imports)]
use rand::prelude::*;
use std::time::Instant;


#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Input {
    n: usize,
    neighbors: Vec<Vec<usize>>,
    neighbor_pairs: Vec<Vec<Vec<(usize, usize)>>>,
    neighbor_pairs2: Vec<Vec<Vec<(usize, usize)>>>,
    neighbor_path_2dist: Vec<HashMap<usize, Vec<usize>>>,
    d: Vec<usize>,
    mindistmap: Vec<Vec<usize>>, // (h, w)への最短経路で、(h_, w_)から次に向かう単一の座標
}

impl Input {
    fn read_input() -> Self {
        input! {
            n: usize,
            h_: [Chars; n-1],
            v_: [Chars; n],
            d: [usize; n*n],
        }
        let mut neighbors = vec![vec![]; n*n];
        for h in 0..n-1{
            for w in 0..n{
                if h_[h][w]=='0'{
                    let idx = h*n+w;
                    let idx_ = idx+n;
                    neighbors[idx].push(idx_);
                    neighbors[idx_].push(idx);
                }
            }
        }
        for h in 0..n{
            for w in 0..n-1{
                if v_[h][w]=='0'{
                    let idx = h*n+w;
                    let idx_ = idx+1;
                    neighbors[idx].push(idx_);
                    neighbors[idx_].push(idx);
                }
            }
        }
        let mut neighbor_path_2dist = vec![HashMap::new(); n*n];
        for idx in 0..n*n{
            for &idx1 in neighbors[idx].iter(){
                for &idx2 in neighbors[idx1].iter(){
                    neighbor_path_2dist[idx].entry(idx2).or_insert(vec![]).push(idx1);
                }
            }
        }
        let mut neighbor_pairs = vec![vec![vec![]; n*n]; n*n];
        let mut neighbor_pairs2 = vec![vec![vec![]; n*n]; n*n];
        for h in 0..n-1{
            for w in 0..n-1{
                if h_[h][w]=='0' && v_[h][w]=='0' && h_[h][w+1]=='0' && v_[h+1][w]=='0'{
                    let idx00 = h*n+w;
                    let idx01 = idx00+n;
                    let idx10 = idx00+1;
                    let idx11 = idx00+n+1;
                    neighbor_pairs[idx00][idx01].push((idx10, idx11));
                    neighbor_pairs[idx00][idx10].push((idx01, idx11));
                    neighbor_pairs[idx01][idx00].push((idx11, idx10));
                    neighbor_pairs[idx01][idx11].push((idx00, idx10));
                    neighbor_pairs[idx10][idx00].push((idx11, idx01));
                    neighbor_pairs[idx10][idx11].push((idx00, idx01));
                    neighbor_pairs[idx11][idx01].push((idx10, idx00));
                    neighbor_pairs[idx11][idx10].push((idx01, idx00));

                    neighbor_pairs2[idx00][idx11].push((idx01, idx10));
                    neighbor_pairs2[idx11][idx00].push((idx01, idx10));
                    neighbor_pairs2[idx01][idx10].push((idx00, idx11));
                    neighbor_pairs2[idx10][idx01].push((idx00, idx11));
                }
            }
        }
        let mut mindistmap = vec![vec![n*n; n*n]; n*n];
        for h in 0..n{
            for w in 0..n{
                let idx = h*n+w;
                let mut q = VecDeque::new();
                let mut appeared = vec![false; n*n];
                let mut mindist = vec![usize::MAX; n*n];
                mindist[idx] = 0;
                q.push_front((idx, 0));
                while !q.is_empty(){
                    let (idx_, d) = q.pop_front().unwrap();
                    if appeared[idx_]{
                        continue;
                    }
                    appeared[idx_] = true;
                    for &nidx in neighbors[idx_].iter(){
                        if mindist[nidx]>=d+1{
                            if mindistmap[idx][nidx]==n*n{
                                mindistmap[idx][nidx] = idx_;
                            }
                            // mindistmap[idx][nidx].push(idx_);
                            mindist[nidx] = d+1;
                            q.push_back((nidx, d+1));
                        }
                    }
                }
            }
        }
        Self { n, neighbors, neighbor_pairs, neighbor_pairs2, neighbor_path_2dist, d, mindistmap}
    }

    fn stamps2score(&self, stamp: &Vec<usize>, idx: usize, l: usize) -> f64{
        if stamp.is_empty(){
            return 10000000000.0;
        }
        let mut s = 0;
        let d = self.d[idx];
        let mut sorted_stamp = stamp.clone();
        sorted_stamp.sort();
        for i in 0..sorted_stamp.len()-1{
            let a = sorted_stamp[i];
            let b = sorted_stamp[i+1];
            s += d*(b-a)*(b-a-1)/2;
        }
        let a = sorted_stamp[sorted_stamp.len()-1];
        let b = sorted_stamp[0]+l;
        s += d*(b-a)*(b-a-1)/2;
        s as f64
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
        for loop_i in 0..30-input.n/2{
        // for loop_i in 0..15{
                // ランダムに未訪問の点を選んで向かう。途中でできるだけ未訪問の点を選ぶ
            let mut unvisited = HashSet::new();
            for h in 0..input.n{
                for w in 0..input.n{
                    unvisited.insert(h*input.n+w);
                }
            }
            unvisited.remove(&cur);
            while !unvisited.is_empty(){
                let mut to = *unvisited.iter().choose(&mut rng).unwrap();
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
                let idxs = input.neighbors[cur].clone();
                // if loop_i%2==0{
                // // if loop_i>(30-input.n/2)/2{
                //     idxs.reverse();
                // }
                // idxs.shuffle(&mut rng);
                // idxs = [&idxs[loop_i%idxs.len()..], &idxs[..loop_i%idxs.len()]].concat();
                for &idx in idxs.iter(){
                    if unvisited.contains(&idx){
                        to = idx;
                        break;
                    }
                }
                while cur!=to{
                    let next = input.mindistmap[to][cur];
                    // eprintln!("cur: {:?}, to: {:?}, next: {:?}", cur, to, next);
                    unvisited.remove(&next);
                    pathes.push(next);
                    cur = next;
                }
            }
            let to = rng.gen::<usize>()%(input.n*input.n);
            while cur!=to{
                let next = input.mindistmap[to][cur];
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
            let next = input.mindistmap[to][cur];
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

    fn update(&mut self, params: &mut Neighbor){
        // pathesのfrからtoまでのpathをparams.pathesに更新する
        for &idx in self.pathes[params.fr..params.to].iter(){
            self.cnts[idx] -= 1;
        }
        if params.to-params.fr==params.new_pathes.len(){
            for (i, idx) in params.new_pathes.iter().enumerate(){
                self.pathes[i+params.fr] = *idx;
            }
        } else {
            //let mut new_pathes = Vec::with_capacity(params.new_pathes.len() + self.pathes.len() - params.to);
            params.new_pathes.extend_from_slice(&self.pathes[params.to..]);
            // new_pathes.extend_from_slice(&self.pathes[..params.fr]);
            // new_pathes.extend_from_slice(&params.new_pathes);
            // new_pathes.extend_from_slice(&self.pathes[params.to..]);
            self.pathes.truncate(params.fr);
            self.pathes.append(&mut params.new_pathes);
            // self.pathes = new_pathes;
        }
        // self.pathes = [&self.pathes[..params.fr], &params.new_pathes, &self.pathes[params.to..]].concat();
        for &idx in params.new_pathes.iter(){
            self.cnts[idx] += 1;
        }
    }

    fn optimize_fixed_length_iter(&mut self, input: &Input, timer: &Instant, tl: f64){
        let l = self.pathes.len()-1;
        let mut stamps = vec![vec![]; input.n*input.n];
        for (i, &idx) in self.pathes.iter().enumerate(){
            stamps[idx].push(i);
        }
        let mut cnt = 0;
        loop {
            if timer.elapsed().as_secs_f64()>tl{
                return;
            }
            let mut idxs = (0..self.pathes.len()-4).collect_vec();
            idxs.shuffle(&mut rand::thread_rng());
            for &fr in idxs.iter(){
                cnt += 1;
                if timer.elapsed().as_secs_f64()>tl{
                    eprintln!("cnt: {}", cnt);
                    return;
                }
                let mut candidate = vec![];
                let idx = self.pathes[fr];
                let idx_ = self.pathes[fr+4];
                let mut idxs = HashSet::new();
                for (midx, pathes1) in input.neighbor_path_2dist[idx].iter(){
                    if let Some(pathes2) = input.neighbor_path_2dist[idx_].get(midx){
                        idxs.insert(*midx);
                        for path1 in pathes1.iter(){
                            idxs.insert(*path1);
                            for path2 in pathes2.iter(){
                                candidate.push(vec![*path1, *midx, *path2]);
                                idxs.insert(*path2);
                            }
                        }
                    }
                }
                if candidate.len()==1{
                    continue;
                }
                for (i, idx__) in self.pathes[fr+1..fr+4].iter().enumerate(){
                    stamps[*idx__].retain(|&x| x!=i+fr+1);
                }
                let mut best_path = vec![];
                let mut best_score = f64::MAX;
                for path in candidate.into_iter(){
                    let mut score = 0.0;
                    for (i, idx__) in path.iter().enumerate(){
                        score -= input.stamps2score(&stamps[*idx__], *idx__, l);
                        stamps[*idx__].push(i+fr+1);
                        score += input.stamps2score(&stamps[*idx__], *idx__, l);
                    }
                    for (i, idx__) in path.iter().enumerate(){
                        stamps[*idx__].pop();
                    }
                    if score<best_score{
                        best_score = score;
                        best_path = path;
                    }
                }
                for (i, idx__) in best_path.iter().enumerate(){
                    stamps[*idx__].push(i+fr+1);
                    stamps[*idx__].sort();
                    self.cnts[self.pathes[fr+1+i]] -= 1;
                    self.pathes[fr+1+i] = *idx__;
                    self.cnts[self.pathes[fr+1+i]] += 1;
                }
            }
        }
    }

    fn get_neighbor(&mut self, input: &Input, elasped_time: f64) -> Neighbor{
        let mut rng = rand::thread_rng();
        let flg: usize;
        // 1, 3, 4, 5, 8, 9, 12が採用
        // 近傍0: 何もしない
        // 近傍1: (x, y)->(x+1, y)のところで(x, y)->(x, y+1)->(x+1, y+1)->(x+1, y) と寄り道する
        // 近傍2: (x, y)->(x+1, y)->(x,y)と寄り道する
        // 近傍3: (x, y)->(x_, y_)のところで(mx, my)に寄り道する
        // 近傍4: (x, y)->(x_, y_)のところで最短経路でいかせる
        // 近傍5: (x, y)->..(x+1, y)のところを省略する
        // 近傍6: (x, x)->..(x, x)のところを逆順にする
        // 近傍7: クラスタへ寄り道->削除
        // 近傍8: パスのswap
        // 近傍9: (x, x)->..(x, x)のところを逆順にする, 前後同じところは省く
        // 近傍10: (x, x)->..(x, x)のループを移動
        // 近傍11: (x, x)->..(x, x)のループをコピー
        // 近傍12: (x, y)->(x+1, y)->(x+1, y+1)のところで(x, y)->(x, y+1)->(x+1, y+1)と変更
        // 近傍13: pathes[fr..fr+6]を全探索して最良のものを出力
        //                            0   1   2   3   4   5   6   7   8   9, 10, 11, 12, 13
        let flg_idx: [usize; 14];
        if elasped_time<0.2{
            flg = rng.gen::<usize>()%80;
            //         0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,  11,  12,  13
            flg_idx = [0, 20, 20, 40, 60, 80, 90, 90, 95, 100, 110, 110, 120, 130];
        } else if elasped_time<1.0{
            // flg = rng.gen::<usize>()%100;
            flg = rng.gen::<usize>()%90;
            //         0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,  11,  12,  13
            flg_idx = [0, 20, 20, 20, 40, 60, 60, 60, 70, 80, 80, 80, 90, 100];
        } else{
            // flg = rng.gen::<usize>()%90;
            flg = rng.gen::<usize>()%80;
            //         0,  1,  2,  3,  4,  5,  6,  7,   8,   9,  10,  11,  12,  13
            flg_idx = [0, 10, 10, 10, 30, 50, 50, 50, 60, 70, 70, 70, 80, 90];
        // } else{
        //     flg = rng.gen::<usize>()%80;
        //     //         0,  1,  2,  3,  4,  5,  6,  7,   8,   9,  10,  11,  12,  13
        //     flg_idx = [0, 10, 10, 10, 30, 50, 50, 50, 60, 70, 70, 70, 80, 90];
        }
        // const flg_idx: [usize; 10] = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
        // const flg_idx: [usize; 10] = [0, 20, 20, 30, 40, 70, 80, 80, 90, 100, 110];
        if flg<flg_idx[1]{
            // 近傍1: (x, y)->(x+1, y)のところで(x, y)->(x, y+1)->(x+1, y+1)->(x+1, y) と寄り道する
            let fr = rng.gen::<usize>()%(self.pathes.len()-1);
            let to = fr+1;
            let idx = self.pathes[fr];
            let idx_ = self.pathes[to];
            if let Some((nidx, nidx_)) = input.neighbor_pairs[idx][idx_].choose(&mut rng){
                return Neighbor{fr, to, new_pathes: vec![idx, *nidx, *nidx_], neighbortype: 1};
            }
        } else if flg<flg_idx[2]{
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
        }else if flg<flg_idx[3]{
            // 近傍3: (x, y)->(x_, y_)のところで(mx, my)に寄り道する
            let fr = rng.gen::<usize>()%(self.pathes.len()-3);
            // let to = fr+1+rng.gen::<usize>()%(self.pathes.len()-(fr+1)-1);
            let to = cmp::min(self.pathes.len()-1, fr+1+rng.gen::<usize>()%100);
            let idx = self.pathes[fr];
            let idx_ = self.pathes[to];
            let midx = rng.gen::<usize>()%(input.n*input.n);
            let mut new_pathes = vec![];
            let mut cur = idx;
            let mmap = &input.mindistmap[midx];
            while cur!=midx{
                let next = mmap[cur];
                new_pathes.push(next);
                cur = next;
            }
            let mmap = &input.mindistmap[idx_];
            while cur!=idx_{
                let next = mmap[cur];
                new_pathes.push(next);
                cur = next;
            }
            return Neighbor{fr: fr+1, to: to+1, new_pathes, neighbortype: 3};
        } else if flg<flg_idx[4]{
            // 近傍4: (x, y)->(x_, y_)のところで最短経路でいかせる
            let mut fr = rng.gen::<usize>()%(self.pathes.len()-3);
            let to = cmp::min(self.pathes.len()-1, fr+1+rng.gen::<usize>()%100);
            // let to = fr+1+rng.gen::<usize>()%(self.pathes.len()-fr-1);
            // let mut idx = self.pathes[fr];
            let idx_ = self.pathes[to];
            let mut new_pathes = vec![];
            let mut cur = self.pathes[fr];
            let mmap = &input.mindistmap[idx_];
            // TODO: 経路同じだったらfr+=1させる
            while fr<to && mmap[cur]==self.pathes[fr+1]{
                fr += 1;
                // idx = self.pathes[fr];
                cur = mmap[cur]
            }
            while cur!=idx_{
                let next = mmap[cur];
                new_pathes.push(next);
                cur = next;
            }
            if fr==to{
                return Neighbor { fr: 0, to: 0, new_pathes: vec![], neighbortype: 0};
            }
            return Neighbor{fr: fr+1, to: to+1, new_pathes, neighbortype: 4};
        } else if flg<flg_idx[5]{
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
        } else if flg<flg_idx[6]{
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
                    let mut new_pathes = self.pathes[fr+1..to].to_vec();
                    new_pathes.reverse();
                    return Neighbor{fr: fr+1, to, new_pathes, neighbortype: 6};
                }
            }
        } else if flg<flg_idx[8]{
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
                        let mut new_pathes = Vec::with_capacity(to_max-fr_min);
                        new_pathes.extend_from_slice(&self.pathes[fr_max..to_max]);
                        new_pathes.extend_from_slice(&self.pathes[to_min..fr_max]);
                        new_pathes.extend_from_slice(&self.pathes[fr_min..to_min]);
                        // let mut new_pathes = [self.pathes[fr_max..to_max].to_vec(), self.pathes[to_min..fr_max].to_vec(), self.pathes[fr_min..to_min].to_vec()].concat();
                        return Neighbor{fr: fr_min, to: to_max, new_pathes, neighbortype: 8}
                    }
                }
            }
        } else if flg<flg_idx[9]{
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
                    let mut new_pathes = self.pathes[fr+to_f..to-fr_b+1].to_vec();
                    new_pathes.reverse();
                    return Neighbor{fr: fr-fr_b, to: to+to_f+1, new_pathes, neighbortype: 9};
                }
            }
        } else if flg<flg_idx[10]{
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
            if fr<fr2{
                let mut new_pathes = Vec::with_capacity(fr2-fr);
                new_pathes.extend_from_slice(&self.pathes[to..fr2]);
                new_pathes.extend_from_slice(&self.pathes[fr..to]);
                return Neighbor{fr, to: fr2, new_pathes, neighbortype: 10};
            } else {
                let mut new_pathes = Vec::with_capacity(to-fr2);
                new_pathes.extend_from_slice(&self.pathes[fr..to]);
                new_pathes.extend_from_slice(&self.pathes[fr2..fr]);
                return Neighbor{fr: fr2, to, new_pathes, neighbortype: 10};

            }
        } else if flg<flg_idx[11]{
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
            return Neighbor{fr: fr2, to: fr2, new_pathes: self.pathes[fr..to].to_vec(), neighbortype: 10};
        } else if flg<flg_idx[12]{
            // 近傍12: (x, y)->(x+1, y)->(x+1, y+1)のところで(x, y)->(x, y+1)->(x+1, y+1)と変更
            let fr = rng.gen::<usize>()%(self.pathes.len()-2);
            let to = fr+2;
            let idx = self.pathes[fr];
            let idx_ = self.pathes[to];
            if let Some((nidx, nidx_)) = input.neighbor_pairs2[idx][idx_].choose(&mut rng){
                if *nidx==self.pathes[fr+1]{
                    return Neighbor{fr, to, new_pathes: vec![idx, *nidx_], neighbortype: 12};
                } else if *nidx_==self.pathes[fr+1]{
                    return Neighbor{fr, to, new_pathes: vec![idx, *nidx], neighbortype: 12};
                }
            }
        } else if flg<flg_idx[13]{
            // 近傍13: pathes[fr..fr+6]を全探索して最良のものを出力
            let fr = rng.gen::<usize>()%(self.pathes.len()-6);
            let mut candidate = vec![];
            let idx = self.pathes[fr];
            let idx_ = self.pathes[fr+4];
            for (midx, pathes1) in input.neighbor_path_2dist[idx].iter(){
                if let Some(pathes2) = input.neighbor_path_2dist[idx_].get(midx){
                    for path1 in pathes1.iter(){
                        for path2 in pathes2.iter(){
                            candidate.push(vec![*path1, *midx, *path2]);
                        }
                    }
                }
            }
            if candidate.len()<=1{
                return Neighbor { fr: 0, to: 0, new_pathes: vec![], neighbortype: 0};
            }
            candidate.shuffle(&mut rng);
            for path in candidate.into_iter(){
                if path != self.pathes[fr+1..fr+4]{
                    return Neighbor{fr: fr+1, to: fr+4, new_pathes: path, neighbortype: 13};
                }
            }

        }
        Neighbor { fr: 0, to: 0, new_pathes: vec![], neighbortype: 0}
    }

    fn get_score_from_neighbor(&self, input: &Input, neighbor: &Neighbor)->f64{
        let l = self.pathes.len()+neighbor.new_pathes.len()+neighbor.fr-neighbor.to-1;
        let s1 = neighbor.fr;
        let s2 = neighbor.fr+neighbor.new_pathes.len();
        let mut last_visit = vec![usize::MAX; input.n*input.n];
        let mut s = 0;
        let d = &input.d;
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
            let lv = i+l-last_visit[idx];
            s += d[idx]*lv*(lv-1);
            last_visit[idx] = i+l;
        }
        for (i, &idx) in neighbor.new_pathes.iter().enumerate(){
            let lv = i+s1+l-last_visit[idx];
            s += d[idx]*lv*(lv-1);
            last_visit[idx] = i+s1+l;
        }
        for (i, &idx) in self.pathes[neighbor.to..].iter().enumerate(){
            let lv = i+s2+l-last_visit[idx];
            s += d[idx]*lv*(lv-1);
            last_visit[idx] = i+s2+l;
        }
        (s as f64)/((l*2) as f64)
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
    // let mut best_state = init_state.clone();
    // let mut best_score = best_state.get_score(input);
    let mut cur_score = init_state.get_score(input);
    let mut all_iter = 0;
    let mut accepted_cnt = 0;
    let mut rng = rand::thread_rng();
    let start_temp: f64 = 1000.0;
    let end_temp: f64 = 0.1;
    let mut temp: f64;
    // let mut neighbor_cnt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    // let mut neighbor_improve = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
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
        let mut neighbor = state.get_neighbor(input, elasped_time);
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
            // if accepted_cnt%1000000==0{
            //     eprintln!("{} {} {} {:?} {:?}", cur_score, new_score, all_iter, state.pathes.len(), neighbor_improve);
            // }
            // eprintln!("neighbor_cnt  : {:?}", neighbor_cnt);
            // eprintln!("neighbor_improvement  : {:?}", neighbor_improve);
            // neighbor_cnt[neighbor.neighbortype] += 1;
            // neighbor_improve[neighbor.neighbortype] += score_diff as usize;
            // if last_printed+0.1<=elasped_time{
            //     last_printed = elasped_time;
            //     eprintln!("{:?},", neighbor_cnt);
            //     eprintln!("{:?},", neighbor_improve);
            // }
            state.update(&mut neighbor);
            cur_score = new_score;
            // if elasped_time>limit/2.0 && new_score<best_score{
            //     best_state = state.clone();
            //     best_score = new_score;
            // }
        // } else {
        //    state.undo(&neighbor);
        }
    }
    // best_state = state;
    eprintln!("===== simanneal =====");
    eprintln!("all iter   : {}", all_iter);
    eprintln!("accepted   : {}", accepted_cnt);
    eprintln!("score  : {}", cur_score);
    // eprintln!("neighbor_cnt  : {:?}", neighbor_cnt);
    // eprintln!("neighbor_improvement  : {:?}", neighbor_improve);
    eprintln!("");
    state
}

fn main() {
    let timer = Instant::now();
    let input = Input::read_input();
    eprintln!("et: {}", timer.elapsed().as_secs_f64());
    solve(&input, &timer, 1.97);
}

fn solve(input: &Input, timer:&Instant, tl: f64){
    // let mut init_state = State::init_state(input);
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
    let mut best_state = simanneal(input, State::init_state(input), timer, tl);
    //best_state.print(input);
    // best_state = simanneal(input, best_state, timer, tl+10.0);
    // best_state.print(input);
    // best_state.optimize_fixed_length_iter(input, timer, tl);
    best_state.print(input);
}
