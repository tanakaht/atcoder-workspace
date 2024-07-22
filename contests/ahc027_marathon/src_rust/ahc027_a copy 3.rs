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
    mindistmap: Vec<Vec<Vec<usize>>>,// (h, w)への最短経路で、(h_, w_)から次に向かう座標を複数
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
                    neighbors[h*n+w].push((h+1)*n+w);
                    neighbors[(h+1)*n+w].push(h*n+w);
                }
            }
        }
        for h in 0..n{
            for w in 0..n-1{
                if v_[h][w]=='0'{
                    neighbors[h*n+w].push(h*n+w+1);
                    neighbors[h*n+w+1].push(h*n+w);
                }
            }
        }
        let mut mindistmap = vec![vec![vec![]; n*n]; n*n];
        for h in 0..n{
            for w in 0..n{
                let mut q = VecDeque::new();
                let mut appeared = vec![vec![false; n]; n];
                let mut dist = vec![vec![usize::MAX; n]; n];
                dist[h][w] = 0;
                q.push_front((h, w, 0));
                while !q.is_empty(){
                    let (h_, w_, d) = q.pop_front().unwrap();
                    if appeared[h_][w_]{
                        continue;
                    }
                    appeared[h_][w_] = true;
                    for &nidx in neighbors[h_*n+w_].iter(){
                        let (nh, nw) = (nidx/n, nidx%n);
                        if dist[nh][nw]>=d+1{
                            mindistmap[h*n+w][nh*n+nw].push(h_*n+w_);
                            q.push_back((nh, nw, d+1));
                            dist[nh][nw]=d+1;
                        }
                    }
                }
            }
        }
        Self { n, neighbors, d, mindistmap }
    }
}

struct Neighbor{
    fr: usize,
    to: usize,
    new_pathes: Vec<usize>,
    old_pathes: Vec<usize>,
    neighbortype: usize,
}

#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct State{
    pathes: Vec<usize>,
}

impl State{
    fn init_state(input: &Input) -> Self{
        let mut pathes =  vec![];
        let mut rng = rand::thread_rng();
        let mut cur = 0;
        pathes.push(0);
        for loop_i in 0..(30-input.n/2){
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
        Self {pathes}
    }

    fn update(&mut self, params: &Neighbor){
        // pathesのfrからtoまでのpathをparams.pathesに更新する
        self.pathes = [&self.pathes[..params.fr], &params.new_pathes[..], &self.pathes[params.to..]].concat();
    }

    fn undo(&mut self, params: &Neighbor){
        self.pathes = [&self.pathes[..params.fr], &params.old_pathes[..], &self.pathes[params.fr+params.new_pathes.len()..]].concat();
    }

    fn get_neighbor_short_around(&mut self, input: &Input) -> Neighbor{
        let mut rng = rand::thread_rng();
        loop{
            let fr = rng.gen::<usize>()%(self.pathes.len()-1);
            let to = fr+1;
            let idx = self.pathes[fr];
            let idx_ = self.pathes[to];
            // ちょいきもい
            let mut neighbor_pairs = vec![];
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
                    return Neighbor{fr, to, new_pathes: vec![idx, *nidx, *nidx_], old_pathes: vec![idx], neighbortype: 0};
                }
            }
        }
    }

    fn get_neighbor_return_around(&mut self, input: &Input) -> Neighbor{
        let mut rng = rand::thread_rng();
        // (x, y)->(x+1, y)のところで寄り道する
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
                let nidx = input.neighbors[idx].choose(&mut rng).unwrap().clone();
                if nidx==idx_ || nidx==idx__{
                    continue;
                }
                return Neighbor{fr, to, new_pathes: vec![idx, nidx, idx], old_pathes: vec![idx], neighbortype: 0};
            }
        }
    }

    fn get_neighbor_long_around(&mut self, input: &Input) -> Neighbor{
        let mut rng = rand::thread_rng();
        loop{
            // (x, y)->(x_, y_)のところで(mx, my)に寄り道する
            let fr = rng.gen::<usize>()%(self.pathes.len()-3);
            let to = fr+1+rng.gen::<usize>()%(self.pathes.len()-(fr+1)-1);
            // let to = cmp::min(self.pathes.len()-1, fr+1+rng.gen::<usize>()%100);
            let idx = self.pathes[fr];
            let idx_ = self.pathes[to];
            let midx = rng.gen::<usize>()%(input.n*input.n);
            if idx==idx_ || idx==midx || idx_==midx{
                continue;
            }
            let mut new_pathes = vec![];
            let mut cur = idx;
            while cur!=midx{
                let next = input.mindistmap[midx][cur].choose(&mut rng).unwrap().clone();
                new_pathes.push(next);
                cur = next;
            }
            while cur!=idx_{
                let next = input.mindistmap[idx_][cur].choose(&mut rng).unwrap().clone();
                new_pathes.push(next);
                cur = next;
            }
            return Neighbor{fr: fr+1, to: to+1, new_pathes, old_pathes: self.pathes[fr+1..to+1].to_vec(), neighbortype: 1};
        }
    }

    fn get_neighbor_long_shortcut(&mut self, input: &Input) -> Neighbor{
        // (x, y)->(x_, y_)のところで最短経路でいかせる
        let mut rng = rand::thread_rng();
        let fr = rng.gen::<usize>()%(self.pathes.len()-3);
        let to = cmp::min(self.pathes.len()-1, fr+1+rng.gen::<usize>()%100);
        let idx = self.pathes[fr];
        let idx_ = self.pathes[to];
        let mut new_pathes = vec![];
        let mut cur = idx;
        while cur!=idx_{
            let next = input.mindistmap[idx_][cur].choose(&mut rng).unwrap().clone();
            new_pathes.push(next);
            cur = next;
        }
        return Neighbor{fr: fr+1, to: to+1, new_pathes, old_pathes: self.pathes[fr+1..to+1].to_vec(), neighbortype: 2};
    }

    fn get_neighbor_short_shortcut(&mut self, input: &Input) -> Neighbor{
        let mut rng = rand::thread_rng();
        // (x, y)->..(x+1, y)のところを省略する
        loop{
            let fr = rng.gen::<usize>()%(self.pathes.len()-2);
            let idx = self.pathes[fr];
            // let mut idxs = (fr+2..self.pathes.len()).collect::<Vec<_>>();
            // idxs.shuffle(&mut rng);
            for to in fr+2..self.pathes.len(){
                let idx_ = self.pathes[to];
                if input.neighbors[idx].contains(&idx_){
                    return Neighbor{fr: fr+1, to, new_pathes: vec![], old_pathes: self.pathes[fr+1..to].to_vec(), neighbortype: 3};
                }
            }
        }
    }

    fn get_neighbor_reverse(&mut self, input: &Input) -> Neighbor{
        let mut rng = rand::thread_rng();
        // (x, x)->..(x, x)のところを逆順にする
        loop{
            let fr = rng.gen::<usize>()%(self.pathes.len()-2);
            let idx = self.pathes[fr];
            // let mut idxs = (fr+2..self.pathes.len()).collect::<Vec<_>>();
            // idxs.shuffle(&mut rng);
            for to in fr+2..self.pathes.len(){
                let idx_ = self.pathes[to];
                if idx==idx_{
                    let mut new_pathes = self.pathes[fr+1..to].to_vec().clone();
                    new_pathes.reverse();
                    return Neighbor{fr: fr+1, to, new_pathes, old_pathes: self.pathes[fr+1..to].to_vec(), neighbortype: 4};
                }
            }
        }
    }

    fn get_neighbor(&mut self, input: &Input) -> Neighbor{
        let mut rng = rand::thread_rng();
        let mut flg = rng.gen::<usize>()%100;
        match flg{
            0..=10 => {
                // (x, y)->(x+1, y)のところで寄り道する
                self.get_neighbor_short_around(input)
            },
            11..=20 => {
                // (x, y)->(x+1, y)のところで寄り道する
                self.get_neighbor_return_around(input)
            },
            21..=40 => {
                // (x, y)->(x_, y_)のところで(mx, my)に寄り道する
                self.get_neighbor_long_around(input)
            },
            41..=70 => {
                // (x, y)->(x_, y_)のところで最短経路でいかせる
                self.get_neighbor_long_shortcut(input)
            },
            71..=99 => {
                // (x, x)->..(x, x)のところを逆順にする
                self.get_neighbor_reverse(input)
            },
            _ => {
                Neighbor { fr: 0, to: 0, new_pathes: vec![], old_pathes: vec![], neighbortype: 5}
            }
        }
    }

    fn get_score_from_neighbor(&self, input: &Input, neighbor: &Neighbor)->f64{
        let l = self.pathes.len()-1+neighbor.new_pathes.len()-neighbor.old_pathes.len();
        let mut stamps = vec![vec![]; input.n*input.n];
        for (i, &idx) in self.pathes[..neighbor.fr].iter().enumerate(){
            stamps[idx].push(i);
        }
        for (i, &idx) in neighbor.new_pathes.iter().enumerate(){
            stamps[idx].push(i+neighbor.fr);
        }
        for (i, &idx) in self.pathes[neighbor.to..].iter().enumerate(){
            stamps[idx].push(i+neighbor.to);
        }
        let mut s = 0;
        for (idx, stamp) in stamps.iter().enumerate(){
            if stamp.len()==0{
                return f64::MAX;
            }
            let d = input.d[idx];
            for i in 0..stamp.len()-1{
                let a = stamp[i];
                let b = stamp[i+1];
                s += d*(b-a)*(b-a-1)/2;
            }
            let a = stamp[stamp.len()-1];
            let b = stamp[0]+l;
            s += d*(b-a)*(b-a-1)/2;
        }
        (s as f64)/(l as f64)

    }

    fn get_score(&self, input: &Input) -> f64{
        let l = self.pathes.len()-1;
        let mut last_visit = vec![None; input.n*input.n];
        let mut first_visit = vec![None; input.n*input.n];
        let mut s = 0;
        for (i, &idx) in self.pathes.iter().enumerate(){
            if let Some(last) = last_visit[idx]{
                s += input.d[idx]*(i-last)*(i-last-1)/2;
            }
            if first_visit[idx].is_none(){
                first_visit[idx] = Some(i);
            }
            last_visit[idx] = Some(i);
        }
        for (idx, d) in input.d.iter().enumerate(){
            if let Some(first) = first_visit[idx]{
                if let Some(last) = last_visit[idx]{
                    s += d*(l+first-last)*(l+first-last-1)/2;
                }
            } else {
                return f64::MAX;
            }
        }
        (s as f64)/(l as f64)
    }

    fn get_score_old(&self, input: &Input) -> f64{
        let l = self.pathes.len()-1;
        let mut stamps = vec![vec![]; input.n*input.n];
        for (i, &idx) in self.pathes.iter().enumerate(){
            stamps[idx].push(i);
        }
        let mut s = 0;
        for (idx, stamp) in stamps.iter().enumerate(){
            if stamp.len()==0{
                return f64::MAX;
            }
            let d = input.d[idx];
            for i in 0..stamp.len()-1{
                let a = stamp[i];
                let b = stamp[i+1];
                s += d*(b-a)*(b-a-1)/2;
            }
            let a = stamp[stamp.len()-1];
            let b = stamp[0]+l;
            s += d*(b-a)*(b-a-1)/2;
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
    let mut last_updated = 0;
    let mut neighbor_cnt = [0, 0, 0, 0, 0];
    let mut neighbor_improve = [0, 0, 0, 0, 0];
    eprintln!("simanneal start at {}", timer.elapsed().as_secs_f64());
    loop {
        let elasped_time = timer.elapsed().as_secs_f64();
        if elasped_time >= limit{
            break;
        }
        all_iter += 1;
        let neighbor = state.get_neighbor(input);
        if neighbor.fr==neighbor.to{
            continue;
        }
        state.update(&neighbor);
        let new_score = state.get_score(input);
        let score_diff = cur_score-new_score;
        // start_temp = (start_temp*(all_iter-1)+score_diff)/all_iter;
        temp = start_temp + (end_temp-start_temp)*elasped_time/limit;
        if score_diff>=0.0 || rng.gen_bool((score_diff/temp).exp()){
            // state.update(&neighbor);
            // state.print(input);
            accepted_cnt += 1;
            if accepted_cnt%100==0{
                eprintln!("{} {} {} {:?} {}", cur_score, new_score, all_iter, state.pathes.len(), input.n*input.n);
            }
            // eprintln!("neighbor_cnt  : {:?}", neighbor_cnt);
            // eprintln!("neighbor_improvement  : {:?}", neighbor_improve);
            neighbor_cnt[neighbor.neighbortype] += 1;
            neighbor_improve[neighbor.neighbortype] += score_diff as usize;
            cur_score = new_score;
            last_updated = all_iter;
            if new_score<best_score{
                best_state = state.clone();
                best_score = new_score;
            }
        } else {
           state.undo(&neighbor);
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
    solve(&input, &timer, 1.85);
}

fn solve(input: &Input, timer:&Instant, tl: f64){
    let mut init_state = State::init_state(input);
    let mut best_state = simanneal(input, init_state, timer, tl);
    best_state.print(input);
}
