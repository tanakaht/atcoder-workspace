#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused)]
use std::collections::{HashSet, HashMap, VecDeque};
use std::process::exit;
use std::vec;
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
    neighbors: Vec<Vec<Vec<(usize, usize)>>>,
    d: Vec<Vec<usize>>,
    mindistmap: Vec<Vec<Vec<Vec<Vec<(usize, usize)>>>>>,// (h, w)への最短経路で、(h_, w_)から次に向かう座標を複数
}

impl Input {
    fn read_input() -> Self {
        input! {
            n: usize,
            h_: [Chars; n-1],
            v_: [Chars; n],
            d: [[usize; n]; n],
        }
        let mut neighbors = vec![vec![vec![]; n]; n];
        for h in 0..n-1{
            for w in 0..n{
                if h_[h][w]=='0'{
                    neighbors[h][w].push((h+1, w));
                    neighbors[h+1][w].push((h, w));
                }
            }
        }
        for h in 0..n{
            for w in 0..n-1{
                if v_[h][w]=='0'{
                    neighbors[h][w].push((h, w+1));
                    neighbors[h][w+1].push((h, w));
                }
            }
        }
        let mut mindistmap = vec![vec![vec![vec![vec![]; n]; n]; n]; n];
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
                    for &(nh, nw) in neighbors[h_][w_].iter(){
                        if dist[nh][nw]>=d+1{
                            mindistmap[h][w][nh][nw].push((h_, w_));
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
    new_pathes: Vec<(usize, usize)>,
    old_pathes: Vec<(usize, usize)>,
}

#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct State{
    pathes: Vec<(usize, usize)>,
}

impl State{
    fn init_state(input: &Input) -> Self{
        let mut pathes =  vec![];
        let mut visited = vec![vec![false; input.n]; input.n];
        let mut q = VecDeque::new();
        q.push_front((1, 0, 0));
        while !q.is_empty(){
            let (is_go, h, w) = q.pop_front().unwrap();
            if is_go==1 && visited[h][w]{
                continue;
            }
            if is_go==0 && visited[h][w]{
                let (h_, w_) = pathes.last().unwrap();
                if !(h_==&h && w_==&w){
                    pathes.push((h, w));
                }
                continue;
            }
            pathes.push((h, w));
            visited[h][w] = true;
            for &(nh, nw) in input.neighbors[h][w].iter(){
                if !visited[nh][nw]{
                    q.push_front((0, h, w));
                    q.push_front((1, nh, nw));
                }
            }
        }
        pathes.append(&mut pathes[1..].to_vec().clone());
        Self {pathes}
    }

    fn update(&mut self, params: &Neighbor){
        // pathesのfrからtoまでのpathをparams.pathesに更新する
        self.pathes = [&self.pathes[..params.fr], &params.new_pathes[..], &self.pathes[params.to..]].concat();
    }

    fn undo(&mut self, params: &Neighbor){
        self.pathes = [&self.pathes[..params.fr], &params.old_pathes[..], &self.pathes[params.fr+params.new_pathes.len()..]].concat();
    }

    fn get_neighbor(&mut self, input: &Input) -> Neighbor{
        let mut rng = rand::thread_rng();
        let mut flg = rng.gen::<usize>()%100;
        if flg<40{
            // (x, y)->(x+1, y)のところで寄り道する
            let fr = rng.gen::<usize>()%(self.pathes.len()-1);
            let to = fr+1;
            let (x, y) = self.pathes[fr];
            let (x_, y_) = self.pathes[to];
            // ちょいきもい
            let mut neighbor_pairs = vec![];
            for &(nh, nw) in input.neighbors[x][y].iter(){
                if nh==x_ && nw==y_{
                    continue;
                }
                for &(nh_, nw_) in input.neighbors[x_][y_].iter(){
                    if nh_==x && nw_==y{
                        continue;
                    }
                    neighbor_pairs.push(((nh, nw), (nh_, nw_)));
                }
            }
            neighbor_pairs.shuffle(&mut rng);
            for ((nh, nw), (nh_, nw_)) in neighbor_pairs.iter(){
                if input.neighbors[*nh][*nw].contains(&(*nh_, *nw_)){
                    return Neighbor{fr, to, new_pathes: vec![(x, y), (*nh, *nw), (*nh_, *nw_)], old_pathes: vec![(x, y)]};
                }
            }
        } else if flg<80{
            // (x, y)->..(x+1, y)のところを省略する
            let fr = rng.gen::<usize>()%(self.pathes.len()-2);
            let (x, y) = self.pathes[fr];
            let mut idxs = (fr+2..self.pathes.len()).collect::<Vec<_>>();
            idxs.shuffle(&mut rng);
            for &to in idxs.iter(){
                let (x_, y_) = self.pathes[to];
                if input.neighbors[x][y].contains(&(x_, y_)){
                    return Neighbor{fr: fr+1, to, new_pathes: vec![], old_pathes: self.pathes[fr+1..to].to_vec()};
                }
            }
        } else{
            // (x, x)->..(x, x)のところを逆順にする
            let fr = rng.gen::<usize>()%(self.pathes.len()-2);
            let (x, y) = self.pathes[fr];
            let mut idxs = (fr+2..self.pathes.len()).collect::<Vec<_>>();
            idxs.shuffle(&mut rng);
            for &to in idxs.iter(){
                let (x_, y_) = self.pathes[to];
                if x==x_ && y==y_{
                    let mut new_pathes = self.pathes[fr+1..to].to_vec().clone();
                    new_pathes.reverse();
                    return Neighbor{fr: fr+1, to, new_pathes, old_pathes: self.pathes[fr+1..to].to_vec()};
                }
            }
        }
        Neighbor { fr: 0, to: 0, new_pathes: vec![], old_pathes: vec![] }
    }

    fn get_score(&self, input: &Input) -> f64{
        let l = self.pathes.len()-1;
        let mut stamps = vec![vec![]; input.n*input.n];
        for (i, &(h, w)) in self.pathes.iter().enumerate(){
            stamps[h*input.n+w].push(i);
        }
        let mut s = 0;
        for (idx, stamp) in stamps.iter().enumerate(){
            if stamp.len()==0{
                return f64::MAX;
            }
            let d = input.d[idx/input.n][idx%input.n];
            for i in 0..stamp.len(){
                let a = stamp[i];
                let b = stamp[(i+1)%stamp.len()];
                if b<=a{
                    s += d*(b+l-a)*(b+l-a-1)/2;
                } else {
                    s += d*(b-a)*(b-a-1)/2;
                }
            }
        }
        (s as f64)/(l as f64)
    }

    fn print(&mut self, input: &Input){
        let mut ans = "".to_string();
        for i in 0..self.pathes.len()-1{
            let (x, y) = self.pathes[i];
            let (x_, y_) = self.pathes[i+1];
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
    let start_temp: f64 = 10.0;
    let end_temp: f64 = 0.1;
    let mut temp = start_temp;
    let mut last_updated = 0;
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
        if score_diff>=0.0 || rng.gen_bool((score_diff as f64/temp).exp()){
            state.print(input);
            accepted_cnt += 1;
            eprintln!("{} {} {} {:?}", cur_score, new_score, all_iter, neighbor.new_pathes.len());
            cur_score = new_score;
            last_updated = all_iter;
            //state.print(input);
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
    eprintln!("");
    best_state
}

fn main() {
    let timer = Instant::now();
    let input = Input::read_input();
    solve(&input, &timer, 1.8);
}

fn solve(input: &Input, timer:&Instant, tl: f64){
    let mut init_state = State::init_state(input);
    let mut best_state = simanneal(input, init_state, timer, tl);
    best_state.print(input);
}
