#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused)]
use std::collections::HashSet;
use itertools::Itertools;
use num::{range, ToPrimitive};
use petgraph::matrix_graph::Neighbors;
use proconio::*;
use std::iter::FromIterator;
use std::collections::BinaryHeap;
use std::f64::consts::PI;
use std::fmt::Display;
use std::cmp::{Reverse, min};
#[allow(unused_imports)]
use rand::prelude::*;
use std::time::Instant;

const INF: usize = 1 << 31;
const DEFAULTDIST: usize = 1000000000;
const SEED: u128 = 42;

#[derive(Debug, Clone, Copy)]
struct Point{
    x: usize,
    y: usize,
}

impl Point{
    fn get_neighbor(&self, N: usize) -> Vec<Point>{
        let mut neighbors: Vec<(i32, i32)> = vec![((self.x as i32), (self.y as i32)+1), ((self.x as i32)+1, (self.y as i32)), ((self.x as i32), (self.y as i32)-1), ((self.x as i32)-1, (self.y as i32))];
        neighbors = neighbors.into_iter().filter(|(xd, yd)| 0<=*xd && *xd<(N as i32) && 0<=*yd && *yd<(N as i32)).collect_vec();
        let mut ret = vec![];
        for (xd, yd) in neighbors{
            ret.push(Point{x:(xd as usize), y:(yd as usize)})
        }
        ret
    }

    fn get_random_point(N: usize, rng: &mut ThreadRng) -> Point{
        Point{x: rng.gen::<usize>()%N, y: rng.gen::<usize>()%N}
    }
}

#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Input {
    N: usize,
    M: usize,
    A: Vec<Vec<usize>>,
}

impl Input {
    fn read_input() -> Self {
        input! {
            N: usize,
            M: usize,
            A: [[usize; N]; N]
        }
        Self { N, M, A }
    }
}


#[derive(Debug, Clone)]
struct State {
    N: usize,
    M: usize,
    A_init: Vec<Vec<usize>>,
    A_ans: Vec<Vec<usize>>,
}

impl State {
    fn init(input: &Input, seed: u128) -> Self {
        let mut rng = rand::thread_rng();
        Self { N: input.N, M: input.M, A_init: input.A.clone(), A_ans: input.A.clone() }
    }

    fn get_init_v(&self, p: &Point) -> usize{
        self.A_init[p.x][p.y]
    }

    fn get_ans_v(&self, p: &Point) -> usize{
        self.A_ans[p.x][p.y]
    }

    fn get_neighbor(&self) -> (Point, usize){
        let mut rng = rand::thread_rng();
        let flg = rng.gen::<usize>()%100;
        //
        for _ in 0..100{
            let p = Point::get_random_point(self.N, &mut rng);
            if flg < 5{
                // +1
                if self.get_ans_v(&p)==9{
                    continue;
                }
                return (p, self.get_ans_v(&p)+1);
            } else if flg < 10{
                // -1
                if self.get_ans_v(&p)==self.get_init_v(&p){
                    continue;
                }
                return (p, self.get_ans_v(&p)-1);
            } else{
                let neighbor_ps = p.get_neighbor(self.N);
                let p2 = neighbor_ps[flg%neighbor_ps.len()];
                if self.get_init_v(&p)>self.get_ans_v(&p2){
                    continue;
                }
                if self.get_ans_v(&p)==self.get_ans_v(&p2){
                    continue;
                }
                return (p, self.get_ans_v(&p2));
            }
        }
        (Point{x: 0, y: 0}, self.A_ans[0][0])
    }

    fn update(&mut self, params: &(Point, usize)){
        let (p, v) = params;
        self.A_ans[p.x][p.y] = *v;
    }

    fn get_cluster(&self) -> Vec<(usize, Vec<Point>)>{
        // dfs
        let mut appeared = vec![vec![false; self.N]; self.N];
        let mut clusters: Vec<(usize, Vec<Point>)> = vec![];
        for si in 0..self.N{
            for sj in 0..self.N{
                if appeared[si][sj]{
                    continue;
                }
                appeared[si][sj] = true;
                let mut q: Vec<Point> = vec![];
                let mut cluster: Vec<Point> = vec![];
                let cur = Point{x: si, y: sj};
                let v = self.get_ans_v(&cur);
                q.push(cur);
                while let Some(p)=q.pop(){
                    cluster.push(p);
                    for p2 in p.get_neighbor(self.N){
                        if appeared[p2.x][p2.y]{
                            continue;
                        }
                        if self.get_ans_v(&p2)!=v{
                            continue;
                        }
                        appeared[p2.x][p2.y] = true;
                        q.push(p2);
                    }
                }
                if cluster.len()>=v{
                    clusters.push((v*cluster.len(), cluster));
                } else {
                    clusters.push((0, cluster));
                }
            }
        }
        clusters.sort_by_key(|(a, b)| INF-*a);
        clusters
    }

    fn dfs(&self, p: &Point) -> HashSet<(usize, usize)>{
        let mut q: Vec<Point> = vec![];
        let mut cluster: HashSet<(usize, usize)> = HashSet::new();
        let v = self.get_ans_v(&p);
        q.push(p.clone());
        while let Some(p)=q.pop(){
            cluster.insert((p.x, p.y));
            for p2 in p.get_neighbor(self.N){
                if self.get_ans_v(&p2)!=v || cluster.contains(&(p2.x, p2.y)){
                    continue;
                }
                cluster.insert((p2.x, p2.y));
                q.push(p2);
            }
        }
        cluster
    }

    fn get_score(&self) -> i32 {
        let mut diff = 0;
        for i in 0..self.N{
            for j in 0..self.N{
                diff += self.A_ans[i][j]-self.A_init[i][j];
            }
        }
        if diff>self.M{
            return 0
        }
        let mut clusters = self.get_cluster();
        let mut score: i32 = 0;
        for i in 0..min(self.M-diff, clusters.len()){
            score += clusters[i].0 as i32;
        }
        eprintln!("{} {}", score, diff);
        score
    }
}

impl Display for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut ops: Vec<(usize, usize, usize)> = vec![];
        let mut diff = 0;
        for i in 0..self.N{
            for j in 0..self.N{
                diff += self.A_ans[i][j]-self.A_init[i][j];
                for _ in 0..(self.A_ans[i][j]-self.A_init[i][j]){
                    ops.push((1, i, j));
                }
            }
        }
        let mut clusters = self.get_cluster();
        if diff<=self.M{
            for i in 0..min(self.M-diff, clusters.len()){
                let p = clusters[i].1[0];
                ops.push((2, p.x, p.y));
            }
        }
        for (op, i, j) in ops{
            write!(f, "{} {} {}\n", op, i, j)?;
        }
        Ok(())
    }
}

fn simanneal(input: &Input, init_state: State, timer: &Instant, limit: f64) -> State{
    let mut state = init_state.clone();
    let mut score = state.get_score();
    let mut all_iter = 0;
    let mut accepted_cnt = 0;
    let mut rng = rand::thread_rng();
    let start_temp: f64 = 10000.0;
    let end_temp: f64 = 0.1;
    let mut temp = start_temp;
    eprintln!("simanneal start at {}", timer.elapsed().as_secs_f64());
    loop {
        let elasped_time = timer.elapsed().as_secs_f64();
        if elasped_time >= limit{
            break;
        }
        all_iter += 1;
        let (p, v) = state.get_neighbor();
        let pre_v = state.get_ans_v(&p);
        state.update(&(p, v));
        let new_score = state.get_score();
        let score_diff = new_score-score;
        temp = start_temp + (end_temp-start_temp)*elasped_time/limit;
    if score_diff>=0{// || ((score_diff as f64)/temp).exp() > rng.gen::<f64>(){
            accepted_cnt += 1;
            score = new_score;
            // if mode=="move"{
            //     movecnt += 1;
            // } else {
            //     swapcnt += 1;
            // }
            //eprintln!("{} {} {}", timer.elapsed().as_secs_f32(), movecnt, swapcnt);
        } else {
            state.update(&(p, pre_v));
        }
    }
    eprintln!("===== simanneal =====");
    eprintln!("all iter   : {}", all_iter);
    eprintln!("accepted   : {}", accepted_cnt);
    eprintln!("");
    state
}


fn simanneal2(input: &Input, init_state: State, timer: &Instant, limit: f64) -> State{
    let mut state = init_state.clone();
    let mut score = state.get_score();
    let mut all_iter = 0;
    let mut accepted_cnt = 0;
    let mut rng = rand::thread_rng();
    let start_temp: f64 = 10000.0;
    let end_temp: f64 = 0.1;
    let mut temp = start_temp;
    eprintln!("simanneal start at {}", timer.elapsed().as_secs_f64());
    loop {
        let elasped_time = timer.elapsed().as_secs_f64();
        if elasped_time >= limit{
            break;
        }
        all_iter += 1;
        let (p, v) = state.get_neighbor();
        let pre_score = state.dfs(&p).len() as i32;
        let pre_v = state.get_ans_v(&p);
        state.update(&(p, v));
        let new_score = state.dfs(&p).len() as i32;
        let score_diff = new_score-pre_score;
        temp = start_temp + (end_temp-start_temp)*elasped_time/limit;
    if (score_diff>=0 || ((score_diff as f64)/temp).exp() > rng.gen::<f64>()){
            accepted_cnt += 1;
            // if mode=="move"{
            //     movecnt += 1;
            // } else {
            //     swapcnt += 1;
            // }
            //eprintln!("{} {} {}", timer.elapsed().as_secs_f32(), movecnt, swapcnt);
        } else {
            state.update(&(p, pre_v));
        }
    }
    eprintln!("===== simanneal =====");
    eprintln!("all iter   : {}", all_iter);
    eprintln!("accepted   : {}", accepted_cnt);
    eprintln!("");
    state
}


fn main() {
    let timer = Instant::now();
    let input = Input::read_input();
    let state = solve(&input, &timer);
    println!("{}", &state);
    eprintln!("end at {}", timer.elapsed().as_secs_f64());
}


fn solve(input: &Input, timer:&Instant) -> State {
    let init_state = State::init(input, SEED);
    simanneal2(input, init_state, timer, 1.8)
}
