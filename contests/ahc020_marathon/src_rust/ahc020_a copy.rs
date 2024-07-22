#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused)]
use std::collections::{HashSet, HashMap};
use std::process::exit;
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

const INF: i64 = 1 << 31;

#[allow(unused_macros)]
macro_rules! chmin {
    ($base:expr, $($cmps:expr),+ $(,)*) => {{
        let cmp_min = min!($($cmps),+);
        if $base > cmp_min {
            $base = cmp_min;
            true
        } else {
            false
        }
    }};
}

#[allow(unused_macros)]
macro_rules! chmax {
    ($base:expr, $($cmps:expr),+ $(,)*) => {{
        let cmp_max = max!($($cmps),+);
        if $base < cmp_max {
            $base = cmp_max;
            true
        } else {
            false
        }
    }};
}

#[allow(unused_macros)]
macro_rules! min {
    ($a:expr $(,)*) => {{
        $a
    }};
    ($a:expr, $b:expr $(,)*) => {{
        std::cmp::min($a, $b)
    }};
    ($a:expr, $($rest:expr),+ $(,)*) => {{
        std::cmp::min($a, min!($($rest),+))
    }};
}

#[allow(unused_macros)]
macro_rules! max {
    ($a:expr $(,)*) => {{
        $a
    }};
    ($a:expr, $b:expr $(,)*) => {{
        std::cmp::max($a, $b)
    }};
    ($a:expr, $($rest:expr),+ $(,)*) => {{
        std::cmp::max($a, max!($($rest),+))
    }};
}

#[allow(unused_macros)]
macro_rules! mat {
    ($e:expr; $d:expr) => { vec![$e; $d] };
    ($e:expr; $d:expr $(; $ds:expr)+) => { vec![mat![$e $(; $ds)*]; $d] };
}

#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Pos {
    x: i64,
    y: i64,
}

impl Pos {
    pub fn dist(&self, oppo: &Pos) -> i64{
        ((self.x-oppo.x).pow(2)+(self.y-oppo.y).pow(2)).sqrt()
    }
}

struct UnionFind {
    n: usize,
    parent_or_size: Vec<i32>,
}

impl UnionFind {
    pub fn new(size: usize) -> Self {
        Self {
            n: size,
            parent_or_size: vec![-1; size],
        }
    }

    pub fn union(&mut self, a: usize, b: usize) -> usize {
        assert!(a < self.n);
        assert!(b < self.n);
        let (mut x, mut y) = (self.find(a), self.find(b));
        if x == y {
            return x;
        }
        if -self.parent_or_size[x] < -self.parent_or_size[y] {
            std::mem::swap(&mut x, &mut y);
        }
        self.parent_or_size[x] += self.parent_or_size[y];
        self.parent_or_size[y] = x as i32;
        x
    }

    pub fn same(&mut self, a: usize, b: usize) -> bool {
        assert!(a < self.n);
        assert!(b < self.n);
        self.find(a) == self.find(b)
    }

    pub fn find(&mut self, a: usize) -> usize {
        assert!(a < self.n);
        if self.parent_or_size[a] < 0 {
            return a;
        }
        self.parent_or_size[a] = self.find(self.parent_or_size[a] as usize) as i32;
        self.parent_or_size[a] as usize
    }

    pub fn size(&mut self, a: usize) -> usize {
        assert!(a < self.n);
        let x = self.find(a);
        -self.parent_or_size[x] as usize
    }

    pub fn groups(&mut self) -> Vec<Vec<usize>> {
        let mut find_buf = vec![0; self.n];
        let mut group_size = vec![0; self.n];
        for i in 0..self.n {
            find_buf[i] = self.find(i);
            group_size[find_buf[i]] += 1;
        }
        let mut result = vec![Vec::new(); self.n];
        for i in 0..self.n {
            result[i].reserve(group_size[i]);
        }
        for i in 0..self.n {
            result[find_buf[i]].push(i);
        }
        result
            .into_iter()
            .filter(|x| !x.is_empty())
            .collect::<Vec<Vec<usize>>>()
    }
}


#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Input {
    N: usize,
    M: usize,
    K: usize,
    Pos_tower: Vec<Pos>,
    UVWi: Vec<(usize, usize, usize, usize)>,
    Pos_people: Vec<Pos>,
    G: Vec<Vec<(usize, usize)>>,
}

impl Input {
    fn read_input() -> Self {
        input! {
            N: usize,
            M: usize,
            K: usize,
            XY_: [[i64; 2]; N],
            UVW_: [[usize; 3]; M],
            AB_: [[i64; 2]; K]
        }
        let mut G: Vec<Vec<(usize, usize)>> = vec![vec![]; 100];
        let mut UVWi: Vec<(usize, usize, usize, usize)> = vec![];
        for (i, uvw) in UVW_.iter().enumerate(){
            let u = uvw[0]-1;
            let v = uvw[1]-1;
            let w = uvw[2];
            G[u].push((v, w));
            G[v].push((u, w));
            UVWi.push((u, v, w, i));
        }
        UVWi.sort_by_key(|k| k.2);
        let Pos_tower: Vec<Pos> = XY_.iter().map(|xy| Pos{x: xy[0], y: xy[1]}).collect();
        let Pos_people: Vec<Pos> = AB_.iter().map(|xy| Pos{x: xy[0], y: xy[1]}).collect();
        Self { N, M, K, Pos_tower, UVWi, Pos_people, G}
    }
}


#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct State{
    used_tower: Vec<bool>,
}

impl State{
    fn init_state(input: &Input) -> Self{
        let mut used_tower: Vec<bool> = vec![];
        let mut rng = rand::thread_rng();
        used_tower.push(true);
        for _ in 1..input.N{
            if rng.gen::<usize>()%2==0{
                used_tower.push(false);
            } else {
                used_tower.push(true)
            }
        }
        Self {used_tower}
    }

    fn update(&mut self, param: &Vec<(usize, bool)>){
        for (i, flg) in param.iter(){
            self.used_tower[*i] = *flg;
        }
    }

    fn undo(&mut self, param: &Vec<(usize, bool)>){
        for (i, flg) in param.iter(){
            self.used_tower[*i] = !flg;
        }
    }

    fn get_mst(&mut self, input: &Input) -> Vec<bool>{
        let mut used_edge = vec![false; input.M];
        let mut uf = UnionFind::new(input.N);
        for (u, v, w, i) in input.UVWi.iter(){
            if (!self.used_tower[*u] || self.used_tower[*v]) {
                continue;
            }
            if uf.find(*u)!=uf.find(*v){
                used_edge[*i] = true;
                uf.union(*u, *v);
            }
        }
        let mut flg = true;
        for i in 0..input.N{
            if self.used_tower[i] && uf.find(0)!=uf.find(i){
                flg = false;
                break;
            }
        }
        if flg{
            return used_edge;
        }
        for (u, v, w, i) in input.UVWi.iter(){
            if (used_edge[*i]) {
                continue;
            }
            if uf.find(*u)!=uf.find(*v){
                used_edge[*i] = true;
                uf.union(*u, *v);
            }
        }
        used_edge
    }

    fn get_neighbor(&mut self) -> Vec<(usize, bool)>{
        let mut rng = rand::thread_rng();
        let i = rng.gen::<usize>()%self.used_tower.len();
        vec![(i, !self.used_tower[i])]
    }

    fn get_score(&mut self, input: &Input) -> i64{
        let used_edge = self.get_mst(input);
        let mut P = vec![0_usize; input.N];
        let mut score = 0;
        for (u, v, w, i) in input.UVWi.iter(){
            if used_edge[*i]{
                score += w;
            }
        }
        let mut used_tower_ids = vec![];
        for i in 0..input.N{
            if self.used_tower[i]{
                used_tower_ids.push(i);
            }
        }
        for p_people in input.Pos_people.iter(){
            let (mut min_d, mut min_i) = (INF, 0_usize);
            for i in used_tower_ids.iter(){
                let d = input.Pos_tower[*i].dist(p_people);
                if d<min_d{
                    min_d = d;
                    min_i = *i;
                }
            }
            P[min_i] = max!(min_d as usize+1, P[min_i]);
        }
        for p in P.iter(){
            score += p*p;
        }
        score as i64
    }

    fn print(&mut self, input: &Input){
        let used_edge = self.get_mst(input);
        let mut P = vec![0_usize; input.N];
        let mut score = 0;
        let mut used_tower_ids = vec![];
        for i in 0..input.N{
            if self.used_tower[i]{
                used_tower_ids.push(i);
            }
        }
        for p_people in input.Pos_people.iter(){
            let (mut min_d, mut min_i) = (INF, 0_usize);
            for i in used_tower_ids.iter(){
                let d = input.Pos_tower[*i].dist(p_people);
                if d<min_d{
                    min_d = d;
                    min_i = *i;
                }
            }
            P[min_i] = max!(min_d as usize+1, P[min_i]);
        }
        for p in P.iter(){
            score += p*p;
        }
        let dst: Vec<String> = P.iter().map(|x| x.to_string()).collect();
        println!("{}", dst.join(" "));
        let dst: Vec<String> = used_edge.iter().map(|x| if *x {String::from("1")} else {String::from("0")}).collect();
        println!("{}", dst.join(" "));
    }
}


fn simanneal(input: &Input, init_state: State, timer: &Instant, limit: f64) -> State{
    let mut state = init_state.clone();
    let mut best_state = init_state.clone();
    let mut all_iter = 0;
    let mut accepted_cnt = 0;
    let mut rng = rand::thread_rng();
    let start_temp: f64 = 1000000.0;
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
        let pre_score = state.get_score(input);
        let mut params = state.get_neighbor();
        state.update(&params);
        let new_score = state.get_score(input);
        let score_diff = new_score-pre_score;
        temp = start_temp + (end_temp-start_temp)*elasped_time/limit;
        if (score_diff<=0 || ((-score_diff as f64)/temp).exp() > rng.gen::<f64>()){
            accepted_cnt += 1;
            last_updated = all_iter;
            state.print(input);
            // eprintln!("{}", new_score);
            if state.get_score(input)<best_state.get_score(input){
                best_state = state.clone();
            }
        } else {
            state.undo(&params);
            // state.undo(input, params);
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
    // let mut best_state = State::init_state(&input);
    // for i in 0..1{
    //     let mut state = solve(&input, &timer, 5.5*((i+1) as f64));
    //     if state.get_score(1)<best_state.get_score(1){
    //         best_state = state;
    //     }
    // }
    // best_state.print();
}

fn solve(input: &Input, timer:&Instant, tl: f64){
    let mut state = State::init_state(input);
    let mut best_state = simanneal(input, state, timer, tl);
    best_state.print(input);
    // best_state.print();
    // eprintln!("{:?}", state);
    // let mut block1 = Block::single_cube(0, (1, 1, 1));
    // block1.update_cubes(&vec![(1, (1, 2, 1)), (1, (1, 2, 2)), (0, (1, 1, 1))]);
    // //block1.add_cubes(&vec![(1, 2, 1), (2, 1, 1), (1, 1, 2)]);
    // let mut block2 = Block::single_cube(0, (8, 8, 8));
    // block2.add_cubes(&vec![(8, 7, 8), (8, 8, 7), (7, 8, 8)]);
    // eprintln!("{:?} {}", block1, block1.serial);
    // eprintln!("{:?} {}", block2, block2.serial);
}

// fn solve(input: &Input, timer:&Instant, tl: f64) -> State{
//     let init_state = State::init_state(input);
//     let best_state = simanneal(input, init_state, &timer, tl);
//     //println!("{}", best_state);
//     eprintln!("{}", timer.elapsed().as_secs_f64());
//     best_state
// }
