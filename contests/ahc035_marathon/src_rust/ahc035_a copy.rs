#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused)]
use std::collections::{HashSet, HashMap};
use std::io::BufReader;
use std::process::exit;
use itertools::Itertools;
use num::{range, ToPrimitive};
use num_integer::Roots;
use proconio::source::line::LineSource;
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

const N: usize = 6;
const M: usize = 15;
const T: usize = 10;
const BREND_TEST_CNT: usize = 200;


#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Seed{
    spec: [usize; M]
}

impl Seed{
    fn new(v: Vec<usize>) -> Self{
        let mut spec = [0; M];
        spec.clone_from_slice(&v);
        Self{spec}
    }

    fn read_seeds() -> Vec<Self>{
        let stdin = std::io::stdin();
        let mut source = LineSource::new(BufReader::new(stdin.lock()));
        input! {
            from &mut source,
            X: [[usize; M]; 2*N*(N-1)]
        }
        let mut seeds = vec![];
        for x in X{
            seeds.push(Seed::new(x));
        }
        seeds
    }

    fn score(&self) -> usize{
        self.spec.iter().sum()
    }

    fn score_weighted(&self, w: &Vec<f64>) -> f64{
        let mut score = 0.0;
        for i in 0..M{
            score += self.spec[i] as f64 * w[i];
        }
        score
    }

    fn best_brend(&self, other: &Self) -> Self{
        let mut spec = [0; M];
        for i in 0..M{
            spec[i] = if self.spec[i]>other.spec[i]{self.spec[i]}else{other.spec[i]};
        }
        Self{spec}
    }

    fn same_idxs(&self, other: &Self) -> Vec<usize>{
        let mut ret = vec![];
        for i in 0..M{
            if self.spec[i]==other.spec[i]{
                ret.push(i);
            }
        }
        ret
    }

    fn brend(&self, other: &Self, rng: &mut ThreadRng) -> Self{
        let mut spec = [0; M];
        for i in 0..M{
            spec[i] = if rng.gen_bool(0.5){self.spec[i]}else{other.spec[i]};
        }
        Self{spec}
    }
}

#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Seeds{
    seeds: Vec<Seed>,
    best_seed: Seed,
    addj_scores: Vec<Vec<f64>>,
    weight: Vec<f64>
}

impl Seeds{
    fn new(seeds: Vec<Seed>, t: usize) -> Self{
        let mut rng = rand::thread_rng();
        let mut best_seed = Seed::new(vec![0; M]);
        for seed in &seeds{
            best_seed = best_seed.best_brend(seed);
        }
        let mut max_cnt = vec![0; M];
        for seed in &seeds{
            for i in 0..M{
                if seed.spec[i]==best_seed.spec[i]{
                    max_cnt[i] += 1;
                }
            }
        }
        let mut weight = max_cnt.iter().map(|x| 1.0+10.0/(*x as f64)).collect::<Vec<f64>>();
        let mut addj_scores = vec![vec![0.0; 2*N*(N-1)]; 2*N*(N-1)];
        for i in 0..2*N*(N-1){
            for j in i+1..2*N*(N-1){
                for _ in 0..BREND_TEST_CNT{
                    let seed = seeds[i].brend(&seeds[j], &mut rng);
                    if t<5{
                        addj_scores[i][j] += ((seed.score_weighted(&weight) as f64)/BREND_TEST_CNT as f64).powf(5.0);
                    } else {
                        addj_scores[i][j] += ((seed.score_weighted(&weight) as f64)/BREND_TEST_CNT as f64).powf(0.5);
                    }
                }
                addj_scores[j][i] = addj_scores[i][j];
            }
        }
        Self{seeds, best_seed, addj_scores, weight}
    }

    fn read_seeds(t: usize) -> Self{
        let stdin = std::io::stdin();
        let mut source = LineSource::new(BufReader::new(stdin.lock()));
        input! {
            from &mut source,
            X: [[usize; M]; 2*N*(N-1)]
        }
        let mut seeds = vec![];
        for x in X{
            seeds.push(Seed::new(x));
        }
        Self::new(seeds, t)
    }
}



#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Input {
    n: usize,
    m: usize,
    t: usize,
    seeds: Seeds,
}

impl Input {
    fn read_input() -> Self {
        let stdin = std::io::stdin();
        let mut source = LineSource::new(BufReader::new(stdin.lock()));
        input! {
            from &mut source,
            n: usize,
            m: usize,
            t: usize,
            X: [[usize; M]; 2*N*(N-1)]
            //m: usize,
            //a: [i32; n], // a is Vec<i32>, n-array.
            //ab: [[i32; n]; m] // `a` is Vec<Vec<i32>>, (m, n)-matrix.
        }
        let mut seeds = vec![];
        for x in X{
            seeds.push(Seed::new(x));
        }
        Self { n, m, t, seeds: Seeds::new(seeds, 0)}
    }
}

struct Neighbor{
    c0: Coordinate,
    idx0: usize,
    c1: Coordinate,
    idx1: usize,
}

#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct State{
    seeds: Seeds,
    best_cnt: [usize; M],
    map: Map2d<usize>,
    rests: HashSet<usize>,
    score: f64,
    t: usize
}

impl State{
    fn init_state(input: &Input, seeds: &Seeds, t: usize) -> Self{
        let mut rng = rand::thread_rng();
        let mut accept_idxs = vec![];
        let mut best_cnt = [0; M];
        let mut rests: HashSet<usize> = (0..2*N*(N-1)).into_iter().collect();
        // mが最大のを一つずつ採用
        for m in 0..M{
            for (i, seed) in seeds.seeds.iter().enumerate().sorted_by_key(|x| usize::MAX-x.1.score()){
                if !rests.contains(&i){
                    continue;
                }
                if seeds.best_seed.spec[m]==seed.spec[m]{
                    accept_idxs.push(i);
                    rests.remove(&i);
                    for j in 0..M{
                        if seeds.seeds[i].spec[j]==seeds.best_seed.spec[j]{
                            best_cnt[j] += 1;
                        }
                    }
                    break;
                }
            }
        }
        // 面積大きいのを採用
        for (i, seed) in seeds.seeds.iter().enumerate().sorted_by_key(|x| usize::MAX-x.1.score()){
            if accept_idxs.len()>=N*N{
                break;
            }
            if !rests.contains(&i){
                continue;
            }
            accept_idxs.push(i);
            rests.remove(&i);
            for j in 0..M{
                if seeds.seeds[i].spec[j]==seeds.best_seed.spec[j]{
                    best_cnt[j] += 1;
                }
            }
        }
        accept_idxs.shuffle(&mut rng);
        let mut map = Map2d::new(accept_idxs);
        let mut score = 0.0;
        // 縦
        for i in 0..N-1{
            for j in 0..N{
                let idx1 = map[Coordinate::new(i, j)];
                let idx2 = map[Coordinate::new(i+1, j)];
                score += seeds.addj_scores[idx1][idx2];
            }
        }
        for i in 0..N{
            for j in 0..N-1{
                let idx1 = map[Coordinate::new(i, j)];
                let idx2 = map[Coordinate::new(i, j+1)];
                score += seeds.addj_scores[idx1][idx2];
            }
        }
        Self {seeds: seeds.clone(), map, rests, score, t, best_cnt}
    }

    fn update(&mut self, params: &Neighbor){
        // c0を消す
        for c0_ in params.c0.get_adjacents(){
            self.score -= self.seeds.addj_scores[params.idx0][self.map[c0_]];
        }
        // c1を消す
        if params.c1.in_map(){
            for c1_ in params.c1.get_adjacents(){
                self.score -= self.seeds.addj_scores[params.idx1][self.map[c1_]];
            }
        }
        self.map[params.c0] = params.idx1;
        if params.c1.in_map(){
            self.map[params.c1] = params.idx0;
        } else {
            self.rests.remove(&params.idx1);
            self.rests.insert(params.idx0);
        }
        // c0を追加
        for c0_ in params.c0.get_adjacents(){
            self.score += self.seeds.addj_scores[params.idx1][self.map[c0_]];
        }
        // c1を追加
        if params.c1.in_map(){
            for c1_ in params.c1.get_adjacents(){
                self.score += self.seeds.addj_scores[params.idx0][self.map[c1_]];
            }
        }
    }

    fn undo(&mut self, params: &Neighbor){
        self.update(&Neighbor{idx0: params.idx1, c0: params.c0, idx1: params.idx0, c1: params.c1});
    }

    fn get_neighbor(&mut self, input: &Input) -> Neighbor{
        let n = input.n;
        let mut rng = rand::thread_rng();
        let mode_flg = rng.gen::<usize>()%100;
        if mode_flg<50{
            let mut c0 = Coordinate::new(rng.gen::<usize>()%N, rng.gen::<usize>()%N);
            let c1 = *c0.get_adjacents().iter().choose(&mut rng).unwrap();
            Neighbor{idx0: self.map[c0], c0, idx1: self.map[c1], c1}
        } else if mode_flg<95{
            let mut c0 = Coordinate::new(rng.gen::<usize>()%N, rng.gen::<usize>()%N);
            let mut c1 = Coordinate::new(rng.gen::<usize>()%N, rng.gen::<usize>()%N);
            while c0==c1{
                c1 = Coordinate::new(rng.gen::<usize>()%N, rng.gen::<usize>()%N);
            }
            Neighbor{idx0: self.map[c0], c0, idx1: self.map[c1], c1}
        } else {
            let c0 = Coordinate::new(rng.gen::<usize>()%N, rng.gen::<usize>()%N);
            let idx1 = *self.rests.iter().choose(&mut rng).unwrap();
            Neighbor{idx0: self.map[c0], c0, idx1, c1: Coordinate::new(!0, !0)}
        }
    }

    fn get_score(&self) -> f64{
        self.score
    }

    fn print(&mut self, input: &Input){
        for i in 0..N{
            println!("{}", self.map[i].iter().map(|x| x.to_string()).join(" "));
        }
    }

    // fn brend(&self) -> Seeds{
    //     let mut rng = rand::thread_rng();
    //     let mut seeds = vec![];
    //     // 縦
    //     for i in 0..N-1{
    //         for j in 0..N{
    //             let idx1 = self.idxs[Coordinate::new(i, j)];
    //             let idx2 = self.idxs[Coordinate::new(i+1, j)];
    //             seeds.push(self.seeds.seeds[idx1].brend(&self.seeds.seeds[idx2], &mut rng));
    //         }
    //     }
    //     for i in 0..N{
    //         for j in 0..N-1{
    //             let idx1 = self.idxs[Coordinate::new(i, j)];
    //             let idx2 = self.idxs[Coordinate::new(i, j+1)];
    //             seeds.push(self.seeds.seeds[idx1].brend(&self.seeds.seeds[idx2], &mut rng));
    //         }
    //     }
    //     Seeds::new(seeds)
    // }

    // fn get_score(&mut self) -> f64{
    //     if let Some(score) = self.score{
    //         return score;
    //     }
    //     let cnt = 10;
    //     let mut score = 0.0;
    //     for _ in 0..cnt{
    //         let seeds = self.brend();
    //         score += seeds.score(self.t, &self.seeds.best_seed);
    //     }
    //     score /= cnt as f64;
    //     self.score = Some(score);
    //     score
    // }
}


fn simanneal(input: &Input, init_state: State, timer: &Instant, limit: f64) -> State{
    let mut state = init_state.clone();
    let mut cur_score = state.get_score();
    let mut all_iter = 0;
    let mut accepted_cnt = 0;
    let mut rng = rand::thread_rng();
    let start_temp: f64 = 100.0;
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
        state.update(&neighbor);
        let new_score = state.get_score();
        let score_diff = (new_score-cur_score);
        // start_temp = (start_temp*(all_iter-1)+score_diff)/all_iter;
        temp = start_temp + (end_temp-start_temp)*elasped_time/limit;
        if score_diff>=0.0 || rng.gen_bool((score_diff as f64/temp).exp()){
            accepted_cnt += 1;
            //eprintln!("{} {} {:?}", cur_score, new_score, turns);
            cur_score = new_score;
            last_updated = all_iter;
            //state.print(input);
        } else {
            state.undo(&neighbor);
        }
    }
    eprintln!("===== simanneal =====");
    eprintln!("all iter   : {}", all_iter);
    eprintln!("accepted   : {}", accepted_cnt);
    eprintln!("score  : {}", state.get_score());
    eprintln!("");
    state
}

fn main() {
    let timer = Instant::now();
    let input = Input::read_input();
    solve(&input, &timer, 1.8);
}

fn solve(input: &Input, timer:&Instant, tl: f64){
    // let mut init_state = State::init_state(input);
    // let mut best_state = simanneal(input, init_state, timer, tl);
    // best_state.print(input);
    let mut seeds = input.seeds.clone();
    for t in 0..T{
        if t>7{
            for i in 0..2*N*(N-1){
                for j in 0..2*N*(N-1){
                    seeds.addj_scores[i][j] = seeds.addj_scores[i][j].powf(10.0);
                }
            }
        }
        let mut init_state = State::init_state(input, &seeds, t);
        let limit = if (t<8){0.0} else {tl*(t-8) as f64/2 as f64};//tl*((t+1) as f64/T as f64)}};
        let mut best_state = simanneal(input, init_state, timer, limit);
        best_state.print(input);
        seeds = Seeds::read_seeds(t);
    }
}


use grid::{Coordinate, Map2d, ADJACENTS};

#[allow(dead_code)]
mod grid {
    const N: usize = 6;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub struct Coordinate {
        pub row: usize,
        pub col: usize,
    }

    impl Coordinate {
        pub const fn new(row: usize, col: usize) -> Self {
            Self { row, col }
        }

        pub fn in_map(&self) -> bool {
            self.row < N && self.col < N
        }

        pub const fn to_index(&self) -> usize {
            self.row * N + self.col
        }

        pub const fn dist(&self, other: &Self) -> usize {
            Self::dist_1d(self.row, other.row) + Self::dist_1d(self.col, other.col)
        }

        const fn dist_1d(x0: usize, x1: usize) -> usize {
            (x0 as i64 - x1 as i64).abs() as usize
        }

        pub fn get_adjacents(&self) -> Vec<Coordinate>{
            let mut adjacents = vec![];
            for &diff in &ADJACENTS{
                let new_coord = *self + diff;
                if new_coord.in_map(){
                    adjacents.push(new_coord);
                }
            }
            adjacents
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub struct CoordinateDiff {
        pub dr: usize,
        pub dc: usize,
    }

    impl CoordinateDiff {
        pub const fn new(dr: usize, dc: usize) -> Self {
            Self { dr, dc }
        }

        pub const fn invert(&self) -> Self {
            Self::new(0usize.wrapping_sub(self.dr), 0usize.wrapping_sub(self.dc))
        }
    }

    impl std::ops::Add<CoordinateDiff> for Coordinate {
        type Output = Coordinate;

        fn add(self, rhs: CoordinateDiff) -> Self::Output {
            Coordinate::new(self.row.wrapping_add(rhs.dr), self.col.wrapping_add(rhs.dc))
        }
    }

    pub const ADJACENTS: [CoordinateDiff; 4] = [
        CoordinateDiff::new(!0, 0),
        CoordinateDiff::new(0, 1),
        CoordinateDiff::new(1, 0),
        CoordinateDiff::new(0, !0),
    ];

    pub const DIRECTIONS: [char; 4] = ['U', 'R', 'D', 'L'];

    #[derive(Debug, Clone)]
    pub struct Map2d<T> {
        map: Vec<T>,
    }

    impl<T> Map2d<T> {
        pub fn new(map: Vec<T>) -> Self {
            debug_assert!(N * N == map.len());
            Self { map }
        }
    }

    impl<T> std::ops::Index<Coordinate> for Map2d<T> {
        type Output = T;

        #[inline]
        fn index(&self, coordinate: Coordinate) -> &Self::Output {
            &self.map[coordinate.row * N + coordinate.col]
        }
    }

    impl<T> std::ops::IndexMut<Coordinate> for Map2d<T> {
        #[inline]
        fn index_mut(&mut self, coordinate: Coordinate) -> &mut Self::Output {
            &mut self.map[coordinate.row * N + coordinate.col]
        }
    }

    impl<T> std::ops::Index<&Coordinate> for Map2d<T> {
        type Output = T;

        #[inline]
        fn index(&self, coordinate: &Coordinate) -> &Self::Output {
            &self.map[coordinate.row * N + coordinate.col]
        }
    }

    impl<T> std::ops::IndexMut<&Coordinate> for Map2d<T> {
        #[inline]
        fn index_mut(&mut self, coordinate: &Coordinate) -> &mut Self::Output {
            &mut self.map[coordinate.row * N + coordinate.col]
        }
    }

    impl<T> std::ops::Index<usize> for Map2d<T> {
        type Output = [T];

        #[inline]
        fn index(&self, row: usize) -> &Self::Output {
            let begin = row * N;
            let end = begin + N;
            &self.map[begin..end]
        }
    }

    impl<T> std::ops::IndexMut<usize> for Map2d<T> {
        #[inline]
        fn index_mut(&mut self, row: usize) -> &mut Self::Output {
            let begin = row * N;
            let end = begin + N;
            &mut self.map[begin..end]
        }
    }
}
