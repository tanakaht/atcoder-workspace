#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused)]
use std::collections::{HashSet, HashMap};
use std::hash::Hash;
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
use grid::{Coordinate, Map2d, ADJACENTS, CoordinateDiff, STUMPDIFF};


const MOD: usize = 998244353;

#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Input {
    n: usize,
    m: usize,
    k: usize,
    A: Map2d<usize>,
    S: Vec<Vec<usize>>,
}

impl Input {
    fn read_input() -> Self {
        input! {
            n: usize,
            m: usize,
            k: usize,
            //m: usize,
            //a: [i32; n], // a is Vec<i32>, n-array.
            A: [usize; n*n], // `a` is Vec<Vec<i32>>, (m, n)-matrix.
            S: [[usize; 9]; m] // `a` is Vec<Vec<i32>>, (m, n)-matrix.
        }
        Self { n, m, k, A: Map2d::new(A, n) , S}
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone)]
struct Neighbor{
    params: Vec<(usize, usize, Coordinate)>
}

#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct State{
    M: Map2d<usize>,
    cnts: HashMap<(usize, Coordinate), usize>,
    cnt: usize,
    score: usize,
    score_weighted: usize,
}

impl State{
    fn init_state(input: &Input) -> Self{
        let mut score = 0;
        let mut score_weighted = 0;
        for i in 0..input.n{
            for j in 0..input.n{
                let c = Coordinate::new(i, j);
                score += input.A[c];
                score_weighted += input.A[c]*c.weight();
            }
        }
        let mut ret = Self {
            M: input.A.clone(),
            cnts: HashMap::new(),
            cnt: 0,
            score,
            score_weighted,
        };
        ret
    }
    fn init_state2(input: &Input) -> Self{
        let mut score = 0;
        let mut score_weighted = 0;
        for i in 0..input.n{
            for j in 0..input.n{
                let c = Coordinate::new(i, j);
                score += input.A[c];
                score_weighted += input.A[c]*c.weight();
            }
        }
        let mut ret = Self {
            M: input.A.clone(),
            cnts: HashMap::new(),
            cnt: 0,
            score,
            score_weighted,
        };
        for i in 0..input.n-2{
            for j in 0..input.n-2{
                let c = Coordinate::new(i, j);
                let mut best_score = 0;
                let mut best_m = 0;
                for m in 0..input.m{
                    let score = (input.S[m][0]+ret.M[c])%MOD;
                    if score>best_score{
                        best_score = score;
                        best_m = m;
                    }
                }
                ret.update(input, &Neighbor{params: vec![(0, best_m, c)]});
            }
        }
        ret
    }

    fn update(&mut self, input: &Input, neighbor: &Neighbor){
        for (flg, m, p) in neighbor.params.iter(){
            if *flg==0{
                self.stump(input, *m, *p);
            } else {
                self.unstump(input, *m, *p);
            }
        }
    }

    fn undo(&mut self, input: &Input, neighbor: &Neighbor){
        for (flg, m, p) in neighbor.params.iter().rev(){
            if *flg==0{
                self.unstump(input, *m, *p);
            } else {
                self.stump(input, *m, *p);
            }
        }
    }

    fn stump(&mut self, input: &Input, m: usize, p: Coordinate){
        // if self.cnt>=input.k{
        //     return;
        // }
        // assert!(self.cnt<input.k);
        for (d, pd) in input.S[m].iter().zip(STUMPDIFF.iter()){
            let p_ = p+*pd;
            self.score -= self.M[p_]%MOD;
            self.score_weighted -= (self.M[p_]%MOD)*p_.weight();
            self.M[p_] += d;
            self.score += self.M[p_]%MOD;
            self.score_weighted += (self.M[p_]%MOD)*p_.weight();
        }
        *self.cnts.entry((m, p)).or_insert(0) += 1;
        self.cnt += 1;
    }

    fn unstump(&mut self, input: &Input, m: usize, p: Coordinate){
        *self.cnts.get_mut(&(m, p)).unwrap() -= 1;
        if *self.cnts.get(&(m, p)).unwrap() == 0{
            self.cnts.remove(&(m, p));
        }
        self.cnt -= 1;
        for (d, pd) in input.S[m].iter().zip(STUMPDIFF.iter()){
            let p_ = p+*pd;
            self.score -= self.M[p_]%MOD;
            self.score_weighted -= (self.M[p_]%MOD)*p_.weight();
            self.M[p_] -= d;
            self.score += self.M[p_]%MOD;
            self.score_weighted += (self.M[p_]%MOD)*p_.weight();
        }
    }

    fn get_all_neighbor_params(&mut self, input: &Input) -> Vec<(usize, (usize, usize, Coordinate))>{
        let mut ret = vec![];
        let mut appeared = HashSet::new();
        for _ in 0..20{
            let m = rand::thread_rng().gen::<usize>()%input.m;
            let i = rand::thread_rng().gen::<usize>()%(input.n-2);
            let j = rand::thread_rng().gen::<usize>()%(input.n-2);
            if appeared.contains(&(m, i, j)){
                continue;
            }
            appeared.insert((m, i, j));
            let p = Coordinate::new(i, j);
            self.update(input, &Neighbor{params: vec![(0, m, p)]});
            ret.push((self.score, (0, m, p)));
            self.undo(input, &Neighbor{params: vec![(0, m, p)]});
        }
        ret
    }

    fn get_all_neighbor_params2(&mut self, input: &Input) -> Vec<(usize, (usize, usize, Coordinate))>{
        let mut ret = vec![];
        for m in 0..input.m{
            for i in 0..input.n-2{
                for j in 0..input.n-2{
                    let p = Coordinate::new(i, j);
                    self.update(input, &Neighbor{params: vec![(0, m, p)]});
                    ret.push((self.score, (0, m, p)));
                    self.undo(input, &Neighbor{params: vec![(0, m, p)]});
                }
            }
        }
        ret
    }

    fn get_neighbor(&mut self, input: &Input) -> Neighbor{
        let mut rng = rand::thread_rng();
        let mut params = vec![];
        let mut flg = rng.gen::<usize>()%50;
        if flg<10{
            // 押す
            if self.cnt<input.k{
                let i = rng.gen::<usize>()%(input.n-2);
                let j = rng.gen::<usize>()%(input.n-2);
                params.push((0, rng.gen::<usize>()%input.m, Coordinate::new(i, j)));
            }
        } else if flg<20{
            // 引く
            if self.cnt>0{
                let (m, p) = self.cnts.keys().choose(&mut rng).unwrap();
                params.push((1, *m, *p));
            }
        } else if flg<50{
            // swap
            if self.cnt>0{
                let (m, p) = self.cnts.keys().choose(&mut rng).unwrap();
                let mut m2 = rng.gen::<usize>()%input.m;
                while *m==m2{
                    m2 = rng.gen::<usize>()%input.m;
                }
                params.push((1, *m, *p));
                params.push((0, m2, *p));
            }
        } else {
            // ある点から右下を貪欲に押す
            let i_start = rng.gen::<usize>()%(input.n-2);
            let j_start = rng.gen::<usize>()%(input.n-2);
            for i in i_start..input.n-2{
                for j in j_start..input.n-2{
                    if self.cnt>=input.k{
                        break;
                    }
                    let c = Coordinate::new(i, j);
                    let mut best_score = 0;
                    let mut best_m = 0;
                    for m in 0..input.m{
                        let score = (input.S[m][0]+self.M[c])%MOD;
                        if score>best_score{
                            best_score = score;
                            best_m = m;
                        }
                    }
                    if best_score>self.M[c]%MOD{
                        self.update(input, &Neighbor{params: vec![(0, best_m, c)]});
                        params.push((0, best_m, c));
                    }
                }
            }
            self.undo(input, &Neighbor{params: params.clone()});
            // let amount = rng.gen::<usize>()%self.cnt;
            // let changes = self.cnts.keys().choose_multiple(&mut rng, amount);
            // for (m, p) in changes.iter(){
            //     let mut m2 = rng.gen::<usize>()%input.m;
            //     while *m==m2{
            //         m2 = rng.gen::<usize>()%input.m;
            //     }
            //     params.push((1, *m, *p));
            //     params.push((0, m2, *p));
            // }

        // } else {
            // // 2回modする直前まで押す
            // let i = rng.gen::<usize>()%(input.n-2);
            // let j = rng.gen::<usize>()%(input.n-2);
            // let p = Coordinate::new(i, j);
            // let mut vals = [self.M[p]%MOD, self.M[p+1]%MOD, self.M[p+2]%MOD, self.M[p+input.n]%MOD, self.M[p+input.n+1]%MOD, self.M[p+input.n+2]%MOD, self.M[p+2*input.n]%MOD, self.M[p+2*input.n+1]%MOD, self.M[p+2*input.n+2]%MOD];
            // while self.cnt+params.len()<input.k{
            //     let m = rng.gen::<usize>()%input.m;
            //     for (i, &val) in input.S[m].iter().enumerate(){
            //         vals[i] += val;
            //     }
            //     if *vals.iter().max().unwrap()>2*MOD{
            //         break;
            //     }
            //     params.push((0, m, p));
            // }
        }
        Neighbor{params}
    }

    fn get_score(&self, input: &Input) -> (usize, usize){
        (self.score, self.score_weighted)
    }

    fn print(&mut self, input: &Input){
        println!("{}", self.cnts.values().sum::<usize>());
        for ((m, p), cnt) in self.cnts.iter(){
            for _ in 0..*cnt{
                println!("{} {} {}", m, p.row, p.col);
            }
        }
    }
}

fn metrix2score(metrix: (usize, usize), t: f64) -> f64{
    let (score, score_mul) = metrix;
    0.1*(score) as f64 + t*t*score_mul as f64
    // metrix.0 as f64
}

fn chokudai_search(input: &Input, chokudai_width: usize, timer:&Instant, tl: f64) -> State {
    let mut init_state = State::init_state(input);
    let mut heaps = vec![BinaryHeap::new(); 82];
    let mut best_score = 0;
    let mut best_neighbor = Neighbor{params: vec![]};
    for (s, param) in init_state.get_all_neighbor_params(input){
        heaps[1].push((s, Neighbor{params: vec![param]}));
        if s>=best_score{
            best_score = s;
            best_neighbor = Neighbor{params: vec![param]};
        }
    }
    // heaps[1].push((0, vec![0]));
    let mut cnt = 0;
    loop {
        if timer.elapsed().as_secs_f64() >= tl {
            break;
        }
        for t in 1..81 {
            if timer.elapsed().as_secs_f64() >= tl {
                break;
            }
            for _ in 0..chokudai_width {
                if heaps[t].is_empty() {
                    break;
                }
                let (_, neighbor) = heaps[t].pop().unwrap();
                let mut state = init_state.clone();
                state.update(input, &neighbor);
                for (s, param) in state.get_all_neighbor_params(input){
                    cnt += 1;
                    let mut neighbor2 = neighbor.clone();
                    neighbor2.params.push(param);
                    if s>=best_score{
                        best_score = s;
                        best_neighbor = neighbor2.clone();
                    }
                    heaps[t + 1].push((s, neighbor2));
                    if heaps[t].is_empty() {
                        break;
                    }
                }
            }
        }
    }
    eprintln!("cnt: {}", cnt);
    let mut best_state = init_state.clone();
    best_state.update(input, &best_neighbor);
    best_state
}


fn simanneal(input: &Input, init_state: State, timer: &Instant, limit: f64) -> State{
    let mut state = init_state.clone();
    let mut best_state = init_state.clone();
    let mut best_score = best_state.get_score(input).0;
    let mut cur_score = best_state.get_score(input);
    let mut all_iter = 0;
    let mut accepted_cnt = 0;
    let mut rng = rand::thread_rng();
    let start_temp: f64 = 1000000.0;
    // let start_temp: f64 = 1000000000.0;
    let end_temp: f64 = 0.1;
    let mut temp = start_temp;
    let mut last_updated = 0;
    eprintln!("simanneal start at {}", timer.elapsed().as_secs_f64());
    loop {
        let elasped_time = timer.elapsed().as_secs_f64();
        if elasped_time >= limit{
            break;
        }
        let neighbor = state.get_neighbor(input);
        if neighbor.params.len()==0{
            continue;
        }
        all_iter += 1;
        state.update(input, &neighbor);
        let new_score = state.get_score(input);
        let score_diff = metrix2score(new_score, limit-elasped_time)-metrix2score(cur_score, limit-elasped_time);
        // start_temp = (start_temp*(all_iter-1)+score_diff)/all_iter;
        temp = start_temp + (end_temp-start_temp)*elasped_time/limit;
        if score_diff>=0.0 || rng.gen_bool((score_diff as f64/temp).exp()){
            accepted_cnt += 1;
            eprintln!("{}: {:?} {:?} {:?}", elasped_time, cur_score, new_score, neighbor.params);
            cur_score = new_score;
            last_updated = all_iter;
            //state.print(input);
            if new_score.0>best_score{
                best_state = state.clone();
                best_score = new_score.0;
            }
        } else {
            state.undo(input, &neighbor);
        }
    }
    eprintln!("===== simanneal =====");
    eprintln!("all iter   : {}", all_iter);
    eprintln!("accepted   : {}", accepted_cnt);
    eprintln!("score  : {}", best_state.get_score(input).0);
    eprintln!("");
    best_state
}

fn main() {
    let timer = Instant::now();
    let input = Input::read_input();
    solve(&input, &timer, 1.8);
}

fn solve(input: &Input, timer:&Instant, tl: f64){
    // let mut init_state = State::init_state(input);
    // let mut best_state = simanneal(input, init_state, timer, tl);
    let mut best_state = chokudai_search(input, 3, timer, tl+1.0);
    eprintln!("{}", timer.elapsed().as_secs_f64());
    best_state.print(input);
}



#[allow(dead_code)]
mod grid {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct Coordinate {
        pub row: usize,
        pub col: usize,
    }

    impl Coordinate {
        pub const fn new(row: usize, col: usize) -> Self {
            Self { row, col }
        }

        pub fn in_map(&self, size: usize) -> bool {
            self.row < size && self.col < size
        }

        pub fn in_map2(&self, size_row: usize, size_col: usize) -> bool {
            self.row < size_row && self.col < size_col
        }


        pub const fn to_index(&self, size: usize) -> usize {
            self.row * size + self.col
        }

        pub const fn dist(&self, other: &Self) -> usize {
            Self::dist_1d(self.row, other.row) + Self::dist_1d(self.col, other.col)
        }

        const fn dist_1d(x0: usize, x1: usize) -> usize {
            (x0 as i64 - x1 as i64).abs() as usize
        }
        pub fn weight(&self) -> usize{
            18-self.row-self.col
            // 81 - self.row*9-self.col
        }

        pub fn get_adjs(&self, size: usize) -> Vec<Coordinate> {
            let mut result = Vec::with_capacity(4);
            for cd in super::ADJACENTS.iter() {
                if (*self+*cd).in_map(size) {
                    result.push(*self+*cd);
                }
            }
            result
        }

        pub fn get_adjs2(&self, size_row: usize, size_col: usize) -> Vec<Coordinate> {
            let mut result = Vec::with_capacity(4);
            for cd in super::ADJACENTS.iter() {
                if (*self+*cd).in_map2(size_row, size_col) {
                    result.push(*self+*cd);
                }
            }
            result
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

    pub const STUMPDIFF: [CoordinateDiff; 9] = [
        CoordinateDiff::new(0, 0),
        CoordinateDiff::new(0, 1),
        CoordinateDiff::new(0, 2),
        CoordinateDiff::new(1, 0),
        CoordinateDiff::new(1, 1),
        CoordinateDiff::new(1, 2),
        CoordinateDiff::new(2, 0),
        CoordinateDiff::new(2, 1),
        CoordinateDiff::new(2, 2)
    ];

    pub const DIRECTIONS: [char; 4] = ['U', 'R', 'D', 'L'];

    #[derive(Debug, Clone)]
    pub struct Map2d<T> {
        pub width: usize,
        pub height: usize,
        map: Vec<T>,
    }

    impl<T> Map2d<T> {
        pub fn new(map: Vec<T>, width: usize) -> Self {
            let height = map.len() / width;
            debug_assert!(width * height == map.len());
            Self { width, height, map }
        }
    }

    impl<T> std::ops::Index<Coordinate> for Map2d<T> {
        type Output = T;

        #[inline]
        fn index(&self, coordinate: Coordinate) -> &Self::Output {
            &self.map[coordinate.row * self.width + coordinate.col]
        }
    }

    impl<T> std::ops::IndexMut<Coordinate> for Map2d<T> {
        #[inline]
        fn index_mut(&mut self, coordinate: Coordinate) -> &mut Self::Output {
            &mut self.map[coordinate.row * self.width + coordinate.col]
        }
    }

    impl<T> std::ops::Index<&Coordinate> for Map2d<T> {
        type Output = T;

        #[inline]
        fn index(&self, coordinate: &Coordinate) -> &Self::Output {
            &self.map[coordinate.row * self.width + coordinate.col]
        }
    }

    impl<T> std::ops::IndexMut<&Coordinate> for Map2d<T> {
        #[inline]
        fn index_mut(&mut self, coordinate: &Coordinate) -> &mut Self::Output {
            &mut self.map[coordinate.row * self.width + coordinate.col]
        }
    }

    impl<T> std::ops::Index<usize> for Map2d<T> {
        type Output = [T];

        #[inline]
        fn index(&self, row: usize) -> &Self::Output {
            let begin = row * self.width;
            let end = begin + self.width;
            &self.map[begin..end]
        }
    }

    impl<T> std::ops::IndexMut<usize> for Map2d<T> {
        #[inline]
        fn index_mut(&mut self, row: usize) -> &mut Self::Output {
            let begin = row * self.width;
            let end = begin + self.width;
            &mut self.map[begin..end]
        }
    }
}
