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
use std::rc::*;
use std::cell::UnsafeCell;



const MOD: usize = 998244353;

#[derive(Debug, Clone)]
struct Stump{
    n: usize,
    S: Vec<usize>,
    sidx: Vec<usize>,
}

#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Input {
    n: usize,
    m: usize,
    k: usize,
    A: Map2d<usize>,
    S: Vec<Vec<usize>>,
    stumps: Vec<Stump>,
    turn2metadata: Vec<(Coordinate, Vec<usize>, usize)> // スタンプ押す位置, 決まる位置、使っていいスタンプ数
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
        let mut stumps = vec![];
        let mut cnt = 0;
        stumps.push(Stump{n: 0, S:vec![0; 9], sidx: vec![]});
        for _ in 0..3
        {
            let mut new_stumps = stumps.clone();
            for Stump{n, S: S_, sidx: sidx_} in stumps.iter(){
                for (i, s) in S.iter().enumerate(){
                    if let Some(x) = sidx_.last(){
                        if *x>i{
                            continue;
                        }
                    }
                    let S__ = S_.iter().zip(s).map(|(x, y)| (x+y)%MOD).collect::<Vec<usize>>();
                    let mut sidx__ = sidx_.clone();
                    sidx__.push(i);
                    new_stumps.push(Stump{n: *n+1, S: S__, sidx: sidx__});
                }
            }
            stumps = new_stumps;
        }
        let mut turn2metadata = vec![];
        let mut cnt = 1.0;
        for i in 0..n-2{
            for j in 0..n-2{
                let c = Coordinate::new(i, j);
                let fixed_p = if i==n-3 && j==n-3{vec![0, 1, 2, 3, 4, 5, 6, 7, 8]}
                else if j==n-3{vec![0, 1, 2]} else if i==n-3{vec![0, 3, 6]} else{vec![0]};
                if fixed_p.len()==1{
                    cnt += 1.5;
                } else if fixed_p.len()==3{
                    cnt += 2.0;
                } else {
                    cnt = 81.0;
                }
                turn2metadata.push((c, fixed_p, cnt as usize));
            }
        }
        Self { n, m, k, A: Map2d::new(A, n) , S, stumps, turn2metadata}
    }
}



struct State{
    M: Map2d<usize>,
    score: usize,
    cnt: usize,
    turn: usize,
}

impl State{
    fn new(input: &Input) -> Self{
        let mut score = 0;
        for i in 0..input.n{
            for j in 0..input.n{
                let c = Coordinate::new(i, j);
                score += input.A[c];
            }
        }
        Self{
            turn: 0,
            M: input.A.clone(),
            cnt: 0,
            score: 0,
        }
    }

    fn score(&self)->usize{
        self.score
    }

    fn get_neighbor(&self, input: &Input)->Vec<u128>{
        let (c, fixed_p, cnt) = &input.turn2metadata[self.turn];
        let mut ret = vec![];
        let cidx = c.row*input.n+c.col;
        for (i, stump) in input.stumps.iter().enumerate(){
            if self.cnt+stump.n>*cnt || stump.n>fixed_p.len(){
                break;
            }
            ret.push((i*81+cidx) as u128);
        }
        ret
    }

    // スコアとハッシュの差分計算
    // 状態は更新しない
    fn try_apply(&self, input:&Input, op: u128)->usize{
        let (c, fixed_p, cnt) = &input.turn2metadata[self.turn];
        let mut score = self.score;
        let stump = &input.stumps[(op/81) as usize];
        for i in fixed_p{
            score += (self.M[*c+STUMPDIFF[*i]]+stump.S[*i])%MOD;
        }
        score
    }

    // 状態を更新する
    // 元の状態に戻すための情報を返す
    fn apply(&mut self, input:&Input, op: u128)->u128{
        let (c, fixed_p, cnt) = &input.turn2metadata[self.turn];
        let mut score = self.score;
        let stump = &input.stumps[(op/81) as usize];
        for i in fixed_p{
            score += (self.M[*c+STUMPDIFF[*i]]+stump.S[*i])%MOD;
        }
        self.score = score;
        for (i, cd) in STUMPDIFF.iter().enumerate(){
            self.M[*c+*cd] = (self.M[*c+*cd]+stump.S[i])%MOD;
        }
        self.turn += 1;
        self.cnt += stump.n;
        op
    }

    // applyから返された情報をもとに状態を元に戻す
    fn back(&mut self, input:&Input, op: u128){
        let (c, fixed_p, cnt) = &input.turn2metadata[self.turn-1];
        let mut score = self.score;
        let stump = &input.stumps[(op/81) as usize];
        for i in fixed_p{
            score -= self.M[*c+STUMPDIFF[*i]]%MOD;
        }
        self.score = score;
        for (i, cd) in STUMPDIFF.iter().enumerate(){
            self.M[*c+*cd] = (MOD+self.M[*c+*cd]-stump.S[i])%MOD;
        }
        self.turn -= 1;
        self.cnt -= stump.n;
    }
}


struct Candidate{
    op: u128,
    parent: Rc<Node>,
    score: usize,
    p: usize
}


struct Node{
    parent: Option<(u128,Rc<Node>)>, // 操作、親への参照
    child: UnsafeCell<Vec<(u128 ,Weak<Node>)>>, // 操作、子への参照
    score:usize,
}


// 多スタート用に構造体にまとめておくと楽
struct Tree{
    state: State,
    node: Rc<Node>,
    rank: usize
}

impl Tree{
    // 注意: depthは深くなっていくごとに-1されていく
    // oneはrootから子が1個なところまで省略させるために持つフラグ
    fn dfs(&mut self, input: &Input, next_states: &mut Vec<Candidate>, one:bool, p:&mut usize, depth:usize){
        if depth==0{
            let score = self.node.score;

            for op in self.state.get_neighbor(input){
                let next_score = self.state.try_apply(input, op);
                next_states.push(
                    Candidate{
                        op,
                        parent:self.node.clone(),
                        score:next_score,
                        p: *p
                    }
                );
                *p += 1;
            }
        } else{
            let node = self.node.clone();
            let child = unsafe{&mut *node.child.get()};
            // 有効な子だけにする
            child.retain(|(_, x)| x.upgrade().is_some());

            let next_one = one & (child.len()==1);

            // 定数調整の必要あり
            if depth==5{
                *p=0;
            }
            self.rank+=1;

            for (op, ptr) in child{
                self.node = ptr.upgrade().unwrap();
                let backup = self.state.apply(input, *op);
                self.dfs(input, next_states, next_one, p, depth-1);
                if !next_one{
                    self.state.back(input, backup);
                }
            }

            if !next_one{
                self.node = node.clone();
                self.rank -= 1;
            }
        }
    }
}


fn beam(input: &Input) -> Vec<u128>{
    const TURN:usize=49;
    const M:usize=3000; // ビーム幅
    eprintln!("start");

    let mut tree = {
        let state = State::new(input);
        let score = state.score();
        Tree{
            state,
            node: Rc::new(
                Node{
                    parent: None,
                    child: UnsafeCell::new(vec![]),
                    score
                }
            ),
            rank: 0
        }
    };

    let mut cur = vec![tree.node.clone()];
    let mut next_states = vec![];

    // let mut set = rustc_hash::FxHashSet::default();

    for i in 0..TURN{
        next_states.clear();
        tree.dfs(input, &mut next_states, true, &mut 0, i-tree.rank);

        if i+1!=TURN{
            // 上位M個を残す
            if next_states.len()>M{
                next_states.select_nth_unstable_by_key(M,|Candidate{score, p, ..}|(Reverse(*score), *p));
                next_states.truncate(M);
            }

            cur.clear();
            // set.clear();
            for Candidate{op,parent,score,..} in &next_states{
                // 重複除去
                if true{//set.insert(*hash){
                    let child=unsafe{&mut *parent.child.get()};
                    let child_ptr=Rc::new(
                        Node{
                            parent: Some((*op,parent.clone())),
                            child: UnsafeCell::new(vec![]),
                            score: *score
                        }
                    );
                    child.push((*op, Rc::downgrade(&child_ptr)));
                    cur.push(child_ptr);
                }
            }
        }
    }

    // 最良の状態を選択
    let Candidate{op,parent:mut ptr,score,..} = next_states.into_iter().max_by_key(|Candidate{score,..}|*score).unwrap();

    let mut ret=vec![op];
    eprintln!("score: {}",score);
    eprintln!("rank: {}",TURN-tree.rank);

    // 操作の復元
    while let Some((op,parent))=ptr.parent.clone(){
        ret.push(op);
        ptr=parent.clone();
    }

    ret.reverse();
    ret
}



fn main() {
    let timer = Instant::now();
    let input = Input::read_input();
    solve(&input, &timer, 1.8);
    eprintln!("time elasped{}", timer.elapsed().as_secs_f64());
}

fn solve(input: &Input, timer:&Instant, tl: f64){
    let best_ops = beam(input);
    let mut cnt = 0;
    for op in best_ops.iter(){
        cnt += input.stumps[(*op/81) as usize].n;
    }
    println!("{}", cnt);
    for (turn, op) in best_ops.iter().enumerate(){
        let (c, ..) = &input.turn2metadata[turn];
        let stump = &input.stumps[(*op/81) as usize];
        for m in stump.sidx.iter(){
            println!("{} {} {}", m, c.row, c.col);
        }
    }
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
