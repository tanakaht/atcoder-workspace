#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused)]
use std::collections::{HashSet, HashMap};
use std::hash::Hash;
use std::process::exit;
use itertools::Itertools;
use nalgebra::QR;
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
use grid::{Coordinate, Map2d, ADJACENTS, CoordinateDiff, SIZE};
use std::rc::*;
use std::cell::UnsafeCell;
use std::collections::hash_map::DefaultHasher;


#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Input {
    n: usize,
    A: Vec<Vec<u8>>,
}

impl Input {
    fn read_input() -> Self {
        input! {
            n: usize,
            A: [[u8; SIZE]; SIZE] // `a` is Vec<Vec<i32>>, (m, n)-matrix.
        }
        Self { n, A}
    }
}

#[derive(Debug, Clone, Hash)]
struct Crane{
    p: Coordinate,
    item: usize,
}

#[derive(Debug, Clone, Hash)]
struct State{
    M: Map2d<u8>,
    cranes: Vec<Crane>,
    next: Vec<Vec<u8>>,
    next_out: Vec<u8>,
    score: usize,
}

// M
// 0~25: 置かれたコンテナ
// 32~57: 持たれたコンテナ(small_crane)
// 64~89: 置かれたコンテナの重み(big_crane)
// 96~101: 何も持たないクレーン
// MAX: 何もない
impl State{
    fn new(input: &Input) -> Self{
        let mut score = 0;
        let mut M = Map2d::new(vec![u8::MAX; SIZE*SIZE]);
        let mut cranes = vec![];
        let mut next = vec![vec![]; SIZE];
        let mut next_out = vec![];
        for i in 0..SIZE{
            cranes.push(Crane{p: Coordinate::new(i, 0), item: usize::MAX});
            next[i] = input.A[i].clone();
            next[i].reverse();
            let v = next[i].pop().unwrap();
            M[i][0] = v;
            next_out[i] = (i*SIZE) as u8;
            if v%SIZE as u8==0{
                score += 100;
            }
        }
        Self{
            M,
            cranes,
            next,
            next_out,
            score,
        }
    }

    fn score(&self)->usize{
        self.score
    }

    // op
    // 0: U
    // 1: R
    // 2: D
    // 3: L
    // 4: P or Q
    // 5: .
    // スコアとハッシュの差分計算
    // 状態は更新しない
    fn try_apply(&self, input:&Input, op: u64, hash: u64)->(usize, u64){
        // opがOKかチェック0:
        let mut ops = vec![];
        for i in 0..SIZE{
            ops.push(i%6)
            let x = op_%6;

            op


        }
        // opを適用
        // 元に戻す
    }

    // 状態を更新する
    // 元の状態に戻すための情報を返す
    fn apply(&mut self, input:&Input, op: u64)->u64{
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
    fn back(&mut self, input:&Input, op: u64){
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
    op: u64,
    parent: Rc<Node>,
    score: usize,
    hash: u64,
    p: usize
}


struct Node{
    parent: Option<(u64,Rc<Node>)>, // 操作、親への参照
    child: UnsafeCell<Vec<(u64 ,Weak<Node>)>>, // 操作、子への参照
    score:usize,
    hash: u64,
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
            let hash = self.node.hash;

            for op in 0..7776{
                let (next_score, next_hash) = self.state.try_apply(input, op, hash);
                next_states.push(
                    Candidate{
                        op,
                        parent: self.node.clone(),
                        score: next_score,
                        hash: next_hash,
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


fn beam(input: &Input) -> Vec<u64>{
    const TURN:usize=200;
    const M:usize=3000; // ビーム幅
    eprintln!("start");
    let mut hasher = DefaultHasher::new();

    let mut tree = {
        let state = State::new(input);
        let score = state.score();
        let hash = hasher.hash(&mut state);
        Tree{
            state,
            node: Rc::new(
                Node{
                    parent: None,
                    child: UnsafeCell::new(vec![]),
                    score,
                    hash
                }
            ),
            rank: 0
        }
    };

    let mut cur = vec![tree.node.clone()];
    let mut next_states = vec![];
    let mut set=rustc_hash::FxHashSet::default();

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
            for Candidate{op,parent,score,hash, ..} in &next_states{
                // 重複除去
                if set.insert(*hash){
                    let child=unsafe{&mut *parent.child.get()};
                    let child_ptr=Rc::new(
                        Node{
                            parent: Some((*op,parent.clone())),
                            child: UnsafeCell::new(vec![]),
                            score: *score,
                            hash: *hash,
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
    solve(&input, &timer, );
    eprintln!("time elasped{}", timer.elapsed().as_secs_f64());
}

fn solve(input: &Input, timer:&Instant, tl: f64){
    let best_ops = beam(input);
    let mut cnt = 0;
    let mut crane_ops = vec![vec![]; SIZE];
    for op in best_ops.iter(){
        for i in 0..SIZE{
            // TODO: 変換
            crane_ops[i].push("a");
        }
    }
    for i in 0..SIZE{
        println!("{}", crane_ops[i].join(""));
    }
}



#[allow(dead_code)]
mod grid {
    pub const SIZE: usize = 5;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct Coordinate {
        pub row: usize,
        pub col: usize,
    }

    impl Coordinate {
        pub const fn new(row: usize, col: usize) -> Self {
            Self { row, col }
        }

        pub fn in_map(&self) -> bool {
            self.row < SIZE && self.col < SIZE
        }

        pub const fn to_index(&self) -> usize {
            self.row * SIZE + self.col
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
                if (*self+*cd).in_map() {
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

    pub const DIRECTIONS: [char; 4] = ['U', 'R', 'D', 'L'];

    #[derive(Debug, Clone, Hash)]
    pub struct Map2d<T> {
        map: Vec<T>,
    }

    impl<T> Map2d<T> {
        pub fn new(map: Vec<T>) -> Self {
            debug_assert!(SIZE*SIZE == map.len());
            Self { map }
        }
    }

    impl<T> std::ops::Index<Coordinate> for Map2d<T> {
        type Output = T;

        #[inline]
        fn index(&self, coordinate: Coordinate) -> &Self::Output {
            &self.map[coordinate.row * SIZE + coordinate.col]
        }
    }

    impl<T> std::ops::IndexMut<Coordinate> for Map2d<T> {
        #[inline]
        fn index_mut(&mut self, coordinate: Coordinate) -> &mut Self::Output {
            &mut self.map[coordinate.row * SIZE + coordinate.col]
        }
    }

    impl<T> std::ops::Index<&Coordinate> for Map2d<T> {
        type Output = T;

        #[inline]
        fn index(&self, coordinate: &Coordinate) -> &Self::Output {
            &self.map[coordinate.row * SIZE + coordinate.col]
        }
    }

    impl<T> std::ops::IndexMut<&Coordinate> for Map2d<T> {
        #[inline]
        fn index_mut(&mut self, coordinate: &Coordinate) -> &mut Self::Output {
            &mut self.map[coordinate.row * SIZE + coordinate.col]
        }
    }

    impl<T> std::ops::Index<usize> for Map2d<T> {
        type Output = [T];

        #[inline]
        fn index(&self, row: usize) -> &Self::Output {
            let begin = row * SIZE;
            let end = begin + SIZE;
            &self.map[begin..end]
        }
    }

    impl<T> std::ops::IndexMut<usize> for Map2d<T> {
        #[inline]
        fn index_mut(&mut self, row: usize) -> &mut Self::Output {
            let begin = row * SIZE;
            let end = begin + SIZE;
            &mut self.map[begin..end]
        }
    }
}
