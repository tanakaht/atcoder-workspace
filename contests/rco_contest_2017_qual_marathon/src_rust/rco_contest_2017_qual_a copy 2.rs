#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused)]
use std::collections::{HashMap, HashSet, VecDeque};
use std::process::exit;
use itertools::Itertools;
use num::{range, ToPrimitive};
use num_integer::Roots;
use permutohedron::Heap;
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
use grid::{Coordinate, Map2d, ADJACENTS, CoordinateDiff};
use std::rc::*;
use std::cell::UnsafeCell;



fn find_polyominoes(n: usize) -> Vec<Vec<Coordinate>> {
    let center_cell = Coordinate::new(0, 0);
    let mut candidates = VecDeque::new();
    let mut selected = HashSet::new();
    let mut appeared = HashSet::new();
    fn dfs(candidates: &mut VecDeque<Coordinate>, selected: &mut HashSet<Coordinate>, appeared: &mut HashSet<Coordinate>, last: Coordinate, depth: usize)->Vec<Vec<Coordinate>>{
        if depth==1{
            return vec![vec![last]];
        }
        for &adj in &ADJACENTS{
            let nc = last+adj;
            if nc.row<200 && (nc.row!=0 || (nc.row==0 && nc.col<200)) && !appeared.contains(&nc){
                candidates.push_back(nc);
                appeared.insert(nc);
            }
        }
        let mut ret = vec![];
        while let Some(c) = candidates.pop_front(){
            selected.insert(c);
            for mut vs in dfs(&mut candidates.clone(), &mut selected.clone(), &mut appeared.clone(), c, depth-1){
                vs.push(last);
                ret.push(vs);
            }
        }
        ret
    }
    appeared.insert(center_cell);
    dfs(&mut candidates, &mut selected, &mut appeared, center_cell, n)
}


#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Input {
    h: usize,
    w: usize,
    k: usize,
    S: Map2d<i64>,
    candidates: Vec<(i64, CoordinateDiff, usize)>,
    minos: Vec<Mino>,
}

impl Input {
    fn read_input() -> Self {
        input! {
            h: usize,
            w: usize,
            k: usize,
            S_: [Chars; h]
        }
        let mut S_map = vec![];
        for i in 0..h{
            for j in 0..w{
                S_map.push(S_[i][j] as i64 - 48);
            }
        }
        let S = Map2d::new(S_map);
        let mut minos = vec![];
        for cs_ in find_polyominoes(8){
            let mut cs = [Coordinate::new(0, 0); 8];
            for (i, c) in cs_.iter().enumerate(){
                cs[i] = *c;
            }
            minos.push(Mino{cs});
        }
        let mut candidates = vec![];
        for i in 0..h{
            for j in 0..w{
                let cd = CoordinateDiff::new(i, j);
                for (mino_idx, mino) in minos.iter().enumerate(){
                    let score = mino.score(&S, cd);
                    if score>0{
                        candidates.push((score, cd, mino_idx));
                    }
                }
            }
        }
        // let mut rng = rand::thread_rng();
        // candidates.shuffle(&mut rng);
        candidates.sort_by_key(|(score, ..)|-*score);
        Self { h, w, k, S, minos, candidates }
    }
}

#[derive(Debug, Clone, Hash)]
struct Mino{
    cs: [Coordinate; 8],
}

impl Mino{
    fn score(&self, S: &Map2d<i64>, cd: CoordinateDiff)->i64{
        let mut score = 1;
        for &c in &self.cs{
            let nc = c + cd;
            if nc.in_map(){
                score *= S[nc];
            } else {
                score = 0;
            }
        }
        score
    }

    fn can_place(&self, is_used: &Map2d<bool>, cd: CoordinateDiff)->bool{
        for &c in &self.cs{
            let nc = c + cd;
            if is_used[nc]{
                return false;
            }
        }
        true
    }

    fn place(&self, is_used: &mut Map2d<bool>, cd: CoordinateDiff){
        for &c in &self.cs{
            let nc = c + cd;
            is_used[nc] = true;
        }
    }

    fn unplace(&self, is_used: &mut Map2d<bool>, cd: CoordinateDiff){
        for &c in &self.cs{
            let nc = c + cd;
            is_used[nc] = false;
        }
    }
}


struct State{
    is_used: Map2d<bool>,
    score: i64,
    last_idx: usize,
    turn: usize,
}

impl State{
    fn new(input: &Input) -> Self{
        let mut is_used = Map2d::new(vec![false; 200*200]);
        for i in 0..input.h{
            for j in 0..input.w{
                let c = Coordinate::new(i, j);
                if input.S[c]==0{
                    is_used[c] = true;
                }
            }
        }
        Self{
            turn: 0,
            is_used,
            last_idx: 0,
            score: 0,
        }
    }

    fn score(&self)->i64{
        self.score
    }

    fn get_neighbor(&self, input: &Input)->Vec<usize>{
        let mut ret = vec![];
        for i in self.last_idx..std::cmp::min(1000000, input.candidates.len()){
            let (_, cd, mino_idx) = input.candidates[i];
            if input.minos[mino_idx].can_place(&self.is_used, cd){
                ret.push(i);
            }
            if ret.len()==100{
                break
            }
        }
        ret
    }

    // スコアとハッシュの差分計算
    // 状態は更新しない
    fn try_apply(&self, input:&Input, op: usize)->i64{
        if op==usize::MAX{
            self.score
        } else {
            self.score + input.candidates[op].0
        }
    }

    // 状態を更新する
    // 元の状態に戻すための情報を返す
    fn apply(&mut self, input:&Input, op: usize)->usize{
        let (score, cd, mino_idx) = input.candidates[op];
        self.score += score;
        input.minos[mino_idx].place(&mut self.is_used, cd);
        self.turn += 1;
        self.last_idx = op+1;
        op+input.candidates.len()*self.last_idx
    }

    // applyから返された情報をもとに状態を元に戻す
    fn back(&mut self, input:&Input, op: usize){
        let (score, cd, mino_idx) = input.candidates[op%input.candidates.len()];
        self.score -= score;
        input.minos[mino_idx].unplace(&mut self.is_used, cd);
        self.turn -= 1;
        self.last_idx = op/input.candidates.len();
    }
}


struct Candidate{
    op: usize,
    parent: Rc<Node>,
    score: i64,
    p: usize
}


struct Node{
    parent: Option<(usize,Rc<Node>)>, // 操作、親への参照
    child: UnsafeCell<Vec<(usize ,Weak<Node>)>>, // 操作、子への参照
    score: i64,
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


fn beam(input: &Input) -> Vec<usize>{
    const TURN:usize=300;
    const M:usize=50; // ビーム幅
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
    let mut best_ops = vec![];
    let mut best_score = 0;

    // let mut set = rustc_hash::FxHashSet::default();

    for i in 0..TURN{
        eprintln!("turn: {}, {}", i, next_states.len());
        next_states.clear();
        tree.dfs(input, &mut next_states, true, &mut 0, i-tree.rank);

        if i+1!=TURN{
            // 上位M個を残す
            if next_states.len()>M{
                next_states.select_nth_unstable_by_key(M,|Candidate{score, p, ..}|(Reverse(*score), *p));
                next_states.truncate(M);
            }

            if next_states.is_empty(){
                break;
            }
            cur.clear();
            // set.clear();
            for Candidate{op,parent,score,..} in &next_states{
                if best_score<*score{
                    let mut ptr = parent.clone();
                    best_score = *score;
                    best_ops=vec![*op];
                    // 操作の復元
                    while let Some((op,parent))=ptr.parent.clone(){
                        best_ops.push(op);
                        ptr=parent.clone();
                    }
                }
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
    eprintln!("best_score: {}", best_score);
    eprintln!("{}", input.candidates.len());

    best_ops
}



fn main() {
    let timer = Instant::now();
    let input = Input::read_input();
    solve(&input, &timer, 1.8);
    eprintln!("time elasped{}", timer.elapsed().as_secs_f64());
}

fn solve(input: &Input, timer:&Instant, tl: f64){
    let mut best_ops = beam(input);
    best_ops.retain(|&op|op!=usize::MAX);
    // 飛ばしたおけるのがあれば置いていく
    let mut is_used = Map2d::new(vec![false; 200*200]);
    for &op in &best_ops{
        let (score, cd, mino_idx) = input.candidates[op];
        input.minos[mino_idx].place(&mut is_used, cd);
    }
    for i in 0..input.candidates.len(){
        let (score, cd, mino_idx) = input.candidates[i];
        if input.minos[mino_idx].can_place(&is_used, cd){
            best_ops.push(i);
            input.minos[mino_idx].place(&mut is_used, cd);
        }
    }
    println!("{}", best_ops.len());
    for &op in &best_ops{
        let (score, cd, mino_idx) = input.candidates[op];
        for c_ in &input.minos[mino_idx].cs{
            let c = *c_ + cd;
            println!("{} {}", c.row+1, c.col+1);
        }
    }
}

#[allow(dead_code)]
mod grid {
    const SIZE: usize = 50;

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
        pub width: usize,
        pub height: usize,
        map: Vec<T>,
    }

    impl<T> Map2d<T> {
        pub fn new(map: Vec<T>) -> Self {
            debug_assert!(SIZE*SIZE == map.len());
            Self { width: SIZE, height: SIZE, map }
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
