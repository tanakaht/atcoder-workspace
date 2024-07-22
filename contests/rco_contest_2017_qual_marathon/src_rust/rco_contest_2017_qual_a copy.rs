#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused)]
use std::collections::{HashSet, HashMap};
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


#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Input {
    h: usize,
    w: usize,
    k: usize,
    S: Map2d<i64>,
    candidates: Vec<(i64, Coordinate, usize)>,
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
        let mut cs_vec = vec![vec![Coordinate::new(0, 10)]];
        // 4**7なのでめっちゃ適当, 重複排除は最後にやる
        for _ in 1..8{
            let mut new_cs_vec = vec![];
            for cs in cs_vec.iter(){
                for adj in ADJACENTS.iter(){
                    let mut new_c = *cs.last().unwrap()+*adj;
                    if new_c.in_map() && (new_c.row!=0 || new_c.col>10) && !cs.contains(&new_c){
                        let mut new_cs = cs.clone();
                        new_cs.push(new_c);
                        new_cs_vec.push(new_cs);
                    }
                }
            }
            cs_vec = new_cs_vec;
        }
        let mut cs_set = HashSet::new();
        for cs in cs_vec.iter(){
            let mut new_cs = cs.clone();
            new_cs.sort();
            cs_set.insert(new_cs);
        }
        let mut minos = vec![];
        for cs in cs_set.iter(){
            let mut cd = [CoordinateDiff::new(0, 0); 8];
            for (i, c) in cs.iter().enumerate(){
                cd[i] = CoordinateDiff::new(c.row.wrapping_sub(0), c.col.wrapping_sub(10));
            }
            minos.push(Mino{cd});
        }
        let mut candidates = vec![];
        for i in 0..h{
            for j in 0..w{
                let c = Coordinate::new(i, j);
                for (mino_idx, mino) in minos.iter().enumerate(){
                    let score = mino.score(&S, c);
                    if score>0{
                        candidates.push((score, c, mino_idx));
                    }
                }
            }
        }
        let mut rng = rand::thread_rng();
        candidates.shuffle(&mut rng);
        candidates.sort_by_key(|(score, ..)|-*score);
        Self { h, w, k, S, minos, candidates }
    }
}

#[derive(Debug, Clone)]
struct Mino{
    cd: [CoordinateDiff; 8],
}

impl Mino{
    fn score(&self, S: &Map2d<i64>, c: Coordinate)->i64{
        let mut score = 0;
        for &d in &self.cd{
            let nc = c + d;
            if nc.in_map(){
                score *= S[nc];
            } else {
                score = 0;
            }
        }
        score
    }

    fn can_place(&self, is_used: &Map2d<bool>, c: Coordinate)->bool{
        for &d in &self.cd{
            let nc = c + d;
            if !nc.in_map() || !is_used[nc]{
                return false;
            }
        }
        true
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
