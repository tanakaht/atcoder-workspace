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
use lazy_static::lazy_static;


#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Input {
    n: usize,
    A: Vec<Vec<usize>>,
}

impl Input {
    fn read_input() -> Self {
        input! {
            n: usize,
            A_: [[usize; SIZE]; SIZE] // `a` is Vec<Vec<i32>>, (m, n)-matrix.
        }
        let mut A = A_.clone();
        for i in 0..SIZE{
            A[i].reverse();
        }
        Self {n, A}
    }
}

const SPACEIDX2COORDINATE: [Coordinate; 13] = [
    Coordinate::new(0, 0),
    Coordinate::new(1, 0),
    Coordinate::new(2, 0),
    Coordinate::new(3, 0),
    Coordinate::new(4, 0),
    Coordinate::new(0, 2),
    Coordinate::new(1, 2),
    Coordinate::new(2, 2),
    Coordinate::new(3, 2),
    Coordinate::new(0, 3),
    Coordinate::new(1, 3),
    Coordinate::new(2, 3),
    Coordinate::new(3, 3),
];

#[derive(Debug, Clone, Hash)]
struct Task{
    target_item: usize,
    target_item_p: Coordinate,
    target_place_p: Coordinate,
    status: usize, // 0: pickしに移動中、1: pickした直後, 2: pickして移動中, 3: 終了
}

lazy_static!{
    // 4, 5はのぞいた許される移動
    static ref OP_CANDIDATE0: Map2d<Vec<usize>> = Map2d::new([
        vec![1], vec![2, 1, 3], vec![3, 1], vec![2, 1, 3], vec![3],
        vec![0, 1], vec![2, 1, 3], vec![0, 3, 1], vec![2, 1, 3], vec![0, 3],
        vec![0, 1], vec![2, 1, 3], vec![0, 3, 1], vec![2, 1, 3], vec![0, 3],
        vec![0, 1], vec![2, 1, 3], vec![0, 3, 1], vec![2, 1, 3], vec![0, 3],
        vec![0, 1], vec![1, 3], vec![1, 0], vec![1, 0], vec![0],
    ]);

    // ピックor配置時の移動
    static ref OP_CANDIDATE0_PICK: Map2d<Vec<usize>> = Map2d::new([
        vec![4, 1], vec![4, 2, 1, 3], vec![4, 3, 1], vec![4, 2, 1, 3], vec![4, 3],
        vec![4, 0, 1], vec![4, 2, 1, 3], vec![4, 0, 3, 1], vec![4, 2, 1, 3], vec![4, 0, 3],
        vec![4, 0, 1], vec![4, 2, 1, 3], vec![4, 0, 3, 1], vec![4, 2, 1, 3], vec![4, 0, 3],
        vec![4, 0, 1], vec![4, 2, 1, 3], vec![4, 0, 3, 1], vec![4, 2, 1, 3], vec![4, 0, 3],
        vec![4, 0, 1], vec![4, 1, 3], vec![4, 1, 0], vec![4, 1, 0], vec![4, 0],
    ]);

    // タスク何もない時の移動
    static ref OP_CANDIDATE0_STAY: Map2d<Vec<usize>> = Map2d::new([
        vec![5, 1], vec![5, 2, 1, 3], vec![5, 3, 1], vec![5, 2, 1, 3], vec![5, 3],
        vec![5, 0, 1], vec![5, 2, 1, 3], vec![5, 0, 3, 1], vec![5, 2, 1, 3], vec![5, 0, 3],
        vec![5, 0, 1], vec![5, 2, 1, 3], vec![5, 0, 3, 1], vec![5, 2, 1, 3], vec![5, 0, 3],
        vec![5, 0, 1], vec![5, 2, 1, 3], vec![5, 0, 3, 1], vec![5, 2, 1, 3], vec![5, 0, 3],
        vec![5, 0, 1], vec![5, 1, 3], vec![5, 1, 0], vec![5, 1, 0], vec![5, 0],
    ]);

    // 上に行きたい時
    static ref OP_CANDIDATE0_0: Map2d<Vec<usize>> = Map2d::new([
        vec![5, 1], vec![5, 1, 3, 2], vec![5, 3, 1], vec![5, 1, 3, 2], vec![5, 3],
        vec![0, 5, 1], vec![5, 1, 3, 2], vec![0, 5, 3, 1], vec![5, 1, 3, 2], vec![0, 5, 3],
        vec![0, 5, 1], vec![5, 1, 3, 2], vec![0, 5, 3, 1], vec![5, 1, 3, 2], vec![0, 5, 3],
        vec![0, 5, 1], vec![5, 1, 3, 2], vec![0, 5, 3, 1], vec![5, 1, 3, 2], vec![0, 5, 3],
        vec![0, 5, 1], vec![5, 1, 3], vec![0, 5, 1], vec![0, 5, 1], vec![0, 5],
    ]);

    // 右に行きたい時
    static ref OP_CANDIDATE0_1: Map2d<Vec<usize>> = Map2d::new([
        vec![1, 5], vec![1, 5, 2, 3], vec![1, 5, 3], vec![1, 5, 2, 3], vec![5, 3],
        vec![1, 5, 0], vec![1, 5, 2, 3], vec![1, 5, 0, 3], vec![1, 5, 2, 3], vec![5, 0, 3],
        vec![1, 5, 0], vec![1, 5, 2, 3], vec![1, 5, 0, 3], vec![1, 5, 2, 3], vec![5, 0, 3],
        vec![1, 5, 0], vec![1, 5, 2, 3], vec![1, 5, 0, 3], vec![1, 5, 2, 3], vec![5, 0, 3],
        vec![1, 5, 0], vec![1, 3], vec![1, 5, 0], vec![1, 5, 0], vec![5, 0],
    ]);

    // 下に行きたい時
    static ref OP_CANDIDATE0_2: Map2d<Vec<usize>> = Map2d::new([
        vec![1, 5], vec![2, 5, 1, 3], vec![1, 5, 3], vec![2, 5, 1, 3], vec![3, 5],
        vec![1, 5, 0], vec![2, 5, 1, 3], vec![1, 5, 3, 0], vec![2, 5, 1, 3], vec![3, 0, 5],
        vec![1, 5, 0], vec![2, 5, 1, 3], vec![1, 5, 3, 0], vec![2, 5, 1, 3], vec![3, 0, 5],
        vec![1, 5, 0], vec![2, 5, 1, 3], vec![3, 1, 5, 0], vec![3, 2, 5, 1], vec![3, 0, 5],
        vec![1, 5, 0], vec![1, 3, 5], vec![1, 0, 5], vec![1, 0, 5], vec![0, 5],
    ]);

    // 左に行きたい時
    static ref OP_CANDIDATE0_3: Map2d<Vec<usize>> = Map2d::new([
        vec![5, 1], vec![3, 5, 2, 1], vec![3, 1], vec![3, 5, 2, 1], vec![3, 5],
        vec![0, 5, 1], vec![3, 5, 2, 1], vec![3, 5, 0, 1], vec![3, 5, 2, 1], vec![3, 5, 0],
        vec![0, 5, 1], vec![3, 5, 2, 1], vec![3, 5, 0, 1], vec![3, 5, 2, 1], vec![3, 5, 0],
        vec![0, 5, 1], vec![3, 5, 2, 1], vec![3, 5, 0, 1], vec![3, 5, 2, 1], vec![3, 5, 0],
        vec![0, 5, 1], vec![3, 5, 1], vec![0, 5, 1], vec![0, 5, 1], vec![0, 5],
    ]);

    static ref OP_CANDIDATE1: Map2d<Vec<usize>> = Map2d::new([
        vec![1, 5], vec![2, 5], vec![3, 5], vec![1, 5], vec![5],
        vec![1, 5], vec![2, 5], vec![3, 5], vec![1, 5], vec![0, 5],
        vec![1, 5], vec![2, 5], vec![3, 5], vec![1, 5], vec![0, 5],
        vec![1, 5], vec![2, 5], vec![2, 5], vec![2, 5], vec![0, 5],
        vec![1, 5], vec![1, 5], vec![1, 5], vec![1, 5], vec![0, 5],
    ]);
}

#[derive(Debug, Clone, Hash)]
struct Crane{
    is_big: bool,
    p: Coordinate,
    task: Task,
    ops: Vec<usize>, // 0: U, 1: R, 2: D, 3: L, 4: P or Q, 5: .
    has_item: bool,
}

impl Crane{
    fn new(is_big: bool, p: Coordinate) -> Self{
        Self {is_big, p, task: Task{target_item: 0, target_item_p: p, target_place_p: p, status: 3}, ops: vec![], has_item: false}
    }

    fn assign_task(&mut self, task: Task){
        self.task = task;
    }

    fn get_op_candidate(&self, M: &Map2d<usize>) -> Vec<usize>{
        let mut ret = vec![];
        if self.task.status==0{
            // コンテナに移動
            if self.p==self.task.target_item_p{
                if M[self.p]==self.task.target_item{
                    ret = OP_CANDIDATE0_PICK[self.p].clone();
                } else {
                    ret = OP_CANDIDATE0[self.p].clone();
                }
            } else if self.p.row > self.task.target_item_p.row{
                ret = OP_CANDIDATE0_0[self.p].clone();
            } else if self.p.col < self.task.target_item_p.col{
                ret = OP_CANDIDATE0_1[self.p].clone();
            } else if self.p.row < self.task.target_item_p.row{
                ret = OP_CANDIDATE0_2[self.p].clone();
            } else if self.p.col > self.task.target_item_p.col{
                ret = OP_CANDIDATE0_3[self.p].clone();
            } else{
                panic!();
                ret = vec![];
            }
        } else if self.task.status<3 {
            // コンテナを輸送
            if self.p == self.task.target_place_p{
                ret = OP_CANDIDATE0_PICK[self.p].clone();
            } else if self.p.row==self.task.target_place_p.row{
                if self.p.col==1 && self.task.target_place_p.col==2{
                    ret = vec![1, 5];
                } else if self.p.col==4 && self.task.target_place_p.col==3{
                    ret = vec![3, 5]
                } else {
                    ret = OP_CANDIDATE1[self.p].clone();
                }
            } else if self.p.row==4 && self.task.target_place_p.row == 3 && self.p.col==self.task.target_place_p.col{
                ret = vec![0, 5];
            } else {
                ret = OP_CANDIDATE1[self.p].clone();
            }
        } else {
            // 何もしていない
            ret = OP_CANDIDATE0_STAY[self.p].clone();
        }
        ret
    }

    fn operate(&mut self, op: usize) -> (Option<Coordinate>, Option<(Coordinate, usize)>){
        self.ops.push(op);
        let mut ret = if op<4 && self.task.status==1{
            self.task.status = 2;
            (Some(self.p), None)
        } else {
            (None, None)
        };
        match op{
            0 => self.p.row -= 1,
            1 => self.p.col += 1,
            2 => self.p.row += 1,
            3 => self.p.col -= 1,
            4 => {
                if self.has_item{
                    self.has_item = false;
                    self.task.status = 3;
                    ret = (None, Some((self.p, self.task.target_item)));
                } else {
                    self.has_item = true;
                    self.task.status = 1;
                }
            },
            _ => {},
        }
        ret
    }
}

struct ConflictSolver{
    is_ok: Vec<bool>, // pidx1, pidx2, op1, op2 -> ok?
}

impl ConflictSolver{
    fn new() -> Self{
        let mut is_ok = vec![];
        for pidx1 in 0..SIZE*SIZE{
            for pidx2 in 0..SIZE*SIZE{
                for op1 in 0..6{
                    for op2 in 0..6{
                        let mut flg = true;
                        let mut c1 = Coordinate::new(pidx1/SIZE*2, pidx1%SIZE*2);
                        let mut c2 = Coordinate::new(pidx2/SIZE*2, pidx2%SIZE*2);
                        let cd1 = match op1{
                            0 => CoordinateDiff::new(!0, 0),
                            1 => CoordinateDiff::new(0, 1),
                            2 => CoordinateDiff::new(1, 0),
                            3 => CoordinateDiff::new(0, !0),
                            _ => CoordinateDiff::new(0, 0),
                        };
                        let cd2 = match op2{
                            0 => CoordinateDiff::new(!0, 0),
                            1 => CoordinateDiff::new(0, 1),
                            2 => CoordinateDiff::new(1, 0),
                            3 => CoordinateDiff::new(0, !0),
                            _ => CoordinateDiff::new(0, 0),
                        };
                        if c1+cd1==c2+cd2 || c1+cd1+cd1==c2+cd2+cd2{
                            is_ok.push(false);
                        } else {
                            is_ok.push(true);
                        }
                    }
                }
            }
        }
        Self {is_ok}
    }

    fn check(&self, pidx1: usize, pidx2: usize, op1: usize, op2: usize) -> bool{
        self.is_ok[pidx1*SIZE*SIZE*6*6+pidx2*6*6+op1*6+op2]
    }

    fn find_ok_pair(&self, op_candidates: Vec<Vec<usize>>, cranes: &Vec<Crane>) -> Vec<usize>{
        let mut order = vec![];
        // TODO:
        for i in 0..cranes.len(){
            order.push(i);
        }
        let mut idx = vec![0; cranes.len()];
        loop{
            let mut flg = true;
            let mut appeared = vec![];
            for &i in order.iter(){
                let pidx = cranes[i].p.to_index();
                let op = op_candidates[i][idx[i]];
                for &(pidx_, op_) in appeared.iter(){
                    let flg2: bool = self.is_ok[pidx*SIZE*SIZE*6*6+pidx_*6*6+op*6+op_];
                    if !flg2{
                        idx[i] += 1;
                        flg = false;
                        break;
                    }
                }
                if !flg{
                    break;
                }
                appeared.push((pidx, op));
            }
            if flg{
                let mut ret = vec![];
                for i in 0..cranes.len(){
                    ret.push(op_candidates[i][idx[i]]);
                }
                return ret;
            } else {
                for i_ in (0..order.len()).rev(){
                    if idx[order[i_]]>=op_candidates[order[i_]].len(){
                        if i_==0{
                            return vec![5; cranes.len()];
                        }
                        idx[order[i_]] = 0;
                        idx[order[i_-1]] += 1;
                    }
                }
            }
        }
    }

}


struct Neighbor{
    i: usize,
    j: usize,
    score: Option<f64>,
}

#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct State{
    order: Vec<usize>,
    score: Option<f64>,
}

impl State{
    fn init_state(input: &Input) -> Self{
        let mut order = vec![0;SIZE*SIZE];
        let mut order_len = 100;
        let mut v2idx = vec![(0, 0); SIZE*SIZE];
        let mut rng = rand::thread_rng();
        for i in 0..SIZE{
            for j in 0..SIZE{
                v2idx[input.A[i][SIZE-j-1]] = (i, j);
            }
        }
        for i in 0..SIZE{
            let mut tmp_order = vec![];
            let mut cur = [0, 0, 0, 0, 0];
            for j in 0..SIZE{
                let (r, c) = v2idx[i*SIZE+j];
                while cur[r]<c{
                    tmp_order.push(r);
                    cur[r] += 1;
                }
                if cur[r]==c{
                    cur[r]+=1;
                }
            }
            if order_len>tmp_order.len(){
                order_len = tmp_order.len();
                order = tmp_order;
                let mut rest_order = vec![];
                for i in 0..SIZE{
                    while cur[i]<5{
                        rest_order.push(i);
                        cur[i] += 1;
                    }
                }
                rest_order.shuffle(&mut rng);
                order.append(&mut rest_order);
            }
        }
        order.reverse();
        Self {order, score: None}
    }

    fn update(&mut self, params: &Neighbor){
        self.order.swap(params.i, params.j);
        self.score = None;
    }

    fn undo(&mut self, params: &Neighbor){
        self.order.swap(params.i, params.j);
        self.score = params.score;
    }

    fn get_neighbor(&mut self, input: &Input) -> Neighbor{
        let mut rng = rand::thread_rng();
        // let mode_flg = rng.gen::<usize>()%100;
        let mut i: usize;
        let mut j: usize;
        loop{
            i = rng.gen::<usize>()%self.order.len();
            j = rng.gen::<usize>()%self.order.len();
            if self.order[i]!=self.order[j]{
                break;
            }
        }
        Neighbor{i, j, score: self.score}
    }

    fn simulate(&self, input: &Input, conflictsolver: &ConflictSolver) -> Vec<Crane>{
        let mut task_assigner = TaskAssigner::new(input, self.order.clone());
        let mut M = Map2d::new([usize::MAX; SIZE*SIZE]);
        let mut A = input.A.clone();
        let mut next_out = vec![];
        let mut spaces = vec![];
        let mut place_tasks = vec![];
        let mut out_tasks = vec![];
        for i in self.order.iter(){
            place_tasks.push(*i);
        }
        for i in 0..SIZE-1{
            spaces.push(Coordinate::new(i, 2));
            spaces.push(Coordinate::new(i, 3));
        }
        for i in 0..SIZE{
            M[i][0] = A[i].pop().unwrap();
            next_out.push(i*5);
            if M[i][0]%SIZE==0{
                out_tasks.push(Coordinate::new(i, 0));
            }
        }
        let mut cranes = vec![
            Crane::new(true, Coordinate::new(0, 0)),
            Crane::new(false, Coordinate::new(1, 0)),
            Crane::new(false, Coordinate::new(2, 0)),
            Crane::new(false, Coordinate::new(3, 0)),
            Crane::new(false, Coordinate::new(4, 0)),
        ];
        for turn in 0..200{
            // クレーンにタスクを割り当て
            let mut cant_use = vec![];
            for crane in cranes.iter(){
                if crane.task.status<2 && crane.task.target_item_p.col==0{
                    cant_use.push(crane.task.target_item_p.row);
                }
            }
            for crane in cranes.iter_mut(){
                if crane.task.status==3{
                    if let Some(c) = out_tasks.pop(){
                        // out_task優先
                        let target_item = M[c.row][c.col];
                        let target_place_p = Coordinate::new(target_item/SIZE, 4);
                        crane.assign_task(Task{target_item, target_item_p: c, target_place_p, status: 0});
                    } else {
                        // place_task
                        let mut reinserted = vec![];
                        while let Some(i) = place_tasks.pop(){
                            let target_item = M[i][0];
                            if target_item==usize::MAX{
                                continue;
                            }
                            if cant_use.contains(&i){
                                reinserted.push(i);
                                continue;
                            }
                            let target_item_p = Coordinate::new(i, 0);
                            let mut found = false;
                            for c in spaces.clone(){
                                if (c.row==3) || (c.col==2 && i<=c.row) || (c.col==3 && target_item/SIZE<=c.row){
                                    crane.assign_task(Task{target_item, target_item_p, target_place_p: c, status: 0});
                                    cant_use.push(i);
                                    spaces.retain(|&x| x!=c);
                                    found = true;
                                    break;
                                }
                            }
                            if found{
                                break;
                            } else {
                                reinserted.push(i);
                                break;
                            }
                        }
                        reinserted.reverse();
                        place_tasks.append(&mut reinserted);
                    }
                }
            }
            // eprintln!("cranes{:?}", cranes);
            // 次のムーブを取得
            let mut next_op_candidate = vec![];
            for crane in cranes.iter(){
                next_op_candidate.push(crane.get_op_candidate(&M));
            }
            // eprintln!("cands:{:?}", next_op_candidate);
            // 次のムーブを決定(コンフリクト解決)
            let next_ops = conflictsolver.find_ok_pair(next_op_candidate, &cranes);
            // ムーブして、シミュレーションのステートを更新
            for i in 0..SIZE{
                let (c_moved, placed_info) = cranes[i].operate(next_ops[i]);
                if let Some(c_moved) = c_moved{
                    if c_moved.col==0{
                        // 次のコンテナが出てくる
                        if let Some(v) = A[c_moved.row].pop(){
                            M[c_moved.row][c_moved.col] = v;
                            for j in 0..SIZE{
                                if v==next_out[j]{
                                    out_tasks.push(c_moved);
                                }
                            }
                        } else {
                            M[i][c_moved.col] = usize::MAX;
                        }
                    } else {
                        M[i][c_moved.col] = usize::MAX;
                        spaces.push(c_moved);
                    }
                }
                if let Some((placed_p, item)) = placed_info{
                    if placed_p.col==4{
                        let mut new_next_out_v = item+1;
                        if new_next_out_v%SIZE==0{
                            new_next_out_v = usize::MAX;
                        }
                        next_out[placed_p.row] = new_next_out_v;
                        if new_next_out_v!=usize::MAX{
                            for j in 0..SIZE-1{
                                if M[0][j]==new_next_out_v{
                                    out_tasks.push(Coordinate::new(0, j));
                                }
                                if M[2][j]==new_next_out_v{
                                    out_tasks.push(Coordinate::new(2, j));
                                }
                                if M[3][j]==new_next_out_v{
                                    out_tasks.push(Coordinate::new(3, j));
                                }
                            }
                            if M[0][SIZE-1]==new_next_out_v{
                                out_tasks.push(Coordinate::new(0, SIZE-1));
                            }
                        }
                    } else {
                        M[placed_p.row][placed_p.col] = item;
                        for j in 0..SIZE{
                            if item==next_out[j]{
                                out_tasks.push(placed_p);
                            }
                        }
                    }
                }
            }
            // 全部終わってたら終了
            if *next_out.iter().min().unwrap() == usize::MAX{
                break;
            }
        }
        cranes
    }

    fn cal_score(&mut self, input: &Input, conflictsolver: &ConflictSolver) -> f64{
        if let Some(score) = self.score{
            return score;
        }
        let cranes = self.simulate(input, conflictsolver);
        let score = cranes[0].ops.len() as f64;
        self.score = Some(score);
        score
    }

    fn print(&mut self, input: &Input, conflictsolver: &ConflictSolver){
        let cranes = self.simulate(input, conflictsolver);
        // eprintln!("cranes{:?}", cranes);
        for crane in cranes{
            let mut ops_chars = vec![];
            let mut flg = false;
            for &op in crane.ops.iter(){
                match op{
                    0 => ops_chars.push('U'),
                    1 => ops_chars.push('R'),
                    2 => ops_chars.push('D'),
                    3 => ops_chars.push('L'),
                    4 => {
                        if flg{
                            ops_chars.push('Q');
                        } else {
                            ops_chars.push('P');
                        }
                        flg = !flg;
                    },
                    _ => ops_chars.push('.'),
                }
            }
            let s: String = ops_chars.into_iter().collect();
            println!("{}", s);
        }
    }
}

struct TaskAssigner{
    placed_tasks: Vec<Task>,
    container_state: Vec<(Coordinate, usize)>,
    out_tasks: Vec<Task>,
}

impl TaskAssigner{
    fn new(input: &Input, placed_order: Vec<usize>) -> Self{
        let mut placed_tasks = vec![];
        let mut container_state = vec![];
        let mut out_tasks = vec![];
        Self {placed_tasks, container_state, out_tasks}
    }

    fn update(&mut self, infos: Vec<(Option<usize>, (Option<(usize, usize)>))>){
    }

    fn task_assign(&mut self, cranes: &mut Vec<Crane>){
        let mut ret = vec![];
        for crane in cranes.iter(){
            if crane.task.status==3{
                if let Some(t) = self.placed_tasks.pop(){
                    ret.push(Some(t));
                }
            }
        }
        ret
    }
}


fn simanneal(input: &Input, init_state: State, timer: &Instant, limit: f64, conflictsolver: &ConflictSolver) -> State{
    let mut state = init_state.clone();
    let mut best_state = init_state.clone();
    let mut best_score = best_state.cal_score(input, &conflictsolver);
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
        state.update(&neighbor);
        let new_score = state.cal_score(input, conflictsolver);
        let score_diff = new_score-cur_score;
        // start_temp = (start_temp*(all_iter-1)+score_diff)/all_iter;
        temp = start_temp + (end_temp-start_temp)*elasped_time/limit;
        if score_diff>=0.0 || rng.gen_bool((score_diff as f64/temp).exp()){
            accepted_cnt += 1;
            //eprintln!("{} {} {:?}", cur_score, new_score, turns);
            cur_score = new_score;
            last_updated = all_iter;
            //state.print(input);
            if new_score>best_score{
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
    eprintln!("score  : {}", best_state.cal_score(input, conflictsolver));
    eprintln!("");
    best_state
}

fn main() {
    let timer = Instant::now();
    let input = Input::read_input();
    solve(&input, &timer, 2.8);
}

fn solve(input: &Input, timer:&Instant, tl: f64){
    let conflictsolver = ConflictSolver::new();
    let mut init_state = State::init_state(input);
    let mut best_state = simanneal(input, init_state, timer, tl, &conflictsolver);
    best_state.print(input, &conflictsolver);
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
        map: [T; SIZE*SIZE],
    }

    impl<T> Map2d<T> {
        pub fn new(map: [T; SIZE*SIZE]) -> Self {
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
