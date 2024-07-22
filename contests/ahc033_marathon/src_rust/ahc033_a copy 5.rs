#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused)]
use core::task;
use std::collections::{HashSet, HashMap};
use std::f32::consts::E;
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
use std::cell::{RefCell, UnsafeCell};
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

const SPACEIDX2COORDINATE: [Coordinate; 8] = [
    Coordinate::new(0, 2),
    Coordinate::new(0, 3),
    Coordinate::new(1, 2),
    Coordinate::new(1, 3),
    Coordinate::new(2, 2),
    Coordinate::new(2, 3),
    Coordinate::new(3, 2),
    Coordinate::new(3, 3),
];

const COORDINATE2SPACEIDX: [usize; SIZE*SIZE] = [
    usize::MAX, usize::MAX, 0, 1, usize::MAX,
    usize::MAX, usize::MAX, 2, 3, usize::MAX,
    usize::MAX, usize::MAX, 4, 5, usize::MAX,
    usize::MAX, usize::MAX, 6, 7, usize::MAX,
    usize::MAX, usize::MAX, 0, 1, usize::MAX,
];


lazy_static!{
    // タスクの進捗
    static ref PROGRESS_MAP: Map2d<usize> = Map2d::new([
        12, 11, 12, 1, 0,
        11, 10, 11, 2, 1,
        10, 9, 10, 3, 2,
        9, 8, 7, 6, 3,
        8, 7, 6, 5, 4,
    ]);

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
        vec![1, 5], vec![3, 5, 1, 2], vec![3, 5, 1], vec![3, 5, 1, 2], vec![3, 5],
        vec![0, 5, 1], vec![3, 5, 1, 2], vec![3, 5, 0, 1], vec![3, 5, 1, 2], vec![3, 5, 0],
        vec![0, 5, 1], vec![3, 5, 1, 2], vec![3, 5, 0, 1], vec![3, 5, 1, 2], vec![3, 5, 0],
        vec![0, 5, 1], vec![3, 5, 1, 2], vec![3, 5, 0, 1], vec![3, 5, 1, 2], vec![3, 5, 0],
        vec![0, 5, 1], vec![1, 5, 3], vec![1, 5, 0], vec![1, 5, 0], vec![0, 5],
    ]);

    // タスク何もない時の移動
    static ref OP_CANDIDATE0_STAY2: Map2d<Vec<usize>> = Map2d::new([
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
        let mut priority = vec![0; cranes.len()];
        for i in 0..cranes.len(){
            if op_candidates[i][0]==4{
                priority[i] += 10000;
            }
            if cranes[i].has_item{
                priority[i] += 1000;
            }
            if let Some(ref task) = *cranes[i].task.borrow(){
                if task.status<3{
                    priority[i] += 100;
                }
            }
            if (cranes[i].p.col==1 || cranes[i].p.col==4 || (cranes[i].p.row==4&&(cranes[i].p.col==2 || cranes[i].p.col==3))){
                priority[i] += 10;
            }
        }
        let order = (0..cranes.len()).sorted_by_key(|&i| (Reverse(priority[i]), i)).collect::<Vec<usize>>();
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
    mode: usize,
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
                let mut cnts = vec![0; SIZE];
                for &o in order.iter(){
                    cnts[o] += 1;
                }
                let mut rest_order = vec![];
                for i in 0..SIZE{
                    while cnts[i]<SIZE-1{
                        rest_order.push(i);
                        cnts[i] += 1;
                    }
                }
                rest_order.shuffle(&mut rng);
                order.append(&mut rest_order);
            }
        }
        Self {order, score: None}
    }

    fn update(&mut self, params: &Neighbor){
        if params.mode==0{
            self.order.swap(params.i, params.j);
        } else {
            let v = self.order.remove(params.i);
            self.order.insert(params.j, v);
        }
        self.score = None;
    }

    fn undo(&mut self, params: &Neighbor){
        if params.mode==0{
            self.order.swap(params.i, params.j);
        } else {
            let v = self.order.remove(params.i);
            self.order.insert(params.j, v);
        }
        self.score = params.score;
    }

    fn get_neighbor(&mut self, input: &Input) -> Neighbor{
        let mut rng = rand::thread_rng();
        let mode_flg = rng.gen::<usize>()%100;
        if mode_flg<50{
            let mut i: usize;
            let mut j: usize;
            loop{
                i = rng.gen::<usize>()%self.order.len();
                j = rng.gen::<usize>()%self.order.len();
                if self.order[i]!=self.order[j]{
                    break;
                }
            }
            Neighbor{i, j, score: self.score, mode: 0}
        } else {
            let mut i: usize;
            let mut j: usize;
            loop{
                i = rng.gen::<usize>()%self.order.len();
                j = rng.gen::<usize>()%self.order.len();
                if i!=j{//self.order[i]!=self.order[j]{
                    break;
                }
            }
            Neighbor{i, j, score: self.score, mode: 1}
        }
    }

    fn simulate(&self, input: &Input, conflictsolver: &ConflictSolver) -> (Vec<Crane>, usize, usize){
        let mut task_assigner = TaskAssigner::new(input, &self.order);
        let mut cranes = vec![
            Crane::new(true, Coordinate::new(0, 0)),
            Crane::new(false, Coordinate::new(1, 0)),
            Crane::new(false, Coordinate::new(2, 0)),
            Crane::new(false, Coordinate::new(3, 0)),
            Crane::new(false, Coordinate::new(4, 0)),
        ];
        for turn in 0..200{
            // if turn ==100{
            //     eprintln!("t:{}", turn);
            // }
            // クレーンにタスクを割り当て
            task_assigner.task_assign(&mut cranes);
            // eprintln!("cranes{:?}", cranes);
            // 次のムーブを取得
            let mut next_op_candidate = vec![];
            for crane in cranes.iter(){
                next_op_candidate.push(crane.get_op_candidate(&task_assigner.M));
            }
            // eprintln!("cands:{:?}", next_op_candidate);
            // 次のムーブを決定(コンフリクト解決)
            let next_ops = conflictsolver.find_ok_pair(next_op_candidate, &cranes);
            // ムーブして、シミュレーションのステートを更新
            let mut update_infos = vec![];
            for i in 0..cranes.len(){
                let (c_moved, placed_info) = cranes[i].operate(next_ops[i]);
                update_infos.push((c_moved, placed_info));
            }
            task_assigner.update(update_infos);
            // 全部終わってたら終了
            if task_assigner.is_finished(){
                break;
            }
        }
        (cranes, task_assigner.out_cnt, task_assigner.task_queue.len())
    }

    fn cal_score(&mut self, input: &Input, conflictsolver: &ConflictSolver) -> f64{
        if let Some(score) = self.score{
            return score;
        }
        let (cranes, out_cnt, rest_place) = self.simulate(input, conflictsolver);
        let score = cranes[0].ops.len() as f64 + (SIZE*SIZE-out_cnt) as f64 * 1000.0 + rest_place as f64;
        self.score = Some(score);
        score
    }

    fn print(&mut self, input: &Input, conflictsolver: &ConflictSolver){
        let (cranes, out_cnt, rest_place) = self.simulate(input, conflictsolver);
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


#[derive(Debug, Clone)]
struct Crane{
    is_big: bool,
    p: Coordinate,
    task: Option<Rc<RefCell<Option<Task>>>,
    ops: Vec<usize>, // 0: U, 1: R, 2: D, 3: L, 4: P or Q, 5: .
    has_item: bool,
}

impl Crane{
    fn new(is_big: bool, p: Coordinate) -> Self{
        Self {is_big, p, task: Rc::new(RefCell::new(None)), ops: vec![], has_item: false}
    }

    fn assign_task(&mut self, task: Rc<RefCell<Option<Task>>>){
        self.task = task;
        if let Some(ref mut task) = *self.task.borrow_mut(){
            assert!(task.is_assigned==false);
            task.is_assigned = true;
        }
    }

    fn get_op_candidate(&self, M: &Map2d<usize>) -> Vec<usize>{
        let mut ret = vec![];
        if let Some(ref task) = *self.task.borrow(){
            if task.status==0{
                // コンテナに移動
                if self.p==task.target_item_p{
                    if task.is_ready(M){
                        ret = OP_CANDIDATE0_PICK[self.p].clone();
                    } else {
                        ret = OP_CANDIDATE0[self.p].clone();
                    }
                } else if self.p.row > task.target_item_p.row{
                    ret = OP_CANDIDATE0_0[self.p].clone();
                } else if self.p.col < task.target_item_p.col{
                    ret = OP_CANDIDATE0_1[self.p].clone();
                } else if self.p.row < task.target_item_p.row{
                    ret = OP_CANDIDATE0_2[self.p].clone();
                } else if self.p.col > task.target_item_p.col{
                    ret = OP_CANDIDATE0_3[self.p].clone();
                } else{
                    panic!();
                    ret = vec![];
                }
            } else if task.status<3 {
                // コンテナを輸送
                if self.p == task.target_place_p{
                    ret = OP_CANDIDATE0_PICK[self.p].clone();
                } else if self.p.row==task.target_place_p.row{
                    if self.p.col==1 && task.target_place_p.col==2{
                        ret = vec![1, 5];
                    } else if self.p.col==4 && task.target_place_p.col==3{
                        ret = vec![3, 5]
                    } else {
                        ret = OP_CANDIDATE1[self.p].clone();
                    }
                } else if self.p.row==4 && task.target_place_p.row == 3 && self.p.col==task.target_place_p.col{
                    ret = vec![0, 5];
                } else {
                    ret = OP_CANDIDATE1[self.p].clone();
                }
            } else {
                // 何もしていない
                ret = OP_CANDIDATE0_STAY[self.p].clone();
            }
        } else {
            // 何もしていない
            ret = OP_CANDIDATE0_STAY[self.p].clone();
        }
        ret
    }

    fn operate(&mut self, op: usize) -> (Option<Coordinate>, Option<(Coordinate, usize)>){
        let mut ret = (None, None);
        self.ops.push(op);
        if let Some(ref mut task) = *self.task.borrow_mut(){
            if op<4 && task.status==1{
                task.status = 2;
                ret = (Some(self.p), None);
            }
            match op{
                0 => self.p.row -= 1,
                1 => self.p.col += 1,
                2 => self.p.row += 1,
                3 => self.p.col -= 1,
                4 => {
                    if self.has_item{
                        self.has_item = false;
                        task.status = 3;
                        task.is_assigned = false;
                        ret = (None, Some((self.p, task.target_item)));
                    } else {
                        self.has_item = true;
                        task.status = 1;
                    }
                },
                _ => {},
            }
            task.set_progress(self.p);
        } else {
            match op{
                0 => self.p.row -= 1,
                1 => self.p.col += 1,
                2 => self.p.row += 1,
                3 => self.p.col -= 1,
                4 => {panic!()},
                _ => {},
            }
        }
        let mut flg = false;
        if let Some(ref task) = *self.task.borrow(){
            flg = task.status==3;
        }
        if flg{
            let new_task = Rc::new(RefCell::new(None));
            self.task = new_task;
        }
        ret
    }
}


#[derive(Debug, Clone)]
struct Task{
    target_item: usize,
    target_item_p: Coordinate,
    target_place_p: Coordinate,
    status: usize, // 0: pickしに移動中、1: pickした直後, 2: pickして移動中, 3: 終了
    progress: usize, // progress_map + 1000*(is_not_picked)
    is_assigned: bool,
    previous_out_task: Rc<RefCell<Option<Task>>>,
}

impl Task{
    fn is_out_task(&self) -> bool{
        self.target_place_p.col == SIZE-1
    }

    fn set_progress(&mut self, p: Coordinate){
        if self.status == 0{
            self.progress = PROGRESS_MAP[p] + 1000;
        } else {
            self.progress = PROGRESS_MAP[p];
        }
    }

    fn set_as_out_task(&mut self){
        self.target_place_p = Coordinate::new(self.target_item/SIZE, SIZE-1);
    }

    fn set_as_placed_task(&mut self, target_place_p: Coordinate){
        self.target_place_p = target_place_p;
    }

    fn should_done_by_big(&self) -> bool{
        self.is_out_task() && PROGRESS_MAP[self.target_item_p] >= PROGRESS_MAP[self.target_place_p]+1
    }

    fn is_ready(&self, M: &Map2d<usize>) -> bool{
        // 置き場所決めた?
        // space空いている?
        // itemは出現済み?
        if !(self.target_place_p.in_map() && (self.is_out_task() || M[self.target_place_p]==usize::MAX) && M[self.target_item_p] != self.target_item){
            false
        } else if self.is_out_task(){
            // out_taskの場合は直前のout_taskは自身よりも進んでいる?
            if let Some(ref previous_out_task) = self.previous_out_task.borrow().as_ref(){
                previous_out_task.is_out_task() && previous_out_task.progress <= (self.progress % 1000)
            } else {
                true
            }
        } else {
            true
        }
    }

    fn is_ready_for_out(&self) -> bool{
        // 直前のout_taskは終わってる?
        if let Some(ref previous_out_task) = self.previous_out_task.borrow().as_ref(){
            previous_out_task.is_out_task() && previous_out_task.progress <= (self.progress % 1000)
        } else {
            true
        }
    }
}


struct ContainerState{
    p: Coordinate,
    state: usize, // 0: 在庫にある, 1: 搬入口, 2: クレーンが持っている, 3: 置かれている, 4: 搬出済み
    is_out_ok: bool,
    is_assigned: bool,
    is_tmporary_assigned: bool,
}

struct TaskAssigner{
    tasks: Vec<Rc<RefCell<Option<Task>>>>,
    task_queue: Vec<usize>, // 優先度順に並べたtaskのidx
    out_task_queue: Vec<usize>, // 出庫するtaskのidx
    place_assign_task_queue: Vec<usize>, // とりあえずアサインしたplace_taskで、spaceのアサインまちなもの
    M: Map2d<usize>, // Map管理
    item_on_M: HashSet<usize>, // フィールドにあるアイテム
    A: Vec<Vec<usize>>, // 在庫
    out_cnt: usize, // 出庫済みの数
    is_space_available: [bool; 8], // spaceが使えるかどうか
}

impl TaskAssigner{
    fn new(input: &Input, place_order: &Vec<usize>) -> Self{
        let mut tasks = vec![];
        let mut val_row = vec![];
        for i in 0..SIZE{
            for &v in input.A[i].iter(){
                val_row.push((v, i));
            }
        }
        val_row.sort();
        for (v, i) in val_row{
            let target_item = v;
            let target_item_p = Coordinate::new(i, 0);
            let target_place_p = Coordinate::new(usize::MAX, usize::MAX);
            let status = 0;
            let progress = PROGRESS_MAP[target_item_p] + 100;
            let is_assigned = false;
            let previous_out_task = Rc::new(RefCell::new(None));
            tasks.push(Rc::new(RefCell::new(Some(Task{target_item, target_item_p, target_place_p, status, progress, is_assigned, previous_out_task}))));
        }
        for i in 0..SIZE{
            for j in 1..SIZE{
                let mut borrowed_task = tasks[i*SIZE+j].borrow_mut();
                if let Some(ref mut task) = *borrowed_task{
                    task.previous_out_task = Rc::clone(&tasks[i*SIZE+j-1]);
                }
            }
        }
        let mut task_queue = vec![];
        let mut M = Map2d::new([usize::MAX; SIZE*SIZE]);
        let mut item_on_M = HashSet::new();
        let mut A = input.A.clone();
        // マップ初期化
        for i in 0..SIZE{
            let v = A[i].pop().unwrap();
            M[i][0] = v;
            item_on_M.insert(v);
        }
        // place_tasks初期化
        let mut A_tmp = input.A.clone();
        for &i in place_order.iter(){
            let v = A_tmp[i].pop().unwrap();
            task_queue.push(v);
        }
        task_queue.reverse();
        Self {tasks, task_queue, out_task_queue: vec![], place_assign_task_queue: vec![], M, item_on_M, A, out_cnt: 0, is_space_available: [true; 8]}
    }

    fn is_finished(&self) -> bool{
        self.out_cnt == SIZE*SIZE
    }

    fn update(&mut self, update_infos: Vec<(Option<Coordinate>, (Option<(Coordinate, usize)>))>){
        for (picked_info, placed_info) in update_infos{
            if let Some(picked) = picked_info{
                let v = self.M[picked];
                if picked.col==0{
                    // 搬入口を開けた
                    // 次のアイテムを出す
                    if let Some(v_next) = self.A[picked.row].pop(){
                        self.M[picked] = v_next;
                        self.item_on_M.insert(v_next);
                    } else {
                        self.M[picked] = usize::MAX;
                    }
                    // out_tasksに入れれたら入れる
                } else {
                    // spaceを開けた
                    self.M[picked] = usize::MAX;
                    self.is_space_available[COORDINATE2SPACEIDX[picked.to_index()]] = true;
                }
            }
            if let Some((placed, v)) = placed_info{
                if placed.col==SIZE-1{
                    // 搬出した
                    self.out_cnt += 1;
                    // マップ上のアイテムを消す
                    self.item_on_M.remove(&v);
                } else {
                    // スペースに置いた
                    self.M[placed] = v;
                    self.is_space_available[COORDINATE2SPACEIDX[placed.to_index()]] = false;
                }
            }
        }
    }

    fn refresh_tasks(&mut self){
        // taskがout_taskにできないかチェック
        // out_task_queueの更新
        for &v in self.item_on_M.iter(){
            let mut borrowed_task = self.tasks[v].borrow_mut();
            if let Some(ref mut task) = *borrowed_task{
                if !task.is_out_task() && task.is_ready_for_out(){
                    if task.is_assigned{
                        if task.target_place_p.in_map(){
                            self.is_space_available[COORDINATE2SPACEIDX[task.target_place_p.to_index()]] = true;
                        }
                        task.set_as_out_task();
                    } else {
                        self.out_task_queue.push(v);
                        task.set_as_out_task();
                    }
                }
            }
        }
    }

    // #[derive(Debug, Clone)]
    // struct Task{
    //     target_item: usize,
    //     target_item_p: Coordinate,
    //     target_place_p: Coordinate,
    //     status: usize, // 0: pickしに移動中、1: pickした直後, 2: pickして移動中, 3: 終了
    //     progress: usize, // progress_map + 1000*(is_not_picked)
    //     is_assigned: bool,
    //     previous_out_task: Rc<RefCell<Option<Task>>>,
    // }

    fn task_assign(&mut self, cranes: &mut Vec<Crane>){
        // タスクの見直し
        self.refresh_tasks();
        // タスクがないクレーンをとってくる & 仮アサインでspace割り当てまちであればspace割り当てを試みる
        let mut no_task_crane_idxs = vec![];
        let mut temporary_assigned_crane_idxs = vec![];
        for i in 0..cranes.len(){
            if let Some(ref mut task) = *cranes[i].task.borrow_mut(){
                if !task.target_place_p.in_map(){
                    let mut space_idx = usize::MAX;
                    for i in 0..8{
                        if self.is_space_available[i]{
                            let c = SPACEIDX2COORDINATE[i];
                            if (c.row==3) || (c.col==2 && c.row>=task.target_item_p.row) || (c.col==3 && c.row>=task.target_item/SIZE){
                                space_idx = i;
                                break;
                            }
                        }
                    }
                    if space_idx==usize::MAX{
                        task.set_as_placed_task(SPACEIDX2COORDINATE[space_idx]);
                    } else {
                        temporary_assigned_crane_idxs.push(i);
                    }
                }
            } else {
                no_task_crane_idxs.push(i);
            }
        }
        // タスクがないクレーンにタスクを割り当てる
        while !no_task_crane_idxs.is_empty(){
            if let Some(item_v) = self.out_task_queue.pop(){
                let mut assign_info = None;
                if let Some(ref mut task) = *self.tasks[item_v].borrow_mut(){
                    // out_taskがあれば先に割り当て
                    // 一番近いcraneを探す
                    let mut min_dist = usize::MAX;
                    let mut min_dist_idx = 0;
                    for i in no_task_crane_idxs.iter(){
                        let dist = cranes[*i].p.dist(&task.target_item_p);
                        if dist<min_dist{
                            min_dist = dist;
                            min_dist_idx = *i;
                        }
                    }
                    // 割り当ててもろもろ(no_task_crane_idxs, self.out_tasks)更新してcontinue
                    assign_info = Some((min_dist_idx, item_v));
                }
                if let Some((min_dist_idx, item_v)) = assign_info{
                    cranes[min_dist_idx].assign_task(Rc::clone(&self.tasks[item_v]));
                    no_task_crane_idxs.retain(|&x| x!=min_dist_idx);
                    continue;
                }
            }
            // 搬出OKもしくは割り当て済みのコンテナのplace_taskを削除
            while let Some((item)) = self.task_queue.last(){
                let mut borrowed_task = self.tasks[*item].borrow_mut();
                if let Some(ref mut task) = *borrowed_task{
                    if task.is_out_task(){
                        self.task_queue.pop();
                    } else {
                        break;
                    }
                }
            }
            if let Some(item) = self.task_queue.pop(){
                // placeタスクの割り当て
                // spaceを探す。なければbreak
                let mut assign_info = None;
                if let Some(ref mut task) = *self.tasks[item].borrow_mut(){
                    let mut space_idx = usize::MAX;
                    for i in 0..8{
                        if self.is_space_available[i]{
                            let c = SPACEIDX2COORDINATE[i];
                            if (c.row==3) || (c.col==2 && c.row>=task.target_item_p.row) || (c.col==3 && c.row>=item/SIZE){
                                space_idx = i;
                                break;
                            }
                        }
                    }
                    if space_idx==usize::MAX{
                        self.task_queue.push(item);
                        break;
                    }
                    // task作成
                    task.set_as_placed_task(SPACEIDX2COORDINATE[space_idx]);
                    // 一番近いcraneを探す
                    let mut min_dist = usize::MAX;
                    let mut min_dist_idx = 0;
                    for i in no_task_crane_idxs.iter(){
                        let dist = cranes[*i].p.dist(&task.target_item_p);
                        if dist<min_dist{
                            min_dist = dist;
                            min_dist_idx = *i;
                        }
                    }
                    // 割り当ててもろもろ(no_task_crane_idxs, self.place_tasks)更新してcontinue
                    assign_info = Some((min_dist_idx, item));
                    self.is_space_available[space_idx] = false;
                }
                if let Some((min_dist_idx, item_v)) = assign_info{
                    cranes[min_dist_idx].assign_task(Rc::clone(&self.tasks[item_v]));
                    no_task_crane_idxs.retain(|&x| x!=min_dist_idx);
                    continue;
                }

            } else {
                // 割り当てるタスクなければbreak
                break;
            }
        }
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
    let end_temp: f64 = 0.001;
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
        let score_diff = cur_score-new_score;
        // start_temp = (start_temp*(all_iter-1)+score_diff)/all_iter;
        temp = start_temp + (end_temp-start_temp)*elasped_time/limit;
        if score_diff>=0.0 || rng.gen_bool((score_diff as f64/temp).exp()){
            accepted_cnt += 1;
            // eprintln!("{} {} {}", all_iter, cur_score, new_score);
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
    eprintln!("score  : {}", best_state.cal_score(input, conflictsolver));
    eprintln!("");
    best_state
}

fn main() {
    let timer = Instant::now();
    let input = Input::read_input();
    solve(&input, &timer, 0.0);
}

fn solve(input: &Input, timer:&Instant, tl: f64){
    let conflictsolver = ConflictSolver::new();
    eprintln!("start");
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
