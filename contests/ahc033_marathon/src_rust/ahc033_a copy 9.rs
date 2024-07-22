#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused)]
use core::task;
use std::collections::{HashMap, HashSet, VecDeque};
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
    next_places: Vec<Option<usize>>,
    first_places: Vec<usize>,
}

impl Input {
    fn read_input() -> Self {
        input! {
            n: usize,
            A_: [[usize; SIZE]; SIZE] // `a` is Vec<Vec<i32>>, (m, n)-matrix.
        }
        let mut A = A_.clone();
        let mut first_places = vec![];
        let mut next_places = vec![None; SIZE*SIZE];
        for i in 0..SIZE{
            first_places.push(A[i][0]);
            for j in 1..SIZE{
                next_places[A[i][j-1]] = Some(A[i][j]);
            }
        }
        Self {n, A, next_places, first_places}
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
    usize::MAX, usize::MAX, usize::MAX, usize::MAX, usize::MAX,
];

const SIMULATER_END_TURN: usize = 200;

lazy_static!{
    // タスクの進捗
    static ref PROGRESS_MAP: Map2d<usize> = Map2d::new([
        12, 11, 12, 1, 0,
        11, 10, 11, 2, 1,
        10, 9, 10, 3, 2,
        9, 8, 7, 6, 3,
        8, 7, 6, 5, 4,
    ]);

    static ref PROGRESS_MAP2: Map2d<usize> = Map2d::new([
        12, 11, 12, 1, 0,
        11, 10, 11, 2, 1,
        10, 9, 10, 3, 2,
        9, 8, 7, 4, 3,
        8, 7, 6, 5, 4,
    ]);

    // 4, 5はのぞいた許される移動
    static ref OP_CANDIDATE0: Map2d<Vec<usize>> = Map2d::new([
        vec![1], vec![2, 1, 3], vec![3, 1], vec![2, 1, 3], vec![3, 2],
        vec![0, 1], vec![2, 1, 3], vec![0, 3, 1], vec![2, 1, 3], vec![0, 3],
        vec![0, 1], vec![2, 1, 3], vec![0, 3, 1], vec![2, 1, 3], vec![0, 3],
        vec![0, 1], vec![2, 1, 3], vec![0, 3, 1], vec![2, 1, 3], vec![0, 3],
        vec![0, 1], vec![1, 3], vec![1, 0], vec![1, 0], vec![0],
    ]);

    // ピックor配置時の移動
    static ref OP_CANDIDATE0_PICK: Map2d<Vec<usize>> = Map2d::new([
        vec![4, 1], vec![4, 2, 1, 3], vec![4, 3, 1], vec![4, 2, 1, 3], vec![4, 3, 2],
        vec![4, 0, 1], vec![4, 2, 1, 3], vec![4, 0, 3, 1], vec![4, 2, 1, 3], vec![4, 0, 3],
        vec![4, 0, 1], vec![4, 2, 1, 3], vec![4, 0, 3, 1], vec![4, 2, 1, 3], vec![4, 0, 3],
        vec![4, 0, 1], vec![4, 2, 1, 3], vec![4, 0, 3, 1], vec![4, 2, 1, 3], vec![4, 0, 3],
        vec![4, 0, 1], vec![4, 1, 3], vec![4, 1, 0], vec![4, 1, 0], vec![4, 0],
    ]);

    // タスク何もない時の移動
    static ref OP_CANDIDATE0_STAY: Map2d<Vec<usize>> = Map2d::new([
        vec![1, 5], vec![3, 5, 1, 2], vec![3, 5, 1], vec![3, 5, 1, 2], vec![3, 5, 2],
        vec![0, 5, 1], vec![3, 5, 1, 2], vec![3, 5, 0, 1], vec![3, 5, 1, 2], vec![3, 5, 0],
        vec![0, 5, 1], vec![3, 5, 1, 2], vec![3, 5, 0, 1], vec![3, 5, 1, 2], vec![3, 5, 0],
        vec![0, 5, 1], vec![3, 5, 1, 2], vec![3, 5, 0, 1], vec![3, 5, 1, 2], vec![3, 5, 0],
        vec![0, 5, 1], vec![1, 5, 3], vec![1, 5, 0], vec![1, 5, 0], vec![0, 5],
    ]);

    // タスク何もない時の移動
    static ref OP_CANDIDATE0_STAY2: Map2d<Vec<usize>> = Map2d::new([
        vec![5, 1], vec![5, 2, 1, 3], vec![5, 3, 1], vec![5, 2, 1, 3], vec![5, 3, 2],
        vec![5, 0, 1], vec![5, 2, 1, 3], vec![5, 0, 3, 1], vec![5, 2, 1, 3], vec![5, 0, 3],
        vec![5, 0, 1], vec![5, 2, 1, 3], vec![5, 0, 3, 1], vec![5, 2, 1, 3], vec![5, 0, 3],
        vec![5, 0, 1], vec![5, 2, 1, 3], vec![5, 0, 3, 1], vec![5, 2, 1, 3], vec![5, 0, 3],
        vec![5, 0, 1], vec![5, 1, 3], vec![5, 1, 0], vec![5, 1, 0], vec![5, 0],
    ]);

    // 上に行きたい時
    static ref OP_CANDIDATE0_0: Map2d<Vec<usize>> = Map2d::new([
        vec![5, 1], vec![5, 1, 3, 2], vec![5, 3, 1], vec![5, 1, 3, 2], vec![5, 3, 2],
        vec![0, 5, 1], vec![3, 1, 5, 2], vec![0, 5, 3, 1], vec![3, 1, 5, 2], vec![0, 5, 3],
        vec![0, 5, 1], vec![3, 1, 5, 2], vec![0, 5, 3, 1], vec![3, 1, 5, 2], vec![0, 5, 3],
        vec![0, 5, 1], vec![3, 1, 5, 2], vec![0, 5, 3, 1], vec![3, 1, 5, 2], vec![0, 5, 3],
        vec![0, 5, 1], vec![5, 1, 3], vec![0, 5, 1], vec![0, 5, 1], vec![0, 5],
    ]);

    // 右に行きたい時
    static ref OP_CANDIDATE0_1: Map2d<Vec<usize>> = Map2d::new([
        vec![1, 5], vec![1, 5, 2, 3], vec![1, 5, 3], vec![1, 5, 2, 3], vec![5, 3, 2],
        vec![1, 5, 0], vec![1, 5, 2, 3], vec![1, 5, 0, 3], vec![1, 5, 2, 3], vec![5, 0, 3],
        vec![1, 5, 0], vec![1, 5, 2, 3], vec![1, 5, 0, 3], vec![1, 5, 2, 3], vec![5, 0, 3],
        vec![1, 5, 0], vec![1, 5, 2, 3], vec![1, 5, 0, 3], vec![1, 5, 2, 3], vec![5, 0, 3],
        vec![1, 5, 0], vec![1, 3], vec![1, 5, 0], vec![1, 5, 0], vec![5, 0],
    ]);

    // 下に行きたい時
    static ref OP_CANDIDATE0_2: Map2d<Vec<usize>> = Map2d::new([
        vec![1, 5], vec![2, 5, 1, 3], vec![1, 5, 3], vec![2, 5, 1, 3], vec![3, 5, 2],
        vec![1, 5, 0], vec![2, 5, 1, 3], vec![1, 5, 3, 0], vec![2, 5, 1, 3], vec![3, 0, 5],
        vec![1, 5, 0], vec![2, 5, 1, 3], vec![1, 5, 3, 0], vec![2, 5, 1, 3], vec![3, 0, 5],
        vec![1, 5, 0], vec![2, 5, 1, 3], vec![3, 1, 5, 0], vec![3, 2, 5, 1], vec![3, 0, 5],
        vec![1, 5, 0], vec![1, 3, 5], vec![1, 0, 5], vec![1, 0, 5], vec![0, 5],
    ]);

    // 左に行きたい時
    static ref OP_CANDIDATE0_3: Map2d<Vec<usize>> = Map2d::new([
        vec![5, 1], vec![3, 5, 2, 1], vec![3, 1], vec![3, 5, 2, 1], vec![3, 5, 2],
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

    // 最短経路の候補
    static ref MINDIST_CANDIDATE: Vec<Vec<usize>> = vec![
        vec![5, 1, 2], vec![], vec![5, 2, 1, 3], vec![], vec![5, 3, 1, 2], vec![], vec![5, 2, 1, 3], vec![], vec![3, 5, 2],
        vec![], vec![], vec![], vec![], vec![], vec![], vec![], vec![], vec![],
        vec![5, 0, 1, 2], vec![], vec![5, 2, 1, 3, 0], vec![], vec![5, 0, 3, 1, 2], vec![], vec![5, 2, 1, 3, 0], vec![], vec![0, 3, 5, 2],
        vec![], vec![], vec![], vec![], vec![], vec![], vec![], vec![], vec![],
        vec![5, 0, 1, 2], vec![], vec![5, 2, 1, 3, 0], vec![], vec![5, 0, 3, 1, 2], vec![], vec![5, 2, 1, 3, 0], vec![], vec![0, 3, 5, 2],
        vec![], vec![], vec![], vec![], vec![], vec![], vec![], vec![], vec![],
        vec![5, 0, 1, 2], vec![], vec![5, 2, 1, 3, 0], vec![], vec![5, 0, 3, 1, 2], vec![], vec![5, 2, 1, 3, 0], vec![], vec![0, 3, 5, 2],
        vec![], vec![], vec![], vec![], vec![], vec![], vec![], vec![], vec![],
        vec![5, 0, 1, 2], vec![], vec![5, 1, 3, 0], vec![], vec![5, 1, 0, 3], vec![], vec![5, 1, 0, 3], vec![], vec![0, 3, 5],
    ];

        // 最短経路の候補
    static ref MINDIST_CANDIDATE2: Vec<Vec<usize>> = vec![
        vec![5, 1, 2], vec![], vec![5, 2, 1, 3], vec![], vec![5, 3, 1, 2], vec![], vec![5, 2, 1, 3], vec![], vec![5, 3, 2],
        vec![], vec![], vec![], vec![], vec![], vec![], vec![], vec![], vec![],
        vec![5, 0, 1, 2], vec![], vec![5, 2, 1, 3, 0], vec![], vec![5, 0, 3, 1, 2], vec![], vec![5, 2, 1, 3, 0], vec![], vec![5, 0, 3, 2],
        vec![], vec![], vec![], vec![], vec![], vec![], vec![], vec![], vec![],
        vec![5, 0, 1, 2], vec![], vec![5, 2, 1, 3, 0], vec![], vec![5, 0, 3, 1, 2], vec![], vec![5, 2, 1, 3, 0], vec![], vec![5, 0, 3, 2],
        vec![], vec![], vec![], vec![], vec![], vec![], vec![], vec![], vec![],
        vec![5, 0, 1, 2], vec![], vec![5, 2, 1, 3, 0], vec![], vec![5, 0, 3, 1, 2], vec![], vec![5, 2, 1, 3, 0], vec![], vec![5, 0, 3, 2],
        vec![], vec![], vec![], vec![], vec![], vec![], vec![], vec![], vec![],
        vec![5, 0, 1, 2], vec![], vec![5, 1, 3, 0], vec![], vec![5, 1, 0, 3], vec![], vec![5, 1, 0, 3], vec![], vec![5, 0, 3],
    ];

    static ref OPS2COODINATEDIFF: Vec<CoordinateDiff> = vec![
        CoordinateDiff::new(!0, 0),
        CoordinateDiff::new(0, 1),
        CoordinateDiff::new(1, 0),
        CoordinateDiff::new(0, !0),
        CoordinateDiff::new(0, 0),
        CoordinateDiff::new(0, 0),
        CoordinateDiff::new(0, 0),
    ];

    static ref OPS2COODINATEDIFF2: [usize; 7] = [
        !8, 1, 9, !0, 0, 0, 0,
    ];
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
    place_task_order: Vec<usize>,
    out_task_wait: Vec<usize>,
    score: Option<f64>,
}

impl State{
    fn init_state(input: &Input) -> Self{
        let mut out_task_wait = vec![5; SIZE*SIZE];
        let mut place_task_order = vec![];
        let mut best_score = 100000;
        let mut rng = rand::thread_rng();
        let mut v2row = vec![0; SIZE*SIZE];
        for i in 0..SIZE{
            for j in 0..SIZE{
                v2row[input.A[i][j]] = i;
            }
        }
        for perm in (0..SIZE).permutations(SIZE){
            let mut tmp_order = vec![];
            let mut next_outs = vec![0, 5, 10, 15, 20];
            let mut max_used_space_cnt = 0;
            let mut used_space_cnt = 0;
            let mut next_places = input.first_places.clone();
            let mut on_spaces = vec![];
            for i in perm{
                while next_outs[i]!=usize::MAX{
                    // outできるのがあればoutする
                    let mut outed = false;
                    for i_ in 0..SIZE{
                        if next_places[i_]!=usize::MAX && next_outs.contains(&next_places[i_]){
                            tmp_order.push(next_places[i_]+SIZE*SIZE);
                            if next_outs[next_places[i_]/SIZE]%SIZE==SIZE-1{
                                next_outs[next_places[i_]/SIZE] = usize::MAX;
                            } else {
                                next_outs[next_places[i_]/SIZE] += 1;
                            }
                            if let Some(v) = input.next_places[next_places[i_]]{
                                next_places[i_] = v;
                            } else {
                                next_places[i_] = usize::MAX;
                            }
                            outed = true;
                            break;
                        }
                    }
                    if outed{
                        continue
                    }
                    for &v in on_spaces.iter(){
                        if next_outs.contains(&v){
                            tmp_order.push(v+SIZE*SIZE);
                            if next_outs[v/SIZE]%SIZE==SIZE-1{
                                next_outs[v/SIZE] = usize::MAX;
                            } else {
                                next_outs[v/SIZE] += 1;
                            }
                            used_space_cnt -= 1;
                            outed = true;
                            break;
                        }
                    }
                    if outed{
                        on_spaces.retain(|x| {
                            for v in next_outs.iter(){
                                if x/SIZE==*v/SIZE && x%SIZE<*v%SIZE{
                                    return false;
                                }
                            }
                            true
                        });
                        continue
                    }
                    // 次のoutに向けてplaceする
                    let target_row = v2row[next_outs[i]];
                    tmp_order.push(next_places[target_row]);
                    on_spaces.push(next_places[target_row]);
                    used_space_cnt += 1;
                    max_used_space_cnt = max_used_space_cnt.max(used_space_cnt);
                    if let Some(v) = input.next_places[next_places[target_row]]{
                        next_places[target_row] = v;
                    } else {
                        next_outs[target_row] = usize::MAX;
                    }
                }
            }
            if max_used_space_cnt<best_score{
                best_score = max_used_space_cnt;
                order = tmp_order;
            }
        }
        Self {order, score: None}
    }

    fn get_order(&self, input: &Input) -> (Vec<usize>, usize, usize){
        let mut order = vec![];
        let mut next_outs = vec![0, 5, 10, 15, 20];
        let mut out_queue: BinaryHeap<(Reverse<usize>, usize, usize)> = BinaryHeap::new();
        let mut on_spaces = HashSet::new();
        let mut outed = HashSet::new();
        let mut turn = 0;
        let mut max_used_space_cnt = 0;
        let mut undesired_cnt = 0;
        let mut next_places: Vec<Vec<usize>> = vec![vec![]; SIZE];
        let mut place_order = vec![];
        for i in 0..SIZE{
            for j in (0..SIZE).rev(){
                next_places[i].push(input.A[i][j]);
            }
        }
        let mut next_places_ = next_places.clone();
        for &row in self.place_task_order.iter(){
            place_order.push((row, next_places_[row].pop().unwrap()));
        }
        while outed.len()<SIZE*SIZE{
            turn += 1;
            // やっていいout
            if let Some((Reverse(turn_request), item, row)) = out_queue.peek(){
                if *turn_request<=turn{
                    out_queue.pop();
                    if !on_spaces.remove(item){
                        next_places[*row].pop();
                        if let Some(next_item) = next_places[*row].last(){
                            if next_outs[next_item/SIZE]==*next_item{
                                out_queue.push((Reverse(turn+self.out_task_wait[*next_item]), *next_item, *row));
                            }
                        }
                    }
                    order.push(*item+SIZE*SIZE);
                    outed.insert(*item);
                    if item%SIZE!=SIZE-1{
                        next_outs[item/SIZE] = item+1;
                        for i in 0..SIZE{
                            if let Some(v) = next_places[i].last(){
                                if *v==item+1{
                                    out_queue.push((Reverse(turn+self.out_task_wait[*v]), item+1, i));
                                }
                            }
                        }
                        for v in on_spaces.iter(){
                            if *v==item+1{
                                out_queue.push((Reverse(turn+self.out_task_wait[*v]), item+1, usize::MAX));
                            }
                        }
                    }
                    continue;
                }
            }
            // place
            if on_spaces.len()<8{
                if let Some((row, item)) = place_order.pop(){
                    if !outed.contains(&item){
                        order.push(item);
                        on_spaces.insert(item);
                        max_used_space_cnt = max_used_space_cnt.max(on_spaces.len());
                        next_places[row].pop();
                        if let Some(next_item) = next_places[row].last(){
                            if next_outs[next_item/SIZE]==*next_item{
                                out_queue.push((Reverse(turn+self.out_task_wait[*next_item]), *next_item, row));
                            }
                        }
                        continue;
                    }
                } else {
                    turn -= 1;
                }
            }
            // できれば後でのout
            if let Some((Reverse(turn_request), item, row)) = out_queue.peek(){
                out_queue.pop();
                if !on_spaces.remove(item){
                    next_places[*row].pop();
                    if let Some(next_item) = next_places[*row].last(){
                        if next_outs[next_item/SIZE]==*next_item{
                            out_queue.push((Reverse(turn+self.out_task_wait[*next_item]), *next_item, *row));
                        }
                    }
                }
                outed.insert(*item);
                if item%SIZE!=SIZE-1{
                    next_outs[item/SIZE] = item+1;
                    for i in 0..SIZE{
                        if let Some(v) = next_places[i].last(){
                            if *v==item+1{
                                out_queue.push((Reverse(turn+self.out_task_wait[*v]), item+1, i));
                            }
                        }
                    }
                    for v in on_spaces.iter(){
                        if *v==item+1{
                            out_queue.push((Reverse(turn+self.out_task_wait[*v]), item+1, usize::MAX));
                        }
                    }
                }
                undesired_cnt += 1;
                continue;
            }
            // やむなしのplace
            if let Some((row, item)) = place_order.pop(){
                if !outed.contains(&item){
                    order.push(item);
                    on_spaces.insert(item);
                    max_used_space_cnt = max_used_space_cnt.max(on_spaces.len());
                    next_places[row].pop();
                    if let Some(next_item) = next_places[row].last(){
                        if next_outs[next_item/SIZE]==*next_item{
                            out_queue.push((Reverse(turn+self.out_task_wait[*next_item]), *next_item, row));
                        }
                    }
                    undesired_cnt += 1;
                    continue;
                } else {
                    turn -= 1;
                }
            }
        }
        (order, max_used_space_cnt, undesired_cnt)
    }


    fn init_state_v(input: &Input, priority_order: Vec<usize>) -> Self{
        let mut order = vec![];
        let mut v2idx = vec![(0, 0); SIZE*SIZE];
        let mut rng = rand::thread_rng();
        for i in 0..SIZE{
            for j in 0..SIZE{
                v2idx[input.A[i][SIZE-j-1]] = (i, j);
            }
        }
        let mut cur = [0, 0, 0, 0, 0];
        for i in priority_order{
            for j in 0..SIZE{
                let (r, c) = v2idx[i*SIZE+j];
                while cur[r]<c{
                    order.push(r);
                    cur[r] += 1;
                }
                if cur[r]==c{
                    cur[r]+=1;
                }
            }
        }
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
        Self {order, score: None}
    }


    fn update(&mut self, params: &Neighbor){
        // if params.mode==0{
        //     self.order.swap(params.i, params.j);
        // } else {
        //     let v = self.order.remove(params.i);
        //     self.order.insert(params.j, v);
        // }
        self.score = None;
    }

    fn undo(&mut self, params: &Neighbor){
        // if params.mode==0{
        //     self.order.swap(params.i, params.j);
        // } else {
        //     let v = self.order.remove(params.i);
        //     self.order.insert(params.j, v);
        // }
        self.score = params.score;
    }

    fn get_neighbor(&self, input: &Input) -> Neighbor{
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

    fn simulate(&self, input: &Input) -> (Vec<Crane>, usize, usize){
        let mut task_assigner = TaskAssigner::new(input, &self.order);
        let mut cranes = vec![
            Crane::new(true, Coordinate::new(0, 0)),
            Crane::new(false, Coordinate::new(1, 0)),
            Crane::new(false, Coordinate::new(2, 0)),
            Crane::new(false, Coordinate::new(3, 0)),
            Crane::new(false, Coordinate::new(4, 0)),
        ];
        task_assigner.simulate(&mut cranes);
        (cranes, task_assigner.out_cnt, task_assigner.task_queue.len())
    }

    fn cal_score(&mut self, input: &Input) -> f64{
        if let Some(score) = self.score{
            return score;
        }
        let (cranes, out_cnt, rest_place) = self.simulate(input);
        let score = cranes.iter().map(|c| c.ops.len()).max().unwrap() as f64 + (SIZE*SIZE-out_cnt) as f64 * 1000.0 + rest_place as f64;
        self.score = Some(score);
        score
    }

    fn print(&self, input: &Input){
        let (cranes, out_cnt, rest_place) = self.simulate(input);
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
                    5 => ops_chars.push('.'),
                    6 => ops_chars.push('B'),
                    _ => {panic!();}
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
    ops: Vec<usize>, // 0: U, 1: R, 2: D, 3: L, 4: P or Q, 5: .
    has_item: bool,
}

impl Crane{
    fn new(is_big: bool, p: Coordinate) -> Self{
        Self {is_big, p, ops: vec![], has_item: false}
    }

    fn get_op_candidate(&self, turn: usize) -> Vec<usize>{
        if self.ops.len()<=turn{
            OP_CANDIDATE0_STAY[self.p].clone()
        } else {
            vec![self.ops[turn]]
        }
    }

    fn operate(&mut self, op: usize, turn: usize){
        if turn>=self.ops.len(){
            self.ops.push(op);
            match op{
                0 => self.p.row -= 1,
                1 => self.p.col += 1,
                2 => self.p.row += 1,
                3 => self.p.col -= 1,
                4 => {
                    self.has_item = !self.has_item;
                },
                _ => {},
            }
        }
    }
}


#[derive(Debug, Clone)]
struct Task{
    target_item: usize,
    target_item_p: Coordinate,
    target_place_p: Coordinate,
    start_after: usize,
    end_after: usize,
}

impl Task{
    fn is_out_task(&self) -> bool{
        self.target_place_p.col == SIZE-1
    }

    fn set_as_out_task(&mut self){
        self.target_place_p = Coordinate::new(self.target_item/SIZE, SIZE-1);
    }

    fn set_as_placed_task(&mut self, target_place_p: Coordinate){
        self.target_place_p = target_place_p;
    }
}


struct TaskAssigner{
    place_tasks: Vec<Task>,
    out_tasks: Vec<Task>,
    task_queue: Vec<usize>, // 優先度順に並べたtaskのidx
    next_place: Vec<Option<usize>>,
    crane_map: Vec<bool>, // そのターンにそのマスにいるクレーンがいるか
    container_map: Vec<Vec<(usize, usize)>>, // そのマスにあるコンテナのリスト[from, to)
    out_cnt: usize, // 出庫済みの数
    should_place_after: Vec<usize>, // そのマスにコンテナを置いてよくなるターン
}

impl TaskAssigner{
    fn new(input: &Input, order: &Vec<usize>) -> Self{
        let mut place_tasks = vec![];
        let mut out_tasks = vec![];
        let mut next_place = input.next_places.clone();
        let mut val_row = vec![];
        let mut container_map = vec![vec![(0, 0)]; 81];
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
            let progress = PROGRESS_MAP[target_item_p] + 1000;
            let is_assigned = false;
            place_tasks.push(Task{target_item, target_item_p, target_place_p, start_after: usize::MAX, end_after: 0});
            out_tasks.push(Task{target_item, target_item_p, target_place_p: Coordinate::new(v/SIZE, SIZE-1), start_after: usize::MAX, end_after: usize::MAX});
        }
        for i in 0..SIZE{
            place_tasks[input.first_places[i]].start_after  = 0;
            out_tasks[input.first_places[i]].start_after  = 0;
            container_map[i*9*2][0] = (0, usize::MAX);
            out_tasks[i*SIZE].end_after = 0;
        }
        // place_tasks初期化
        let mut task_queue = order.clone();
        task_queue.reverse();
        // crane_map
        let mut crane_map = vec![false; 81*SIMULATER_END_TURN*2];
        for i in 0..SIZE{
            crane_map[i*9*2] = true;
        }
        Self {place_tasks, out_tasks, task_queue, next_place, crane_map, container_map, out_cnt: 0, should_place_after: vec![0; 81]}
    }

    fn is_finished(&self) -> bool{
        self.out_cnt == SIZE*SIZE
    }

    fn get_min_dist_to_space(&self, fr: usize, after: usize, consider_placed_container: bool) -> Vec<usize>{
        let mut spaces = HashSet::new();
        for p in SPACEIDX2COORDINATE.iter(){
            if self.container_map[p.to_index2()].last().unwrap().1!=usize::MAX{
                spaces.insert(p.to_index2());
            }
        }
        let start_turn = fr/81;
        let mut q = VecDeque::new();
        let mut backtrace = HashMap::new();
        q.push_back(fr);
        while !q.is_empty(){
            let u = q.pop_front().unwrap();
            if u/81>=SIMULATER_END_TURN*2-2{
                break;
            }
            if spaces.contains(&(u%81)) && u/81>=self.container_map[u%81].last().unwrap().1 && u/81>=self.should_place_after[u%81] && !self.crane_map[u+162]{
                // TODO: 経路復元
                let mut path = vec![4];
                let mut v = u;
                while let Some(&(u, op)) = backtrace.get(&v){
                    path.push(op);
                    v = u;
                }
                path.reverse();
                return path;
            }
            for &op in MINDIST_CANDIDATE[u%81].iter(){
                let cd = OPS2COODINATEDIFF2[op];
                let v1 = u+cd+81;
                let v2 = v1+cd+81;
                // 移動がvalid?
                if self.crane_map[v1] || self.crane_map[v2]{
                    continue;
                }
                if consider_placed_container && self.has_container2(v2){
                    continue;
                }
                // bfs
                if let std::collections::hash_map::Entry::Vacant(e) = backtrace.entry(v2) {
                    e.insert((u, op));
                    q.push_back(v2);
                }
            }
        }
        vec![5; SIMULATER_END_TURN]

    }

    fn get_min_dist(&self, fr: usize, to_p: usize, after: usize, consider_placed_container: bool) -> Vec<usize>{
        let start_turn = fr/81;
        let mut q = VecDeque::new();
        let mut backtrace = HashMap::new();
        q.push_back(fr);
        while !q.is_empty(){
            let u = q.pop_front().unwrap();
            if u/81>=SIMULATER_END_TURN*2-2{
                break;
            }
            if u%81==to_p && u/81>=after*2 && !self.crane_map[u+162]{
                // TODO: 経路復元
                let mut path = vec![4];
                let mut v = u;
                while let Some(&(u, op)) = backtrace.get(&v){
                    path.push(op);
                    v = u;
                }
                path.reverse();
                return path;
            }
            for &op in MINDIST_CANDIDATE[u%81].iter(){
                let cd = OPS2COODINATEDIFF2[op];
                let v1 = u+cd+81;
                let v2 = v1+cd+81;
                // 移動がvalid?
                if self.crane_map[v1] || self.crane_map[v2]{
                    continue;
                }
                if consider_placed_container && self.has_container2(v2){
                    continue;
                }
                // bfs
                if let std::collections::hash_map::Entry::Vacant(e) = backtrace.entry(v2) {
                    e.insert((u, op));
                    q.push_back(v2);
                }
            }
        }
        vec![5; SIMULATER_END_TURN]
    }

    fn has_container2(&self, u: usize) -> bool{
        for (fr, to) in self.container_map[u%81].iter(){
            if *fr<=u/81 && u/81<*to{
                return true;
            }
        }
        false
    }

    fn record_crane_move(&mut self, crane: &mut Crane, ops: &Vec<usize>){
        let mut turn = crane.ops.len();
        for &op in ops.iter(){
            let u = crane.p.to_index2()+turn*81*2;
            let cd = OPS2COODINATEDIFF2[op];
            let u1 = u+cd+81;
            let u2 = u1+cd+81;
            self.crane_map[u1] = true;
            self.crane_map[u2] = true;
            self.should_place_after[u2%81] = self.should_place_after[u2%81].max(u2/81+1);
            turn += 1;
            crane.operate(op, turn);
        }
    }

    fn simulate(&mut self, cranes: &mut Vec<Crane>){
        let mut ng_crane_idxs = vec![];
        while !self.task_queue.is_empty(){
            if !self.task_assign(cranes, &mut ng_crane_idxs){
                break;
            }
        }
        let end_turn = cranes.iter().map(|c| c.ops.len()).max().unwrap();
        for crane in cranes.iter_mut(){
            if crane.ops.len()<end_turn{
                crane.operate(6, crane.ops.len());
            }
        }
    }

    fn task_assign(&mut self, cranes: &mut Vec<Crane>, ng_crane_idxs: &mut Vec<usize>) -> bool{
        let mut task = if let Some(item) = self.task_queue.pop(){
            if item<SIZE*SIZE{
                self.place_tasks[item].clone()
            } else {
                self.out_tasks[item-SIZE*SIZE].clone()
            }
        } else {
            panic!();
        };
        let mut best_crane_idx = usize::MAX;
        let mut best_turn = 10000;
        for i in 0..cranes.len(){
            if ng_crane_idxs.contains(&i){
                continue;
            }
            let turn = cranes[i].ops.len()+cranes[i].p.dist(&task.target_item_p);
            if turn<best_turn{
                best_turn = turn;
                best_crane_idx = i;
            }
        }
        if best_crane_idx==usize::MAX{
            return false;
        }
        // タスク割り当て
        let mut crane = &mut cranes[best_crane_idx];
        let start_turn = crane.ops.len();
        // pickしに行く
        let mut ops_to_pick = self.get_min_dist(crane.ops.len()*162+crane.p.to_index2(), task.target_item_p.to_index2(), task.start_after, false);
        if crane.ops.len()+ops_to_pick.len()>=SIMULATER_END_TURN{
            ng_crane_idxs.push(best_crane_idx);
            if task.is_out_task(){
                self.task_queue.push(task.target_item+SIZE*SIZE);
            } else {
                self.task_queue.push(task.target_item);
            }
            return true;
        }
        // pickされた場所のcontainer_mapを更新(仮)
        self.container_map[task.target_item_p.to_index2()].last_mut().unwrap().1 = start_turn+ops_to_pick.len();
        // placeしに行く
        let mut ops_to_place = if task.target_place_p.in_map(){
            self.get_min_dist((crane.ops.len()+ops_to_pick.len())*162+task.target_item_p.to_index2(), task.target_place_p.to_index2(), task.end_after, !crane.is_big)} else {
                self.get_min_dist_to_space((crane.ops.len()+ops_to_pick.len())*162+task.target_item_p.to_index2(), task.end_after, !crane.is_big)
        };
        if crane.ops.len()+ops_to_pick.len()+ops_to_place.len()>=SIMULATER_END_TURN{
            ng_crane_idxs.push(best_crane_idx);
            if task.is_out_task(){
                self.task_queue.push(task.target_item+SIZE*SIZE);
            } else {
                self.task_queue.push(task.target_item);
            }
            return true;
        }
        self.record_crane_move(crane, &ops_to_pick);
        self.record_crane_move(crane, &ops_to_place);
        // もろもろ情報アップデートする
        let mut carry_out_turn = start_turn+ops_to_pick.len();
        for op in ops_to_place.iter(){
            if *op<4{
                break;
            } else {
                carry_out_turn += 1;
            }
        }
        self.container_map[task.target_item_p.to_index2()].last_mut().unwrap().1 = carry_out_turn;
        if task.target_item_p.col==0{
            if let Some(next_task_idx) = self.next_place[task.target_item]{
                // next_place_tasksのアップデート
                self.place_tasks[next_task_idx].start_after = Self::max_without_usizemax(carry_out_turn, self.place_tasks[next_task_idx].start_after);
                self.container_map[task.target_item_p.to_index2()].push((carry_out_turn, usize::MAX));
                // next_placeのout_tasksをアップデート
                self.out_tasks[next_task_idx].start_after = Self::max_without_usizemax(self.out_tasks[next_task_idx].start_after, carry_out_turn);
            }
        }
        if crane.p.col==SIZE-1{
            self.out_cnt += 1;
            if task.target_item%SIZE<4{
                self.out_tasks[task.target_item+1].end_after = crane.ops.len()+1;
            }
        } else {
            self.container_map[crane.p.to_index2()].push((crane.ops.len(), usize::MAX));
            self.out_tasks[task.target_item].target_item_p = crane.p;
            self.out_tasks[task.target_item].start_after = Self::max_without_usizemax(self.out_tasks[task.target_item].start_after, crane.ops.len()+1);
        }
        true
    }

    fn max_without_usizemax(u1: usize, u2: usize) -> usize{
        if u1==usize::MAX{
            u2
        } else if u2==usize::MAX{
            u1
        } else {
            u1.max(u2)
        }
    }
}


fn simanneal(input: &Input, init_state: State, timer: &Instant, limit: f64) -> State{
    let mut state = init_state.clone();
    let mut best_state = init_state.clone();
    let mut best_score = best_state.cal_score(input);
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
        let new_score = state.cal_score(input);
        let score_diff = cur_score-new_score;
        // start_temp = (start_temp*(all_iter-1)+score_diff)/all_iter;
        temp = start_temp + (end_temp-start_temp)*elasped_time/limit;
        if score_diff>=0.0 || rng.gen_bool((score_diff as f64/temp).exp()){
            accepted_cnt += 1;
            // eprintln!("{} {} {} {}", all_iter, cur_score, new_score, elasped_time);
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
    eprintln!("score  : {}", best_state.cal_score(input));
    best_state
}

fn main() {
    let timer = Instant::now();
    let input = Input::read_input();
    solve(&input, &timer, 2.5);
}

fn solve(input: &Input, timer:&Instant, tl: f64){
    let mut init_state = State::init_state(input);
    let mut best_state = init_state.clone();
    // init_stateのパターン試す
    // for i in 0..SIZE{
    //     for j in 0..SIZE{
    //         if i==j{
    //             continue;
    //         }
    //         let mut tmp_state = State::init_state_v(input, vec![i, j]);
    //         let tl_ = (i*SIZE+j) as f64/(SIZE*(SIZE-1)) as f64 * tl/2.0;
    //         let mut tmp_best_state = simanneal(input, tmp_state, timer, tl_, &conflictsolver);
    //         if best_state.score>tmp_best_state.score{
    //             best_state = tmp_best_state;
    //         }
    //     }
    // }
    let mut best_state = simanneal(input, best_state, timer, tl);
    // 多点スタート
    // for i in 0..10{
    //     let tl_ = tl*(i as f64/10.0);
    //     let mut tmp_state = simanneal(input, init_state.clone(), timer, tl_);
    //     if best_state.cal_score(input)>tmp_state.cal_score(input){
    //         best_state = tmp_state;
    //     }
    // }

    best_state.print(input);
}


#[allow(dead_code)]
mod grid {
    pub const SIZE: usize = 5;
    pub const SIZE2: usize = 9;

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

        pub const fn to_index2(&self) -> usize {
            self.row * SIZE2 * 2 + self.col*2
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
