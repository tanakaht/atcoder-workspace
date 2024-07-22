#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused)]
use std::collections::{HashSet, HashMap};
use std::process::exit;
use itertools::Itertools;
use num::{range, ToPrimitive};
use num_integer::Roots;
use proconio::{input, marker::Chars, source::line::LineSource};
use std::io::{stdin, stdout, BufReader, Write};
use rand_core::block;
use std::collections::BinaryHeap;
use std::f64::consts::PI;
use std::fmt::Display;
use std::cmp::{Reverse, Ordering};
use std::hash::{Hash, Hasher};
#[allow(unused_imports)]
use rand::prelude::*;
use std::time::Instant;
use ndarray::prelude::*;
use grid::{Coordinate, Map2d, ADJACENTS, CoordinateDiff};

fn likelihood(x: f64, u: f64, sigma_squared: f64) -> f64 {
    (-((x - u).powi(2) / (2.0 * sigma_squared))).exp()/ ((2.0 * PI * sigma_squared).sqrt())
}

#[derive(Debug, Clone)]
struct Mino{
    cds: Vec<CoordinateDiff>,
    cds_inv: Vec<CoordinateDiff>,
    max_ij: (usize, usize),
    n: usize,
}

impl Mino{
    fn new(cds: Vec<CoordinateDiff>, n: usize)->Self{
        let mut max_i = 0;
        let mut max_j = 0;
        let mut cds_inv = vec![];
        for cd in cds.iter(){
            max_i = max_i.max(cd.dr);
            max_j = max_j.max(cd.dc);
            cds_inv.push(cd.invert());
        }
        Self{cds, cds_inv, max_ij: (n-max_i, n-max_j), n}
    }

    fn get_blocks(&self, c: Coordinate)->Vec<Coordinate>{
        assert!(c.in_map2(self.max_ij.0, self.max_ij.1), "c: {:?} not in map", c);
        let mut ans = vec![];
        for cd in self.cds.iter(){
            ans.push(c+*cd);
        }
        ans
    }

    fn get_coords_placed_c(&self, c: Coordinate, ng_coords: &HashSet<Coordinate>)->Vec<Coordinate>{
        let mut ans = vec![];
        for cd in self.cds_inv.iter(){
            let c_ = c+*cd;
            if !c_.in_map2(self.max_ij.0, self.max_ij.1){
                continue;
            }
            let mut found = false;
            for cd_ in self.cds.iter(){
                let c__ = c_+*cd_;
                if ng_coords.contains(&c__){
                    found = true;
                    break;
                }
            }
            if !found{
                ans.push(c_);
            }
        }
        ans
    }

    fn get_ngs(&self, ng_coords: &HashSet<Coordinate>) -> HashSet<Coordinate>{
        let mut ngs = HashSet::new();
        for c in ng_coords.iter(){
            for cd in self.cds_inv.iter(){
                let c_ = *c+*cd;
                if c_.in_map(self.n){
                    ngs.insert(c_);
                }
            }
        }
        ngs
    }

    fn get_oks(&self, ng_coords: &HashSet<Coordinate>) -> HashSet<Coordinate>{
        let mut oks = HashSet::new();
        let ngs = self.get_ngs(ng_coords);
        for i in 0..self.max_ij.0{
            for j in 0..self.max_ij.1{
                let c = Coordinate::new(i, j);
                if !ngs.contains(&c){
                    oks.insert(c);
                }
            }
        }
        oks
    }
}

#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Input {
    n: usize,
    m: usize,
    e: f64,
    minos: Vec<Mino>,
    oil_cnt: usize,
}

impl Input {
    fn read_input() -> Self {
        let stdin = std::io::stdin();
        let mut source = LineSource::new(BufReader::new(stdin.lock()));
        input! {
            from &mut source,
            n: usize,
            m: usize,
            e: f64,
        }
        let mut minos = vec![];
        let mut oil_cnt = 0;
        for _ in 0..m {
            input! {
                from &mut source,
                d: usize,
                ij_: [(usize, usize); d]
            }
            let mut cds = vec![];
            for (i, j) in ij_.iter(){
                cds.push(CoordinateDiff::new(*i, *j));
            }
            oil_cnt += d;
            minos.push(Mino::new(cds, n));
        }
        Self { n, m, e, minos, oil_cnt }
    }
}


#[derive(Debug, Clone)]
struct Node{
    left: usize,
    right: usize,
    col_idx: usize,
    row_idx: usize,
}

impl Node{
    fn new(left: usize, right: usize, col_idx: usize, row_idx: usize)->Self{
        Self{left, right, col_idx, row_idx}
    }
}

struct Column{
    nodes: Vec<usize>,
}

struct Problem{
    nodes: Vec<Node>,
    rows: Vec<Vec<usize>>, // ミノの配置
    head_nodes: Vec<usize>,
    tail_nodes: Vec<usize>,
    unsatisfied_columns: HashSet<usize>,
    columns_satisfied_cnt: Vec<usize>,
    columns_element_cnt: Vec<usize>,
    rest_columns: usize,
    row2minoidx_coordinates: Vec<Vec<(usize, Coordinate)>>,
}

impl Problem{
    fn new(input: &Input) -> Self{
        let mut ret = Self{nodes: vec![], rows: vec![], head_nodes: vec![], tail_nodes: vec![], unsatisfied_columns: HashSet::new(), columns_satisfied_cnt: vec![], columns_element_cnt: vec![], rest_columns: 0, row2minoidx_coordinates: vec![]};
        for _ in 0..input.m{
            ret.add_column(&vec![], 1);
        }
        for mino_idx in 0..input.m{
            let row = vec![mino_idx];
            let mut minoidx_coordinates = vec![];
            for c in input.minos[mino_idx].get_oks(&HashSet::new()).iter(){
                minoidx_coordinates.push((mino_idx, *c));
            }
            ret.add_row(&row, minoidx_coordinates);
        }
        eprintln!("{:?} {:?} {:?}", ret.nodes, ret.rows, ret.head_nodes);
        ret
    }

    fn is_covered(&self, node_idx: usize)->bool{
        !(self.nodes[self.nodes[node_idx].left].right==node_idx && self.nodes[self.nodes[node_idx].right].left==node_idx)
    }

    fn cover(&mut self, node_idx: usize){
        eprintln!("cover {}", node_idx);
        assert!(!self.is_covered(node_idx), "node_idx: {} is already covered", node_idx);
        self.columns_element_cnt[self.nodes[node_idx].col_idx] -= 1;
        self.__cover(self.nodes[node_idx].left, self.nodes[node_idx].right);
    }

    fn __cover(&mut self, left: usize, right: usize){
        self.nodes[left].right = right;
        self.nodes[right].left = left;
    }

    fn uncover(&mut self, node_idx: usize){
        self.columns_element_cnt[self.nodes[node_idx].col_idx] += 1;
        self.__uncover(self.nodes[node_idx].left, self.nodes[node_idx].right, node_idx);
    }

    fn __uncover(&mut self, left: usize, right: usize, node_idx: usize){
        self.nodes[left].right = node_idx;
        self.nodes[right].left = node_idx;
    }

    fn cover_row(&mut self, row_idx: usize){
        let node_idxs = self.rows[row_idx].to_vec();
        for node_idx in node_idxs{
            self.cover(node_idx);
        }
    }

    fn remove_column(&mut self, node_idx: usize){
        self.unsatisfied_columns.remove(&self.nodes[node_idx].col_idx);
        let head_node_idx = self.head_nodes[self.nodes[node_idx].col_idx];
        let tail_node_idx = self.tail_nodes[self.nodes[node_idx].col_idx];
        let mut row_idxs = vec![self.nodes[node_idx].row_idx];
        let mut node_idx_ = self.nodes[node_idx].left;
        while node_idx_!=head_node_idx{
            row_idxs.push(self.nodes[node_idx_].row_idx);
            node_idx_ = self.nodes[node_idx_].left;
        }
        self.rest_columns -= 1;
        node_idx_ = self.nodes[node_idx].right;
        while node_idx_!=tail_node_idx{
            row_idxs.push(self.nodes[node_idx_].row_idx);
            node_idx_ = self.nodes[node_idx_].right;
        }
        for row_idx in row_idxs{
            self.cover_row(row_idx);
        }
    }

    fn select_row(&mut self, row_idx: usize){
        let node_idxs = self.rows[row_idx].to_vec();
        for node_idx in node_idxs{
            let col_idx = self.nodes[node_idx].col_idx;
            self.columns_satisfied_cnt[col_idx] -= 1;
            if self.columns_satisfied_cnt[col_idx]==0{
                self.remove_column(node_idx);
            }
        }
    }

    fn uncover_row(&mut self, row_idx: usize){
        let node_idxs = self.rows[row_idx].to_vec();
        for node_idx in node_idxs{
            self.uncover(node_idx);
        }
    }

    fn restore_column(&mut self, node_idx: usize){
        self.unsatisfied_columns.insert(self.nodes[node_idx].col_idx);
        let head_node_idx = self.head_nodes[self.nodes[node_idx].col_idx];
        let tail_node_idx = self.tail_nodes[self.nodes[node_idx].col_idx];
        let mut row_idxs = vec![self.nodes[node_idx].row_idx];
        let mut node_idx_ = self.nodes[node_idx].left;
        while node_idx_!=head_node_idx{
            row_idxs.push(self.nodes[node_idx_].row_idx);
            node_idx_ = self.nodes[node_idx_].left;
        }
        node_idx_ = self.nodes[node_idx].right;
        self.rest_columns += 1;
        while node_idx_!=tail_node_idx{
            row_idxs.push(self.nodes[node_idx_].row_idx);
            node_idx_ = self.nodes[node_idx_].right;
        }
        for row_idx in row_idxs{
            self.uncover_row(row_idx);
        }
    }

    fn unselect_row(&mut self, row_idx: usize){
        let node_idxs = self.rows[row_idx].to_vec();
        for node_idx in node_idxs{
            let col_idx = self.nodes[node_idx].col_idx;
            if self.columns_satisfied_cnt[col_idx]==0{
                self.restore_column(node_idx);
            }
            self.columns_satisfied_cnt[col_idx] += 1;
        }
    }

    fn solve_problem(&mut self, ans_cnt: usize, timer:&Instant, tl: f64)->Vec<Vec<usize>>{
        let mut ans = vec![];
        while ans.len()<ans_cnt && timer.elapsed().as_secs_f64()<tl{
            // 未選択の行の中で最も要素数が少ない行を選択
            let mut min_element_cnt = usize::MAX;
            let mut best_column_idx = 0;
            for column_idx in self.unsatisfied_columns.iter(){
                if min_element_cnt>self.columns_element_cnt[*column_idx]{
                    min_element_cnt = self.columns_element_cnt[*column_idx];
                    best_column_idx = *column_idx;
                }
            }
            let mut node_idx = self.nodes[self.head_nodes[best_column_idx]].right;
            while node_idx != self.tail_nodes[best_column_idx]{
                let row_idx = self.nodes[node_idx].row_idx;
                // 列をselect
                self.select_row(row_idx);
                // 終了判定して、終了はしていたらreturn
                if self.unsatisfied_columns.is_empty(){
                    self.unselect_row(row_idx);
                    return vec![vec![row_idx]];
                }
                // ansを追加
                for mut v in self.solve_problem(ans_cnt-ans.len(), timer, tl){
                    v.push(row_idx);
                    ans.push(v);
                }
                // 列をunselect
                self.unselect_row(row_idx);
                node_idx = self.nodes[node_idx].right;
            }
        }
        ans
    }

    fn add_node(&mut self, row_idx: usize, col_idx: usize){
        let node_idx = self.nodes.len();
        let node = Node::new(self.nodes[self.tail_nodes[col_idx]].left, self.tail_nodes[col_idx], col_idx, row_idx);
        self.nodes.push(node);
        self.rows[row_idx].push(node_idx);
        self.uncover(node_idx);
    }

    fn add_row(&mut self, vec: &Vec<usize>, minoidx_coordinates: Vec<(usize, Coordinate)>){
        let row_idx = self.rows.len();
        self.rows.push(vec![]);
        for col_idx in vec.iter(){
            self.add_node(row_idx, *col_idx);
        }
        self.row2minoidx_coordinates.push(minoidx_coordinates);
    }

    fn add_column(&mut self, vec: &Vec<usize>, satisfied_cnt: usize){
        let col_idx = self.head_nodes.len();
        let head_node_idx = self.nodes.len();
        let tail_node_idx = head_node_idx+1;
        let head_node = Node::new(usize::MAX, tail_node_idx, col_idx, usize::MAX);
        let tail_node = Node::new(head_node_idx, usize::MAX, col_idx, usize::MAX);
        self.nodes.push(head_node);
        self.nodes.push(tail_node);
        self.head_nodes.push(head_node_idx);
        self.tail_nodes.push(head_node_idx+1);
        self.unsatisfied_columns.insert(col_idx);
        self.columns_satisfied_cnt.push(satisfied_cnt);
        self.columns_element_cnt.push(vec.len());
        self.rest_columns += 1;
        for row_idx in vec.iter(){
            self.add_node(*row_idx, col_idx);
        }
    }

    fn add_record(&mut self, input: &Input, c: Coordinate, oil_cnt: usize){
        if oil_cnt==0{
            // 今あるのを消す
            for row_idx in 0..self.row2minoidx_coordinates.len(){
                self.row2minoidx_coordinates[row_idx].retain(|(mino_idx, c_)| !input.minos[*mino_idx].get_blocks(*c_).contains(&c));
            }
        } else{
            // 今あるのを01で分岐させる
            let mut columns = vec![];
            let mut rows = vec![];
            let mut rows_minoidx_coordinates = vec![];
            for row_idx in 0..self.row2minoidx_coordinates.len(){
                let mut minoidx_coordinates0 = self.row2minoidx_coordinates[row_idx].to_vec();
                minoidx_coordinates0.retain(|(mino_idx, c_)| !input.minos[*mino_idx].get_blocks(*c_).contains(&c));
                if !minoidx_coordinates0.is_empty(){
                    columns.push(row_idx);
                } else if minoidx_coordinates0.len()<self.row2minoidx_coordinates[row_idx].len(){
                    self.row2minoidx_coordinates[row_idx].retain(|(mino_idx, c_)| input.minos[*mino_idx].get_blocks(*c_).contains(&c));
                    columns.push(row_idx);
                    let mut row = vec![];
                    for node_idx in self.rows[row_idx].iter(){
                        row.push(self.nodes[*node_idx].col_idx);
                    }
                    rows.push(row);
                    rows_minoidx_coordinates.push(minoidx_coordinates0);
                }
            }
            self.add_column(&columns, oil_cnt);
            for i in 0..rows.len(){
                self.add_row(&rows[i], rows_minoidx_coordinates[i].to_vec());
            }
        }
    }
}


struct Neighbor{
    moves: Vec<(usize, Option<Coordinate>)>,
}

#[derive(Debug, Clone)]
struct Candidate{
    ans: Vec<Option<Coordinate>>,
    ans_: Vec<(usize, Coordinate)>,
    V: Map2d<usize>,
    score_1: (usize, f64),
    score_d: (usize, f64)
}

impl Eq for Candidate {
}


impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.ans == other.ans
    }
}

impl Hash for Candidate {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.ans.hash(state);
    }
}

impl Candidate{
    fn randam_candidate(input: &Input, record: &Record) -> Self{
        let mut ret = Self {ans: vec![None; input.m], ans_: vec![], V: Map2d::new(vec![0; input.n*input.n], input.n), score_1: (0, 0.0), score_d: (0, 1.0)};
        // TODO: 適当に置けるところに置く
        ret.construct_block(input, record);
        ret
    }

    fn from_ans(input: &Input, ans: Vec<Option<Coordinate>>) -> Self{
        let mut ret = Self {ans: vec![None; input.m], ans_: vec![], V: Map2d::new(vec![0; input.n*input.n], input.n), score_1: (0, 0.0), score_d: (0, 1.0)};
        let mut neighbor = Neighbor{moves: vec![]};
        for (i, c) in ans.iter().enumerate(){
            if let Some(c_) = c{
                neighbor.moves.push((i, Some(*c_)));
            }
        }
        ret.update(input, &neighbor);
        ret
    }

    fn update(&mut self, input: &Input, neighbor: &Neighbor){
        if neighbor.moves.is_empty(){
            return;
        }
        for (idx, to) in neighbor.moves.iter(){
            if let Some(c) = self.ans[*idx]{
                for c_ in input.minos[*idx].get_blocks(c).iter(){
                    self.V[*c_] -= 1;
                }
            }
            if let Some(c) = to{
                for c_ in input.minos[*idx].get_blocks(*c).iter(){
                    self.V[*c_] += 1;
                }
            }
            self.ans[*idx] = *to;
            self.ans_.retain(|(c, _)| *c!=*idx);
            if let Some(to_) = to{
                self.ans_.push((*idx, *to_));
            }
        }
        self.score_1 = (0, 0.0);
        self.score_d = (0, 1.0);
    }

    fn destruct_and_construct(&mut self, input: &Input, record: &Record, xx: f64){
        self.destruction_block2(input, record);
        self.destruction_block2(input, record);
        self.construct_block(input, record);
    }

    fn construct_block(&mut self, input: &Input, record: &Record){
        // ランダムに置ける限り置く
        let mut rng = rand::thread_rng();
        // 候補ミノ列挙
        let mut cand_mino_idxs = HashSet::new();
        for idx in 0..self.ans.len(){
            cand_mino_idxs.insert(idx);
        }
        for (idx, c) in self.ans_.iter(){
            cand_mino_idxs.remove(idx);
        }
        // ng_coords, oil_coords列挙
        let mut ng_coords = record.ng_coords.clone();
        let mut oil_coords = HashSet::new();
        for (c, cnt) in record.oil_cnts.iter(){
            if self.V[c]==*cnt{
                ng_coords.insert(*c);
            } else {
                oil_coords.insert(*c);

            }
        }
        // 置ける限りおく
        while !cand_mino_idxs.is_empty(){
            let idx = *cand_mino_idxs.iter().choose(&mut rng).unwrap();
            cand_mino_idxs.remove(&idx);
            let mut oil_coords_: Vec<Coordinate> = oil_coords.iter().copied().collect();
            oil_coords_.shuffle(&mut rng);
            let mut is_placed = false;
            for c in oil_coords_.iter(){
                if let Some(c_) = input.minos[idx].get_coords_placed_c(*c, &ng_coords).iter().choose(&mut rng){
                    self.update(input, &Neighbor{moves: vec![(idx, Some(*c_))]});
                    for (c__, cnt) in record.oil_cnts.iter(){
                        if self.V[c__]==*cnt{
                            ng_coords.insert(*c__);
                            oil_coords.remove(c__);
                        }
                    }
                    is_placed = true;
                    break;
                }
            }
            if is_placed{
                continue;
            }
            if let Some(c) = input.minos[idx].get_oks(&ng_coords).iter().choose(&mut rng){
                self.update(input, &Neighbor{moves: vec![(idx, Some(*c))]});
                for (c, cnt) in record.oil_cnts.iter(){
                    if self.V[c]==*cnt{
                        ng_coords.insert(*c);
                        oil_coords.remove(c);
                    }
                }
            }
        }
    }
    fn destruction_block2(&mut self, input: &Input, record: &Record){
        if self.ans_.is_empty(){
            return;
        }
        let mut rng = rand::thread_rng();
        // ミノを強制移動、移動さきに置かれているミノを破壊
        let idx = rng.gen::<usize>()%input.m;
        let mut moves = vec![];
        let c = *input.minos[idx].get_oks(&record.ng_coords).iter().choose(&mut rng).unwrap();
        let blocks = input.minos[idx].get_blocks(c);
        moves.push((idx, Some(c)));
        for (idx_, c_) in self.ans_.iter(){
            for c__ in input.minos[*idx_].get_blocks(*c_).iter(){
                if blocks.contains(c__){
                    moves.push((*idx_, None));
                    break;
                }
            }
        }
        self.update(input, &Neighbor{moves});
    }


    fn destruction_block(&mut self, input: &Input, xx: f64){
        if self.ans_.is_empty(){
            return;
        }
        let mut rng = rand::thread_rng();
        // 破壊するミノを選ぶ。適当に一つ選んで、繋がっているとこをxx%で破壊
        let mut idxs = vec![];
        let mut q = vec![];
        q.push(self.ans_.choose(&mut rng).unwrap().0);
        while !q.is_empty(){
            let idx = q.pop().unwrap();
            idxs.push(idx);
            if rng.gen_bool(xx){
                break;
            }

            let blocks = input.minos[idx].get_blocks(self.ans[idx].unwrap());
            for (idx_, c_) in self.ans_.iter(){
                if idxs.contains(&idx_) || q.contains(&idx_){
                    continue;
                }
                let blocks_ = input.minos[*idx_].get_blocks(*c_);
                let mut is_adj = false;
                for c1 in blocks.iter(){
                    for c2 in blocks_.iter(){
                        if c1.dist(c2)<=1{
                            is_adj = true;
                            break;
                        }
                    }
                    if is_adj{
                        break;
                    }
                }
                if is_adj && !q.contains(idx_) && !idxs.contains(idx_){
                    q.push(*idx_);
                }
            }
        }
        let mut moves = vec![];
        for idx in idxs.iter(){
            moves.push((*idx, None));
        }
        self.update(input, &Neighbor{moves});
    }

    fn check_valid_for_ng_coords(&self, input: &Input, record: &Record)->bool{
        for (i, c) in self.ans_.iter(){
            if input.minos[*i].get_ngs(&record.ng_coords).contains(c){
                return false;
            }
        }
        true
    }

    fn get_score_1(&mut self, input: &Input, record: &Record) -> f64{
        // TODO: 入らないはずなので外す
        // assert!(self.check_valid_for_ng_coords(input, record), "invalid coords in ans: {:?}", self.ans_);
        if self.score_1.0==record.q1.len(){
            return self.score_1.1;
        }
        let mut score_1 = (self.ans_.len()-self.ans.len()) as f64*50.0;
        for (c, v) in record.oil_cnts.iter(){
            score_1 += (self.V[*c].abs_diff(*v)*10) as f64;
        }
        self.score_1 = (record.q1.len(), score_1);
        score_1
    }

    fn get_score_d(&mut self, input: &Input, record: &Record) -> f64{
        if self.score_d.0==record.qd.len(){
            return self.score_d.1;
        }
        let mut score_d = self.score_d.1;
        let e = input.e;
        let qdi = self.score_d.0;
        for (cs, v) in record.qd.iter().skip(qdi){
            let cnt = cs.iter().map(|c| self.V[*c]).sum::<usize>() as f64;
            let k = cs.len() as f64;
            // TODO: ほんとは累積分布関数を使いたい
            score_d *= likelihood(*v as f64, (k-cnt)*e+cnt*(1.0-e), k*e*(1.0-e));
        }
        self.score_d = (record.qd.len(), score_d);
        score_d
    }

    fn is_answered(&self, input: &Input, record: &Record)->bool{
        let mut ans = vec![Coordinate::new(0, 0); input.m];
        for (i, c) in self.ans_.iter(){
            ans[*i] = *c;
        }
        for c in record.a.iter(){
            if ans==*c{
                return true;
            }
        }
        false
    }

    fn check_valid(&mut self, input: &Input, record: &Record) -> bool{
        if self.is_answered(input, record) || !self.check_valid_for_ng_coords(input, record){
            return false;
        } else {
            return self.get_score_1(input, record)==0.0;
        }
    }

    fn get_oil_coords(&self)->Vec<Coordinate>{
        let mut ans = vec![];
        for i in 0..self.V.width{
            for j in 0..self.V.width{
                if self.V[Coordinate::new(i, j)]>0{
                    ans.push(Coordinate::new(i, j));
                }
            }
        }
        ans
    }

    fn print(&self){
        let mut ans = vec![];
        for i in 0..self.V.width{
            for j in 0..self.V.width{
                if self.V[Coordinate::new(i, j)]>0{
                    ans.push(i.to_string());
                    ans.push(j.to_string());
                }
            }
        }
        println!("a {} {}", ans.len()/2, ans.join(" "));
    }
}

#[derive(Debug, Clone)]
struct Record{
    q1: Vec<(Coordinate, usize)>,
    qd: Vec<(Vec<Coordinate>, usize)>,
    a: Vec<Vec<Coordinate>>,
    oil_map: Map2d<usize>,
    oil_cnts: HashMap<Coordinate, usize>,
    ng_coords: HashSet<Coordinate>,
}

#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Solver{
    candidates: Vec<Candidate>,
    record: Record,
    turn: usize,
    n_candidates: usize,
    measured: HashSet<Coordinate>,
    total_iter: usize,
}

impl Solver{
    fn new(input: &Input, n_candidates: usize) -> Self{
        let candidates = vec![];
        let record = Record{q1: vec![], qd: vec![], a: vec![], oil_map: Map2d::new(vec![usize::MAX; input.n*input.n], input.n), oil_cnts: HashMap::new(), ng_coords: HashSet::new()};
        Self {candidates, record, turn:0, n_candidates, measured: HashSet::new(), total_iter: 0}
    }

    fn solve2(&mut self, input: &Input){
        eprintln!("solve2");
        let mut oil_cnt = 0;
        let mut oil_coords = vec![];
        for (c, v) in self.record.q1.iter(){
            oil_cnt += v;
            if *v>0{
                oil_coords.push(*c);
            }
        }
        let mut rng = rand::thread_rng();
        let mut q = HashSet::new();
        for c in oil_coords.iter(){
            for c_ in c.get_adjs(input.n){
                if !self.measured.contains(&c_){
                    q.insert(c_);
                }
            }
        }
        let mut c = Coordinate::new(5, 5);
        while oil_cnt<input.oil_cnt{
            while self.measured.contains(&c) && !q.is_empty(){
                c = *q.iter().next().unwrap();
                q.remove(&c);
            }
            while self.measured.contains(&c){
                c = Coordinate::new(rng.gen::<usize>()%input.n, rng.gen::<usize>()%input.n);
            }
            let resp = self.measure(input, 0, vec![c]);
            oil_cnt += resp;
            if resp>0{
                oil_coords.push(c);
                for c_ in c.get_adjs(input.n){
                    if !self.measured.contains(&c_){
                        q.insert(c_);
                    }
                }
            }
        }
        println!("a {} {}", oil_coords.len(), oil_coords.iter().map(|c| format!("{} {}", c.row, c.col)).join(" "));
        let stdin = std::io::stdin();
        let mut source = LineSource::new(BufReader::new(stdin.lock()));
        input!{
            from &mut source,
            resp: usize
        }
    }

    fn solve(&mut self, input: &Input, timer: &Instant, tl_: f64){
        for i in 0..input.n.sqrt(){
            for j in 0..input.n.sqrt(){
                // self.measure(input, 0, vec![Coordinate::new(i*input.n.sqrt()+input.n.sqrt()/2, j*input.n.sqrt()+input.n.sqrt()/2)]);
            }
        }
        let tl = tl_-timer.elapsed().as_secs_f64();
        let max_turn = 100;
        while self.turn<max_turn{
            if timer.elapsed().as_secs_f64()>=tl{
                break;
            }
            eprintln!("n_candidate_pre: {}", self.candidates.len());
            self.update_candidates(input, self.n_candidates, timer, (tl*(self.turn+1) as f64)/max_turn as f64);
            eprintln!("n_candidate_aft: {}", self.candidates.len());
            // 候補が0ならtlまで探索
            if self.candidates.is_empty(){
                self.update_candidates(input, 2, timer, tl);
                eprintln!("n_candidate_aft2: {}", self.candidates.len());
            }
            // 候補1しか見つからないなら回答
            if self.candidates.len()==1{
                let mut ans = vec![Coordinate::new(0, 0); input.m];
                for (i, c) in self.candidates[0].ans_.iter(){
                    ans[*i] = *c;
                }
                self.measure(input, 2, ans);
                continue;
            }
            // 候補が0なら掘らせる
            if self.candidates.is_empty(){
                break;
            }
            if !self.record.qd.is_empty(){
                for candidate in self.candidates.iter_mut(){
                    candidate.get_score_d(input, &self.record);
                }
                self.candidates.sort_by(|a, b| (-a.score_d.1).partial_cmp(&(-b.score_d.1)).unwrap());
            }
            // 良さそうな点を選ぶ
            let mut best_c = Coordinate::new(0, 0);
            let mut best_score = 100000;
            for i in 0..input.n{
                for j in 0..input.n{
                    let c = Coordinate::new(i, j);
                    if self.measured.contains(&c){
                        continue;
                    }
                    let mut cnts: HashMap<usize, usize> = HashMap::new();
                    for candidate in self.candidates.iter().take(if self.record.qd.is_empty() {self.candidates.len()} else {(self.n_candidates*2/3).max(self.candidates.len()*2/3)}){
                        if let Some(cnt) = cnts.get(&candidate.V[c]){
                            cnts.insert(candidate.V[c], cnt+1);
                        } else {
                            cnts.insert(candidate.V[c], 1);
                        }
                    }
                    let score = *cnts.values().max().unwrap();
                    if score<best_score{
                        best_score = score;
                        best_c = c;
                    }
                }
            }
            self.measure(input, 0, vec![best_c]);
        }
        // 時間なくなったらとにかく掘らせてoilをすべて見つける
        eprintln!("total_iter: {}", self.total_iter);
        self.solve2(input);
    }

    fn measure(&mut self, input: &Input, flg: usize, cs: Vec<Coordinate>)->usize{
        let stdin = std::io::stdin();
        let mut source = LineSource::new(BufReader::new(stdin.lock()));
        self.turn += 1;
        if flg==0{
            let c = cs[0];
            println!("q 1 {} {}", c.row, c.col);
            input!{
                from &mut source,
                resp: usize
            }
            self.measured.insert(c);
            self.record.q1.push((c, resp));
            self.record.oil_map[c] = resp;
            if resp>0{
                self.record.oil_cnts.insert(c, resp);
            } else {
                self.record.ng_coords.insert(c);
            }
            self.filter_candidates(input);
            resp
        } else if flg==1{
            println!("q {} {}", cs.len(), cs.iter().map(|c| format!("{} {}", c.row, c.col)).join(" "));
            input!{
                from &mut source,
                resp: usize
            }
            self.record.qd.push((cs, resp));
            self.filter_candidates(input);
            resp
        } else {
            let cs_ = cs.iter().map(|c| Some(*c)).collect_vec();
            let candidate = Candidate::from_ans(input, cs_);
            let oil_coords = candidate.get_oil_coords();
            println!("a {} {}", oil_coords.len(), oil_coords.iter().map(|c| format!("{} {}", c.row, c.col)).join(" "));
            input!{
                from &mut source,
                resp: usize
            }
            if resp==0{
                self.record.a.push(cs);
            } else {
                eprintln!("total_iter: {}", self.total_iter);
                std::process::exit(0);
            }
            self.filter_candidates(input);
            resp
        }
    }

    fn update_candidates(&mut self, input: &Input, n_candidates: usize, timer: &Instant, tl: f64){
        // let mut cnt = 0;
        // for i in 0..input.m{
        //     cnt += input.minos[i].get_oks(&self.record.ng_coords).len();
        // }
        // eprintln!("place ptn: {}", cnt);
        // let mut maps = HashMap::new();
        // for (c, _) in self.record.oil_cnts.iter(){
        //     maps.insert(c, 0);
        //     for (idx, mino) in input.minos.iter().enumerate(){
        //         for _ in mino.get_coords_placed_c(*c, &self.record.ng_coords).iter(){
        //             maps.insert(c, maps[&c]+1);
        //         }
        //     }
        // }
        // eprintln!("{:?}", maps);
        // let mut cnt_tmp = 0;
        // for (idx, mino) in input.minos.iter().enumerate(){
        //     let oks = mino.get_oks(&self.record.ng_coords);
        //     for c in oks.iter(){
        //         for c_ in mino.get_blocks(*c).iter(){
        //             if self.record.oil_cnts.contains_key(c_){
        //                 cnt_tmp += 1;
        //                 break;
        //             }
        //         }
        //     }
        // }
        // eprintln!("cnt_tmp: {:?}", cnt_tmp);
        let mut rng = rand::thread_rng();
        let mut all_iter = 0;
        let mut accepted = 0;
        while self.candidates.len()<n_candidates{
            let start_time = timer.elapsed().as_secs_f64();
            if start_time >= tl{
                break;
            }
            let mut state = Candidate::randam_candidate(input, &self.record);
            let mut cur_score = state.get_score_1(input, &self.record);
            let start_temp: f64 = 10.0;
            let end_temp: f64 = 0.1;
            let mut temp = start_temp;
            let mut break_point = all_iter+1000;
            loop{
                let elasped_time = timer.elapsed().as_secs_f64();
                if elasped_time >= tl || all_iter>break_point || self.candidates.len()>=n_candidates*2{
                    break;
                }
                all_iter += 1;
                let mut new_state = state.clone();
                new_state.destruct_and_construct(input, &self.record, 0.1);
                let new_score = new_state.get_score_1(input, &self.record);
                let score_diff = cur_score-new_score;
                // start_temp = (start_temp*(all_iter-1)+score_diff)/all_iter;
                temp = start_temp + (end_temp-start_temp)*(elasped_time-start_time)/(tl-start_time);
                if score_diff>=0.0 || rng.gen_bool((score_diff/temp).exp()){
                    accepted += 1;
                    // eprintln!("{}->{}", cur_score, new_score);
                    cur_score = new_score;
                    state = new_state;
                    if new_score == 0.0 && !self.candidates.contains(&state) && !state.is_answered(input, &self.record) {
                        self.candidates.push(state.clone());
                        break;
                        break_point = all_iter+10;
                    }
                } else {
                    // state.undo(input,&neighbor);
                }
            }
        }
        self.total_iter += all_iter;
        eprintln!("all_iter: {}, accepted: {}", all_iter, accepted);
    }

    fn filter_candidates(&mut self, input: &Input){
        let mut is_valid = vec![false; self.candidates.len()];
        for (i, candidate) in self.candidates.iter_mut().enumerate(){
            is_valid[i] = candidate.check_valid(input, &self.record);
        }
        let mut idx = 0;
        self.candidates.retain(|_| {
            idx += 1;
            is_valid[idx-1]
        });
    }
}

fn main() {
    let timer = Instant::now();
    let input = Input::read_input();
    solve(&input, &timer, 29.8);
}

fn solve(input: &Input, timer:&Instant, tl: f64){
    let mut solver = Solver::new(input, 10);
    let mut prob = Problem::new(input);
    let ans = prob.solve_problem(10, timer, tl);
    eprintln!("ans: {:?}", ans);
    solver.solve(input, timer, tl);
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
