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

#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Input {
    n: usize,
    m: usize,
    e: f64,
    D: Vec<Vec<CoordinateDiff>>,
    max_cs: Vec<Coordinate>,
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
        let mut D = vec![];
        let mut max_cs = vec![];
        let mut oil_cnt = 0;
        for _ in 0..m {
            input! {
                from &mut source,
                d: usize,
                ij_: [(usize, usize); d]
            }
            let mut cs = vec![];
            let mut c = (0, 0);
            for (i, j) in ij_.iter(){
                c.0 = c.0.max(*i);
                c.1 = c.1.max(*j);
                cs.push(CoordinateDiff::new(*i, *j));
            }
            oil_cnt += d;
            D.push(cs);
            max_cs.push(Coordinate::new(n-c.0, n-c.1));
        }
        Self { n, m, e, D, max_cs, oil_cnt }
    }
}

struct Neighbor{
    moves: Vec<(usize, Coordinate, Coordinate)>,
    pre_score: ((usize, f64), (usize, f64)),
}

#[derive(Debug, Clone)]
struct Candidate{
    ans: Vec<Coordinate>,
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
    fn randam_candidate(input: &Input) -> Self{
        let mut rng = rand::thread_rng();
        let mut V = Map2d::new(vec![0; input.n*input.n], input.n);
        let mut ans = vec![];
        for idx in 0..input.m{
            let i = rng.gen::<usize>()%input.max_cs[idx].row;
            let j = rng.gen::<usize>()%input.max_cs[idx].col;
            let c = Coordinate::new(i, j);
            for cd in input.D[idx].iter(){
                V[c+*cd] += 1;
            }
            ans.push(c);
        }
        Self{ans, V, score_1: (0, 0.0), score_d: (0, 1.0)}
    }

    fn from_ans(input: &Input, ans: Vec<Coordinate>) -> Self{
        let mut V = Map2d::new(vec![0; input.n*input.n], input.n);
        for idx in 0..input.m{
            let c = ans[idx];
            for cd in input.D[idx].iter(){
                V[c+*cd] += 1;
            }
        }
        Self{ans, V, score_1: (0, 0.0), score_d: (0, 1.0)}
    }


    fn get_neighbor(&self, input: &Input) -> Neighbor{
        let mut rng = rand::thread_rng();
        let flg = rng.gen::<usize>()%100;
        if flg<90{
            // 1ますいどう
            let i = rng.gen::<usize>()%input.m;
            let pre_c = self.ans[i];
            let new_c = *pre_c.get_adjs2(input.max_cs[i].row, input.max_cs[i].col).choose(&mut rng).unwrap();
            let moves = vec![(i, pre_c, new_c)];
            Neighbor{moves, pre_score: (self.score_1, self.score_d)}
        } else if flg<99 {
            // みの交換
            let mut idx1s: Vec<usize> = (0..input.m).collect();
            let mut idx2s: Vec<usize> = (0..input.m).collect();
            idx1s.shuffle(&mut rng);
            idx2s.shuffle(&mut rng);
            for &i in idx1s.iter(){
                for &j in idx2s.iter(){
                    if i==j{
                        continue;
                    }
                    let mut idxis: Vec<usize> = (0..input.D[i].len()).collect();
                    let mut idxjs: Vec<usize> = (0..input.D[j].len()).collect();
                    idxis.shuffle(&mut rng);
                    idxjs.shuffle(&mut rng);
                    for &idxi in idxis.iter(){
                        for &idxj in idxjs.iter(){
                            let ci = self.ans[j]+input.D[j][idxj]+input.D[i][idxi].invert();
                            let cj = self.ans[i]+input.D[i][idxi]+input.D[j][idxj].invert();
                            if (input.max_cs[i].row>ci.row && input.max_cs[i].col>ci.col && input.max_cs[j].row>cj.row && input.max_cs[j].col>cj.col){
                                let moves = vec![(i, self.ans[i], ci), (j, self.ans[j], cj)];
                                return Neighbor{moves, pre_score: (self.score_1, self.score_d)}
                            }
                        }
                    }
                }
            }
            Neighbor{moves: vec![], pre_score: (self.score_1, self.score_d)}
    } else{
            // 1みの吹っ飛ばすいどう
            let i = rng.gen::<usize>()%input.m;
            let pre_c = self.ans[i];
            let mut new_c = Coordinate::new(rng.gen::<usize>()%input.max_cs[i].row, rng.gen::<usize>()%input.max_cs[i].col);
            while pre_c==new_c {
                new_c = Coordinate::new(rng.gen::<usize>()%input.max_cs[i].row, rng.gen::<usize>()%input.max_cs[i].col);
            }
            let moves = vec![(i, pre_c, new_c)];
            Neighbor{moves, pre_score: (self.score_1, self.score_d)}
        }
    }

    fn update(&mut self, input: &Input, neighbor: &Neighbor, record: &Record){
        if neighbor.moves.is_empty(){
            return;
        }
        for (i, from, to) in neighbor.moves.iter(){
            for cd in input.D[*i].iter(){
                self.V[*from+*cd] -= 1;
                self.V[*to+*cd] += 1;
            }
            self.ans[*i] = *to;
        }
        self.score_1 = (0, 0.0);
        self.score_d = (0, 1.0);
    }

    fn undo(&mut self, input: &Input, neighbor: &Neighbor){
        if neighbor.moves.is_empty(){
            return;
        }
        for (i, from, to) in neighbor.moves.iter(){
            for cd in input.D[*i].iter(){
                self.V[*to+*cd] -= 1;
                self.V[*from+*cd] += 1;
            }
            self.ans[*i] = *from;
        }
        self.score_1 = neighbor.pre_score.0;
        self.score_d = neighbor.pre_score.1;
    }

    fn get_score_1(&mut self, input: &Input, record: &Record) -> f64{
        if self.score_1.0==record.q1.len(){
            return self.score_1.1;
        }
        let mut score_1 = 0.0;
        let mut over_cs = vec![];
        let mut under_cs = vec![];
        for i in 0..input.n{
            for j in 0..input.n{
                let c = Coordinate::new(i, j);
                if record.oil_map[c]==usize::MAX{
                    over_cs.push(c);
                } else if self.V[c]<record.oil_map[c]{
                    score_1 += (self.V[c].abs_diff(record.oil_map[c])*100) as f64;
                    under_cs.push(c);
                } else if self.V[c]>record.oil_map[c]{
                    score_1 += (self.V[c].abs_diff(record.oil_map[c])*100) as f64;
                    over_cs.push(c);
                }
            }
        }
        for c in under_cs.iter(){
            let mut best_diff = usize::MAX;
            for c_ in over_cs.iter(){
                let diff = c.dist(c_);
                if diff<best_diff{
                    best_diff = diff;
                }
            }
            score_1 += best_diff as f64;
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
        for c in record.a.iter(){
            if self.ans==*c{
                return true;

            }
        }
        false
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
    invalid: Vec<HashSet<Coordinate>>,
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
        let x = Candidate::randam_candidate(input);
        let s: HashSet<Candidate> = HashSet::new();
        let candidates = vec![];
        let record = Record{q1: vec![], qd: vec![], a: vec![], oil_map: Map2d::new(vec![usize::MAX; input.n*input.n], input.n), oil_cnts: HashMap::new(), invalid: vec![HashSet::new(); input.m]};
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
        let tl = tl_-timer.elapsed().as_secs_f64();
        let max_turn = input.n*input.n/2;
        for turn in 0..max_turn{
            if timer.elapsed().as_secs_f64()>=tl{
                break;
            }
            eprintln!("n_candidate_pre: {}", self.candidates.len());
            self.update_candidates(input, self.n_candidates, timer, (tl*(turn+1) as f64)/max_turn as f64);
            eprintln!("n_candidate_aft: {}", self.candidates.len());
            // 候補が0ならtlまで探索
            if self.candidates.is_empty(){
                self.update_candidates(input, self.n_candidates/2, timer, tl);
                eprintln!("n_candidate_aft2: {}", self.candidates.len());
            }
            // 候補1しか見つからないなら回答
            if self.candidates.len()==1{
                self.measure(input, 2, self.candidates[0].ans.clone());
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
                for i in 0..input.m{
                    for cd in input.D[i].iter(){
                        let c_ = c+cd.invert();
                        if c_.in_map2(input.max_cs[i].row, input.max_cs[i].col){
                            self.record.invalid[i].insert(c_);
                        }
                    }
                }
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
            let candidate = Candidate::from_ans(input, cs.clone());
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
        let mut rng = rand::thread_rng();
        let mut all_iter = 0;
        let mut accepted = 0;
        while self.candidates.len()<n_candidates{
            let start_time = timer.elapsed().as_secs_f64();
            if start_time >= tl{
                break;
            }
            let mut state = Candidate::randam_candidate(input);
            let mut cur_score = state.get_score_1(input, &self.record);
            let start_temp: f64 = 10.0;
            let end_temp: f64 = 0.1;
            let mut temp = start_temp;
            let mut break_point = all_iter+10000;
            loop{
                let elasped_time = timer.elapsed().as_secs_f64();
                if elasped_time >= tl || all_iter>break_point || self.candidates.len()>=n_candidates*2{
                    break;
                }
                all_iter += 1;
                let neighbor = state.get_neighbor(input);
                state.update(input, &neighbor, &self.record);
                let new_score = state.get_score_1(input, &self.record);
                let score_diff = cur_score-new_score;
                // start_temp = (start_temp*(all_iter-1)+score_diff)/all_iter;
                temp = start_temp + (end_temp-start_temp)*(elasped_time-start_time)/(tl-start_time);
                if score_diff>=0.0 || rng.gen_bool((score_diff/temp).exp()){
                    accepted += 1;
                    // eprintln!("{}->{}", cur_score, new_score);
                    cur_score = new_score;
                    if new_score == 0.0 && !self.candidates.contains(&state) && !state.is_answered(input, &self.record) {
                        break_point = all_iter+1000;
                        self.candidates.push(state.clone());
                    }
                } else {
                    state.undo(input,&neighbor);
                }
            }
        }
        self.total_iter += all_iter;
        eprintln!("all_iter: {}, accepted: {}", all_iter, accepted);
    }

    fn filter_candidates(&mut self, input: &Input){
        for candidate in self.candidates.iter_mut(){
            let x = candidate.get_score_1(input, &self.record);
        }
        self.candidates.retain(|candidate| candidate.score_1.1==0.0);
        if let Some(cs) = self.record.a.last(){
            self.candidates.retain(|candidate| candidate.ans!=*cs);
        }
    }
}

fn main() {
    let timer = Instant::now();
    let input = Input::read_input();
    solve(&input, &timer, 2.8);
}

fn solve(input: &Input, timer:&Instant, tl: f64){
    let mut solver = Solver::new(input, 10);
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
