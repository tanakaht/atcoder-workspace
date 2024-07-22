#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused)]
use std::collections::{HashSet, HashMap, BinaryHeap};
use std::process::exit;
use itertools::Itertools;
use libm::ceil;
use proconio::{input, marker::Chars, source::line::LineSource};
use std::io::{stdin, stdout, BufReader, Write};
use std::fmt::Display;
use std::cmp::{Reverse, Ordering, max, min};
#[allow(unused_imports)]
use rand::prelude::*;
use std::time::Instant;
use ndarray::prelude::*;
use grid::{Coordinate, CoordinateDiff, Map2d, ADJACENTS};

struct UnionFind {
    n: usize,
    parent_or_size: Vec<i32>,
}

impl UnionFind {
    pub fn new(size: usize) -> Self {
        Self {
            n: size,
            parent_or_size: vec![-1; size],
        }
    }

    pub fn union(&mut self, a: usize, b: usize) -> usize {
        assert!(a < self.n);
        assert!(b < self.n);
        let (mut x, mut y) = (self.find(a), self.find(b));
        if x == y {
            return x;
        }
        if -self.parent_or_size[x] < -self.parent_or_size[y] {
            std::mem::swap(&mut x, &mut y);
        }
        self.parent_or_size[x] += self.parent_or_size[y];
        self.parent_or_size[y] = x as i32;
        x
    }

    pub fn same(&mut self, a: usize, b: usize) -> bool {
        assert!(a < self.n);
        assert!(b < self.n);
        self.find(a) == self.find(b)
    }

    pub fn find(&mut self, a: usize) -> usize {
        assert!(a < self.n);
        if self.parent_or_size[a] < 0 {
            return a;
        }
        self.parent_or_size[a] = self.find(self.parent_or_size[a] as usize) as i32;
        self.parent_or_size[a] as usize
    }

    pub fn size(&mut self, a: usize) -> usize {
        assert!(a < self.n);
        let x = self.find(a);
        -self.parent_or_size[x] as usize
    }

    pub fn groups(&mut self) -> Vec<Vec<usize>> {
        let mut find_buf = vec![0; self.n];
        let mut group_size = vec![0; self.n];
        for i in 0..self.n {
            find_buf[i] = self.find(i);
            group_size[find_buf[i]] += 1;
        }
        let mut result = vec![Vec::new(); self.n];
        for i in 0..self.n {
            result[i].reserve(group_size[i]);
        }
        for i in 0..self.n {
            result[find_buf[i]].push(i);
        }
        result
            .into_iter()
            .filter(|x| !x.is_empty())
            .collect::<Vec<Vec<usize>>>()
    }
}


#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Input {
    l: usize,
    n: usize,
    s: usize,
    //a: [i32; n], // a is Vec<i32>, n-array.
    xy: Vec<Coordinate> // `a` is Vec<Vec<i32>>, (m, n)-matrix.
}

impl Input {
    fn read_input() -> Self {
        let stdin = std::io::stdin();
        let mut source = LineSource::new(BufReader::new(stdin.lock()));
        input! {
            from &mut source,
            l: usize,
            n: usize,
            s: usize
        }
        let mut xy = vec![];
        for _ in 0..n{
            input! {
                from &mut source,
                x: usize,
                y: usize
            }
            xy.push(Coordinate::new(x, y, l));
        }
        Self { l, n, s, xy }
    }
}

// n個のm　bitのidを作成する
fn get_unoverlap_n_ids(n: usize, m:usize, same_points: &Vec<((usize, usize), (usize, usize))>, timer: &Instant, tl: f64) -> Vec<usize>{
    eprintln!("start: {}", timer.elapsed().as_secs_f64());
    let mut vars = vec![];
    let mut var_val = vec![];
    let mut uf = UnionFind::new(n*m);
    let mut used_ids: HashMap<usize, usize> = HashMap::new();
    let mut ids = vec![0; n];
    let mut rng = rand::thread_rng();
    let mut cnt = 0;
    for ((i1, j1), (i2, j2)) in same_points.iter(){
        uf.union(*i1*m + *j1, *i2*m + j2);
    }
    for (i, gr) in uf.groups().iter().enumerate(){
        let mut var: Vec<(usize, usize)> = vec![];
        for x in gr.iter(){
            let (i, j) = (x/m, x%m);
            var.push((i, 1<<j));
        }
        vars.push(var);
        var_val.push(rng.gen::<usize>()%2);
    }
    for (var, v) in vars.iter().zip(var_val.iter()){
        for (i, x) in var.iter(){
            ids[*i] |= v * x;
        }
    }
    for id in ids.iter(){
        *used_ids.entry(*id).or_insert(0) += 1;
    }
    // climbing
    loop {
        cnt += 1;
        // 終了判定
        let elasped_time = timer.elapsed().as_secs_f64();
        let n_id = used_ids.len();
        if elasped_time >= tl || n_id==n{
            break;
        }
        // idを変更する
        let i = rng.gen::<usize>()%vars.len();
        var_val[i] ^= 1;
        for (i, x) in vars[i].iter(){
            *used_ids.entry(ids[*i]).or_insert(0) -= 1;
            if used_ids.get(&ids[*i]) == Some(&0){
                used_ids.remove(&ids[*i]);
            }
            ids[*i] ^= x;
            *used_ids.entry(ids[*i]).or_insert(0) += 1;
        }
        // 悪化したら元に戻す
        if n_id > used_ids.len(){
            var_val[i] ^= 1;
            for (i, x) in vars[i].iter(){
                *used_ids.entry(ids[*i]).or_insert(0) -= 1;
                if used_ids.get(&ids[*i]) == Some(&0){
                    used_ids.remove(&ids[*i]);
                }
                ids[*i] ^= x;
                *used_ids.entry(ids[*i]).or_insert(0) += 1;
                }
        }
    }
    eprintln!("end: {}, cnt: {}, n_overlap: {}", timer.elapsed().as_secs_f64(), cnt, n-used_ids.len());
    ids
}


fn get_used_coordinates(hole_point: &Vec<Coordinate>, id_points: &Vec<CoordinateDiff>) -> HashSet<Coordinate>{
    let mut used_coordinates = HashSet::new();
    for c in hole_point.iter(){
        for cd in id_points.iter(){
            used_coordinates.insert(*c+*cd);
        }
    }
    used_coordinates
}

struct NeighborParam{
    added: Vec<(Coordinate, usize)>,
    deleted: Vec<(Coordinate, usize)>,
    id_points: (Vec<CoordinateDiff>, Option<Vec<CoordinateDiff>>), // before, after
    score: (Option<f64>, Option<f64>), // before, after
}

#[derive(Debug, Clone)]
struct State{
    key_points: HashSet<(Coordinate, usize)>,
    id_points: Vec<CoordinateDiff>,
    score: Option<f64>,
}

// TODO: 差分計算
impl State{
    fn init_state(input: &Input, timer: &Instant, tl: f64) -> Self{
        let mut rng = rand::thread_rng();
        let id_points = vec![
            CoordinateDiff::from_i64(0, 0),
            CoordinateDiff::from_i64(0, -2),
            CoordinateDiff::from_i64(1, 1),
            CoordinateDiff::from_i64(0, 2),
            CoordinateDiff::from_i64(-1, -1),
            CoordinateDiff::from_i64(2, 0),
            CoordinateDiff::from_i64(-2, 0),
        ];
        let mut key_points: HashSet<(Coordinate, usize)> = get_used_coordinates(&input.xy, &id_points).iter().map(|c| (*c, 1000*(rng.gen::<usize>()%2))).collect();
        let mut n_overlap = Self::get_id_overlap_cnt(input, &id_points, &key_points);
        // 山登り
        loop {
            let elasped_time = timer.elapsed().as_secs_f64();
            if elasped_time >= tl{
                break;
            }
            let (c, t) = key_points.iter().choose(&mut rng).unwrap().clone();
            key_points.remove(&(c, t));
            key_points.insert((c, 1000-t));
            let n_overlap_new = Self::get_id_overlap_cnt(input, &id_points, &key_points);
            if n_overlap_new==0{
                break;
            } else if n_overlap_new<=n_overlap{
                n_overlap = n_overlap_new;
            } else {
                key_points.remove(&(c, 1000-t));
                key_points.insert((c, t));
            }
        }
        eprintln!("{}", n_overlap);
        Self {key_points, id_points, score: None}
    }


    fn find_new_id_points(&self, input: &Input) -> Option<Vec<CoordinateDiff>>{
        let mut rng = rand::thread_rng();
        let mut id_points_candidates = HashSet::new();
        let c0 = Coordinate::new(0, 0, input.l);
        for i in -3..4{
            for j in -3..4{
                let cd = CoordinateDiff::from_i64(i, j);
                if c0.dist(&(c0+cd))<=3{
                    id_points_candidates.insert(cd);
                }
            }
        }
        let key_points: HashSet<Coordinate> = self.key_points.iter().map(|(c, t)| *c).collect();
        for c in input.xy.iter(){
            let mut removed = vec![];
            for cd in id_points_candidates.iter(){
                if !key_points.contains(&(*c+*cd)){
                    removed.push(*cd);
                }
            }
            for cd in removed{
                id_points_candidates.remove(&cd);
            }
        }
        if id_points_candidates.len()<7{
            return None;
        }
        let mut id_points_candidates_vec: Vec<CoordinateDiff> = id_points_candidates.into_iter().collect();
        id_points_candidates_vec.shuffle(&mut rng);
        let mut cnt = 0;
        for id_points in id_points_candidates_vec.into_iter().combinations(7){
            cnt += 1;
            if cnt>=1000{
                return None;
            }
            if self.is_id_points_ok(input, &id_points){
                return Some(id_points);
            }
        }
        None
    }

    fn is_id_points_ok(&self, input: &Input, id_points: &Vec<CoordinateDiff>) -> bool{
        let mut used_id: HashSet<usize> = HashSet::new();
        let coords2bit: HashMap<Coordinate, bool> = self.key_points.iter().map(|(c, x)| (*c, *x>=500)).collect();
        for c in input.xy.iter(){
            let mut id = 0;
            for (i, cd) in id_points.iter().enumerate(){
                let c_ = *c + *cd;
                // c_がcoord2bitにあればその値を返し、なければfalseを返しreturnする
                if let Some(b) = coords2bit.get(&c_){
                    if *b{
                        id += (1<<i);
                    }
                } else {
                    return false;
                }
            }
            if used_id.contains(&id){
                return false;
            }
            used_id.insert(id);
        }
        true
    }

    fn get_ids(input: &Input, id_points: &Vec<CoordinateDiff>, key_points: &HashSet<(Coordinate, usize)>) -> Vec<usize>{
        let mut ids: Vec<usize> = vec![];
        let coords2bit: HashMap<Coordinate, bool> = key_points.iter().map(|(c, x)| (*c, *x>=500)).collect();
        for c in input.xy.iter(){
            let mut id = 0;
            for (i, cd) in id_points.iter().enumerate(){
                let c_ = *c + *cd;
                // c_がcoord2bitにあればその値を返し、なければfalseを返しreturnする
                if let Some(b) = coords2bit.get(&c_){
                    if *b{
                        id += (1<<i);
                    }
                }
            }
            ids.push(id);
        }
        ids
    }

    fn get_id_overlap_cnt(input: &Input, id_points: &Vec<CoordinateDiff>, key_points: &HashSet<(Coordinate, usize)>) -> usize{
        let mut used_id: HashSet<usize> = HashSet::new();
        let mut overlap_cnt: usize = 0;
        let coords2bit: HashMap<Coordinate, bool> = key_points.iter().map(|(c, x)| (*c, *x>=500)).collect();
        for c in input.xy.iter(){
            let mut id = 0;
            for (i, cd) in id_points.iter().enumerate(){
                let c_ = *c + *cd;
                // c_がcoord2bitにあればその値を返し、なければfalseを返しreturnする
                if let Some(b) = coords2bit.get(&c_){
                    if *b{
                        id += (1<<i);
                    }
                } else {
                    // id取れてないので特大罰則
                    overlap_cnt += 10000000000;
                }
            }
            if used_id.contains(&id){
                overlap_cnt += 1;
            }
            used_id.insert(id);
        }
        overlap_cnt
    }

    fn get_score(&mut self, input: &Input) -> f64{
        if let Some(x) = self.score{
            return x;
        }
        let map = self.get_map(input);
        // p_costの算出
        let mut p_cost = 0.0;
        let (cd1, cd2) = (CoordinateDiff::new(0, 1), CoordinateDiff::new(1, 0));
        for i in 0..input.l{
            for j in 0..input.l{
                let c = Coordinate::new(i, j, input.l);
                p_cost += (map[c] as f64 - map[c+cd1] as f64).powf(2.0)+(map[c] as f64 - map[c+cd2] as f64).powf(2.0);
            }
        }
        let c = Coordinate::new(0, 0, input.l);
        let mean_dist = self.id_points.iter().fold(0, |sum, cd| sum+c.dist(&(c+*cd))) as f64;
        // x*sqrt(n_measure)/input.s = 3とp_cost=m_costから算出
        // TODO: /2.0?
        let x = num::Float::sqrt(1500.0/2.0*(input.s as f64)*num::Float::sqrt(7.0*(input.n as f64)*100.0*(10.0+mean_dist)/p_cost));
        // p_costをスケール. 0, 1000の前提
        p_cost *= (x/500.0)*(x/500.0);
        // m_costの算出
        let n_measure = (3.0*x/(input.s as f64)).powf(2.0);
        let m_cost = 7.0*(input.n as f64)*100.0*(10.0+mean_dist)*n_measure;
        // TODO: w厳密にやる?
        let w_cnt = 0.0;
        let score = 100_000_000_000_000.0*(0.8_f64.powf(w_cnt))/(p_cost+m_cost+100_000.0);
        self.score = Some(score);
        //eprintln!("p_cost={}, m_cost={}, x={}", p_cost, m_cost, x);

        score
    }

    fn get_neighbor_param(&self, input: &Input) -> NeighborParam{
        let mut rng = rand::thread_rng();
        let mut added = vec![];
        let mut deleted = vec![];
        let mut new_id_points = None;
        let mode_flg = rng.gen::<usize>()%100;
        if (mode_flg<99){
            // temp反転
            let (c, t) = *self.key_points.iter().choose(&mut rng).unwrap();
            deleted.push((c, t));
            added.push((c, 1000-t));
        } else if (mode_flg<70){
            // ketpoint移動
            let (c, t) = *self.key_points.iter().choose(&mut rng).unwrap();
            let cd = CoordinateDiff::from_i64(rng.gen::<i64>()%3-1, rng.gen::<i64>()%3-1);
            deleted.push((c, t));
            added.push((c+cd, t));
        } else if (mode_flg<90){
            // keypoint削除
            let mut unused_keypoint: HashMap<Coordinate, usize> = self.key_points.iter().map(|x| *x).collect();
            for c in input.xy.iter(){
                for cd in self.id_points.iter(){
                    if unused_keypoint.contains_key(&(*c+*cd)){
                        unused_keypoint.remove(&(*c+*cd));
                    }
                }
            }
            if let Some(c) = unused_keypoint.keys().choose(&mut rng){
                let t = unused_keypoint.get(c).unwrap();
                deleted.push((*c, *t));
            }
        } else if (mode_flg<100){
            // id_pointsの移動
            // TODO
        }
        NeighborParam { added, deleted, id_points: (self.id_points.clone(), new_id_points), score: (self.score, None) }
    }

    fn update(&mut self, input: &Input, arg: &NeighborParam) -> bool{
        // updateする
        for (c, t) in arg.deleted.iter(){
            self.key_points.remove(&(*c, *t));
        }
        for (c, t) in arg.added.iter(){
            self.key_points.insert((*c, *t));
        }
        if let (_, Some(new_id_point)) = arg.id_points.clone(){
            self.id_points = new_id_point;
        }
        // id_pointsのアップデート(必要なら)
        if !self.is_id_points_ok(input, &self.id_points){
            if let Some(new_id_points) = self.find_new_id_points(input){
                self.id_points = new_id_points;
            } else {
                return false;
            }
            // 有効なid_pointsがなければfalse
        }
        self.score = arg.score.1;
        true
    }

    fn undo(&mut self, input: &Input, arg: &NeighborParam){
        for (c, t) in arg.added.iter(){
            self.key_points.remove(&(*c, *t));
        }
        for (c, t) in arg.deleted.iter(){
            self.key_points.insert((*c, *t));
        }
        let old_id_points = arg.id_points.0.clone();
        self.id_points = old_id_points;
        self.score = arg.score.0;
    }

    fn get_map(&self, input: &Input) -> Map2d<usize>{
        let mut map = Map2d::<usize>::new(vec![500; input.l*input.l], input.l);
        let key_points: HashSet<Coordinate> = self.key_points.iter().map(|(c, _)| *c).collect();
        for (c, templature) in self.key_points.iter() {
            map[c] = *templature;
        }
        for _ in 0..100{
            for x in 0..input.n{
                for y in 0..input.n{
                    let c = Coordinate::new(x, y, input.l);
                    if key_points.contains(&c) {continue;}
                    let mut mean_neighbor_xy = c.get_adj().iter().map(|c| map[c]).sum::<usize>()/4;
                    map[c] = mean_neighbor_xy;
                }
            }
        }
        map
    }

    fn get_x(&self, input: &Input) -> f64{
        let map = self.get_map(input);
        // p_costの算出
        let mut p_cost = 0.0;
        let (cd1, cd2) = (CoordinateDiff::new(0, 1), CoordinateDiff::new(1, 0));
        for i in 0..input.l{
            for j in 0..input.l{
                let c = Coordinate::new(i, j, input.l);
                p_cost += (map[c] as f64 - map[c+cd1] as f64).powf(2.0)+(map[c] as f64 - map[c+cd2] as f64).powf(2.0);
            }
        }
        let c = Coordinate::new(0, 0, input.l);
        let mean_dist = self.id_points.iter().fold(0, |sum, cd| sum+c.dist(&(c+*cd))) as f64;
        // x*sqrt(n_measure)/input.s = 3とp_cost=m_costから算出
        // TODO: /2.0?
        let x = num::Float::sqrt(1500.0/2.0*(input.s as f64)*num::Float::sqrt(7.0*(input.n as f64)*100.0*(10.0+mean_dist)/p_cost));
        x
    }

    fn scale_x(&mut self, input: &Input){
        // x*sqrt(n_measure)/input.s = 3とp_cost=m_costから算出
        let x = self.get_x(input) as usize;
        // xでスケールさせる
        self.key_points = self.key_points.iter().map(|(c, t)| (*c, 500+ *t*x/500 - x)).collect();
    }

}



fn simanneal(input: &Input, init_state: State, timer: &Instant, limit: f64) -> State{
    let mut state = init_state.clone();
    let mut best_state = init_state.clone();
    let mut best_score = best_state.get_score(input);
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
        let neighbor_param = state.get_neighbor_param(input);
        let flg = state.update(input, &neighbor_param);
        let new_score = state.get_score(input);
        let score_diff = new_score-cur_score;
        // start_temp = (start_temp*(all_iter-1)+score_diff)/all_iter;
        temp = start_temp + (end_temp-start_temp)*elasped_time/limit;
        if flg && (score_diff>=0.0 || rng.gen_bool((score_diff as f64/temp).exp())){
            accepted_cnt += 1;
            eprintln!("{} {} {} {} {:?} {:?}", elasped_time, cur_score, new_score, all_iter, neighbor_param.added, neighbor_param.deleted);
            cur_score = new_score;
            last_updated = all_iter;
            //state.print(input);
            if new_score>best_score{
                best_state = state.clone();
                best_score = new_score;
            }
        } else {
            state.undo(input, &neighbor_param);
        }
    }
    eprintln!("===== simanneal =====");
    eprintln!("all iter   : {}", all_iter);
    eprintln!("accepted   : {}", accepted_cnt);
    eprintln!("score  : {}", best_state.get_score(input));
    eprintln!("");
    best_state
}


struct Arrangement{
    map: Map2d<usize>,
    x: usize,
    s: usize,
    l: usize,
    n_hole: usize,
    n_planed_meansure: usize,
    id_points: Vec<CoordinateDiff>,
    id_for_output_cell: Vec<usize>,
}

impl Arrangement{
    fn new(input: &Input, timer: &Instant, tl: f64) -> Self{
        let init_state = State::init_state(input, timer, tl);
        let mut best_state = simanneal(input, init_state, timer, tl);
        let x = best_state.get_x(input) as usize;
        best_state.scale_x(input);
        let map = best_state.get_map(input);
        eprintln!("x = {}", x);
        // TODO: ガバ
        let mut n_planed_meansure = (3*input.s/x)*(3*input.s/x);
        let id_for_output_cell = State::get_ids(input, &best_state.id_points, &best_state.key_points);
        let id_points = best_state.id_points;
        Self { map, x, s:input.s, l:input.l, n_hole: input.n, n_planed_meansure, id_points, id_for_output_cell }
    }

    fn print_map(&self){
        for x in 0..self.l{
            for y in 0..self.l{
                print!("{} ", self.map[Coordinate::new(x, y, self.l)]);
            }
            println!();
        }
    }
}

struct Measure{
    arrangement: Arrangement,
    measured_templature: Vec<Vec<Vec<usize>>>,
    candidate_for_measure_point: BinaryHeap<(Reverse<usize>, (usize, usize))>,
    n_measured: usize,
}

impl Measure{
    fn new(arrangement: Arrangement) -> Self{
        let mut measured_templature = vec![vec![vec![]; arrangement.id_points.len()]; arrangement.n_hole];
        let mut candidate_for_measure_point = BinaryHeap::new();
        for i in 0..arrangement.n_hole{
            for j in 0..arrangement.id_points.len(){
                candidate_for_measure_point.push((Reverse(0usize), (i, j)));
            }
        }
        Self { arrangement, measured_templature, candidate_for_measure_point, n_measured: 0}
    }

    fn solve(&mut self, input: &Input, timer:&Instant, tl: f64){
        // 99.7%ライン
        let target_v = f64::ceil(((3*self.arrangement.s*self.arrangement.s) as f64)/(self.arrangement.x as f64)) as usize;
        while self.n_measured < 10000{
            let (Reverse(idx), (i, j)) = self.candidate_for_measure_point.pop().unwrap();
            if (idx>=target_v){break;}
            let temp = self.measure_and_record(i, j);
        }
        println!("-1 -1 -1");
        let mut id2output_hole: HashMap<usize, usize> = self.arrangement.id_for_output_cell.iter().enumerate().rev().map(|(i, id)| (*id, i)).collect();
        for i in 0..self.arrangement.n_hole{
            let mut id = 0;
            for (j, measured_templatures) in self.measured_templature[i].iter().enumerate(){
                if (measured_templatures.iter().fold(0, |sum, a| sum+*a) >= 500*measured_templatures.len()){
                    id += 1<<j;
                }
            }
            let output_hole = id2output_hole.entry(id).or_insert(0);
            println!("{}", output_hole);
        }
    }

    // i番目のホールのj番目のIDを測定、記録する。
    fn measure_and_record(&mut self, i: usize, j: usize) -> usize{
        let cd = self.arrangement.id_points.get(j).unwrap();

        let stdin = std::io::stdin();
        let mut source = LineSource::new(BufReader::new(stdin.lock()));
        let (dr, dc) = cd.get_idx_as_i64();
        println!("{} {} {}", i, dr, dc);
        input!{
            from &mut source,
            temp: usize
        }
        // self.measured_templature[i, j]にtempをpush
        self.measured_templature[i][j].push(temp);
        self.n_measured += 1;
        let idx = ((self.measured_templature[i][j].len()*500) as i64 - (self.measured_templature[i][j].iter().fold(0, |sum, a| sum+*a) ) as i64).abs() as usize;
        self.candidate_for_measure_point.push((Reverse(idx), (i, j)));
        temp
    }

}


fn main() {
    let timer = Instant::now();
    let input = Input::read_input();
    // get arrangement
    let arrangement = Arrangement::new(&input, &timer, 3.2);
    // print arrangement by input.l
    arrangement.print_map();
    // 計測, 回答
    let mut measure = Measure::new(arrangement);
    measure.solve(&input, &timer, 3.8);
}


//借用して改変
#[allow(dead_code)]
mod grid {
    use itertools::Itertools;
    use std::cmp::min;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct Coordinate {
        pub row: usize,
        pub col: usize,
        pub size: usize
    }

    impl Coordinate {
        pub const fn new(row: usize, col: usize, size: usize) -> Self {
            // ガバだけど, diffが100＊sizee以下ならOKなはず
            Self { row: (row.wrapping_add(100*size))%size, col: (col.wrapping_add(100*size))%size, size }
        }

        pub const fn to_index(&self) -> usize {
            self.row * self.size + self.col
        }

        pub fn dist(&self, other: &Self) -> usize {
            self.dist_1d(self.row, other.row) + self.dist_1d(self.col, other.col)
        }

        pub fn get_adj(&self) -> Vec<Coordinate> {
            ADJACENTS.iter().map(|&diff| *self + diff).collect()
        }

        fn dist_1d(&self, x0: usize, x1: usize) -> usize {
            min::<usize>((x0 as i64 - x1 as i64).abs() as usize, self.size-(x0 as i64 - x1 as i64).abs() as usize)
        }

    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct CoordinateDiff {
        pub dr: usize,
        pub dc: usize,
    }

    impl CoordinateDiff {
        pub const fn new(dr: usize, dc: usize) -> Self {
            Self { dr, dc }
        }

        pub const fn from_i64(dr: i64, dc: i64) -> Self {
            Self { dr: dr as usize, dc: dc as usize }
        }

        pub const fn get_idx_as_i64(&self) -> (i64, i64) {
            (self.dr as i64, self.dc as i64)
        }

        pub const fn invert(&self) -> Self {
            Self::new(0usize.wrapping_sub(self.dr), 0usize.wrapping_sub(self.dc))
        }
    }

    impl std::ops::Add<CoordinateDiff> for Coordinate {
        type Output = Coordinate;

        fn add(self, rhs: CoordinateDiff) -> Self::Output {
            Coordinate::new(self.row.wrapping_add(rhs.dr), self.col.wrapping_add(rhs.dc), self.size)
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
            &self.map[coordinate.to_index()]
        }
    }

    impl<T> std::ops::IndexMut<Coordinate> for Map2d<T> {
        #[inline]
        fn index_mut(&mut self, coordinate: Coordinate) -> &mut Self::Output {
            &mut self.map[coordinate.to_index()]
        }
    }

    impl<T> std::ops::Index<&Coordinate> for Map2d<T> {
        type Output = T;

        #[inline]
        fn index(&self, coordinate: &Coordinate) -> &Self::Output {
            &self.map[coordinate.to_index()]
        }
    }

    impl<T> std::ops::IndexMut<&Coordinate> for Map2d<T> {
        #[inline]
        fn index_mut(&mut self, coordinate: &Coordinate) -> &mut Self::Output {
            &mut self.map[coordinate.to_index()]
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
