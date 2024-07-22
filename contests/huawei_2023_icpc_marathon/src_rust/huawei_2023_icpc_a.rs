#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused)]
use std::collections::{HashSet, HashMap};
use std::process::exit;
use std::collections::BinaryHeap;
use std::f64::consts::PI;
use std::fmt::Display;
use std::cmp::{Reverse, Ordering};
#[allow(unused_imports)]
use std::time::Instant;
use std::io;
use std::str::FromStr;

#[derive(Debug, Clone)]
pub struct PCG32si {
    state: u32,
}

impl PCG32si {
    const PCG_DEFAULT_MULTIPLIER_32: u32 = 747796405;
    const PCG_DEFAULT_INCREMENT_32: u32 = 2891336453;

    fn pcg_oneseq_32_step_r(&mut self) {
        self.state = self
            .state
            .wrapping_mul(Self::PCG_DEFAULT_MULTIPLIER_32)
            .wrapping_add(Self::PCG_DEFAULT_INCREMENT_32);
    }

    fn pcg_output_rxs_m_xs_32_32(state: u32) -> u32 {
        let word = ((state >> ((state >> 28).wrapping_add(4))) ^ state).wrapping_mul(277803737);
        (word >> 22) ^ word
    }

    pub fn new(seed: u32) -> Self {
        let mut rng = Self { state: seed };
        rng.pcg_oneseq_32_step_r();
        rng.state = rng.state.wrapping_add(seed);
        rng.pcg_oneseq_32_step_r();
        rng
    }

    pub fn next_u32(&mut self) -> u32 {
        let old_state = self.state;
        self.pcg_oneseq_32_step_r();
        Self::pcg_output_rxs_m_xs_32_32(old_state)
    }

    pub fn next_f32(&mut self) -> f32 {
        const FLOAT_SIZE: u32 = core::mem::size_of::<f32>() as u32 * 8;
        const PRECISION: u32 = 23 + 1;
        const SCALE: f32 = 1.0 / (1 << PRECISION) as f32;
        const SHIFT: u32 = FLOAT_SIZE - PRECISION;

        let value = self.next_u32();
        let value = value >> SHIFT;
        SCALE * value as f32
    }

    pub fn next_f32_range(&mut self, min: f32, max: f32) -> f32 {
        min + (max - min) * self.next_f32()
    }
}


#[allow(dead_code)]
fn read_line() -> String {
    let mut buffer = String::new();
    io::stdin()
        .read_line(&mut buffer)
        .expect("failed to read line");

    buffer
}

#[allow(dead_code)]
fn read<T : FromStr>() -> Result<T, T::Err>{
    read_line().trim().parse::<T>()
}

#[allow(dead_code)]
fn read_vec<T : FromStr>() -> Result< Vec<T>, T::Err>{
    read_line().split_whitespace().map(|x| x.parse::<T>()).collect()
}

#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Input {
    n: usize,
    X: Vec<f64>,
    real_val: f64,
    thres_f16: f64,
    thres_f32: f64,
}

impl Input {
    fn read_input() -> Self {
        let eles = read_vec::<f64>().unwrap();
        let n = eles[0] as usize;
        let X = eles[1..].to_vec();
        let mut X_ = X.clone();
        X_.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap());
        let mut real_val = 0.0;
        for x in X_{
            real_val += x;
        }
        let thres_f16 = (real_val.abs()/(2.0_f64.powi(42)*n as f64/2.0)).min(F16MAX/2.0);
        let thres_f32 = (real_val.abs()/(2.0_f64.powi(23)*n as f64/2.0)).min(f32::MAX as f64/2.0);
        Self { n, X, real_val, thres_f16, thres_f32 }
    }
}

const F16MAX: f64 = 65504.0;
const F16MIN: f64 = 5.96e-8;

// TODO: がば
fn round_to_f16(value: f64) -> f64 {
    if value.abs() >= F16MAX{
        return f64::INFINITY;
    }
    if value.abs() <= F16MIN{
        return 0.0;
    }
    f64::from_bits(value.to_bits() & 0b1111111111111111111111000000000000000000000000000000000000000000)
}

fn round_to_f32(value: f64) -> f64 {
    if value.abs() >= f32::MAX as f64{
        return f64::INFINITY;
    }
    if value.abs() <= f32::MIN_POSITIVE as f64{
        return 0.0;
    }
    f64::from_bits(value.to_bits() & 0b1111111111111111111111111111111111100000000000000000000000000000)
}

fn add_f64(a: f64, b: f64) -> f64{
    a+b
}

fn add_f32(a: f64, b: f64) -> f64{
    round_to_f32(a+b)
}

fn add_f16(a: f64, b: f64) -> f64{
    round_to_f16(a+b)
}

#[derive(Debug, Clone)]
struct Group{
    x: Vec<usize>,
    real_val: f64,
    best_val: Option<f64>,
    best_p: Option<usize>,
    best_w: Option<usize>,
}

impl Group{
    fn new(x: Vec<usize>, input: &Input) -> Self{
        let mut real_val = 0.0;
        let mut x_ = x.clone();
        x_.sort_by(|a, b| input.X[*b].abs().partial_cmp(&input.X[*a].abs()).unwrap());
        for &i in &x_{
            real_val += input.X[i];
        }
        Self{x, real_val, best_val: None, best_p: None, best_w: None}
    }

    // コスト最小の組み合わせを探す。
    fn get_score(&mut self, input: &Input, thres_f16: f64, thres_f32: f64) -> (f64, usize, usize){
        if let Some(v) = self.best_val{
            return ((v-self.real_val).abs(), self.best_p.unwrap(), self.best_w.unwrap());
        }
        // TODO: 16^2->16に
        let mut best_p = 20;
        let mut best_p_i = 0;
        for &i in &self.x{
            let mut p = 0;
            for &j in &self.x{
                if i.abs_diff(j) > 15{
                    p += 1;
                }
            }
            if p<best_p{
                best_p = p;
                best_p_i = i;
            }
        }
        let mut w = 0;
        let mut val = 0.0;
        let mut vals = self.x.iter().map(|i| input.X[*i]).collect::<Vec<_>>();
        vals.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap());
        while vals.len()>=2{
            let a = vals.pop().unwrap();
            let b = vals.pop().unwrap();
            let v: f64;
            if b.abs() < thres_f16{
                v = add_f16(a, b);
                w += 1;
            } else if b.abs()<thres_f32{
                v = add_f32(a, b);
                w += 2;
            } else {
                v = add_f64(a, b);
                w += 4;
            }
            val = v;
            vals.push(v);
            vals.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap());
        }
        self.best_val = Some(val);
        self.best_p = Some(best_p);
        self.best_w = Some(w);
        ((self.best_val.unwrap()-self.real_val).abs(), self.best_p.unwrap(), self.best_w.unwrap())
    }

    fn swap(&mut self, x_add: usize, x_rm: usize, input: &Input){
        self.real_val -= input.X[x_rm];
        self.real_val += input.X[x_add];
        self.x.retain(|&i| i!=x_rm);
        self.x.push(x_add);
        self.best_val = None;
        self.best_p = None;
        self.best_w = None;
    }

    fn get_expr(&self,  input: &Input, thres_f16: f64, thres_f32: f64) -> String{
        let mut best_p = 20;
        let mut best_p_i = 0;
        for &i in &self.x{
            let mut p = 0;
            for &j in &self.x{
                if i.abs_diff(j) > 15{
                    p += 1;
                }
            }
            if p<best_p{
                best_p = p;
                best_p_i = i;
            }
        }
        let mut vals = self.x.iter().map(|i| (input.X[*i], (i+1).to_string())).collect::<Vec<_>>();
        vals.sort_by(|a, b| b.0.abs().partial_cmp(&a.0.abs()).unwrap());
        while vals.len()>=2{
            let (a, a_expr) = vals.pop().unwrap();
            let (b, b_expr) = vals.pop().unwrap();
            let v: f64;
            let sumtype: String;
            if b.abs() < thres_f16{
                v = add_f16(a, b);
                sumtype = "h".to_string();
            } else if b.abs()<thres_f32{
                v = add_f32(a, b);
                sumtype = "s".to_string();
            } else {
                v = add_f64(a, b);
                sumtype = "d".to_string();
            }
            if (b_expr==best_p_i.to_string() ||
            b_expr.starts_with(format!("{{d:{},", best_p_i).as_str()) ||
            b_expr.starts_with(format!("{{s:{},", best_p_i).as_str()) ||
            b_expr.starts_with(format!("{{h:{},", best_p_i).as_str())){
                let expr = format!("{{{}:{},{}}}", sumtype, b_expr, a_expr);
                vals.push((v, expr));
            } else {
                let expr = format!("{{{}:{},{}}}", sumtype, a_expr, b_expr);
                vals.push((v, expr));
            }
            vals.sort_by(|a, b| b.0.abs().partial_cmp(&a.0.abs()).unwrap());
        }
        vals[0].1.clone()
    }

}

#[derive(Debug, Clone)]
struct Neighbor{
    g1: usize,
    g2: usize,
    x1: usize,
    x2: usize,
}

#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct State{
    groups: Vec<Group>,
    score: f64,
    errors: f64,
    p: f64,
    w: f64,
    rng: PCG32si,
}

impl State{
    fn init_state(input: &Input) -> Self{
        let mut groups = vec![];
        let mut errors = 0.0;
        let mut p = 0.0;
        let mut w = 0.0;
        for i in 0..(input.n-1)/16+1{
            let til = std::cmp::min(input.n-16*i, 16);
            let mut g = Group::new((0..til).map(|j| i*16+j).collect(), input);
            let (e, p_, w_) = g.get_score(input, input.thres_f16, input.thres_f32);
            errors += e;
            p += p_ as f64;
            w += w_ as f64;
            groups.push(g);
        }
        let mut ret = Self {groups, score: 0.0, errors, p, w, rng: PCG32si::new(42)};
        ret.score = ret.get_score(input);
        ret
    }

    fn get_score(&self, input: &Input) -> f64{
        let a = 1e-20_f64.max((self.errors.abs())/input.real_val.abs().max(1e-200)).powf(0.05);
        let c = (self.p+self.w) as f64/(input.n-1) as f64;
        let d = 10.0/(c+0.5).powf(0.5);
        let score = d/a;
        score
    }

    fn update(&mut self, params: &Neighbor, input: &Input){
        if params.g1==params.g2{
            return;
        }
        let (e, p, w) = self.groups[params.g1].get_score(&input, input.thres_f16, input.thres_f32);
        self.errors -= e;
        self.p -= p as f64;
        self.w -= w as f64;
        let (e, p, w) = self.groups[params.g2].get_score(&input, input.thres_f16, input.thres_f32);
        self.errors -= e;
        self.p -= p as f64;
        self.w -= w as f64;
        self.groups[params.g1].swap(params.x2, params.x1, input);
        self.groups[params.g2].swap(params.x1, params.x2, input);
        let (e, p, w) = self.groups[params.g1].get_score(&input, input.thres_f16, input.thres_f32);
        self.errors += e;
        self.p += p as f64;
        self.w += w as f64;
        let (e, p, w) = self.groups[params.g2].get_score(&input, input.thres_f16, input.thres_f32);
        self.errors += e;
        self.p += p as f64;
        self.w += w as f64;
    }

    fn undo(&mut self, params: &Neighbor, input: &Input){
        self.update(&Neighbor{g1: params.g1, g2: params.g2, x1: params.x2, x2: params.x1}, input)
    }

    fn get_neighbor(&mut self, input: &Input) -> Neighbor{
        let n = input.n;
        // let mode_flg = rng.gen::<usize>()%100;
        let g1 = self.rng.next_u32() as usize % (self.groups.len()-1);
        let g2 = g1+1;
        let x1 = self.groups[g1].x[self.rng.next_u32() as usize % self.groups[g1].x.len()];
        let x2 = self.groups[g2].x[self.rng.next_u32() as usize % self.groups[g2].x.len()];
        Neighbor{g1, g2, x1, x2}
    }

    fn print(&mut self, input: &Input){
        if self.groups.len()==1{
            println!("{}", self.groups[0].get_expr(input, input.thres_f16, input.thres_f32));
            return;
        }
        let exprs = self.groups.iter().map(|g| g.get_expr(input, input.thres_f16, input.thres_f32)).collect::<Vec<_>>();
        let joined_exprs = exprs.join(",");
        println!("{{d:{}}}", joined_exprs);
    }
}


fn simanneal(input: &Input, init_state: State, timer: &Instant, limit: f64) -> State{
    let mut state = init_state.clone();
    let mut best_state = init_state.clone();
    let mut best_score = best_state.get_score(input);
    let mut cur_score = best_score;
    let mut all_iter = 0;
    let mut accepted_cnt = 0;
    let start_temp: f64 = 0.00001;
    let end_temp: f64 = 0.00000001;
    let mut temp = start_temp;
    let mut last_updated = 0;
    let mut rng = PCG32si::new(42);
    eprintln!("simanneal start at {}", timer.elapsed().as_secs_f64());
    loop {
        if state.groups.len()==1{
            break;
        }
        let elasped_time = timer.elapsed().as_secs_f64();
        if elasped_time >= limit{
            break;
        }
        all_iter += 1;
        let neighbor = state.get_neighbor(input);
        state.update(&neighbor, input);
        let new_score = state.get_score(input);
        let score_diff = new_score-cur_score;
        // start_temp = (start_temp*(all_iter-1)+score_diff)/all_iter;
        temp = start_temp + (end_temp-start_temp)*elasped_time/limit;
        if score_diff>=0.0{ // || rng.next_f32() as f64<=(score_diff as f64/temp).exp(){
            accepted_cnt += 1;
            // eprintln!("{} {} {:?}", all_iter, cur_score, new_score);
            cur_score = new_score;
            last_updated = all_iter;
            //state.print(input);
            if new_score>best_score{
                best_state = state.clone();
                best_score = new_score;
            }
        } else {
            state.undo(&neighbor, input);
        }
    }
    eprintln!("===== simanneal =====");
    eprintln!("all iter   : {}", all_iter);
    eprintln!("accepted   : {}", accepted_cnt);
    eprintln!("score  : {}", best_state.get_score(input));
    eprintln!("");
    best_state
}

fn main() {
    let timer = Instant::now();
    let input = Input::read_input();
    eprintln!("time: {} sec", timer.elapsed().as_secs_f64());
    solve(&input, &timer, 8.0);
}

fn solve(input: &Input, timer:&Instant, tl: f64){
    let mut init_state = State::init_state(input);
    let mut best_state = simanneal(input, init_state, timer, tl);
    best_state.print(input);
    eprintln!("{} {} {}", best_state.errors, best_state.p, best_state.w);
    eprintln!("{} {} {}", input.real_val, input.thres_f16, input.thres_f32);
    eprintln!("time_elapsed: {} sec", timer.elapsed().as_secs_f64());
}
