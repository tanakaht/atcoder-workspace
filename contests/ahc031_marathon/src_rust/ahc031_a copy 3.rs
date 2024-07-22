#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused)]
use std::collections::{HashSet, HashMap};
use std::process::exit;
use std::vec;
use itertools::Itertools;
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


#[allow(unused_variables)]
const W: usize = 1000;

#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Input {
    D: usize,
    N: usize,
    A: Vec<Vec<usize>>,
    rest_rates: Vec<usize>,
}

impl Input {
    fn read_input() -> Self {
        input! {
            _: usize,
            D: usize,
            N: usize,
            A: [[usize; N]; D], // a is Vec<i32>, n-array.
            //m: usize,
            //a: [i32; n], // a is Vec<i32>, n-array.
            //ab: [[i32; n]; m] // `a` is Vec<Vec<i32>>, (m, n)-matrix.
        }
        let mut rest_rates: Vec<usize> = vec![];
        for d in 0..D {
            rest_rates.push((W * W - A[d].iter().sum::<usize>()) / N);
        }
        Self { D, N, A, rest_rates }
    }
}

struct Neighbor{
    x: usize
}

#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Group{
    rect: (usize, usize, usize, usize), // (x1, y1, x2, y2)
    start_day: usize,
    end_day: usize,
    member_idxs: Vec<Vec<usize>>,
    member_sums: Vec<usize>,
    child_group: Vec<Group>,
    score: Option<f64>
}

impl Group{
    fn new(input: &Input, rect: (usize, usize, usize, usize), start_day: usize, end_day: usize, member_idxs: Vec<Vec<usize>>) -> Self{
        let mut member_sums = vec![];
        for d in start_day..end_day{
            member_sums.push(member_idxs[d-start_day].iter().map(|idx| input.A[d][*idx]).sum());
        }
        Self{rect, start_day, end_day, member_idxs, member_sums, child_group: vec![], score: None}
    }

    fn divide(&mut self, input: &Input){
        let mut rng = rand::thread_rng();
        if self.member_idxs[0].len() == 1{
            return;
        }
        if self.end_day-self.start_day==1{
            let divide1 = vec![self.member_idxs[0][0..1].to_vec()];
            let divide2 = vec![self.member_idxs[0][1..].to_vec()];
            let w = self.rect.2-self.rect.0;
            let h = self.rect.3-self.rect.1;
            let a = input.A[self.start_day][self.member_idxs[0][0]];
            if w>h{
                let w1 = std::cmp::min(w-1, a/h+1);
                let child_group1: Group = Group::new(input, (self.rect.0, self.rect.1, self.rect.0+w1, self.rect.3), self.start_day, self.end_day, divide1);
                let child_group2: Group = Group::new(input, (self.rect.0+w1, self.rect.1, self.rect.2, self.rect.3), self.start_day, self.end_day, divide2);
                self.child_group = vec![child_group1, child_group2];
            } else {
                let h1 = std::cmp::min(h-1, a/w+1);
                let child_group1: Group = Group::new(input, (self.rect.0, self.rect.1, self.rect.2, self.rect.1+h1), self.start_day, self.end_day, divide1);
                let child_group2: Group = Group::new(input, (self.rect.0, self.rect.1+h1, self.rect.2, self.rect.3), self.start_day, self.end_day, divide2);
                self.child_group = vec![child_group1, child_group2];
            }
            for g in self.child_group.iter_mut(){
                g.divide(input);
            }
            return;
        }
        // 全時間帯で分割できればする。できなければ、適当な時間帯で分割する。
        let mut best_score = usize::MAX;
        let mut best_params = (vec![vec![0]; self.end_day - self.start_day], 0, true); // (divide, w1, is_tate)
        for n_member in 1..(self.member_idxs[0].len()/2){
            for _ in 0..100{
                let mut divides = vec![vec![]; self.end_day - self.start_day];
                let mut divide_sums = vec![0; self.end_day - self.start_day];
                for d in self.start_day..self.end_day{
                    for idx in self.member_idxs[d-self.start_day].iter().choose_multiple(&mut rng, n_member){
                        divides[d-self.start_day].push(*idx);
                        divide_sums[d-self.start_day] += input.A[d][*idx];
                    }
                    // for i in 0..n_member{
                    //     divides[d-self.start_day].push(self.member_idxs[d-self.start_day][i]);
                    //     divide_sums[d-self.start_day] += input.A[d][self.member_idxs[d-self.start_day][i]];
                    // }
                }
                let mut rates = vec![];
                for d in self.start_day..self.end_day{
                    rates.push((divide_sums[d-self.start_day] as f64)/(self.member_sums[d-self.start_day] as f64));
                }
                let mut mid_rate = rates.iter().sum::<f64>()/(rates.len() as f64);
                // 一番平均から遠いところを探して、分割変更させる。改善できなくなったら終わり。
                // TODO: もうちょいいい感じのありそうな。
                loop {
                    let mut max_diff = 0.0;
                    let mut max_diff_d = 0;
                    for d in self.start_day..self.end_day{
                        let diff = (rates[d-self.start_day] - mid_rate).abs();
                        if diff > max_diff{
                            max_diff = diff;
                            max_diff_d = d;
                        }
                    }
                    let target_diff = (mid_rate-rates[max_diff_d-self.start_day])*self.member_sums[max_diff_d-self.start_day] as f64;
                    let mut best_diff = 0.0;
                    let mut best_diff_idx = (0, 0);
                    for i1 in self.member_idxs[max_diff_d-self.start_day].iter(){
                        if divides[max_diff_d-self.start_day].contains(i1){
                            continue;
                        }
                        for i2 in divides[max_diff_d-self.start_day].iter(){
                            let diff = (input.A[max_diff_d][*i1] as f64 - input.A[max_diff_d][*i2] as f64);
                            if (target_diff-diff).abs()<(target_diff-best_diff).abs(){
                                best_diff = diff;
                                best_diff_idx = (*i1, *i2);
                            }
                        }
                    }
                    if best_diff == 0.0{
                        break;
                    }
                    divides[max_diff_d-self.start_day].retain(|&x| x != best_diff_idx.1);
                    divides[max_diff_d-self.start_day].push(best_diff_idx.0);
                    divide_sums[max_diff_d-self.start_day] = divide_sums[max_diff_d-self.start_day] + input.A[max_diff_d][best_diff_idx.0] - input.A[max_diff_d][best_diff_idx.1];
                    rates[max_diff_d-self.start_day] = (divide_sums[max_diff_d-self.start_day] as f64)/(self.member_sums[max_diff_d-self.start_day] as f64);
                    mid_rate = rates.iter().sum::<f64>()/(rates.len() as f64);
                }
                let mut score = 0;
                let max1: usize = *divide_sums.iter().max().unwrap();
                let max2: usize = divide_sums.iter().enumerate().map(|(i, x)| self.member_sums[i] - *x).max().unwrap();
                let w = self.rect.2-self.rect.0;
                let h = self.rect.3-self.rect.1;
                // 分割時の幅を決める
                let tmp_w1 = (w as f64*max1 as f64/(max1+max2) as f64) as usize;
                let tmp_w2 = w-tmp_w1;
                let tmp_h1 = (h as f64*max1 as f64/(max1+max2) as f64) as usize;
                let tmp_h2 = h-tmp_h1;
                let tmp_yohaku1_w = n_member.pow(2)*std::cmp::min(tmp_w1, h);
                let tmp_yohaku2_w = (self.member_idxs[0].len()-n_member).pow(2)*std::cmp::min(tmp_w2, h);
                let tmp_yohaku1_h = n_member.pow(2)*std::cmp::min(w, tmp_h1);
                let tmp_yohaku2_h = (self.member_idxs[0].len()-n_member).pow(2)*std::cmp::min(w, tmp_h2);
                let w1 = ((w as f64*(max1+tmp_yohaku1_w) as f64)/(max1+max2+tmp_yohaku1_w+tmp_yohaku2_w) as f64) as usize;
                let w2 = w - w1;
                let h1 = ((h as f64*(max1+tmp_yohaku1_h) as f64)/(max1+max2+tmp_yohaku1_h+tmp_yohaku2_h) as f64) as usize;
                let h2 = h - h1;
                let yohaku1_w = n_member*std::cmp::min(w1, h);
                let yohaku2_w = (self.member_idxs[0].len()-n_member)*std::cmp::min(w2, h);
                let yohaku1_h = n_member*std::cmp::min(w, h1);
                let yohaku2_h = (self.member_idxs[0].len()-n_member)*std::cmp::min(w, h2);
                // 縦に入れる(wを分割)
                score = max1+max2;
                score += std::cmp::max(w1, w2)*1;
                if w1*h<max1+yohaku1_w || w2*h<max2+yohaku2_w{
                    score = usize::MAX;
                }
                if score < best_score{
                    best_score = score;
                    best_params = (divides.clone(), w1, true);
                }
                // 横に入れる(hを分割)
                score = max1+max2;
                score += std::cmp::max(h1, h2)*1;
                if w*h1<max1+yohaku1_h || w*h2<max2+yohaku2_h{
                    score = usize::MAX;
                }
                if score < best_score{
                    best_score = score;
                    best_params = (divides, h1, false);
                }
            }
        }
        if best_score<usize::MAX{
            let (divide1, w1, is_tate) = best_params;
            let mut divide2 = vec![vec![]; self.end_day - self.start_day];
            for d in self.start_day..self.end_day{
                for &idx in self.member_idxs[d-self.start_day].iter(){
                    if !divide1[d-self.start_day].contains(&idx){
                        divide2[d-self.start_day].push(idx);
                    }
                }
            }
            let child_group1: Group;
            let child_group2: Group;
            if is_tate{
                child_group1 = Group::new(input, (self.rect.0, self.rect.1, self.rect.0+w1, self.rect.3), self.start_day, self.end_day, divide1);
                child_group2 = Group::new(input, (self.rect.0+w1, self.rect.1, self.rect.2, self.rect.3), self.start_day, self.end_day, divide2);
            } else {
                child_group1 = Group::new(input, (self.rect.0, self.rect.1, self.rect.2, self.rect.1+w1), self.start_day, self.end_day, divide1);
                child_group2 = Group::new(input, (self.rect.0, self.rect.1+w1, self.rect.2, self.rect.3), self.start_day, self.end_day, divide2);
            }
            self.child_group = vec![child_group1, child_group2];
            eprintln!("divide group: {} {}/{}", w1, self.child_group[0].member_idxs[0].len(), self.member_idxs[0].len());
        } else {
            let divide_d = (self.start_day+self.end_day)/2;
            let divide1 = self.member_idxs[0..divide_d-self.start_day].to_vec();
            let divide2 = self.member_idxs[divide_d-self.start_day..self.end_day-self.start_day].to_vec();
            let child_group1: Group = Group::new(input, self.rect, self.start_day, divide_d, divide1);
            let child_group2: Group = Group::new(input, self.rect, divide_d, self.end_day, divide2);
            self.child_group = vec![child_group1, child_group2];
            eprintln!("divide days: {} {}", self.start_day, self.end_day);
        }
        for g in self.child_group.iter_mut(){
            g.divide(input);
        }
    }

    fn get_score(&mut self) -> f64{
        if let Some(score) = self.score{
            return score;
        }
        let mut score = 0.0;
        // TODO: score計算
        self.score = Some(score);
        score
    }

    fn solve(&self, input: &Input) -> HashMap<(usize, usize), (usize, usize, usize, usize)>{
        let mut ans = HashMap::new();
        if self.member_idxs[0].len() == 1{
            for d in self.start_day..self.end_day{
                ans.insert((d, self.member_idxs[d-self.start_day][0]), (self.rect.0, self.rect.1, self.rect.2, self.rect.3));
            }
            return ans;
        } else {
            for g in self.child_group.iter(){
                let child_ans = g.solve(input);
                for (k, v) in child_ans.iter(){
                    ans.insert(*k, *v);
                }
            }
        }
        ans
    }
}

fn main() {
    let timer = Instant::now();
    let input = Input::read_input();
    solve(&input, &timer, 1.8);
}

fn solve(input: &Input, timer:&Instant, tl: f64){
    let mut group = Group::new(input, (0, 0, W, W), 0, input.D, (0..input.D).map(|d| (0..input.N).collect()).collect());
    group.divide(input);
    let ans = group.solve(input);
    eprintln!("{:?}", ans.keys().sorted());
    for d in 0..input.D{
        for n in 0..input.N{
            if !ans.contains_key(&(d, n)){
                eprintln!("{} {}", d, n);
            }
            let (x1, y1, x2, y2) = ans[&(d, n)];
            println!("{} {} {} {}", x1, y1, x2, y2);
        }
    }
    // let mut init_state = State::init_state(input);
    // let mut best_state = simanneal(input, init_state, timer, 0.0);
    // best_state.print(input);
}
