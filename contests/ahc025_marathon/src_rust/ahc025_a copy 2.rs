#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused)]
use std::collections::{HashSet, HashMap, BinaryHeap};
use std::process::exit;
use itertools::Itertools;
use libm::ceil;
use proconio::{input, source::line::LineSource};
use std::io::{stdin, stdout, BufReader, Write};
use std::fmt::Display;
use std::cmp::{Reverse, Ordering, max, min};
#[allow(unused_imports)]
use rand::prelude::*;
use std::time::Instant;
use ndarray::prelude::*;



#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Measure{
    measured_record: Vec<(Vec<usize>, Vec<usize>, usize)>, // (l, r, flg). 基本, l>r. flg=1ならl=r
    sample_weights: Vec<Vec<usize>>,
    n: usize,
    d: usize,
    q: usize,
    turn: usize,
}

impl Measure{
    fn new() -> Self{
        let stdin = std::io::stdin();
        let mut source = LineSource::new(BufReader::new(stdin.lock()));
        input! {
            from &mut source,
            n: usize,
            d: usize,
            q: usize
        }
        let mut ret = Self { measured_record: vec![], sample_weights: vec![], n, d, q, turn: 0 };
        let mut rng = rand::thread_rng();
        for _ in 0..10{
            ret.sample_weights.push(ret.sample_exponentials(&mut rng, n));
        }
        ret
    }

    fn sample_exponentials(&self, rng: &mut ThreadRng, n: usize) -> Vec<usize>{
        let lambda = 100000.0;
        let max_v = (lambda*(self.n as f64)/(self.d as f64)) as usize;
        let mut ret = vec![];
        for _ in 0..n{
            loop {
                let x = rng.gen::<f64>()%1.0;
                let y = (-lambda * (1.0-x).ln()) as usize;
                if y<max_v{
                    ret.push(y);
                    break;
                }
            }
        }
        ret
    }

    fn solve(&mut self, timer:&Instant, tl: f64){
        for _ in 0..self.q{
            let (ls, rs) = self.make_query();
            self.measure_and_record(&ls, &rs);
        }
        let ans = self.make_group(timer, tl);
        println!("{}", ans.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(" "));
    }

    fn n_invalid_record(&self, sample_weight: &Vec<usize>) -> usize{
        let mut ans = 0;
        for (ls, rs, flg) in self.measured_record.iter(){
            let mut l = ls.iter().map(|&i| sample_weight[i]).sum::<usize>();
            let mut r = rs.iter().map(|&i| sample_weight[i]).sum::<usize>();
            if (flg==&0 && l>r) || (flg==&1 && l==r){
                ans += 1;
            }
        }
        ans
    }

    fn estimate(&mut self, timer:&Instant, tl: f64) -> (Vec<usize>, Vec<usize>){
        let mut rng = rand::thread_rng();
        let ts = timer.elapsed().as_secs_f64();
        for sample_idx in 0..self.sample_weights.len(){
            let mut score = self.n_invalid_record(&self.sample_weights[sample_idx]);
            // 山登りっぽく
            loop{
                if timer.elapsed().as_secs_f64()>(tl-ts)*((sample_idx+1) as f64)/(self.sample_weights.len() as f64){
                    break;
                }
                if score==0{
                    break;
                }
                if sample_idx%2==0{
                    let idx = rng.gen::<usize>()%self.n;
                    let pre_weight = self.sample_weights[sample_idx][idx];
                    let new_weight = self.sample_exponentials(&mut rng, 1)[0];
                    self.sample_weights[sample_idx][idx] = new_weight;
                    let new_score = self.n_invalid_record(&self.sample_weights[sample_idx]);
                    if new_score>=score{
                        self.sample_weights[sample_idx][idx] = pre_weight;
                    } else{
                        score = new_score;
                    }
                } else {
                    let idx1 = rng.gen::<usize>()%self.n;
                    let idx2 = rng.gen::<usize>()%self.n;
                    self.sample_weights[sample_idx].swap(idx1, idx2);
                    let new_score = self.n_invalid_record(&self.sample_weights[sample_idx]);
                    if new_score>=score{
                        self.sample_weights[sample_idx].swap(idx1, idx2);
                    } else{
                        score = new_score;
                    }

                }
            }
            eprint!("{} ", score);
        }
        let mut means = vec![];
        let mut vars = vec![];
        for i in 0..self.n{
            let mut sum = 0;
            for sample_idx in 0..self.sample_weights.len(){
                sum += self.sample_weights[sample_idx][i];
            }
            means.push(sum/self.sample_weights.len());
            let mut sum = 0;
            for sample_idx in 0..self.sample_weights.len(){
                sum += (self.sample_weights[sample_idx][i]-means[i])*(self.sample_weights[sample_idx][i]-means[i]);
            }
            vars.push(sum/self.sample_weights.len());
        }
        (means, vars)
    }

    fn make_query(&mut self) -> (Vec<usize>, Vec<usize>){
        let mut rng = rand::thread_rng();
        loop{
            let mut ls = vec![];
            let mut rs = vec![];
            for _ in 0..2{
                ls.push(rng.gen::<usize>()%self.n);
            }
            for _ in 0..2{
                rs.push(rng.gen::<usize>()%self.n);
            }
            if ls.iter().collect::<HashSet<_>>().intersection(&rs.iter().collect()).next().is_none() && ls.iter().collect::<HashSet<_>>().len()==ls.len() && rs.iter().collect::<HashSet<_>>().len()==rs.len(){
                return (ls, rs);
            }
        }
    }

    fn make_group(&mut self, timer:&Instant, tl: f64) -> Vec<usize>{
        let (means, vars) = self.estimate(timer, tl);
        let mut sorted_idx = (0..self.n).collect::<Vec<usize>>();
        sorted_idx.sort_by(|&i, &j| means[i].cmp(&means[j]));
        let mut bh: BinaryHeap<(Reverse<usize>, Vec<usize>)> = BinaryHeap::new();
        for i in 0..self.d{
            bh.push((Reverse(0), vec![]));
        }
        for idx in sorted_idx.iter(){
            if let Some((Reverse(weight_sum), mut group)) = bh.pop(){
                group.push(*idx);
                bh.push((Reverse(weight_sum+means[*idx]), group));
            }
        }

        let mut ans = vec![0; self.n];
        for (i, (_, idxs)) in bh.iter().enumerate(){
            for idx in idxs.iter(){
                ans[*idx] = i;
            }
        }
        ans
    }

    fn measure_and_record(&mut self, ls: &Vec<usize>, rs: &Vec<usize>){
        // TODO: lsとrsのintersectionが空であることチェック
        assert!(ls.iter().collect::<HashSet<_>>().intersection(&rs.iter().collect()).next().is_none(), "ls: {:?}, rs: {:?}", ls, rs);

        let stdin = std::io::stdin();
        let mut source = LineSource::new(BufReader::new(stdin.lock()));
        let s = format!("{} {}", ls.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(" "), rs.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(" "));
        println!("{} {} {}", ls.len(), rs.len(), s);
        input!{
            from &mut source,
            flg_s: char
        }
        if flg_s=='>'{
            self.measured_record.push((ls.clone(), rs.clone(), 0));
        } else if flg_s=='<'{
            self.measured_record.push((rs.clone(), ls.clone(), 0));
        } else{
            self.measured_record.push((ls.clone(), rs.clone(), 1));
        }
        // let mut new_sample_weights = vec![];
        // for sample_weight in self.sample_weights.iter(){
        //     let mut l = ls.iter().map(|&i| sample_weight[i]).sum::<usize>();
        //     let mut r = rs.iter().map(|&i| sample_weight[i]).sum::<usize>();
        //     if (flg_s=='>' && l>r) || (flg_s=='<' && l<r) || (flg_s=='=' && l==r){
        //         new_sample_weights.push(sample_weight.clone());
        //     }
        // }
        // self.sample_weights = new_sample_weights;
        self.turn += 1;
        // self.estimate();
        eprintln!("turn: {} end", self.turn);
    }

}


fn main() {
    let timer = Instant::now();
    let mut measure = Measure::new();
    let tl = 1.5;
    measure.solve(&timer, tl);
    eprintln!("{}", timer.elapsed().as_secs_f64());
}
