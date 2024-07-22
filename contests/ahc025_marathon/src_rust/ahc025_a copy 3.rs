#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused)]
use std::collections::{HashSet, HashMap, BinaryHeap, VecDeque};
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


fn is_subset(base: &HashSet<usize>, other: &HashSet<usize>) -> bool{
    base.intersection(&other).count()==other.len()
}

fn most_frequent(items: Vec<usize>) -> Option<usize> {
    let mut occurrences: HashMap<usize, usize> = HashMap::new();
    for item in items.iter() {
        *occurrences.entry(item.clone()).or_insert(0) += 1;
    }
    occurrences.into_iter()
        .max_by(|a, b| a.1.cmp(&b.1))
        .map(|(key, _)| key)
}


#[allow(unused_variables)]
#[derive(Debug, Clone)]
struct Measure{
    measured_record: Vec<(Vec<usize>, Vec<usize>, usize)>, // (l, r, flg). 基本, l>r. flg=1ならl=r
    groups: Vec<HashSet<usize>>,
    n: usize,
    d: usize,
    q: usize,
    turn: usize,
    phase: usize,
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
        let mut groups = vec![HashSet::new(); d];
        for i in 0..n{
            groups[i%d].insert(i);
        }
        Self { measured_record: vec![], groups, n, d, q, turn: 0, phase: 0 }
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


    fn get_sample_weight(&mut self, timer:&Instant, tl: f64) -> Vec<usize>{
        let mut rng = rand::thread_rng();
        let ts = timer.elapsed().as_secs_f64();
        let mut sample_weight = self.sample_exponentials(&mut rng, self.n);
        // 山登りっぽく
        loop{
            if timer.elapsed().as_secs_f64()>tl{
                break;
            }
            let mut ng_cnt = 0;
            let mut measured_record = self.measured_record.clone();
            measured_record.shuffle(&mut rng);
            for (ls, rs, flg) in measured_record{
                let mut l = ls.iter().map(|&i| sample_weight[i]).sum::<usize>();
                let mut r = rs.iter().map(|&i| sample_weight[i]).sum::<usize>();
                if (flg==0 && l>r) || (flg==1 && l==r){
                    continue;
                }
                ng_cnt += 1;
                if rng.gen::<usize>()%2==0{
                    for li in ls.iter(){
                        sample_weight[*li] += (r-l)/ls.len()+10;
                    }
                } else {
                    for ri in rs.iter(){
                        sample_weight[*ri] = sample_weight[*ri]-(r-l)/rs.len()-10;
                    }
                }
            }
            if ng_cnt == 0{
                break;
            }
            eprint!("{} ", ng_cnt);
        }
        sample_weight
    }


    fn solve(&mut self, timer:&Instant, tl: f64){
        loop{
            let lrss = self.make_querys();
            for (ls, rs) in lrss.iter(){
                self.measure_and_record(&ls, &rs);
                if self.turn==self.q{
                    break;
                }
            }
            if self.turn==self.q{
                break;
            }
        }
        self.print_group(false);
        let mut ans = vec![0; self.n];
        for (i, group) in self.groups.iter().enumerate(){
            for &idx in group.iter(){
                ans[idx] = i;
            }
        }
        let sample_weight = self.get_sample_weight(timer, tl);
        eprintln!("{:?}", sample_weight);
        println!("{}", ans.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(" "));
    }

    fn toporogical_sort_for_group(&self) -> Vec<usize>{
        let mut g = vec![HashSet::new(); self.d];
        let mut g_rev = vec![HashSet::new(); self.d];
        let mut same = vec![vec![]; self.d];
        for (ls, rs, flg)  in self.measured_record.iter(){
            let ls_set = ls.iter().map(|&x| x).collect::<HashSet<_>>();
            let rs_set = rs.iter().map(|&x| x).collect::<HashSet<_>>();
            for ls_idx in 0..self.d{
                if !is_subset(&self.groups[ls_idx], &ls_set){
                    continue;
                }
                for rs_idx in 0..self.d{
                    if ls_idx==rs_idx || !is_subset(&rs_set, &self.groups[rs_idx]){
                        continue;
                    }
                    if *flg==1{
                        same[ls_idx].push(rs_idx);
                        same[rs_idx].push(ls_idx);
                        continue;
                    }
                    g[ls_idx].insert(rs_idx);
                    g_rev[rs_idx].insert(ls_idx);
                }
            }
        }
        let mut ret = vec![];
        let mut q = VecDeque::new();
        for i in 0..self.d{
            if g[i].is_empty(){
                q.push_back(i);
            }
        }
        while !q.is_empty(){
            let i = q.pop_front().unwrap();
            ret.push(i);
            for &j in g_rev[i].iter(){
                g[j].remove(&i);
                if g[j].is_empty(){
                    q.push_back(j);
                }
            }
            for &j in same[i].iter(){
                if !ret.contains(&j){
                    q.push_front(j);
                }
            }
        }
        ret
    }

    fn print_group(&self, debug: bool){
        let mut ans = vec![0; self.n];
        for (i, group) in self.groups.iter().enumerate(){
            for &idx in group.iter(){
                ans[idx] = i;
            }
        }
        if debug{
            println!("#c {}", ans.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(" "));
        } else {
            println!("{}", ans.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(" "));
        }
    }

    fn make_querys(&mut self) -> Vec<(Vec<usize>, Vec<usize>)>{
        let mut rng = rand::thread_rng();
        if self.turn<3{
            // 0..3で最も軽いのを見つける
            let mut lrss = vec![(vec![0], vec![1]), (vec![0], vec![2]), (vec![1], vec![2])];
            return lrss;
        } else if self.turn<self.n{
            // 軽いのに目星つける
            let mut lrss = vec![];
            let minidx = most_frequent(self.measured_record.iter().take(3).map(|(ls, rs, flg)| rs[0]).collect_vec()).unwrap();
            for i in 3..self.n{
                lrss.push((vec![minidx], vec![i]));
            }
            return lrss;
        } else if self.turn<self.n+(self.q-self.n)/4{
            // 重いのでまずグループ分け
            // 適当に初期化
            if self.phase==0{
                let minidx = most_frequent(self.measured_record.iter().take(3).map(|(ls, rs, flg)| rs[0]).collect_vec()).unwrap();
                let mut heavys = HashSet::new();
                for (ls, rs, flg) in self.measured_record.iter(){
                    if rs[0]==minidx{
                        heavys.insert(ls[0]);
                    }
                }
                self.groups = vec![HashSet::new(); self.d];
                for (i, x) in heavys.iter().enumerate(){
                    self.groups[i%self.d].insert(*x);
                }
                self.phase +=1;
            }
            // toposo
            let sorted_idx = self.toporogical_sort_for_group();
            // 右から左へ渡す
            eprintln!("{:?}", sorted_idx);
            let ls_idx = *sorted_idx.first().unwrap();
            let rs_idx = *sorted_idx.last().unwrap();
            let item_idx = *self.groups[rs_idx].iter().choose(&mut rng).unwrap();
            self.groups[ls_idx].insert(item_idx);
            self.groups[rs_idx].remove(&item_idx);
            // 真ん中と比較させる
            let mut lrss = vec![];
            if self.d==2{
                lrss.push((self.groups[ls_idx].iter().map(|&x| x).collect_vec(), self.groups[rs_idx].iter().map(|&x| x).collect_vec()));
            } else {
                let mid_idx = *sorted_idx.get(self.d/2).unwrap();
                lrss.push((self.groups[mid_idx].iter().map(|&x| x).collect_vec(), self.groups[rs_idx].iter().map(|&x| x).collect_vec()));
                lrss.push((self.groups[mid_idx].iter().map(|&x| x).collect_vec(), self.groups[ls_idx].iter().map(|&x| x).collect_vec()));
            }
            return lrss;
        } else {
            // 軽いの含めてグループ分け
            // 軽いの求める
            let minidx = most_frequent(self.measured_record.iter().take(3).map(|(ls, rs, flg)| rs[0]).collect_vec()).unwrap();
            let mut lights = HashSet::new();
            lights.insert(minidx);
            for (ls, rs, flg) in self.measured_record.iter().take(self.n){
                if ls[0]==minidx{
                    lights.insert(ls[0]);
                }
            }
            // 適当に初期化
            if self.phase==1{
                for (i, x) in lights.iter().enumerate(){
                    self.groups[i%self.d].insert(*x);
                }
                self.phase +=1;
            }
            // toposo
            let sorted_idx = self.toporogical_sort_for_group();
            // 右から左へ渡す
            eprintln!("{:?}", sorted_idx);
            let ls_idx = *sorted_idx.first().unwrap();
            let rs_idx = *sorted_idx.last().unwrap();
            if let Some(&item_idx) = lights.intersection(&self.groups[rs_idx]).choose(&mut rng){
                self.groups[ls_idx].insert(item_idx);
                self.groups[rs_idx].remove(&item_idx);
            } else {
                let item_idx = *self.groups[rs_idx].iter().choose(&mut rng).unwrap();
                self.groups[ls_idx].insert(item_idx);
                self.groups[rs_idx].remove(&item_idx);
            }
            // 真ん中と比較させる
            let mut lrss = vec![];
            if self.d==2{
                lrss.push((self.groups[ls_idx].iter().map(|&x| x).collect_vec(), self.groups[rs_idx].iter().map(|&x| x).collect_vec()));
            } else {
                let mid_idx = *sorted_idx.get(self.d/2).unwrap();
                lrss.push((self.groups[mid_idx].iter().map(|&x| x).collect_vec(), self.groups[rs_idx].iter().map(|&x| x).collect_vec()));
                lrss.push((self.groups[mid_idx].iter().map(|&x| x).collect_vec(), self.groups[ls_idx].iter().map(|&x| x).collect_vec()));
            }
            return lrss;
        }
        vec![]
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
        self.print_group(true);
    }

}


fn main() {
    let timer = Instant::now();
    let mut measure = Measure::new();
    let tl = 1.5;
    measure.solve(&timer, tl);
    eprintln!("{}", timer.elapsed().as_secs_f64());
}
