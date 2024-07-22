#![allow(non_snake_case)]

use proconio::{marker::Chars, *};
use rand::prelude::*;

fn main() {
    get_time();
    let input = read_input();
    let out = solve(&input);
    write_output(&input, &out);
    eprintln!("Time = {:.3}", get_time());
}

const TL: f64 = 1.9;
const T0: f64 = 0.01;
const T1: f64 = 0.001;

fn solve(input: &Input) -> Vec<char> {
    let n = input.n;
    let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(4732809);
    let mut s = State {
        bs: input.bs.clone(),
        n: input.n,
        si: input.si,
        sj: input.sj,
        visited: vec![0; n * n * 4],
        iter: 0,
        w: 0.0,
        score: 0,
    };
    let mut crt_score = s.evaluate(1.0);
    let mut best = s.bs.clone();
    let mut best_score = s.score;
    eprintln!("{:.3}: {:.0}", get_time(), best_score);
    loop {
        let t = get_time() / TL;
        if t >= 1.0 {
            break;
        }
        let n0 = (n as f64 * t * 0.9).round() as usize;
        let i = rng.gen_range(0, n);
        let j = rng.gen_range(0, n);
        let p = i * n + j;
        if p == input.si * n + input.sj || s.bs[p] == '#' || i.min(n - 1 - i) > n0 && j.min(n - 1 - j) > n0 {
            continue;
        }
        if s.bs[p] == '.' && rng.gen_bool(0.5) {
            continue;
        }
        let T = T0.powf(1.0 - t) * T1.powf(t);
        s.change(i, j);
        let score = s.evaluate((1.0 - t * 1.1).max(0.0));
        if crt_score < score || rng.gen_bool(((score - crt_score) as f64 / (crt_score as f64 + 0.1) / T).exp()) {
            crt_score = score;
            if best_score.setmax(s.score) {
                best = s.bs.clone();
                eprintln!("{:.3}: {}", get_time(), s.score);
            }
        } else {
            s.change(i, j);
        }
    }
    eprintln!("Iter = {}", s.iter);
    best
}

struct State {
    n: usize,
    si: usize,
    sj: usize,
    bs: Vec<char>,
    visited: Vec<i32>,
    iter: i32,
    w: f64,
    score: i32,
}

const W: f64 = 0.3;
const DIJ: [(usize, usize); 4] = [(0, 1), (1, 0), (0, !0), (!0, 0)];

impl State {
    fn change(&mut self, i: usize, j: usize) {
        let p = i * self.n + j;
        if self.bs[p] == '.' {
            self.bs[p] = 'o';
            if 2 <= i && i < self.n - 2 && 2 <= j && j < self.n - 2 {
                self.w -= self.n as f64 * W;
            }
        } else {
            self.bs[p] = '.';
            if 2 <= i && i < self.n - 2 && 2 <= j && j < self.n - 2 {
                self.w += self.n as f64 * W;
            }
        }
    }
    fn evaluate(&mut self, weight: f64) -> f64 {
        self.iter += 1;
        let n = self.n;
        let mut dir = 0;
        let mut pi = self.si;
        let mut pj = self.sj;
        let mut t = 0;
        loop {
            if !self.visited[(pi * n + pj) * 4 + dir].setmax(self.iter) {
                break;
            }
            let qi = pi + DIJ[dir].0;
            let qj = pj + DIJ[dir].1;
            if qi < n && qj < n && self.bs[qi * n + qj] == '.' {
                pi = qi;
                pj = qj;
                t += 1;
            } else {
                dir = (dir + 1) % 4;
            }
        }
        self.score = t;
        if weight > 0.0 {
            t as f64 + self.w as f64 * (weight * 10.0).round() * 1e-1
        } else {
            t as f64
        }
    }
}

// 入出力と得点計算

#[derive(Clone, Debug)]
struct Input {
    n: usize,
    si: usize,
    sj: usize,
    bs: Vec<char>,
}

fn read_input() -> Input {
    input! {
        n: usize,
        si: usize,
        sj: usize,
        bs: [Chars; n],
    }
    Input {
        n,
        si,
        sj,
        bs: bs.into_iter().flatten().collect(),
    }
}

fn write_output(input: &Input, bs: &Vec<char>) {
    let n = input.n;
    let mut out = vec![];
    for i in 0..n {
        for j in 0..n {
            if input.bs[i * n + j] != bs[i * n + j] {
                out.push((i, j));
            }
        }
    }
    println!("{}", out.len());
    for (i, j) in out {
        println!("{} {}", i, j);
    }
}

// ここからライブラリ

pub trait SetMinMax {
    fn setmin(&mut self, v: Self) -> bool;
    fn setmax(&mut self, v: Self) -> bool;
}
impl<T> SetMinMax for T
where
    T: PartialOrd,
{
    fn setmin(&mut self, v: T) -> bool {
        *self > v && {
            *self = v;
            true
        }
    }
    fn setmax(&mut self, v: T) -> bool {
        *self < v && {
            *self = v;
            true
        }
    }
}

#[macro_export]
macro_rules! mat {
	($($e:expr),*) => { vec![$($e),*] };
	($($e:expr,)*) => { vec![$($e),*] };
	($e:expr; $d:expr) => { vec![$e; $d] };
	($e:expr; $d:expr $(; $ds:expr)+) => { vec![mat![$e $(; $ds)*]; $d] };
}

pub fn get_time() -> f64 {
    static mut STIME: f64 = -1.0;
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap();
    let ms = t.as_secs() as f64 + t.subsec_nanos() as f64 * 1e-9;
    unsafe {
        if STIME < 0.0 {
            STIME = ms;
        }
        // ローカル環境とジャッジ環境の実行速度差はget_timeで吸収しておくと便利
        #[cfg(feature = "local")]
        {
            (ms - STIME) * 1.3
        }
        #[cfg(not(feature = "local"))]
        {
            ms - STIME
        }
    }
}
