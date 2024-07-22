use grid::{Coordinate, Map2d, ADJACENTS};
use proconio::marker::Chars;
#[allow(unused_imports)]
use proconio::*;
#[allow(unused_imports)]
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use std::time::Instant;

pub trait ChangeMinMax {
    fn change_min(&mut self, v: Self) -> bool;
    fn change_max(&mut self, v: Self) -> bool;
}

impl<T: PartialOrd> ChangeMinMax for T {
    fn change_min(&mut self, v: T) -> bool {
        *self > v && {
            *self = v;
            true
        }
    }

    fn change_max(&mut self, v: T) -> bool {
        *self < v && {
            *self = v;
            true
        }
    }
}

#[derive(Debug, Clone)]
struct Input {
    n: usize,
    start: Coordinate,
    map: Map2d<bool>,
    since: Instant,
}

impl Input {
    fn read_input() -> Input {
        input! {
            n: usize,
            si: usize,
            sj: usize
        }

        let since = Instant::now();
        let start = Coordinate::new(si, sj);

        let mut map = Map2d::new(vec![false; n * n], n);

        for row in 0..n {
            input! {
                s: Chars
            }

            for col in 0..n {
                let b = s[col] == '#';
                map[Coordinate::new(row, col)] = b;
            }
        }

        Self {
            n,
            start,
            map,
            since,
        }
    }

    fn to_index(&self, c: Coordinate, dir: usize) -> usize {
        ((c.row * self.n) + c.col) * 4 + dir
    }
}

static mut SEEN: [u32; 10000] = [0; 10000];
static mut ITERATION: u32 = 0;

#[derive(Debug, Clone)]
struct Output {
    map: Map2d<bool>,
}

impl Output {
    fn new(input: &Input) -> Self {
        Self {
            map: input.map.clone(),
        }
    }

    fn calc_score(&self, input: &Input) -> i32 {
        unsafe {
            ITERATION += 1;
            let mut place = input.start;
            let mut dir = 1; // Right

            SEEN[input.to_index(place, dir)] = ITERATION;
            let mut dist = 0;

            'main_loop: loop {
                let prev_dir = dir;

                loop {
                    let next = place + ADJACENTS[dir];
                    if !next.in_map(input.n) || self.map[next] {
                        dir = (dir + 1) % 4;

                        if dir == prev_dir {
                            break 'main_loop;
                        }
                    } else {
                        break;
                    }
                }

                place = place + ADJACENTS[dir];

                if SEEN[input.to_index(place, dir)] == ITERATION {
                    break;
                } else {
                    SEEN[input.to_index(place, dir)] = ITERATION;
                    dist += 1;
                }
            }

            dist
        }
    }

    fn write(&self, input: &Input) {
        let mut results = vec![];

        for row in 0..input.n {
            for col in 0..input.n {
                let c = Coordinate::new(row, col);

                if input.map[c] ^ self.map[c] {
                    results.push(c);
                }
            }
        }

        println!("{}", results.len());

        for c in results.iter() {
            println!("{} {}", c.row, c.col);
        }
    }
}

fn main() {
    let input = Input::read_input();
    let output = annealing(&input, Output::new(&input), 1.9);
    output.write(&input);
}

fn annealing(input: &Input, initial_solution: Output, duration: f64) -> Output {
    let mut solution = initial_solution;
    let mut best_solution = solution.clone();
    let mut current_score = solution.calc_score(input);
    let mut best_score = current_score;
    let init_score = current_score;

    let mut all_iter = 0;
    let mut valid_iter = 0;
    let mut accepted_count = 0;
    let mut update_count = 0;
    let mut rng = rand_pcg::Pcg64Mcg::new(42);

    let duration_inv = 1.0 / duration;
    let since = std::time::Instant::now();

    let temp0 = 3e1;
    let temp1 = 1e-1;
    let mut inv_temp = 1.0 / temp0;

    loop {
        all_iter += 1;
        if (all_iter & ((1 << 4) - 1)) == 0 {
            let time = (std::time::Instant::now() - since).as_secs_f64() * duration_inv;
            let temp = f64::powf(temp0, 1.0 - time) * f64::powf(temp1, time);
            inv_temp = 1.0 / temp;

            if time >= 1.0 {
                break;
            }
        }

        // 変形
        let row = rng.gen_range(0, input.n);
        let col = rng.gen_range(0, input.n);

        let c = Coordinate::new(row, col);

        if c == input.start || input.map[c] {
            continue;
        }

        solution.map[c] ^= true;

        // スコア計算
        let new_score = solution.calc_score(input);
        let score_diff = new_score - current_score;

        if score_diff >= 0 || rng.gen_bool(f64::exp(score_diff as f64 * inv_temp)) {
            // 解の更新
            current_score = new_score;
            accepted_count += 1;

            if best_score.change_max(current_score) {
                best_solution = solution.clone();
                update_count += 1;
            }
        } else {
            solution.map[c] ^= true;
        }

        valid_iter += 1;
    }

    eprintln!("===== annealing =====");
    eprintln!("init score : {}", init_score);
    eprintln!("score      : {}", best_score);
    eprintln!("all iter   : {}", all_iter);
    eprintln!("valid iter : {}", valid_iter);
    eprintln!("accepted   : {}", accepted_count);
    eprintln!("updated    : {}", update_count);
    eprintln!("");

    best_solution
}

#[allow(dead_code)]
mod grid {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
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

        pub const fn to_index(&self, size: usize) -> usize {
            self.row * size + self.col
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
