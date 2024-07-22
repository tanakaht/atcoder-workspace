#![allow(non_snake_case)]

use proconio::*;
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;

fn main() {
    get_time();
    let input = read_input();
    let out = solve(&input);
    eprintln!("Time = {:.3}", get_time());
    write_output(&out);
}

const TL: f64 = 4.9;
const T0: f64 = 0.015;
const T1: f64 = 0.0025;

/// ランダムな領域を破壊して作り直す焼き鈍し
fn solve(input: &Input) -> Output {
    // 外側ほど高い重みを用いると焼き鈍しの途中段階で外側に広げすぎてしまうので序盤は全重み1として最適化
    // random playoutの性能を上げたら最終的にそこまで重要でなかった
    let mut S = 0;
    let mut S2 = 0;
    for i in 0..input.N {
        for j in 0..input.N {
            S += weight((i as i32, j as i32), input.N as i32);
            S2 += weight2((i as i32, j as i32), input.N as i32);
        }
    }
    S2 *= 3; // 全重み1だと正規化時のスコアが高くなるので適当に調整
    let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(4823902);
    let mut crt_state = State::new(input);
    let mut crt: Vec<Rect> = vec![];
    let mut best = vec![];
    let mut best_score = normalized_score(crt_state.score, &input, S);
    let mut iter = 0;
    let mut is_deleted = mat![0; input.N * input.N];
    // Mの小さい入力はぎっしり詰めるより外側に広げる感じの解になるので最初から正規の重みで最適化
    let t0 = if input.ps.len() <= (0.25 * (input.N as f64).powf(1.5)) as usize {
        0.0
    } else {
        0.9
    };
    while get_time() < TL {
        iter += 1;
        let t = get_time() / TL;
        // 小さい入力の方が反復回数が稼げるので温度高めにした
        let T = T0.powf(1.0 - t) * T1.powf(t) * 61.0 / input.N as f64;
        if crt.len() > 0 {
            // ランダムな中心(cx,cy)からランダムなマンハッタン距離wの範囲にある印を全削除
            let NUM = (best_score * 1.2e-4).round() as u32;
            let w = rng.gen_range(0, input.N as i32 / 6);
            let cx = rng.gen_range(w, input.N as i32 - w);
            let cy = rng.gen_range(w, input.N as i32 - w);
            let num0 = (-w..=w)
                .map(|dy| {
                    let w = w - dy.abs();
                    let lx = cx - w;
                    (crt_state.has_point[0][(cy + dy) as usize] >> lx << (64 - 2 * w - 1)).count_ones()
                })
                .sum::<u32>();
            // 沢山消しすぎるとまともに再構築出来る確率が低い上に時間がかかるので選び直し
            if num0 == 0 || num0 >= NUM {
                continue;
            }
            let mut num = 0;
            // 消した印を使う長方形も連鎖的に削除
            for &rect in &crt {
                let (x, y) = rect.0[0];
                if (x - cx).abs() + (y - cy).abs() <= w
                    || rect.0[1..]
                        .iter()
                        .any(|&(x, y)| is_deleted[x as usize * input.N + y as usize] == iter)
                {
                    is_deleted[x as usize * input.N + y as usize] = iter;
                    num += 1;
                    if num >= NUM {
                        break;
                    }
                }
            }
            if num == 0 || num >= NUM {
                continue;
            }
        }
        let deleted_rects = crt
            .iter()
            .filter(|r| is_deleted[r.0[0].0 as usize * input.N + r.0[0].1 as usize] == iter)
            .cloned();
        let mut state = crt_state.clone();
        state.delete_rects(deleted_rects);
        let ps1 = crt
            .iter()
            .filter(|rect| is_deleted[rect.0[0].0 as usize * input.N + rect.0[0].1 as usize] != iter)
            .cloned()
            .collect::<Vec<_>>();
        state.init_cands(input.ps.iter().cloned().chain(ps1.iter().map(|r| r.0[0])));
        // 部分的に削除した状態から、random playoutで解を再構築
        let ps2 = state.playout(&mut rng);
        // 序盤は重み1で最適化して時刻に応じて徐々に正規の重みに近づけている
        let w1 = if t < t0 { t / t0 } else { 1.0 };
        let crt_score =
            normalized_score(crt_state.score, &input, S) * w1 + normalized_score(crt_state.score2, &input, S2) * (1.0 - w1);
        let score = normalized_score(state.score, &input, S) * w1 + normalized_score(state.score2, &input, S2) * (1.0 - w1);
        if best_score.setmax(normalized_score(state.score, &input, S)) {
            best = ps1.clone().into_iter().chain(ps2.clone().into_iter()).collect();
            eprintln!("{:.3}: {:.0}", get_time(), best_score);
        }
        if crt_score <= score || rng.gen_bool(((score - crt_score) as f64 / crt_score / T).exp()) {
            crt_state = state;
            crt = ps1.into_iter().chain(ps2.into_iter()).collect();
        }
    }
    eprintln!("iter = {}", iter);
    best
}

#[derive(Clone, Debug)]
struct State {
    N: i32,
    /// ある範囲に印が存在するかを高速に判定するため、ビットベクトルを用いて状態を表現。
    /// (x,y)に印がある場合、以下のビットが1となる。
    /// ```
    /// has_point[0][y][x]
    /// has_point[1][-x+y][(x+y)/2]
    /// has_point[2][-x][y]
    /// has_point[3][-x-y][(-x+y)/2]
    /// ```
    /// ただし、負のインデックスを回避するために-xはN-1-x、-yはN-1-yで置き換える。
    has_point: [Vec<u64>; 4],
    /// (x,y)に印があり、d方向に長方形の辺が伸びている場合にhas_edge[Nx+y]のdビット目が1となる。
    has_edge: Vec<u8>,
    /// 高速化のために、次に描画出来る長方形(と1点目から見た2点目の向き)の候補を保持しておく。
    /// 新たな長方形を描画後に候補の追加のみで削除はしないため、不可能になったものも含まれており、実際に描画時に可能かを再度判定する。
    /// 良い解では1x1の正方形を多用するため、1x1とそれ以外で区別して保持している。
    cand: [Vec<Rect>; 2],
    /// 正規の重みの和
    score: i32,
    /// 重み1の和(=印の個数)
    score2: i32,
}

impl State {
    fn new(input: &Input) -> Self {
        let mut state = Self {
            N: input.N as i32,
            has_point: [
                vec![0; input.N],
                vec![0; input.N * 2 - 1],
                vec![0; input.N],
                vec![0; input.N * 2 - 1],
            ],
            has_edge: vec![0; input.N * input.N],
            cand: [vec![], vec![]],
            score: 0,
            score2: 0,
        };
        for &p in &input.ps {
            state.insert_point(p);
        }
        state
    }
    fn on_board(&self, (x, y): P) -> bool {
        0 <= x && x < self.N && 0 <= y && y < self.N
    }
    fn has_point(&self, (x, y): P) -> bool {
        self.has_point[0][y as usize] >> x & 1 != 0
    }
    fn has_edge(&self, (x, y): P, mask: u8) -> bool {
        self.has_edge[(x * self.N + y) as usize] & mask != 0
    }
    /// (x, y) に印を付けてスコアを更新
    fn insert_point(&mut self, (x, y): P) {
        let nx = self.N - 1 - x;
        let ny = self.N - 1 - y;
        self.has_point[0][y as usize] |= 1 << x;
        self.has_point[1][(nx + y) as usize] |= 1 << ((x + y) / 2);
        self.has_point[2][nx as usize] |= 1 << y;
        self.has_point[3][(nx + ny) as usize] |= 1 << ((nx + y) / 2);
        self.score += weight((x, y), self.N);
        self.score2 += weight2((x, y), self.N);
    }
    fn insert_edge(&mut self, (x, y): P, mask: u8) {
        self.has_edge[(x * self.N + y) as usize] |= mask;
    }
    /// (x, y) の印と辺を除去してスコアを更新
    fn remove_point(&mut self, (x, y): P) {
        let nx = self.N - 1 - x;
        let ny = self.N - 1 - y;
        self.has_point[0][y as usize] ^= 1 << x;
        self.has_point[1][(nx + y) as usize] ^= 1 << ((x + y) / 2);
        self.has_point[2][nx as usize] ^= 1 << y;
        self.has_point[3][(nx + ny) as usize] ^= 1 << ((nx + y) / 2);
        self.has_edge[(x * self.N + y) as usize] = 0;
        self.score -= weight((x, y), self.N);
        self.score2 -= weight2((x, y), self.N);
    }
    fn remove_edge(&mut self, (x, y): P, mask: u8) {
        self.has_edge[(x * self.N + y) as usize] &= !mask;
    }
    /// 描画可能な長方形の候補を追加する。1x1の場合はtier0にそれ以外はtier1に保存される。
    fn insert_cand(&mut self, (rect, d): Rect) {
        let tier = if (rect[2].0 - rect[0].0).abs() + (rect[2].1 - rect[0].1).abs() == 2 {
            0
        } else {
            1
        };
        self.cand[tier].push((rect, d));
    }
    /// (x, y) と (x2, y2) の間に他の点があるかを判定
    fn has_point_in_range(&self, (x, y): P, (x2, y2): P, d: usize) -> bool {
        let w = (x - x2).abs() + (y - y2).abs();
        let nx = self.N - 1 - x;
        let ny = self.N - 1 - y;
        match d {
            0 => w >= 2 && self.has_point[0][y as usize] >> (x + 1) << (65 - w) != 0,
            1 => w >= 4 && self.has_point[1][(nx + y) as usize] >> ((x + y) / 2 + 1) << (65 - w / 2) != 0,
            2 => w >= 2 && self.has_point[2][nx as usize] >> (y + 1) << (65 - w) != 0,
            3 => w >= 4 && self.has_point[3][(nx + ny) as usize] >> ((nx + y) / 2 + 1) << (65 - w / 2) != 0,
            4 => w >= 2 && self.has_point[0][y as usize] << (64 - x) >> (65 - w) != 0,
            5 => w >= 4 && self.has_point[1][(nx + y) as usize] << (64 - (x + y) / 2) >> (65 - w / 2) != 0,
            6 => w >= 2 && self.has_point[2][nx as usize] << (64 - y) >> (65 - w) != 0,
            7 => w >= 4 && self.has_point[3][(nx + ny) as usize] << (64 - (nx + y) / 2) >> (65 - w / 2) != 0,
            _ => unreachable!(),
        }
    }
    /// pからd方向に進んだ最初の印(pは除く)の座標を返す
    fn next_point(&self, (x, y): P, d: usize) -> Option<P> {
        let nx = self.N - 1 - x;
        let ny = self.N - 1 - y;
        match d {
            0 => {
                let k = (self.has_point[0][y as usize] >> (x + 1)).trailing_zeros();
                if k == 64 {
                    None
                } else {
                    Some((x + 1 + k as i32, y))
                }
            }
            1 => {
                let k = (self.has_point[1][(nx + y) as usize] >> ((x + y) / 2 + 1)).trailing_zeros();
                if k == 64 {
                    None
                } else {
                    Some((x + 1 + k as i32, y + 1 + k as i32))
                }
            }
            2 => {
                let k = (self.has_point[2][nx as usize] >> (y + 1)).trailing_zeros();
                if k == 64 {
                    None
                } else {
                    Some((x, y + 1 + k as i32))
                }
            }
            3 => {
                let k = (self.has_point[3][(nx + ny) as usize] >> ((nx + y) / 2 + 1)).trailing_zeros();
                if k == 64 {
                    None
                } else {
                    Some((x - 1 - k as i32, y + 1 + k as i32))
                }
            }
            4 => {
                if x == 0 {
                    None
                } else {
                    let k = (self.has_point[0][y as usize] << (64 - x)).leading_zeros();
                    if k == 64 {
                        None
                    } else {
                        Some((x - 1 - k as i32, y))
                    }
                }
            }
            5 => {
                if (x + y) / 2 == 0 {
                    None
                } else {
                    let k = (self.has_point[1][(nx + y) as usize] << (64 - (x + y) / 2)).leading_zeros();
                    if k == 64 {
                        None
                    } else {
                        Some((x - 1 - k as i32, y - 1 - k as i32))
                    }
                }
            }
            6 => {
                if y == 0 {
                    None
                } else {
                    let k = (self.has_point[2][nx as usize] << (64 - y)).leading_zeros();
                    if k == 64 {
                        None
                    } else {
                        Some((x, y - 1 - k as i32))
                    }
                }
            }
            7 => {
                if (nx + y) / 2 == 0 {
                    None
                } else {
                    let k = (self.has_point[3][(nx + ny) as usize] << (64 - (nx + y) / 2)).leading_zeros();
                    if k == 64 {
                        None
                    } else {
                        Some((x + 1 + k as i32, y - 1 - k as i32))
                    }
                }
            }
            _ => unreachable!(),
        }
    }
    /// 長方形が描画可能か判定。
    /// rect[0]の盤面内判定とrect[1~3]に印があるかの判定は既に済んでいるとしてここでは行っていない。
    fn can_draw(&self, (rect, d): Rect) -> bool {
        if self.has_point(rect[0]) {
            return false;
        }
        for k in 0..4 {
            let p = rect[k];
            let q = rect[(k + 1) & 3];
            let d = (d + 2 * k) & 7;
            if self.has_edge(p, 1 << d | 1 << ((d + 2) & 7)) || self.has_point_in_range(p, q, d) {
                return false;
            }
        }
        true
    }
    /// 長方形を描画し、辺の情報を更新する
    fn draw(&mut self, (rect, d): Rect) {
        let p = rect[0];
        self.insert_point(p);
        // 既存の長方形の辺上に印を追加した場合に、その長方形の辺を追加
        if let Some(q) = self.next_point(p, (d + 1) & 7) {
            if self.has_edge(q, 1 << ((d + 5) & 7)) {
                self.insert_edge(p, 1 << ((d + 1) & 7) | 1 << ((d + 5) & 7))
            }
        }
        if let Some(q) = self.next_point(p, (d + 3) & 7) {
            if self.has_edge(q, 1 << ((d + 7) & 7)) {
                self.insert_edge(p, 1 << ((d + 3) & 7) | 1 << ((d + 7) & 7))
            }
        }
        // 新たに描画する長方形の辺を追加
        for k in 0..4 {
            self.insert_edge(rect[k], 1 << ((d + k * 2) & 7) | 1 << ((d + k * 2 + 2) & 7));
        }
    }
    /// 新たに追加する印が既に描画されている長方形の辺上に乗っているか判定
    fn on_edge(&self, (rect, d): Rect) -> bool {
        let p = rect[0];
        if let Some(q) = self.next_point(p, (d + 1) & 7) {
            if self.has_edge(q, 1 << ((d + 5) & 7)) {
                return true;
            }
        }
        if let Some(q) = self.next_point(p, (d + 3) & 7) {
            if self.has_edge(q, 1 << ((d + 7) & 7)) {
                return true;
            }
        }
        false
    }
    fn check_and_insert_cand(&mut self, rect: Rect) {
        if self.on_board(rect.0[0]) && self.can_draw(rect) {
            self.insert_cand(rect);
        }
    }
    /// 新たに印を付けた点pを用いることで描画可能となった長方形の候補を追加する
    fn insert_new_cands(&mut self, p: P) {
        let mut next = [None; 8];
        for d in 0..8 {
            if !self.has_edge(p, 1 << d) {
                next[d] = self.next_point(p, d);
            }
        }
        // 新たに追加した印を頂点とする長方形の候補を追加
        for d in 0..8 {
            if let Some(q) = next[d] {
                if let Some(p2) = next[(d + 2) & 7] {
                    let p1 = (p2.0 + q.0 - p.0, p2.1 + q.1 - p.1);
                    self.check_and_insert_cand(([p1, p2, p, q], (d + 4) & 7));
                }
                if let Some(p4) = self.next_point(q, (d + 2) & 7) {
                    let p1 = (p.0 + p4.0 - q.0, p.1 + p4.1 - q.1);
                    self.check_and_insert_cand(([p1, p, q, p4], (d + 6) & 7));
                }
                if let Some(p2) = self.next_point(q, (d + 6) & 7) {
                    let p1 = (p.0 + p2.0 - q.0, p.1 + p2.1 - q.1);
                    self.check_and_insert_cand(([p1, p2, q, p], d));
                }
            }
        }
    }
    /// 描画可能な長方形の候補を一から列挙する。
    /// 印の集合は内部で保持していないので引数で渡す。
    fn init_cands<PS: Iterator<Item = P>>(&mut self, ps: PS) {
        for i in 0..self.cand.len() {
            self.cand[i].clear();
        }
        let mut next = [None; 8];
        for p3 in ps {
            if self.has_edge[(p3.0 * self.N + p3.1) as usize] == 255 {
                continue;
            }
            for d in 0..8 {
                if !self.has_edge(p3, 1 << d) {
                    next[d] = self.next_point(p3, d);
                } else {
                    next[d] = None;
                }
            }
            for d in 0..8 {
                if let (Some(p4), Some(p2)) = (next[d], next[(d + 2) & 7]) {
                    let p1 = (p2.0 + p4.0 - p3.0, p2.1 + p4.1 - p3.1);
                    if self.on_board(p1)
                        && !self.has_point(p1)
                        && !self.has_edge(p2, 1 << d)
                        && !self.has_edge(p4, 1 << ((d + 2) & 7))
                        && !self.has_point_in_range(p2, p1, d)
                        && !self.has_point_in_range(p4, p1, (d + 2) & 7)
                    {
                        let rect = ([p1, p2, p3, p4], (d + 4) & 7);
                        self.insert_cand(rect);
                    }
                }
            }
        }
    }
    /// 指定した長方形を除去する
    fn delete_rects<RS: DoubleEndedIterator<Item = Rect>>(&mut self, deletes: RS) {
        for (rect, d) in deletes.rev() {
            for k in 0..4 {
                self.remove_edge(rect[k], 1 << ((d + k * 2) & 7) | 1 << ((d + k * 2 + 2) & 7));
                if self.has_point_in_range(rect[k], rect[(k + 1) & 3], (d + k * 2) & 7) {
                    let mut p = rect[k];
                    loop {
                        p = self.next_point(p, (d + k * 2) & 7).unwrap();
                        if p == rect[(k + 1) & 3] {
                            break;
                        }
                        self.remove_edge(p, 1 << ((d + k * 2) & 7) | 1 << ((d + k * 2 + 4) & 7));
                    }
                }
            }
            self.remove_point(rect[0]);
        }
    }
    /// 描画可能な長方形のうち、どれを次に描画するかの決定のための評価値を計算する。
    fn eval(&self, (rect, d): Rect) -> i32 {
        let mut score = 0;
        for k in 0..4 {
            // 長方形の各頂点について、180度逆側の2辺が既に描画されている場合、評価値+1
            let p = rect[k];
            let d = (d + 2 * k) & 7;
            if self.has_edge(p, 1 << ((d + 4) & 7)) {
                score += 1;
            }
        }
        // 新たに追加する印が既に描画した辺上の場合、将来的な使用可能回数が2回分減るので評価値-2
        if self.on_edge((rect, d)) {
            score -= 2;
        }
        score
    }
    /// ランダムに最後までプレイする。
    /// 1x1の正方形を優先して使用する。
    fn playout(&mut self, rng: &mut Pcg64Mcg) -> Vec<Rect> {
        let mut out = vec![];
        let mut cand = vec![];
        while self.cand[0].len() + self.cand[1].len() > 0 {
            // 最初の一手は最小サイズを優先しない
            let r = if out.len() == 0 {
                if rng.gen_range(0, self.cand[0].len() + self.cand[1].len()) < self.cand[0].len() {
                    0
                } else {
                    1
                }
            } else {
                0
            };
            for tier in 0..self.cand.len() {
                let tier = (tier + r) % self.cand.len();
                // ランダムに最大4つ候補を取り出す
                while self.cand[tier].len() > 0 && cand.len() < 4 {
                    let k = rng.gen_range(0, self.cand[tier].len());
                    let rect = self.cand[tier].swap_remove(k);
                    if self.can_draw(rect) {
                        cand.push(rect);
                        // 最初の一手は最初に取り出した候補で決定
                        if out.len() == 0 {
                            break;
                        }
                    }
                }
                if cand.len() > 0 {
                    // 評価値の一番高い候補を使用し、残りは戻す
                    let mut rect = cand.pop().unwrap();
                    let mut score = self.eval(rect);
                    while let Some(rect2) = cand.pop() {
                        if score.setmax(self.eval(rect2)) {
                            self.cand[tier].push(rect);
                            rect = rect2;
                        } else {
                            self.cand[tier].push(rect2);
                        }
                    }
                    out.push(rect);
                    self.draw(rect);
                    self.insert_new_cands(rect.0[0]);
                    break;
                }
            }
        }
        out
    }
}

// 入出力と得点計算

// const DXY: [(i32, i32); 8] = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)];

fn weight((x, y): P, N: i32) -> i32 {
    let dx = x - N / 2;
    let dy = y - N / 2;
    dx * dx + dy * dy + 1
}

#[allow(unused)]
fn weight2((x, y): P, N: i32) -> i32 {
    1
}

fn normalized_score(score: i32, input: &Input, S: i32) -> f64 {
    1e6 * input.N as f64 * input.N as f64 / input.ps.len() as f64 * score as f64 / S as f64
}

type P = (i32, i32);
/// 長方形の4頂点(反時計回り)と最初の辺の方向
type Rect = ([P; 4], usize);
type Output = Vec<Rect>;

#[derive(Clone, Debug)]
struct Input {
    N: usize,
    ps: Vec<P>,
}

fn read_input() -> Input {
    input! {
        N: usize, M: usize,
        ps: [(i32, i32); M],
    }
    Input { N, ps }
}

fn write_output(out: &Output) {
    println!("{}", out.len());
    for rect in out {
        for i in 0..4 {
            print!("{} {}", rect.0[i].0, rect.0[i].1);
            if i == 3 {
                println!();
            } else {
                print!(" ");
            }
        }
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
            (ms - STIME) * 1.5
        }
        #[cfg(not(feature = "local"))]
        {
            (ms - STIME)
        }
    }
}
