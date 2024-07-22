#![allow(non_snake_case, unused_macros)]

use std::collections::HashSet;

use itertools::Itertools;
use proconio::{input, marker::Chars};
use rand::prelude::*;
use svg::node::{
    element::{Group, Line, Rectangle, Title, SVG},
    Text,
};

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
	($($e:expr),*) => { Vec::from(vec![$($e),*]) };
	($($e:expr,)*) => { Vec::from(vec![$($e),*]) };
	($e:expr; $d:expr) => { Vec::from(vec![$e; $d]) };
	($e:expr; $d:expr $(; $ds:expr)+) => { Vec::from(vec![mat![$e $(; $ds)*]; $d]) };
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Input {
    pub n: usize,
    pub m: usize,
    pub p: (usize, usize),
    pub a: Vec<Vec<char>>,
    pub t: Vec<String>,
}

impl Input {
    pub fn new(n: usize, m: usize, p: (usize, usize), a: Vec<Vec<char>>, t: Vec<String>) -> Self {
        Self { n, m, p, a, t }
    }
}

impl std::fmt::Display for Input {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{} {}", self.n, self.m)?;
        writeln!(f, "{} {}", self.p.0, self.p.1)?;

        for a in self.a.iter() {
            writeln!(f, "{}", a.iter().collect::<String>())?;
        }

        for t in self.t.iter() {
            writeln!(f, "{}", t)?;
        }

        Ok(())
    }
}

pub fn parse_input(f: &str) -> Input {
    let f = proconio::source::once::OnceSource::from(f);
    input! {
        from f,
        n: usize,
        m: usize,
        p: (usize, usize),
        a: [Chars; n],
        t: [String; m]
    }

    Input::new(n, m, p, a, t)
}

pub fn read<T: Copy + PartialOrd + std::fmt::Display + std::str::FromStr>(
    token: Option<&str>,
    lb: T,
    ub: T,
) -> Result<T, String> {
    if let Some(v) = token {
        if let Ok(v) = v.parse::<T>() {
            if v < lb || ub < v {
                Err(format!("Out of range: {}", v))
            } else {
                Ok(v)
            }
        } else {
            Err(format!("Parse error: {}", v))
        }
    } else {
        Err("Unexpected EOF".to_owned())
    }
}

type Output = Vec<(usize, usize)>;
const MAX_OUT_LEN: usize = 5000;

#[derive(Debug, Clone)]
struct State {
    r: usize,
    c: usize,
    s: String,
    cost: i64,
}

impl State {
    fn new(input: &Input) -> Self {
        let (r, c) = input.p;
        Self {
            r,
            c,
            s: String::new(),
            cost: 0,
        }
    }

    fn move_to(&mut self, input: &Input, r: usize, c: usize) {
        let cost = self.r.abs_diff(r) + self.c.abs_diff(c) + 1;
        self.cost += cost as i64;
        self.r = r;
        self.c = c;
        self.s.push(input.a[r][c]);
    }
}

#[derive(Debug, Clone, Default)]
pub struct VisResult {
    pub s: String,
    pub cost: i64,
}

impl VisResult {
    pub fn new(s: String, cost: i64) -> Self {
        Self { s, cost }
    }
}

pub fn parse_output(input: &Input, f: &str) -> Result<Output, String> {
    let mut out = vec![];
    let mut tokens = f.split_whitespace().peekable();

    while tokens.peek().is_some() {
        let r = read(tokens.next(), 0, input.n - 1)?;
        let c = read(tokens.next(), 0, input.n - 1)?;
        out.push((r, c));
    }

    if out.len() > MAX_OUT_LEN {
        return Err("Too many output".to_owned());
    }

    Ok(out)
}

pub fn get_output_str(output: &Output) -> String {
    let mut out = String::new();

    for &(i, j) in output {
        out.extend(format!("{} {}\n", i, j).chars());
    }

    out
}

pub fn gen(seed: u64) -> Input {
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(seed);
    let n = 15;
    let m = 200;
    let p = (
        rng.gen_range(0..(n as u64)) as usize,
        rng.gen_range(0..(n as u64)) as usize,
    );

    let mut a = mat!['A'; n; n];

    loop {
        let mut set = HashSet::new();

        for i in 0..n {
            for j in 0..n {
                let c = rng.gen_range('A'..='Z');
                set.insert(c);
                a[i][j] = c;
            }
        }

        if set.len() == 26 {
            break;
        }
    }

    let mut s = vec![];

    for _ in 0..m {
        const L: usize = 5;

        loop {
            let mut si = String::new();

            for _ in 0..L {
                si.push(rng.gen_range('A'..='Z'));
            }

            if !s.contains(&si) {
                s.push(si);
                break;
            }
        }
    }

    s.sort_unstable();

    Input::new(n, m, p, a, s)
}

pub fn compute_score(input: &Input, out: &[(usize, usize)]) -> (i64, VisResult) {
    let mut state = State::new(input);

    for &(r, c) in out {
        state.move_to(input, r, c);
    }

    let word_count = input.t.iter().filter(|&t| state.s.contains(t)).count();

    let score = if word_count == input.m {
        (10000 - state.cost).max(1001)
    } else {
        (1000.0 * (word_count + 1) as f64 / input.m as f64).round() as i64
    };

    let vis_result = VisResult::new(state.s, state.cost);

    (score, vis_result)
}

/// 0 <= val <= 1
pub fn color(mut val: f64) -> String {
    val.setmin(1.0);
    val.setmax(0.0);
    let (r, g, b) = if val < 0.5 {
        let x = val * 2.0;
        (
            30. * (1.0 - x) + 144. * x,
            144. * (1.0 - x) + 255. * x,
            255. * (1.0 - x) + 30. * x,
        )
    } else {
        let x = val * 2.0 - 1.0;
        (
            144. * (1.0 - x) + 255. * x,
            255. * (1.0 - x) + 30. * x,
            30. * (1.0 - x) + 70. * x,
        )
    };
    format!(
        "#{:02x}{:02x}{:02x}",
        r.round() as i32,
        g.round() as i32,
        b.round() as i32
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisType {
    Walk(usize),
    Visit,
}

pub fn rect(x: f64, y: f64, w: f64, h: f64, fill: &str) -> Rectangle {
    Rectangle::new()
        .set("x", x)
        .set("y", y)
        .set("width", w)
        .set("height", h)
        .set("fill", fill)
}

fn text(x: f64, y: f64, size: f64, s: &str) -> svg::node::element::Text {
    svg::node::element::Text::new()
        .set("x", x)
        .set("y", y)
        .set("font-size", size)
        .set("text-anchor", "middle")
        .add(svg::node::Text::new(s))
}

pub fn group(title: String) -> Group {
    Group::new().add(Title::new().add(Text::new(title)))
}

pub fn vis_default(
    input: &Input,
    out: &[(usize, usize)],
) -> Result<(i64, VisResult, String), String> {
    vis(input, out, out.len(), VisType::Walk(out.len()))
}

pub const BOARD_SIZE: f64 = 600.0;
pub const VIEW_PADDING: f64 = 5.0;

pub fn vis(
    input: &Input,
    out: &[(usize, usize)],
    t: usize,
    option: VisType,
) -> Result<(i64, VisResult, String), String> {
    let t = t.min(out.len());
    let (score, vis_result) = compute_score(input, &out[..t]);

    let d = BOARD_SIZE / input.n as f64;
    let mut doc = svg::Document::new()
        .set("id", "vis")
        .set(
            "viewBox",
            (
                -VIEW_PADDING,
                -VIEW_PADDING,
                d * input.n as f64 + VIEW_PADDING * 2.0 + 693.,
                d * input.n as f64 + VIEW_PADDING * 2.0,
            ),
        )
        .set("width", d * input.n as f64 + VIEW_PADDING * 2.0 + 693.)
        .set("height", d * input.n as f64 + VIEW_PADDING * 2.0)
        .set("style", "background-color:white");

    doc = draw_board(input, out, t, d, option, doc);
    doc = draw_strings(input, out, t, doc);

    Ok((score, vis_result, doc.to_string()))
}

fn draw_board(
    input: &Input,
    out: &[(usize, usize)],
    t: usize,
    d: f64,
    option: VisType,
    mut doc: SVG,
) -> SVG {
    for i in 0..=input.n {
        doc = doc
            .add(
                Line::new()
                    .set("x1", 0)
                    .set("y1", i as f64 * d)
                    .set("x2", input.n as f64 * d)
                    .set("y2", i as f64 * d),
            )
            .set("stroke", "gray")
            .set("stroke-width", 1);

        doc = doc
            .add(
                Line::new()
                    .set("x1", i as f64 * d)
                    .set("y1", 0)
                    .set("x2", i as f64 * d)
                    .set("y2", input.n as f64 * d),
            )
            .set("stroke", "gray")
            .set("stroke-width", 1);
    }

    doc = match option {
        VisType::Walk(show_route) => draw_walk(input, out, t, show_route, d, doc),
        VisType::Visit => draw_visit(input, out, t, d, doc),
    };

    for i in 0..input.n {
        for j in 0..input.n {
            doc = doc.add(
                text(
                    j as f64 * d + d * 0.5,
                    i as f64 * d + d * 0.8,
                    d as f64 * 0.8,
                    &input.a[i][j].to_string(),
                )
                .set("stroke", "none")
                .set("style", "pointer-events: none"),
            );
        }
    }

    doc
}

fn draw_walk(
    input: &Input,
    out: &[(usize, usize)],
    t: usize,
    show_route: usize,
    d: f64,
    mut doc: SVG,
) -> SVG {
    let out = &out[..t];
    let mut moves = vec![input.p];
    moves.extend_from_slice(out);

    let (i, j) = *moves.last().unwrap();
    let x = j as f64 * d;
    let y = i as f64 * d;
    let c = "#FFF979".to_string();
    let rect = rect(x, y, d, d, &c);
    doc = doc.add(rect);

    let skip_count = t.saturating_sub(show_route);

    for (k, (&(i0, j0), &(i1, j1))) in moves.iter().tuple_windows().enumerate().skip(skip_count) {
        let cost = i0.abs_diff(i1) + j0.abs_diff(j1) + 1;
        let color = color(((cost as f64 - 2.0) / (input.n as f64 / 2.0)).clamp(0.0, 1.0));
        let x0 = j0 as f64 * d + d * 0.5;
        let y0 = i0 as f64 * d + d * 0.5;
        let x1 = j1 as f64 * d + d * 0.5;
        let y1 = i1 as f64 * d + d * 0.5;
        let c0 = input.a[i0][j0];
        let c1 = input.a[i1][j1];
        let thickness = (cost as f64).sqrt() * 2.0;

        let title = Title::new().add(Text::new(format!(
            "move {}\n{} ({}, {}) -> {} ({}, {})\ncost = {}",
            k + 1,
            c0,
            i0,
            j0,
            c1,
            i1,
            j1,
            cost
        )));
        let line = Line::new()
            .set("x1", x0)
            .set("y1", y0)
            .set("x2", x1)
            .set("y2", y1)
            .set("stroke", color)
            .set("stroke-width", thickness);

        doc = doc.add(Group::new().add(title).add(line));
    }

    doc
}

fn draw_visit(input: &Input, out: &[(usize, usize)], t: usize, d: f64, mut doc: SVG) -> SVG {
    let mut count = mat![0; input.n; input.n];

    for &(i, j) in out {
        count[i][j] += 1;
    }

    let max = (*count.iter().flatten().max().unwrap()).max(1);

    let out = &out[..t];
    let mut count = mat![0; input.n; input.n];

    for &(i, j) in out {
        count[i][j] += 1;
    }

    for i in 0..input.n {
        for j in 0..input.n {
            let x = j as f64 * d;
            let y = i as f64 * d;
            let c = count[i][j];
            let color = color(c as f64 / max as f64);
            let rect = rect(x, y, d, d, &color);
            let title = Title::new().add(Text::new(format!(
                "{} ({}, {})\nvisited = {} times",
                input.a[i][j], i, j, c
            )));

            doc = doc.add(Group::new().add(title).add(rect));
        }
    }

    doc
}

fn draw_strings(input: &Input, out: &[(usize, usize)], t: usize, mut doc: SVG) -> SVG {
    const BASE_X: f64 = 615.;
    const BASE_Y: f64 = 4.;
    const DX: f64 = 84.;
    const DY: f64 = 24.;
    const SQ_SIZE: f64 = 15.;
    const COUNT_PER_ROW: usize = 8;

    let out = &out[..t];
    let substr_len = compute_substr_len(input, out);
    let prev_substr_len = compute_substr_len(
        input,
        if out.len() > 0 {
            &out[..(out.len() - 1)]
        } else {
            out
        },
    );

    // 今回新しく訪れた文字のインデックス
    let newly_visited = substr_len
        .iter()
        .zip(&prev_substr_len)
        .zip(&input.t)
        .map(|((&len, &prev_len), t)| {
            if len > prev_len || (0 < len && len < t.len() && len == prev_len) {
                Some(len - 1)
            } else {
                None
            }
        })
        .collect_vec();

    for (row, chunk) in input
        .t
        .iter()
        .zip(substr_len.iter().zip(&newly_visited))
        .enumerate()
        .chunks(COUNT_PER_ROW)
        .into_iter()
        .enumerate()
    {
        let base_y = row as f64 * DY + BASE_Y;

        for (col, (w, (s, (&len, &new)))) in chunk.into_iter().enumerate() {
            let base_x = col as f64 * DX + BASE_X;

            for (i, c) in s.chars().enumerate() {
                let color = if new == Some(i) {
                    "#FFF21E".to_string()
                } else if i < len {
                    color(0.5)
                } else {
                    "#FFFFFF".to_string()
                };

                let x = base_x + i as f64 * SQ_SIZE;
                let y = base_y;
                let rect = rect(x, y, SQ_SIZE, SQ_SIZE, &color);
                let text = text(
                    x + SQ_SIZE as f64 * 0.5,
                    y + SQ_SIZE as f64 * 0.8,
                    SQ_SIZE as f64 * 0.8,
                    &c.to_string(),
                )
                .set("stroke", "none");
                let title = Title::new().add(Text::new(format!("word {}\n{}", w + 1, input.t[w])));

                doc = doc.add(Group::new().add(title).add(rect).add(text));
            }
        }
    }

    doc
}

fn compute_substr_len(input: &Input, out: &[(usize, usize)]) -> Vec<usize> {
    let mut result_str = String::new();

    for &(i, j) in out {
        result_str.push(input.a[i][j]);
    }

    let mut substr_len = vec![];

    for s in &input.t {
        let mut char_len = s.chars().count();

        if result_str.contains(s) {
            substr_len.push(char_len);
            continue;
        }

        let mut s = s.clone();

        while let Some(_) = s.pop() {
            char_len -= 1;

            if result_str.ends_with(&s) {
                substr_len.push(char_len);
                break;
            }
        }
    }

    substr_len
}
