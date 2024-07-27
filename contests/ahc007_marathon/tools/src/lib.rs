#![allow(non_snake_case, unused_macros)]

use rand::prelude::*;
use proconio::input;
use svg::{node::{element::{Circle, Path, Rectangle, path::Data, Line, Group, Title}, Text}, Document};

pub trait SetMinMax {
	fn setmin(&mut self, v: Self) -> bool;
	fn setmax(&mut self, v: Self) -> bool;
}
impl<T> SetMinMax for T where T: PartialOrd {
	fn setmin(&mut self, v: T) -> bool {
		*self > v && { *self = v; true }
	}
	fn setmax(&mut self, v: T) -> bool {
		*self < v && { *self = v; true }
	}
}

#[macro_export]
macro_rules! mat {
	($($e:expr),*) => { Vec::from(vec![$($e),*]) };
	($($e:expr,)*) => { Vec::from(vec![$($e),*]) };
	($e:expr; $d:expr) => { Vec::from(vec![$e; $d]) };
	($e:expr; $d:expr $(; $ds:expr)+) => { Vec::from(vec![mat![$e $(; $ds)*]; $d]) };
}


use std::cell::Cell;

#[derive(Clone, Debug)]
pub struct UnionFind {
	/// size / parent
	ps: Vec<Cell<usize>>,
	pub is_root: Vec<bool>
}

impl UnionFind {
	pub fn new(n: usize) -> UnionFind {
		UnionFind { ps: vec![Cell::new(1); n], is_root: vec![true; n] }
	}
	pub fn find(&self, x: usize) -> usize {
		if self.is_root[x] { x }
		else {
			let p = self.find(self.ps[x].get());
			self.ps[x].set(p);
			p
		}
	}
	pub fn unite(&mut self, x: usize, y: usize) {
		let mut x = self.find(x);
		let mut y = self.find(y);
		if x == y { return }
		if self.ps[x].get() < self.ps[y].get() {
			::std::mem::swap(&mut x, &mut y);
		}
		*self.ps[x].get_mut() += self.ps[y].get();
		self.ps[y].set(x);
		self.is_root[y] = false;
	}
	pub fn same(&self, x: usize, y: usize) -> bool {
		self.find(x) == self.find(y)
	}
	pub fn size(&self, x: usize) -> usize {
		self.ps[self.find(x)].get()
	}
}

pub type Output = Vec<i32>;

const K: usize = 5;
pub const N: usize = 400;
pub const M: usize = (N - 1) * K;

pub struct Input {
	ps: Vec<(i32, i32)>,
	es: Vec<(usize, usize)>,
	cs: Vec<i32>,
}

impl std::fmt::Display for Input {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		for &(x, y) in &self.ps {
			writeln!(f, "{} {}", x, y)?;
		}
		for &(u, v) in &self.es {
			writeln!(f, "{} {}", u, v)?;
		}
		for &c in &self.cs {
			writeln!(f, "{}", c)?;
		}
		Ok(())
	}
}

pub fn parse_input(f: &str) -> Input {
	let f = proconio::source::once::OnceSource::from(f);
	input! {
		from f,
		ps: [(i32, i32); N],
		es: [(usize, usize); M],
		cs: [i32; M],
	}
	Input { ps, es, cs }
}

pub fn parse_output(_input: &Input, f: &str) -> Output {
	f.split_whitespace().map(|a| a.parse().unwrap()).collect()
}

pub fn compute_mst(input: &Input) -> (i32, Vec<usize>) {
	let mut es = (0..M).map(|i| {
		(input.cs[i], input.es[i].0, input.es[i].1, i)
	}).collect::<Vec<_>>();
	es.sort();
	let mut uf = UnionFind::new(N);
	let mut cost = 0;
	let mut mst = vec![];
	for (c, u, v, e) in es {
		if !uf.same(u, v) {
			uf.unite(u, v);
			cost += c;
			mst.push(e);
		}
	}
	(cost, mst)
}

pub fn compute_score(input: &Input, out: &[i32]) -> (i64, String, i32) {
	let mut uf = UnionFind::new(N);
	let mut cost = 0;
	for i in 0..out.len() {
		if out[i] != 0 && out[i] != 1 {
			return (0, format!("illegal output ({})", out[i]), 0);
		}
		if out[i] == 1 {
			uf.unite(input.es[i].0, input.es[i].1);
			cost += input.cs[i];
		}
	}
	if uf.size(0) != N {
		return (0, "not connected".to_owned(), cost);
	}
	let (mst, _) = compute_mst(input);
	let score = (1e8 * mst as f64 / cost as f64).round() as i64;
	(score, String::new(), cost)
}

fn dist2((x1, y1): (i32, i32), (x2, y2): (i32, i32)) -> i32 {
	(x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
}

fn dist(p1: (i32, i32), p2: (i32, i32)) -> i32 {
	f64::sqrt(dist2(p1, p2) as f64).round() as i32
}

pub fn gen(seed: u64) -> Input {
	let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(seed);
	let mut ps = vec![];
	while ps.len() < N {
		let p = (rng.gen_range(0, 801), rng.gen_range(0, 801));
		if ps.iter().any(|&q| dist2(p, q) <= 25) {
			continue;
		}
		ps.push(p);
	}
	let mut fs = vec![];
	for i in 0..N {
		for j in i+1..N {
			fs.push((dist(ps[i], ps[j]), i, j));
		}
	}
	fs.sort();
	let mut used = mat![false; N; N];
	let mut edges = vec![];
	for _ in 0..K {
		let mut uf = UnionFind::new(N);
		for &(_, i, j) in &fs {
			if !used[i][j] && !uf.same(i, j) {
				uf.unite(i, j);
				edges.push((i, j));
				used[i][j] = true;
			}
		}
	}
	edges.shuffle(&mut rng);
	assert_eq!(edges.len(), M);
	let mut es = vec![];
	let mut cs = vec![];
	for (i, j) in edges {
		let d1 = dist(ps[i], ps[j]);
		let d2 = d1 * 3;
		es.push((i, j));
		cs.push(rng.gen_range(d1, d2 + 1));
	}
	Input { ps, es, cs }
}

fn rect(x: i32, y: i32, w: i32, h: i32, fill: &str) -> Rectangle {
	Rectangle::new().set("x", x).set("y", y).set("width", w).set("height", h).set("fill", fill)
}

pub fn vis_default(input: &Input, out: &Output) -> String {
	vis(input, out, true, true, true)
}

const B: [f64; 201] = [
	0.0,
	12.366,
	17.522,
	21.535,
	24.962,
	28.019,
	30.816,
	33.418,
	35.867,
	38.192,
	40.415,
	42.551,
	44.612,
	46.609,
	48.550,
	50.439,
	52.284,
	54.089,
	55.856,
	57.591,
	59.294,
	60.970,
	62.620,
	64.246,
	65.850,
	67.432,
	68.996,
	70.542,
	72.070,
	73.583,
	75.080,
	76.563,
	78.033,
	79.490,
	80.936,
	82.369,
	83.792,
	85.204,
	86.606,
	87.999,
	89.383,
	90.758,
	92.125,
	93.484,
	94.835,
	96.179,
	97.516,
	98.846,
	100.169,
	101.486,
	102.797,
	104.103,
	105.402,
	106.696,
	107.985,
	109.269,
	110.548,
	111.822,
	113.091,
	114.356,
	115.616,
	116.872,
	118.124,
	119.372,
	120.617,
	121.857,
	123.094,
	124.327,
	125.557,
	126.783,
	128.006,
	129.226,
	130.443,
	131.657,
	132.867,
	134.075,
	135.280,
	136.482,
	137.682,
	138.879,
	140.073,
	141.265,
	142.454,
	143.641,
	144.826,
	146.009,
	147.189,
	148.367,
	149.542,
	150.716,
	151.888,
	153.057,
	154.225,
	155.391,
	156.554,
	157.716,
	158.877,
	160.035,
	161.191,
	162.346,
	163.500,
	164.651,
	165.801,
	166.949,
	168.096,
	169.241,
	170.385,
	171.527,
	172.668,
	173.807,
	174.945,
	176.082,
	177.217,
	178.351,
	179.484,
	180.615,
	181.745,
	182.874,
	184.001,
	185.128,
	186.253,
	187.377,
	188.500,
	189.621,
	190.742,
	191.861,
	192.980,
	194.097,
	195.214,
	196.329,
	197.443,
	198.556,
	199.669,
	200.780,
	201.890,
	203.000,
	204.108,
	205.216,
	206.323,
	207.429,
	208.533,
	209.638,
	210.741,
	211.843,
	212.945,
	214.045,
	215.145,
	216.244,
	217.343,
	218.440,
	219.537,
	220.633,
	221.728,
	222.823,
	223.917,
	225.010,
	226.102,
	227.194,
	228.285,
	229.376,
	230.465,
	231.554,
	232.643,
	233.730,
	234.817,
	235.904,
	236.990,
	238.075,
	239.160,
	240.244,
	241.327,
	242.410,
	243.492,
	244.574,
	245.655,
	246.735,
	247.815,
	248.895,
	249.974,
	251.052,
	252.130,
	253.207,
	254.284,
	255.360,
	256.436,
	257.511,
	258.586,
	259.660,
	260.734,
	261.808,
	262.880,
	263.953,
	265.025,
	266.096,
	267.167,
	268.238,
	269.308,
	270.378,
	271.447,
	272.516,
	273.584,
];

fn draw_line(doc: Document, input: &Input, k: usize, show_length: bool, color: &str, width: i32, title: Option<String>, class: Option<String>) -> Document {
	let (i, j) = input.es[k];
	if show_length {
		let sx = input.ps[i].0 as f64;
		let sy = input.ps[i].1 as f64;
		let dx = (input.ps[j].0 as f64 - sx) / 4.0;
		let dy = (input.ps[j].1 as f64 - sy) / 4.0;
		let b = dist(input.ps[i], input.ps[j]);
		let d = B[((input.cs[k] - b) as f64 / b as f64 * 100.0).round() as usize] / 200.0;
		let data = Data::new()
			.move_to(input.ps[i])
			.quadratic_curve_to(((sx + dx * 0.5 - dy * d, sy + dy * 0.5 + dx * d), (sx + dx, sy + dy), (sx + dx * 1.5 + dy * d, sy + dy * 1.5 - dx * d), (sx + dx * 2.0, sy + dy * 2.0), (sx + dx * 2.5 - dy * d, sy + dy * 2.5 + dx * d), (sx + dx * 3.0, sy + dy * 3.0)))
			.smooth_quadratic_curve_to(input.ps[j]);
		let mut path = Path::new().set("d", data).set("fill", "none").set("stroke", color).set("stroke-width", width);
		if let Some(class) = class {
			path = path.set("class", class);
		}
		if let Some(title) = title {
			let group = Group::new().add(path).add(Title::new().add(Text::new(title)));
			doc.add(group)
		} else {
			doc.add(path)
		}
	} else {
		let mut path = Line::new()
			.set("x1", input.ps[i].0).set("y1", input.ps[i].1)
			.set("x2", input.ps[j].0).set("y2", input.ps[j].1)
			.set("stroke", color).set("stroke-width", width);
		if let Some(class) = class {
			path = path.set("class", class);
		}
		if let Some(title) = title {
			let group = Group::new().add(path).add(Title::new().add(Text::new(title)));
			doc.add(group)
		} else {
			doc.add(path)
		}
	}
}

pub fn vis(input: &Input, out: &[i32], show_optimum: bool, show_all_edges: bool, show_length: bool) -> String {
	let mut doc = svg::Document::new().set("id", "vis").set("viewBox", (-20, -20, 840, 840)).set("width", 840).set("height", 840);
	doc = doc.add(rect(-20, -20, 840, 840, "white"));
	let t = out.len();
	for e in if show_all_edges { 0 } else { t }..M {
		doc = draw_line(doc, input, e, show_length, "lightgray", 1, Some(format!("edge {} (d={}, l={})", e, dist(input.ps[input.es[e].0], input.ps[input.es[e].1]), input.cs[e])), Some("L1".to_owned()));
	}
	if show_optimum {
		let (_, mst) = compute_mst(input);
		for e in mst {
			if e < t {
				doc = draw_line(doc, input, e, show_length, "lightcoral", 5, Some(format!("edge {} (d={}, l={})", e, dist(input.ps[input.es[e].0], input.ps[input.es[e].1]), input.cs[e])), Some("L5".to_owned()));
			}
		}
	}
	for e in 0..t {
		if out[e] == 1 {
			doc = draw_line(doc, input, e, show_length, "black", 2, Some(format!("edge {} (d={}, l={})", e, dist(input.ps[input.es[e].0], input.ps[input.es[e].1]), input.cs[e])), Some("L2".to_owned()));
		}
	}
	for i in 0..N {
		let group = Group::new().add(Circle::new().set("cx", input.ps[i].0).set("cy", input.ps[i].1).set("r", 3).set("fill", "black").set("class", "C3"))
			.add(Title::new().add(Text::new(format!("vertex {}", i))));
		doc = doc.add(group);
	}
	doc.to_string()
}
