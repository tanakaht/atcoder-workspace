#!/bin/bash
f=0000
cargo build --manifest-path ./Cargo.toml --bin ahc022_a  --release || exit 1
# cargo build --manifest-path ./Cargo.toml --bin ahc022_a   || exit 1
cd ./tools
# cargo run --release --bin tester ../target/release/ahc022_a  < "./in/${f}.txt" > "./out/${f}.txt" 2> "./err/${f}.txt"
cargo run --release --bin tester ../target/release/ahc022_a  < "./in/${f}.txt" > "./out/${f}.txt"
# cargo run --release --bin tester ../target/debug/ahc022_a  < "./in/${f}.txt" > "./out/${f}.txt"
cd ..
open ./tools/vis.html
