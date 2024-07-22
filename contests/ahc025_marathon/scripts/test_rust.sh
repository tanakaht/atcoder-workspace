#!/bin/bash
f=0001
cargo build --manifest-path ./Cargo.toml --bin ahc025_a --release || exit 1
cd ./tools
cargo run -r --bin tester ../target/release/ahc025_a  < "./in/${f}.txt" > "./out/${f}.txt"
cargo run --release --bin vis "./in/${f}.txt" "./out/${f}.txt"
cd ..
open ./tools/vis.html
