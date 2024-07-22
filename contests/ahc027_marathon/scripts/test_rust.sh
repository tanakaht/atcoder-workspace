#!/bin/bash
f=0005
cargo build --manifest-path ./Cargo.toml --bin ahc027_a --release || exit 1
./target/release/ahc027_a  < "./tools/in/${f}.txt" > "./tools/out/${f}.txt"
# cargo build --manifest-path ./Cargo.toml --bin ahc027_a || exit 1
# ./target/debug/ahc027_a  < "./tools/in/${f}.txt" > "./tools/out/${f}.txt"
cd ./tools
cargo run --release --bin vis "./in/${f}.txt" "./out/${f}.txt"
cd ..
open ./tools/vis.html
python3 ./src_py/vis.py < "./tools/out/${f}.txt"
