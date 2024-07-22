#!/bin/bash
cargo build --manifest-path ./Cargo.toml --bin ahc019_a --release || exit 1
# cargo build --manifest-path ./Cargo.toml --bin ahc019_a || exit 1
f=0000
./target/release/ahc019_a  < "./tools/in/${f}.txt" #> "./tools/out/${f}.txt"
# ./target/debug/ahc019_a  < "./tools/in/${f}.txt"
exit
./target/release/ahc019_a  < "./tools/in/${f}.txt" > "./tools/out/${f}.txt"
cd ./tools
cargo run --release --bin vis "./in/${f}.txt" "./out/${f}.txt"
cd ..
# open ./tools/vis.html
