#!/bin/bash
f=0000
#cargo build --manifest-path ./Cargo.toml --bin ahc035_a || exit 1
# ./target/debug/ahc035_a  < "./tools/in/${f}.txt" > "./tools/out/${f}.txt"
cargo build --manifest-path ./Cargo.toml --bin ahc035_a --release || exit 1
cd ./tools
cargo run -r --bin tester ../target/release/ahc035_a  < "./in/${f}.txt" > "./out/${f}.txt"
cargo run --release --bin vis "./in/${f}.txt" "./out/${f}.txt"
cd ..
open ./tools/vis.html
