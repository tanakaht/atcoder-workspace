#!/bin/bash
f=0002
cargo build --manifest-path ./Cargo.toml --bin ahc024_a --release || exit 1
./target/release/ahc024_a  < "./tools/in/${f}.txt" > "./tools/out/${f}.txt"
cd ./tools
cargo run --release --bin vis "./in/${f}.txt" "./out/${f}.txt"
cd ..
open ./tools/vis.html
