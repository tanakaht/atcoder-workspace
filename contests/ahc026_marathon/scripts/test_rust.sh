#!/bin/bash
f=0034
cargo build --manifest-path ./Cargo.toml --bin ahc026_a --release || exit 1
./target/release/ahc026_a  < "./tools/in/${f}.txt" > "./tools/out/${f}.txt"
cd ./tools
cargo run --release --bin vis "./in/${f}.txt" "./out/${f}.txt"
cd ..
open ./tools/vis.html
