#!/bin/bash
f=0001
pypy3 ./src/ahc014_a.py < "./tools/in/${f}.txt" > "./tools/out/${f}.txt"
cd ./tools
cargo run --release --bin vis "./in/${f}.txt" "./out/${f}.txt"
cd ..
# open ./tools/vis.html
