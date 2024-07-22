#!/bin/bash
pypy3 ./src/rco_contest_2017_qual_a.py < "./tools/in/${f}.txt" > "./tools/out/${f}.txt"
cd ./tools
cargo run --release --bin vis "./in/${f}.txt" "./out/${f}.txt"
cd ..
open ./tools/vis.html
