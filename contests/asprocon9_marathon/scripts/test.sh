#!/bin/bash
f=0002
./tools/judge "pypy3 ./src/asprocon9_a.py" < "./tools/in/${f}.txt" > "./tools/out/${f}.txt"
# pypy3 ./src/asprocon9_a.py < "./tools/in/${f}.txt" > "./tools/out/${f}.txt"
# cd ./tools
# cargo run --release --bin vis "./in/${f}.txt" "./out/${f}.txt"
# cd ..
# open ./tools/vis.html
