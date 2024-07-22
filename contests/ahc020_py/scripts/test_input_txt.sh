#!/bin/bash
problem_name=$1
base_url=${problem_name%_*}

{ts=$(gdate +%s%3N) && env="\e[34mpypy\e[m" && pypy3 "problems/${base_url}/${problem_name}.py" < input.txt;} || {ts=$(gdate +%s%3N); env="\e[31mpython\e[m" && python3 "problems/${base_url}/${problem_name}.py" < input.txt;}
echo "$(echo "scale=3; ($(gdate +%s%3N)-$ts)/1000" | bc) second in ${env}"
