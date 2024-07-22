#!/bin/zsh
. ~/.zshrc
sa atcoder

problem_name=$1
test_dir=./testcases/${problem_name}
base_url=${problem_name%_*}

# make test directory
if [ ! -e ${test_dir} ]; then
    oj dl -d ${test_dir} https://atcoder.jp/contests/${base_url}/tasks/${problem_name//-/_}
fi

(oj test -c "pypy3 ./src/${problem_name}.py" -d ${test_dir} && (echo "\e[34mPassed the test in pypy.\e[m"; sleep 1; ./scripts/submit.sh ${problem_name} pypy)) || (oj test -c "python ./src/${problem_name}.py" -d ${test_dir} && (echo "\e[31mPassed the test in python, not pypy!!!\e[m"; sleep 1; ./scripts/submit.sh ${problem_name} python))
