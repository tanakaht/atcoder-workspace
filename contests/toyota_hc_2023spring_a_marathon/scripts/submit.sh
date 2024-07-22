#!/bin/bash
problem=$1
# 指定あればpythonそうでなければpypyでで提出
if [ "$2" = "python" ]; then
    language=4006
elif [ "$2" = "pypy" ]; then
    language=4047
else
    language=4047
fi
oj submit https://atcoder.jp/contests/${problem%_*}/tasks/${problem} ./src/${problem}.py --language ${language}
