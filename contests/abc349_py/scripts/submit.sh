#!/bin/bash
problem=$1
# 指定あればpythonそうでなければpypyでで提出
if [ "$2" = "python" ]; then
    language=5055
elif [ "$2" = "pypy" ]; then
    language=5078
else
    language=5055
fi
oj submit https://atcoder.jp/contests/${problem%_*}/tasks/${problem} ./src/${problem}.py --language ${language}
