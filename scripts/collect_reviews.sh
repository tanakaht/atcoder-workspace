#!/bin/bash

if [ -e './reviews.md' ]; then
    rm './reviews.md'
    touch './reviews.md'
fi

for file in ./contests/*/*review.md; do
    cat ${file} >> './reviews.md'
    echo '' >> './reviews.md'
done
