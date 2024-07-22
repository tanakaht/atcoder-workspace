path="./"
problem_name=$1
for f in ${path}input/${problem_name}/*.txt; do
    filename="${f##*/}"
    mkdir -p "${path}output/${problem_name}"
    echo ${problem_name}
    pypy3 "${path}${problem_name}.py" < "${f}" > "${path}output/${problem_name}/${filename}";
    echo "input(${f})"
    cat "${f}"
    echo "output"
    cat "${path}output/${problem_name}/${filename}";
done
