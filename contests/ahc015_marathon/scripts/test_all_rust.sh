#[st, en] のseed のファイルを処理する．procsはプロセス数，print_errorはxargs のエラー出力表示
st=0
en=10
procs=8
comment=""
print_error=1
if [ ! -e ./dev ]; then
    mkdir ./dev
fi
touch ./dev/tmp_scores.txt
PATH="$HOME/.cargo/bin:$PATH"
problem_name=$1
cargo build --manifest-path ./Cargo.toml --bin ahc015_a || exit 1

# インタラクティブ処理の関数
f1(){
    #f=$(printf "%04d\n" "${1}")
    f=${1}
    ./target/debug/ahc015_a  < "./tools/in/${f}.txt" > "./tools/out/${f}.txt"
    cd ./tools
    score=$(cargo run --release --bin vis "./in/${f}.txt" "./out/${f}.txt" 2>/dev/null | awk '{ print $3 }')
    cd ..
    echo "${f} ${score}" >> ./dev/tmp_scores.txt
}
# xargs で関数使うための処理
export -f f1

usage(){
  cat <<EOM
使い方：
  -s : 開始 seed
  -e : 終了 seed
  -P : プロセス数
  -d : 指定でエラー出力なし
  -m : comment
EOM
  exit 2
}

while getopts "s:e:P:m:d" optKey; do
  case "$optKey" in
    s)
      st=${OPTARG}
      ;;
    e)
      en=${OPTARG}
      ;;
    P)
      procs=${OPTARG}
      ;;
    m)
      comment=${OPTARG}
      ;;
    d)
      print_error=0
      ;;
    '-h' | '--help' | *)
      usage
      ;;
  esac
done
# 並列処理
if [ $print_error = 0 ]; then
  seq -f '%04g' $st $en | xargs -n1 -P$procs -I{} bash -c "f1 {}"
else
  seq -f '%04g' $st $en | xargs -t -n1 -P$procs -I{} bash -c "f1 {}"
fi
sleep 1
# tmp_score.txt に書き込まれたスコアの計算
# source ~/.venv/atcoder/bin/activate
score=$(python3 ./scripts/format_score.py)
echo "|${score}|${comment}|$(date +%H:%M:%S)|" >> ./results/results.md
echo $score
rm ./dev/tmp_scores.txt
open ./tools/vis.html
