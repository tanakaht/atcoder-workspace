# atcoder-workspace
## 何ができるか

- サンプルケースと自作ケースについて、ショートカットから自動的にテスト実行できる環境(https://qiita.com/chokoryu/items/4b31ffb89dbc8cb86971 みたいなやつ)がvscodeで開くだけでできます。
- コンテスト用のフォルダーを(templatesからコピーして)自動作成します。

## 事前準備

- vscode
  - vscodeのインストール
  - vscodeの拡張機能、Remote-Containersのインストール
- dockerのインストール

## 対応言語

- python(pypy)
- rust

## 使い方

1. forkしてclone
2. vscodeでcloneしたフォルダを開く
3. 右下からポップアップが出るので、`Reopen in container`を選択
4. ターミナルで`./scripts/start_contest.sh <contest_name> <language>` を実行
    - <contest_name>: abcxxx とか、https://atcoder.jp/contests/XXX のXXX
    - <language>: [py, rust(, marathon)]
5. contestのフォルダがvscode上で開かれるので、`./src` 以下のファイルを編集して問題を解く
6. (macでは)`cmd+Shift+b` でサンプルケースのテスト実行、テストを通過した場合、(確認用の文字列を入力して、)自動提出
7. (option): `./XXX＿review.md` にコンテストの感想を書きましょう。各問題について、自分の思考回路を辿るとか、どういった部分が足りなくて解けなかったとか を書いて復習すると伸びる気がします。
