# 実時間信号処理の練習

## 実行手順
1. リポジトリをクローンしてフォルダに入る
```
git clone https://github.com/onolab-tmu/online_dsp.git
cd online_dsp
```

2. 仮想環境の作成からライブラリのインストールまで一括同期する
```
uv sync
```

4. スクリプト（main.py）を実行する
```
uv run online-dsp
```

## 環境
Macbook Proの内蔵マイクを使用しています。

## 内容
実時間処理をしています。入ってきた音をフレーム分割し、FFT→IFFTしています（つまり、何もしない）。出力は今の波形のプロットです。
