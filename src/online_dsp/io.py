import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np

# 描画用の変数
_line = None
_fig = None
_ax = None
# キューを使ってデータを蓄積する
input_queue = []

def callback(indata, frames, time, status):
    if status:
        print(status)
    input_queue.append(indata.copy())

# マイク入力の初期化（チャンネル数は1で固定）
stream = sd.InputStream(channels=1, samplerate=16000, callback=callback)
stream.start()

def audio_in(len_samples, channels):
    """指定されたlen_samples分が溜まるまで待ってから返す"""
    # 必要な分が溜まるまで単純なループで待機（簡易的な実装）
    while True:
        # キュー内の全データを結合して現在の手持ちサイズを確認
        current_data = np.concatenate(input_queue, axis=0) if input_queue else np.zeros((0, channels))
        
        if current_data.shape[0] >= len_samples:
            # 必要な分だけ取り出して残りをキューに戻す
            ret = current_data[:len_samples, :]
            remaining = current_data[len_samples:, :]
            
            input_queue.clear()
            if remaining.shape[0] > 0:
                input_queue.append(remaining)
                
            return ret
        
        # データが足りなければ少し待つ
        sd.sleep(10)

def audio_out(sig):
    """チラつきを抑えた描画関数"""
    global _line, _fig, _ax
    
    # 初回のみ図を作成
    if _fig is None:
        plt.ion()
        _fig, _ax = plt.subplots()
        _ax.set_ylim(-0.20, 0.20)  # 振幅を固定して軸のブレを防ぐ
        _ax.set_xlim(0, sig.shape[0])
        # 初期の線を作成
        _line, = _ax.plot(sig)
    
    # データを更新（既存の図を書き換える）
    _line.set_ydata(sig)
    
    # 描画を更新
    plt.draw()
    plt.pause(0.001)