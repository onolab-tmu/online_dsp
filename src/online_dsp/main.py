import matplotlib.pyplot as plt
import numpy as np

from .io import audio_in, audio_out
from .processor import frame_processing


def main():
    # 1. 変数定義
    frlen = 1024  # フレーム長
    frsft = 256  # フレームシフト
    ch = 1  # チャンネル数

    # 2. 各種バッファ・配列の初期化
    buf0 = np.zeros((frlen, ch))
    buf1 = np.zeros((frlen, ch))
    frm_sig0 = np.zeros((frlen, ch))
    frm_sig1 = np.zeros((frlen, ch))

    wnd_a = np.zeros(frlen)
    wnd_s = np.zeros(frlen)

    # 3. 窓関数の設定（MATLABの1オリジンインデックスに対応）
    for i in range(frlen):
        t = i + 1
        wnd_a[i] = 0.5 - 0.5 * np.cos(2 * np.pi * t / frlen)
        wnd_s[i] = 0.5 - 0.5 * np.cos(2 * np.pi * t / frlen)

    # オーバラップ加算のスケール調整
    sc = np.sum(wnd_a * wnd_s) / frsft
    wnd_s = wnd_s / sc

    # ダミーパラメータの初期化
    param = np.random.randn(ch, ch)

    # 描画の準備
    plt.ion()
    plt.figure()

    print("\n中止する場合は Ctrl+C を押してください。\n")

    # 4. メインループ
    while True:
        try:
            # ① バッファをシフトし、新しいフレームシフト分をゼロクリア
            buf0[0 : frlen - frsft, :] = buf0[frsft:frlen, :]
            buf1[0 : frlen - frsft, :] = buf1[frsft:frlen, :]
            buf0[frlen - frsft : frlen, :] = np.zeros((frsft, ch))
            buf1[frlen - frsft : frlen, :] = np.zeros((frsft, ch))

            # ② フレームシフト分のデータをオーディオ入力から取得
            new_sig = audio_in(frsft, ch)
            buf0[frlen - frsft : frlen, :] = new_sig

            # ③ 分析窓をかける
            for h in range(ch):
                frm_sig0[:, h] = buf0[:, h] * wnd_a

            """
            # ④ フレーム毎の信号処理
            frm_sig1, param = frame_processing(frm_sig0, param)
            """
            # processor.pyを修正したので受け取りも3つに増やす
            frm_sig1, param, spec = frame_processing(frm_sig0, param)

            # ⑤ 合成窓をかける
            for h in range(ch):
                frm_sig1[:, h] = frm_sig1[:, h] * wnd_s

            # ⑥ オーバーラップ加算
            buf1 = buf1 + frm_sig1
            

            """# ⑦ 確定したフレームシフト分のデータをオーディオ出力に転送
            audio_out(buf1[0:frsft, :])
            """
            # ⑦ スペクトルを渡して出力
            audio_out(buf1[0:frsft, :], spec=spec)

        except KeyboardInterrupt:
            print("\n処理を停止しました。")
            break


if __name__ == "__main__":
    main()