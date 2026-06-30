import numpy as np
import sounddevice as sd
import soundfile as sf


# =========================
# パラメータ設定
# =========================
FRLEN = 1024          # フレーム長
FRSFT = 256           # フレームシフト
OUTPUT_WAV = "output.wav"


# =========================
# フレーム処理
# =========================
def frame_processing(frm_sig0, param):
    """
    frm_sig0: shape = (FRLEN, channels)

    現状は Fast Fourier Transform（FFT，高速フーリエ変換）
    -> Inverse FFT（IFFT，逆高速フーリエ変換）するだけのスルー処理。

    ここを書き換えれば，周波数領域処理やBSS処理を入れられる。
    """

    spec = np.fft.rfft(frm_sig0, axis=0)

    # =========================
    # ここに周波数領域処理を書く
    # =========================
    # 例:
    # spec *= 0.5

    frm_sig1 = np.fft.irfft(spec, n=FRLEN, axis=0)

    return frm_sig1.astype(np.float32), param


# 何も処理しない場合はこちらを使う
# def frame_processing(frm_sig0, param):
#     return frm_sig0.astype(np.float32), param


# =========================
# オンラインフレーム処理クラス
# =========================
class RealTimeProcessor:
    def __init__(self, frlen, frsft, channels):
        self.frlen = frlen
        self.frsft = frsft
        self.channels = channels

        # 信号保持用バッファ
        self.buf0 = np.zeros((frlen, channels), dtype=np.float32)
        self.buf1 = np.zeros((frlen, channels), dtype=np.float32)

        # 窓関数
        t = np.arange(1, frlen + 1, dtype=np.float32)
        wnd_a = 0.5 - 0.5 * np.cos(2 * np.pi * t / frlen)
        wnd_s = 0.5 - 0.5 * np.cos(2 * np.pi * t / frlen)

        # オーバーラップ加算後のゲインが1になるように補正
        sc = np.sum(wnd_a * wnd_s) / frsft
        wnd_s = wnd_s / sc

        self.wnd_a = wnd_a[:, None].astype(np.float32)
        self.wnd_s = wnd_s[:, None].astype(np.float32)

        # フレーム処理用のダミーパラメータ
        self.param = np.random.randn(channels, channels).astype(np.float32)

    def process(self, new_sig):
        """
        new_sig: shape = (FRSFT, channels)
        return : shape = (FRSFT, channels)
        """

        # バッファをシフト
        self.buf0[: self.frlen - self.frsft, :] = self.buf0[self.frsft :, :]
        self.buf1[: self.frlen - self.frsft, :] = self.buf1[self.frsft :, :]

        # 後ろのフレームシフト分をゼロクリア
        self.buf0[self.frlen - self.frsft :, :] = 0.0
        self.buf1[self.frlen - self.frsft :, :] = 0.0

        # 新しい入力を格納
        self.buf0[self.frlen - self.frsft :, :] = new_sig

        # 分析窓
        frm_sig0 = self.buf0 * self.wnd_a

        # フレーム処理
        frm_sig1, self.param = frame_processing(frm_sig0, self.param)

        # 合成窓
        frm_sig1 = frm_sig1 * self.wnd_s

        # オーバーラップ加算
        self.buf1 += frm_sig1

        # 先頭のフレームシフト分を出力
        return self.buf1[: self.frsft, :].copy()


# =========================
# デバイス設定用関数
# =========================
def show_devices(kind):
    """
    kind:
        "input"  -> 録音デバイス一覧
        "output" -> 再生デバイス一覧
    """

    devices = sd.query_devices()

    if kind == "input":
        print("\n=== 録音デバイス一覧 ===")
        ch_key = "max_input_channels"
    else:
        print("\n=== 再生デバイス一覧 ===")
        ch_key = "max_output_channels"

    for i, dev in enumerate(devices):
        max_ch = int(dev[ch_key])

        if max_ch > 0:
            print(
                f"[{i}] {dev['name']} "
                f"| channels: {max_ch} "
                f"| default samplerate: {dev['default_samplerate']:.0f} Hz"
            )


def choose_device(kind):
    """
    録音デバイスまたは再生デバイスを選択する。
    Enterを押すとデフォルトデバイスを使う。
    """

    while True:
        show_devices(kind)

        if kind == "input":
            ans = input("\n録音デバイス番号を入力してください。Enterでデフォルト: ").strip()
            ch_key = "max_input_channels"
        else:
            ans = input("\n再生デバイス番号を入力してください。Enterでデフォルト: ").strip()
            ch_key = "max_output_channels"

        if ans == "":
            return None

        try:
            device_id = int(ans)
            dev = sd.query_devices(device_id)

            if int(dev[ch_key]) <= 0:
                print("そのデバイスは選択できません。")
                continue

            return device_id

        except Exception:
            print("有効なデバイス番号を入力してください。")


def get_device_info(input_device, output_device):
    """
    選択した録音・再生デバイスの情報を取得する。
    """

    input_info = sd.query_devices(input_device, kind="input")
    output_info = sd.query_devices(output_device, kind="output")

    return input_info, output_info


def choose_channels(input_info, output_info):
    """
    デバイス情報から入出力チャンネル数を自動決定する。
    """

    input_channels = int(input_info["max_input_channels"])
    output_channels = int(output_info["max_output_channels"])

    print("\n自動設定されたチャンネル数:")
    print(f"録音チャンネル数: {input_channels}")
    print(f"再生チャンネル数: {output_channels}")

    return input_channels, output_channels


def choose_samplerate(input_device, output_device, input_channels, output_channels):
    """
    録音・再生デバイスで共通に使えるサンプリング周波数を自動決定する。
    """

    input_info = sd.query_devices(input_device, kind="input")
    output_info = sd.query_devices(output_device, kind="output")

    input_default_fs = int(input_info["default_samplerate"])
    output_default_fs = int(output_info["default_samplerate"])

    print("\nデバイスのデフォルトサンプリング周波数:")
    print(f"録音デバイス: {input_default_fs} Hz")
    print(f"再生デバイス: {output_default_fs} Hz")

    candidate_fs_list = [
        input_default_fs,
        output_default_fs,
        48000,
        44100,
        96000,
        32000,
        16000,
    ]

    # 重複削除
    candidate_fs_list = list(dict.fromkeys(candidate_fs_list))

    for fs in candidate_fs_list:
        try:
            sd.check_input_settings(
                device=input_device,
                channels=input_channels,
                samplerate=fs,
                dtype="float32",
            )

            sd.check_output_settings(
                device=output_device,
                channels=output_channels,
                samplerate=fs,
                dtype="float32",
            )

            print(f"\n自動設定されたサンプリング周波数: {fs} Hz")
            return fs

        except Exception:
            pass

    raise RuntimeError(
        "録音デバイスと再生デバイスで共通に使えるサンプリング周波数が見つかりませんでした。"
    )


# =========================
# チャンネル変換
# =========================
def convert_channels(y, input_channels, output_channels):
    """
    処理後信号 y を，再生デバイスのチャンネル数に合わせる。
    """

    frames = y.shape[0]
    y_out = np.zeros((frames, output_channels), dtype=np.float32)

    if input_channels == output_channels:
        y_out[:] = y

    elif input_channels == 1:
        # 1ch入力を全出力チャンネルにコピー
        y_out[:] = y[:, [0]]

    elif output_channels <= input_channels:
        # 入力チャンネル数が多い場合は，先頭の出力チャンネル数だけ使う
        y_out[:] = y[:, :output_channels]

    else:
        # 出力チャンネル数が多い場合は，足りない分をゼロ埋め
        ch = min(input_channels, output_channels)
        y_out[:, :ch] = y[:, :ch]

    return y_out


# =========================
# メイン処理
# =========================
def main():
    # 録音・再生デバイスを選択
    input_device = choose_device("input")
    output_device = choose_device("output")

    # デバイス情報を取得
    input_info, output_info = get_device_info(input_device, output_device)

    print("\n使用する録音デバイス:")
    print(input_info)

    print("\n使用する再生デバイス:")
    print(output_info)

    # チャンネル数を自動設定
    input_channels, output_channels = choose_channels(input_info, output_info)

    # サンプリング周波数を自動設定
    fs = choose_samplerate(
        input_device=input_device,
        output_device=output_device,
        input_channels=input_channels,
        output_channels=output_channels,
    )

    # フレーム処理器を作成
    processor = RealTimeProcessor(
        frlen=FRLEN,
        frsft=FRSFT,
        channels=input_channels,
    )

    # 保存するか選択
    ans = input("\n出力音声を wav ファイルに保存しますか？ [y/n]: ").strip().lower()
    save_enabled = ans in ["y", "yes"]

    if save_enabled:
        print(f"保存モード: ON -> {OUTPUT_WAV} に保存します")
        recorded_frames = []
    else:
        print("保存モード: OFF -> ファイル保存しません")
        recorded_frames = None

    def callback(indata, outdata, frames, time, status):
        if status:
            print(status, flush=True)

        if frames != FRSFT:
            outdata[:] = 0
            return

        # 入力信号
        new_sig = indata.astype(np.float32)

        # フレーム処理
        y = processor.process(new_sig)

        # 出力チャンネル数に合わせる
        y_out = convert_channels(
            y=y,
            input_channels=input_channels,
            output_channels=output_channels,
        )

        # スピーカーへ出力
        outdata[:] = y_out

        # 保存モードがONのときだけ記録
        if save_enabled:
            recorded_frames.append(y_out.copy())

    print("\nStart real-time processing.")
    print(f"Sampling rate: {fs} Hz")
    print("Press Ctrl+C to stop.")

    try:
        with sd.Stream(
            samplerate=fs,
            blocksize=FRSFT,
            dtype="float32",
            channels=(input_channels, output_channels),
            device=(input_device, output_device),
            callback=callback,
        ):
            while True:
                sd.sleep(1000)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        if save_enabled:
            if len(recorded_frames) > 0:
                recorded_signal = np.concatenate(recorded_frames, axis=0)
                sf.write(OUTPUT_WAV, recorded_signal, fs)

                print(f"Saved: {OUTPUT_WAV}")
                print(f"Sampling rate: {fs} Hz")
            else:
                print("No audio was recorded.")
        else:
            print("Finished without saving.")


if __name__ == "__main__":
    main()