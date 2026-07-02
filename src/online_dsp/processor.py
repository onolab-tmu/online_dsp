import numpy as np

def frame_processing(frm_sig0, param):
    len_val, ch_val = frm_sig0.shape

    # 振幅スペクトル（表示用）を格納する箱
    # 各チャンネルのFFT結果を積むため、(周波数ビン数, チャンネル数) とする
    freq_bins = len_val // 2
    frm_spec_all = np.zeros((freq_bins, ch_val))
    
    frm_sig1 = np.zeros((len_val, ch_val))

    # 各チャンネルごとにFFT
    for h in range(ch_val):
        # FFT実行
        spc = np.fft.fft(frm_sig0[:, h], len_val)
        
        # 振幅スペクトル（表示用）を保存（前半の半分のみ）
        frm_spec_all[:, h] = np.abs(spc)[:freq_bins]

    # ----------------------------------------------------
    # ※ 将来のオンラインAuxIVA用
    # ----------------------------------------------------

    # 各チャンネルごとにiFFT
    for h in range(ch_val):
        frm_sig1[:, h] = np.fft.ifft(np.fft.fft(frm_sig0[:, h], len_val), len_val).real

    # チャンネル0のスペクトルを表示用として返す（複数ある場合は平均などが必要）
    return frm_sig1, param, frm_spec_all[:, 0]