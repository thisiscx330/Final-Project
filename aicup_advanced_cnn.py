# aicup_advanced_cnn.py
# private分數0.8056059699404513 排名21/633

import os
import re
import math
import json
import numpy as np
import pandas as pd

# TSFresh 相關
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.feature_selection import select_features

# 時頻、小波、DTW 等特徵
import pywt
from scipy.signal import butter, filtfilt
from scipy.fft import rfft, rfftfreq
from scipy.stats import kurtosis, skew
from dtaidistance import dtw

# 深度學習 (1D-CNN)
import tensorflow as tf
from tensorflow.keras import layers, models

# 機器學習
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from lightgbm import LGBMClassifier

# Blend/stacking 偽標工具
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from scipy.optimize import minimize

#Dynamic Time Warping
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# ====== 全域參數 ======
FS     = 85.0    # 感測器取樣頻率 (Hz)
CUTOFF = 20.0    # 低通濾波截止 (Hz)

# ====== 1) 低通濾波器設計 ======
b, a = butter(4, CUTOFF/(FS/2), btype='low')

# ====== 2) 解析 cut_point 格式 ======
def parse_cutpoints(cp_str, length):
    s = re.sub(r"[\[\]\(\)]", "", str(cp_str))
    parts = re.split(r"[,; ]+", s)
    pts = sorted({
        int(float(p)) for p in parts
        if p.replace('.', '', 1).isdigit() and 0 < int(float(p)) < length
    })
    if len(pts) < 27:
        return list(np.linspace(0, length, 28, dtype=int))
    return [0] + pts[:26] + [length]

# ====== 3) 手工 + 進階特徵提取函式 ======

def parse_cutpoints(cp_str, length):
    """
    解析 cut_point 字串，回傳 0..length 之間的 28 個節點索引（若原本不足 27 個，就均分 27 段）。
    """
    s = re.sub(r"[\[\]\(\)]", "", str(cp_str))
    parts = re.split(r"[,; ]+", s)
    pts = sorted({
        int(float(p)) for p in parts
        if p.replace('.', '', 1).isdigit() and 0 < int(float(p)) < length
    })
    if len(pts) < 27:
        return list(np.linspace(0, length, 28, dtype=int))
    return [0] + pts[:26] + [length]

def extract_handcrafted(txt_path, cp_str):
    """
    讀入一個 .txt（欄位為 Ax,Ay,Az,Gx,Gy,Gz），
    先做低通濾波、再根據 cut_point 切成 27 段。
    每一段都提以下特徵（合計 90 維）：
      (A) 時域統計：6 軸 × (mean, var, RMS, max, min, peak-to-peak) → 共 6*6 = 36 維
      (B) 頻域統計：加速度向量/角速度向量做 FFT → abs(A).mean(), abs(G).mean(), PSD_A.mean(), PSD_G.mean(), entropy(PSD_A), entropy(PSD_G), peak_freq(A), peak_freq(G) → 8 維
      (C) Wavelet
      (D) 滑動視窗特徵 (Window size=50, step=25)：算出全部滑窗的統計，再取整體 min/max/mean/std → 4 維
      (E) 倒頻譜 Cepstrum 前 10 個係數 → 10 維
    每段合計 36+8+32+4+10 = 90 維。接著把這 90 維取平均 (段落平均)，
    再把全局(整次測驗)的 12 維時間域統計 + 1 維 DTW 為一個向量，
    回傳長度 = 90 + 12 + 1 = 103 維。
    """
    # 1) 讀原始六軸
    df = pd.read_csv(txt_path, header=None, sep='\s+')
    data = df.values  # shape = (N, 6)
    n = data.shape[0]

    # 2) 低通濾波 (六軸分別做)
    data[:, :3] = filtfilt(b, a, data[:, :3], axis=0)
    data[:, 3:] = filtfilt(b, a, data[:, 3:], axis=0)

    # 3) 切成 27 段
    idx = parse_cutpoints(cp_str, n)
    seg_feats_list = []

    for i in range(len(idx) - 1):
        seg = data[idx[i] : idx[i + 1], :]
        if seg.shape[0] == 0:
            # 如果這段完全沒有樣本，就直接回傳一個 90 維全 0 的向量
            seg_feats_list.append(np.zeros(90, dtype=float))
            continue

        stats = []

        # (A) 時域統計 + peak-to-peak：6 軸 × (mean, var, RMS, max, min, ptp) → 36 維
        for j in range(6):
            col = seg[:, j]
            stats += [
                col.mean(),
                col.var(),
                math.sqrt((col**2).mean()),  # RMS
                col.max(),
                col.min(),
                np.ptp(col),
            ]

        # (B) 加速度向量 / 角速度向量 → FFT + PSD + entropy + peak frequency → 8 維
        accel = np.linalg.norm(seg[:, :3], axis=1)
        ang   = np.linalg.norm(seg[:, 3:], axis=1)

        A = rfft(accel)
        G = rfft(ang)
        freqs = rfftfreq(len(accel), d=1/FS)
        psdA = np.abs(A) ** 2
        psdG = np.abs(G) ** 2

        def ent(psd):
            p = psd / (psd.sum() + 1e-12)
            return -np.sum(p * np.log(p + 1e-12))

        stats += [
            np.abs(A).mean(),
            np.abs(G).mean(),
            psdA.mean(),
            psdG.mean(),
            ent(psdA),
            ent(psdG),
            freqs[np.argmax(np.abs(A))] if len(freqs) > 0 else 0.0,
            freqs[np.argmax(np.abs(G))] if len(freqs) > 0 else 0.0,
        ]


        # (C) 小波 (Wavelet) 特徵 (db4, level=3)
        try:
            coeffs_accel = pywt.wavedec(accel, 'db4', level=3)
            coeffs_ang   = pywt.wavedec(ang,   'db4', level=3)
            for arr in coeffs_accel:
                stats += [arr.mean(), arr.var(), arr.max(), arr.min()]
            for arr in coeffs_ang:
                stats += [arr.mean(), arr.var(), arr.max(), arr.min()]
        except:
            stats += [0.0]*8*2


        # (D) 滑窗：用窗長 50、步長 25，在 accel, ang 上滑動取特徵，然後整體取 min,max,mean,std → 4 維
        sw_feats = []
        L = len(accel)
        wsize, step = 50, 25
        for start in range(0, max(1, L - wsize + 1), step):
            win_a = accel[start : start + wsize]
            win_g = ang[start : start + wsize]
            if win_a.size == 0:
                continue
            sw_feats += [
                win_a.mean(),
                win_a.var(),
                np.ptp(win_a),
                math.sqrt((win_a**2).mean()),
                pd.Series(win_a).skew(),
                pd.Series(win_a).kurt(),
            ]
            sw_feats += [
                win_g.mean(),
                win_g.var(),
                np.ptp(win_g),
                math.sqrt((win_g**2).mean()),
                pd.Series(win_g).skew(),
                pd.Series(win_g).kurt(),
            ]
        if len(sw_feats) == 0:
            sw_feats = [0.0] * 12
        stats += [
            np.min(sw_feats),
            np.max(sw_feats),
            np.mean(sw_feats),
            np.std(sw_feats),
        ]

        # (E) 倒頻譜 (Cepstrum) 前 10 係數
        try:
            fft_vals = np.abs(rfft(accel)) + 1e-12
            log_pow  = np.log(fft_vals)
            cepstr   = np.fft.irfft(log_pow)
            if len(cepstr) >= 10:
                stats += cepstr[:10].tolist()
            else:
                # 不夠 10 維就補 0
                stats += cepstr.tolist() + [0.0] * (10 - len(cepstr))
        except:
            stats += [0.0] * 10

        # 到此，stats 理論上正好是 36 + 8 + 32 + 4 + 10 = 90 維
        seg_feats_list.append(stats)

    # 把所有段落的 90 維做「段落平均」
    seg_feats_arr = np.vstack(seg_feats_list)  # shape = (#segments, 90)
    seg_mean = np.mean(seg_feats_arr, axis=0)   # (90,)

    # (F) 全局特徵：整次測驗的加速度/角速度向量 → 12 維
    accel_all = np.linalg.norm(data[:, :3], axis=1)
    ang_all   = np.linalg.norm(data[:, 3:], axis=1)
    glob_feats = [
        accel_all.mean(),
        accel_all.var(),
        np.ptp(accel_all),
        math.sqrt((accel_all**2).mean()),
        pd.Series(accel_all).skew(),
        pd.Series(accel_all).kurt(),
        ang_all.mean(),
        ang_all.var(),
        np.ptp(ang_all),
        math.sqrt((ang_all**2).mean()),
        pd.Series(ang_all).skew(),
        pd.Series(ang_all).kurt(),
    ]

    # (G) DTW 相似度 placeholder (1 維)

    try:
        REFERENCE_ACCEL_TEMPLATE = np.linspace(0, 1, 300)
        dtw_dist, _ = fastdtw(accel_all, REFERENCE_ACCEL_TEMPLATE, dist=euclidean)
        dtw_feats = [dtw_dist]
    except:
        dtw_feats = [0.0]

    # 最後串起來：段落平均 (90 維) + 全局 (12 維) + DTW (1 維) → 全長 103 維
    final_feats = np.hstack([seg_mean, glob_feats, dtw_feats])
    return final_feats

# ====== 4) 建立「完整 TSFresh 特徵」一次用，並固定各任務欄位 ======
def build_and_save_tsfresh(train_info, train_data_folder):
    """
    第一次執行時呼叫：
     1) 把 train_data 轉成長表 → extract_features()
     2) impute()
     3) 分別對 4 個任務 select_features → 存成 JSON (tsf_cols_*.json)
     4) 把 X_tsf_full.pkl 儲存起來。之後直接讀雲端硬碟的檔案。
    """
    # (1) 構造長表
    records = []
    for _, r in train_info.iterrows():
        uid, cp = r['unique_id'], r['cut_point']
        arr = np.loadtxt(os.path.join(train_data_folder, f"{uid}.txt"))
        for t, row in enumerate(arr):
            # row: [Ax,Ay,Az,Gx,Gy,Gz]
            records.append((uid, t, *row))
    df_all = pd.DataFrame(records, columns=['unique_id','time','Ax','Ay','Az','Gx','Gy','Gz'])

    # (2) TSFresh 特徵
    efs = EfficientFCParameters()
    X_tsf = extract_features(
        df_all,
        column_id='unique_id',
        column_sort='time',
        default_fc_parameters=efs
    )
    X_tsf = impute(X_tsf)

    # (3) 讀 train_info 標籤
    y_gender = (train_info['gender'] == 1).astype(int).values
    y_hand   = (train_info['hold racket handed'] == 1).astype(int).values
    years_le = LabelEncoder().fit(train_info['play years'].values)
    level_le = LabelEncoder().fit(train_info['level'].values)
    y_years_enc = years_le.transform(train_info['play years'].values)
    y_level_enc = level_le.transform(train_info['level'].values)

    # (4) 分別 select_features → 存成 JSON
    tsf_cols_gender = select_features(X_tsf, y_gender).columns.tolist()
    tsf_cols_hand   = select_features(X_tsf, y_hand).columns.tolist()
    tsf_cols_years  = select_features(X_tsf, y_years_enc).columns.tolist()
    tsf_cols_level  = select_features(X_tsf, y_level_enc).columns.tolist()

    json.dump(tsf_cols_gender, open("tsf_cols_gender.json", "w"))
    json.dump(tsf_cols_hand,   open("tsf_cols_hand.json",   "w"))
    json.dump(tsf_cols_years,  open("tsf_cols_years.json",  "w"))
    json.dump(tsf_cols_level,  open("tsf_cols_level.json",  "w"))

    # (5) 把完整 X_tsf 存 pickle，之後直接 load
    pd.to_pickle(X_tsf, "X_tsf_full.pkl")
    print("✅ 已經儲存 X_tsf_full.pkl 及 tsf_cols_*.json，共四個任務的欄位清單。")

# ====== 4a) 載入已經存好的 TSFresh 欄位清單 ======
def load_tsf_cols(prefix="tsf_cols_"):
    """
    讀取同目錄底下的 JSON 檔：tsf_cols_gender.json, tsf_cols_hand.json, tsf_cols_years.json, tsf_cols_level.json
    回傳 dict：{'gender': [...], 'hand': [...], 'years': [...], 'level': [...]}
    """
    out = {}
    for task, fname in [
        ('gender', f"{prefix}gender.json"),
        ('hand',   f"{prefix}hand.json"),
        ('years',  f"{prefix}years.json"),
        ('level',  f"{prefix}level.json")
    ]:
        if os.path.exists(fname):
            out[task] = json.load(open(fname, "r"))
        else:
            raise FileNotFoundError(f"{fname} 不存在；請先執行 build_and_save_tsfresh()")
    return out

# ====== 5) 1D-CNN 定義：用於產生 embedding vector ======
    """
    建立一個簡單的 1D-CNN 分類模型，最後一層 dense 之前作為 embedding。
    input_length: 序列長度
    n_channels:   6
    n_classes:    2 (binary) or >2 (multiclass)
    回傳 keras Model，且 embedding_layer_name="embedding".
    """
def build_cnn_model(input_length, n_channels, n_classes):
    inp = layers.Input(shape=(input_length, n_channels), name="input_signal")
    x = layers.Conv1D(32, kernel_size=7, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.GlobalMaxPool1D()(x)
    x = layers.Dense(128, activation="relu", name="embedding")(x)  # embedding 層
    x = layers.Dropout(0.3)(x)

    if n_classes == 2:
        out = layers.Dense(1, activation="sigmoid", name="output")(x)
        model = models.Model(inputs=inp, outputs=out)
        model.compile(optimizer="adam", loss="binary_crossentropy")
    else:
        out = layers.Dense(n_classes, activation="softmax", name="output")(x)
        model = models.Model(inputs=inp, outputs=out)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model

# ====== 6) 讀所有原始波形並 pad/truncate 到相同長度 ======
def load_and_pad_raw(train_or_test, info_df, data_folder, max_len=None):
    """
    把 train or test 的每個 .txt 讀出，先收集所有長度，若 max_len=None 則先 pass，
    之後再求 max_len；再把每筆 signal pad or truncate 到 (max_len,6)，並回傳 X_raw (n_samples,max_len,6)。
    train_or_test: "train" 或 "test"（只用來印 log）。
    info_df: 搭配 .txt 的 DataFrame，一定要有欄位 'unique_id', 'cut_point' (雖然 cut_point 這裡不使用，只保留原始完整 signal)。
    data_folder: 對應的路徑，如 TRAIN_DATA 或 TEST_DATA。
    max_len: 如果不是 None，則直接用它做 pad/trunc；若 None，則先求出所有 Length，再回傳 max_len。
    回傳 (X_raw, max_len)。如果輸入的 max_len=None，則先掃完再決定；如果輸入已指定，就直接使用。
    """
    # (1) 如果沒有預先提供 max_len，就先掃一次收集 lengths
    lengths = []
    paths   = []
    for _, r in info_df.iterrows():
        uid = r['unique_id']
        pth = os.path.join(data_folder, f"{uid}.txt")
        if os.path.exists(pth):
            arr = np.loadtxt(pth)
            lengths.append(len(arr))
            paths.append(pth)
        else:
            raise FileNotFoundError(f"{pth} 不存在")
    if max_len is None:
        max_len = max(lengths)

    # (2) 再建 X_raw，shape = (n_samples, max_len, 6)
    n_samples = len(paths)
    X_raw = np.zeros((n_samples, max_len, 6), dtype=float)

    for i, pth in enumerate(paths):
        arr = np.loadtxt(pth)  # shape (L,6)
        L   = len(arr)
        if L >= max_len:
            X_raw[i, :, :] = arr[:max_len, :]
        else:
            X_raw[i, :L, :] = arr
            # 後面自動 fill 0

    return X_raw, max_len

# ====== 7) OOF + 融合 (Blending) Tool ======
    """
    對特徵 X, 標籤 y, 以及測試 X_test，分別用 XGB/LGB/CB 做 5-Fold OOF。
    回傳:
      oof_preds = (oof_xgb, oof_lgb, oof_cb)
      test_preds= (test_xgb, test_lgb, test_cb)
    若二分類，oof_xgb shape=(n_train,), test_xgb=(n_test,)
    若多分類，oof_xgb shape=(n_train,n_cls), test_xgb=(n_test,n_cls)
    """
def get_oof_and_test_preds(X, y, X_test, is_multiclass=False, random_seed=42):
    y_arr = np.array(y)
    classes = np.unique(y_arr)
    n_classes = len(classes) if is_multiclass else 2

    n_train = X.shape[0]
    n_test  = X_test.shape[0]

    # init containers
    if not is_multiclass:
        oof_xgb = np.zeros(n_train); test_xgb = np.zeros(n_test)
        oof_lgb = np.zeros(n_train); test_lgb = np.zeros(n_test)
        oof_cb  = np.zeros(n_train); test_cb  = np.zeros(n_test)
    else:
        oof_xgb = np.zeros((n_train, n_classes)); test_xgb = np.zeros((n_test, n_classes))
        oof_lgb = np.zeros((n_train, n_classes)); test_lgb = np.zeros((n_test, n_classes))
        oof_cb  = np.zeros((n_train, n_classes)); test_cb  = np.zeros((n_test, n_classes))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    for tr_idx, va_idx in skf.split(X, y_arr):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y_arr[tr_idx], y_arr[va_idx]

        # === XGB ===
        clf_xgb = XGBClassifier(
            use_label_encoder=False,
            eval_metric='mlogloss' if is_multiclass else 'logloss',
            objective='multi:softprob' if is_multiclass else 'binary:logistic',
            num_class=n_classes if is_multiclass else None,
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=random_seed
        )
        clf_xgb.fit(X_tr, y_tr)
        if not is_multiclass:
            oof_xgb[va_idx] = clf_xgb.predict_proba(X_va)[:, 1]
            test_xgb += clf_xgb.predict_proba(X_test)[:, 1] / skf.n_splits
        else:
            oof_xgb[va_idx, :] = clf_xgb.predict_proba(X_va)
            test_xgb += clf_xgb.predict_proba(X_test) / skf.n_splits

        # === LightGBM ===
        clf_lgb = LGBMClassifier(
            objective='multiclass' if is_multiclass else 'binary',
            num_class=n_classes if is_multiclass else None,
            metric='multi_logloss' if is_multiclass else 'binary_logloss',
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=random_seed
        )
        clf_lgb.fit(X_tr, y_tr)
        if not is_multiclass:
            oof_lgb[va_idx] = clf_lgb.predict_proba(X_va)[:, 1]
            test_lgb += clf_lgb.predict_proba(X_test)[:, 1] / skf.n_splits
        else:
            oof_lgb[va_idx, :] = clf_lgb.predict_proba(X_va)
            test_lgb += clf_lgb.predict_proba(X_test) / skf.n_splits

        # === CatBoost ===
        clf_cb = CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.05,
            verbose=0, random_seed=random_seed,
            loss_function='MultiClass' if is_multiclass else 'Logloss'
        )
        clf_cb.fit(X_tr, y_tr)
        if not is_multiclass:
            oof_cb[va_idx] = clf_cb.predict_proba(X_va)[:, 1]
            test_cb += clf_cb.predict_proba(X_test)[:, 1] / skf.n_splits
        else:
            oof_cb[va_idx, :] = clf_cb.predict_proba(X_va)
            test_cb += clf_cb.predict_proba(X_test) / skf.n_splits

    return (oof_xgb, oof_lgb, oof_cb), (test_xgb, test_lgb, test_cb)

# ====== 8) 找最優加權融合權重 ======
def find_best_weights(oof_preds, y_true, is_multiclass=False):
    """
    oof_preds: (oof_xgb, oof_lgb, oof_cb),
      each shape = (n_train,) for binary, (n_train,n_cls) for multiclass.
    y_true:    shape = (n_train,)
    is_multiclass: True→multi-class AUC, else binary AUC.
    回傳 [w_xgb, w_lgb, w_cb].
    """
    o_xgb, o_lgb, o_cb = oof_preds

    def loss_bin(w):
        w1, w2 = w
        w0 = 1 - w1 - w2
        blend = w0*o_xgb + w1*o_lgb + w2*o_cb
        return -roc_auc_score(y_true, blend)

    def loss_multi(w):
        w1, w2 = w
        w0 = 1 - w1 - w2
        blend = w0*o_xgb + w1*o_lgb + w2*o_cb
        return -roc_auc_score(y_true, blend, multi_class='ovr', average='micro')

    init = [0.33, 0.33]
    bounds = [(0,1),(0,1)]
    cons = (
        {'type':'ineq', 'fun': lambda w:  1 - w[0] - w[1]},  # w0 ≥ 0
        {'type':'ineq', 'fun': lambda w:  w[0] + w[1]}       # w0 ≤ 1
    )

    if is_multiclass:
        res = minimize(loss_multi, init, bounds=bounds, constraints=cons)
    else:
        res = minimize(loss_bin, init, bounds=bounds, constraints=cons)

    w1, w2 = res.x
    w0 = 1 - w1 - w2
    return np.array([w0, w1, w2])

# ====== 9) 主程式 ======
def main():
    # --- (A) 路徑設置 ---
    TRAIN_INFO = '/content/drive/MyDrive/work/tt/39_Training_Dataset/train_info.csv'
    TRAIN_DATA = '/content/drive/MyDrive/work/tt/39_Training_Dataset/train_data'
    TEST_INFO  = '/content/drive/MyDrive/work/tt/39_Test_Dataset/test_info.csv'
    TEST_DATA  = '/content/drive/MyDrive/work/tt/39_Test_Dataset/test_data'

    # --- (B) 讀 train_info, 構造標籤陣列 ---
    train_info = pd.read_csv(TRAIN_INFO)
    y_gender = (train_info['gender'] == 1).astype(int).values
    y_hand   = (train_info['hold racket handed'] == 1).astype(int).values
    years_le = LabelEncoder().fit(train_info['play years'].values)
    level_le = LabelEncoder().fit(train_info['level'].values)
    y_years_enc = years_le.transform(train_info['play years'].values)
    y_level_enc = level_le.transform(train_info['level'].values)

    # --- (C) 提取所有手工特徵 ---
    print(">>> Extract handcrafted features ...")
    hand_feats = []
    for _, r in train_info.iterrows():
        uid, cp = r['unique_id'], r['cut_point']
        path = os.path.join(TRAIN_DATA, f"{uid}.txt")
        hand_feats.append(extract_handcrafted(path, cp))
    X_hand = np.vstack(hand_feats)  # shape (n_train, D_hand)

    # --- (C.1) TSFresh：第一次要跑 build_and_save_tsfresh，之後用 load_tsf_cols() & pickle 讀檔 ---
    print(">>> Extract TSFresh features ...")
    # 如果還沒 build TSFresh就要跑這行
    # build_and_save_tsfresh(train_info, TRAIN_DATA)

    # 讀取 pickled 整個 TSFresh 特徵表
    X_tsf_full = pd.read_pickle("X_tsf_full.pkl")
    tsf_cols   = load_tsf_cols()  # {'gender':[...], 'hand':[...], 'years':[...], 'level':[...]}

    X_tsf_g = X_tsf_full[tsf_cols['gender']].values
    X_tsf_h = X_tsf_full[tsf_cols['hand']].values
    X_tsf_y = X_tsf_full[tsf_cols['years']].values
    X_tsf_l = X_tsf_full[tsf_cols['level']].values

    # 合併：手工 + TSFresh
    Xg = np.hstack([X_hand, X_tsf_g])
    Xh = np.hstack([X_hand, X_tsf_h])
    Xy = np.hstack([X_hand, X_tsf_y])
    Xl = np.hstack([X_hand, X_tsf_l])

    # --- (D) 讀取 test, 提取手工 + TSFresh 特徵 ---
    print(">>> Extract test handcrafted/TSFresh ...")
    test_info = pd.read_csv(TEST_INFO)
    hand_feats_test = []
    for _, r in test_info.iterrows():
        uid, cp = r['unique_id'], r['cut_point']
        hand_feats_test.append(extract_handcrafted(os.path.join(TEST_DATA, f"{uid}.txt"), cp))
    X_hand_test = np.vstack(hand_feats_test)

    # TSFresh test
    records_test = []
    for _, r in test_info.iterrows():
        uid, cp = r['unique_id'], r['cut_point']
        arr = np.loadtxt(os.path.join(TEST_DATA, f"{uid}.txt"))
        for t, row in enumerate(arr):
            records_test.append((uid, t, *row))
    df_test_all = pd.DataFrame(records_test, columns=['unique_id','time','Ax','Ay','Az','Gx','Gy','Gz'])

    efs = EfficientFCParameters()
    X_tsf_test_full = extract_features(
        df_test_all,
        column_id='unique_id',
        column_sort='time',
        default_fc_parameters=efs
    )
    X_tsf_test_full = impute(X_tsf_test_full)

    X_tsf_test_g = X_tsf_test_full[tsf_cols['gender']].values
    X_tsf_test_h = X_tsf_test_full[tsf_cols['hand']].values
    X_tsf_test_y = X_tsf_test_full[tsf_cols['years']].values
    X_tsf_test_l = X_tsf_test_full[tsf_cols['level']].values

    Xg_test = np.hstack([X_hand_test, X_tsf_test_g])
    Xh_test = np.hstack([X_hand_test, X_tsf_test_h])
    Xy_test = np.hstack([X_hand_test, X_tsf_test_y])
    Xl_test = np.hstack([X_hand_test, X_tsf_test_l])

    # --- (E) 特徵標準化（手工+TSFresh）---
    sg = StandardScaler().fit(Xg);   Xg_s      = sg.transform(Xg);   Xg_test_s = sg.transform(Xg_test)
    sh = StandardScaler().fit(Xh);   Xh_s      = sh.transform(Xh);   Xh_test_s = sh.transform(Xh_test)
    sy = StandardScaler().fit(Xy);   Xy_s      = sy.transform(Xy);   Xy_test_s = sy.transform(Xy_test)
    sl = StandardScaler().fit(Xl);   Xl_s      = sl.transform(Xl);   Xl_test_s = sl.transform(Xl_test)

    # --- (F) 準備 1D-CNN Embedding       先 load/pad 原始訊號，然後 train CNN，最後 extract embedding
    print(">>> Build CNN embeddings ...")
    # (F.1) 讀取所有原始訊號，pad/truncate 到同一長度
    X_raw_train, max_len = load_and_pad_raw("train", train_info, TRAIN_DATA, max_len=None)
    X_raw_test, _       = load_and_pad_raw("test",  test_info,  TEST_DATA,  max_len=max_len)
    # shape: (n_train, max_len, 6), (n_test, max_len, 6)

    # (F.2) 分別為四個任務建 CNN model，train on full train，extract embeddings
    def train_cnn_and_extract(X_raw, y, X_raw_test, is_multiclass=False):
        n_classes = len(np.unique(y)) if is_multiclass else 2
        model = build_cnn_model(input_length=max_len, n_channels=6, n_classes=n_classes)
        # callback 只留最小化驗證 loss，可自行微調
        early = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)
        # full-train on all train data
        if not is_multiclass:
            model.fit(X_raw, y, epochs=20, batch_size=32, callbacks=[early], verbose=2)
        else:
            model.fit(X_raw, y, epochs=20, batch_size=32, callbacks=[early], verbose=2)

        # 建立「到 embedding 層」的子 model
        embed_layer = model.get_layer("embedding").output
        embed_model = models.Model(inputs=model.input, outputs=embed_layer)

        # 產生 embedding
        emb_train = embed_model.predict(X_raw, batch_size=32, verbose=0)
        emb_test  = embed_model.predict(X_raw_test, batch_size=32, verbose=0)

        return emb_train, emb_test, model

    # Gender (binary)
    emb_g_train, emb_g_test, cnn_g = train_cnn_and_extract(X_raw_train, y_gender, X_raw_test, is_multiclass=False)
    # Hand   (binary)
    emb_h_train, emb_h_test, cnn_h = train_cnn_and_extract(X_raw_train, y_hand,   X_raw_test, is_multiclass=False)
    # Years  (3-class)
    emb_y_train, emb_y_test, cnn_y = train_cnn_and_extract(X_raw_train, y_years_enc, X_raw_test, is_multiclass=True)
    # Level  (4-class)
    emb_l_train, emb_l_test, cnn_l = train_cnn_and_extract(X_raw_train, y_level_enc, X_raw_test, is_multiclass=True)

    # (F.3) 分別對各 embedding 做標準化
    scaler_emb_g = StandardScaler().fit(emb_g_train); emb_g_train_s = scaler_emb_g.transform(emb_g_train); emb_g_test_s = scaler_emb_g.transform(emb_g_test)
    scaler_emb_h = StandardScaler().fit(emb_h_train); emb_h_train_s = scaler_emb_h.transform(emb_h_train); emb_h_test_s = scaler_emb_h.transform(emb_h_test)
    scaler_emb_y = StandardScaler().fit(emb_y_train); emb_y_train_s = scaler_emb_y.transform(emb_y_train); emb_y_test_s = scaler_emb_y.transform(emb_y_test)
    scaler_emb_l = StandardScaler().fit(emb_l_train); emb_l_train_s = scaler_emb_l.transform(emb_l_train); emb_l_test_s = scaler_emb_l.transform(emb_l_test)

    # (F.4) 合併手工+TSFresh 與 CNN embeddings
    Xg_comb      = np.hstack([Xg_s, emb_g_train_s])
    Xg_test_comb = np.hstack([Xg_test_s, emb_g_test_s])

    Xh_comb      = np.hstack([Xh_s, emb_h_train_s])
    Xh_test_comb = np.hstack([Xh_test_s, emb_h_test_s])

    Xy_comb      = np.hstack([Xy_s, emb_y_train_s])
    Xy_test_comb = np.hstack([Xy_test_s, emb_y_test_s])

    Xl_comb      = np.hstack([Xl_s, emb_l_train_s])
    Xl_test_comb = np.hstack([Xl_test_s, emb_l_test_s])

    # --- (G) OOF + Blending + Pseudo-Labeling（改用 X*_comb） ---
    print(">>> OOF + Blending + Pseudo-Labeling ...")

    # (G.1) Gender（二分類）
    (o_xgb_g, o_lgb_g, o_cb_g), (t_xgb_g, t_lgb_g, t_cb_g) = get_oof_and_test_preds(
        Xg_comb, y_gender, Xg_test_comb, is_multiclass=False
    )
    wg = find_best_weights((o_xgb_g, o_lgb_g, o_cb_g), y_gender, is_multiclass=False)
    print("Gender weights:", wg)
    blend_oof_g = wg[0]*o_xgb_g + wg[1]*o_lgb_g + wg[2]*o_cb_g
    print("Gender OOF AUC:", roc_auc_score(y_gender, blend_oof_g))
    blend_test_g = wg[0]*t_xgb_g + wg[1]*t_lgb_g + wg[2]*t_cb_g

    idx_g, y_g_pseudo = (np.where(blend_test_g > 0.99)[0],
                         (blend_test_g > 0.5).astype(int)[np.where(blend_test_g > 0.99)[0]])
    idx_g_low = np.where(blend_test_g < 0.01)[0]
    idx_g = np.concatenate([idx_g, idx_g_low])
    yg_pseudo = (blend_test_g[idx_g] > 0.5).astype(int)
    Xg_pseudo = Xg_test_comb[idx_g]

    # (G.2) Hand（二分類）
    (o_xgb_h, o_lgb_h, o_cb_h), (t_xgb_h, t_lgb_h, t_cb_h) = get_oof_and_test_preds(
        Xh_comb, y_hand, Xh_test_comb, is_multiclass=False
    )
    wh = find_best_weights((o_xgb_h, o_lgb_h, o_cb_h), y_hand, is_multiclass=False)
    print("Hand weights:", wh)
    blend_oof_h = wh[0]*o_xgb_h + wh[1]*o_lgb_h + wh[2]*o_cb_h
    print("Hand OOF AUC:", roc_auc_score(y_hand, blend_oof_h))
    blend_test_h = wh[0]*t_xgb_h + wh[1]*t_lgb_h + wh[2]*t_cb_h

    idx_h_high = np.where(blend_test_h > 0.99)[0]
    idx_h_low  = np.where(blend_test_h < 0.01)[0]
    idx_h = np.concatenate([idx_h_high, idx_h_low])
    yh_pseudo = (blend_test_h[idx_h] > 0.5).astype(int)
    Xh_pseudo = Xh_test_comb[idx_h]

    # (G.3) Years（3-class）
    (o_xgb_y, o_lgb_y, o_cb_y), (t_xgb_y, t_lgb_y, t_cb_y) = get_oof_and_test_preds(
        Xy_comb, y_years_enc, Xy_test_comb, is_multiclass=True
    )
    wy = find_best_weights((o_xgb_y, o_lgb_y, o_cb_y), y_years_enc, is_multiclass=True)
    print("Years weights:", wy)
    blend_oof_y = wy[0]*o_xgb_y + wy[1]*o_lgb_y + wy[2]*o_cb_y
    print("Years OOF AUC:", roc_auc_score(y_years_enc, blend_oof_y, multi_class='ovr', average='micro'))
    blend_test_y = wy[0]*t_xgb_y + wy[1]*t_lgb_y + wy[2]*t_cb_y

    idx_y = np.where(np.max(blend_test_y, axis=1) > 0.95)[0]
    yy_pseudo = np.argmax(blend_test_y[idx_y], axis=1)
    Xy_pseudo = Xy_test_comb[idx_y]

    # (G.4) Level（4-class）
    (o_xgb_l, o_lgb_l, o_cb_l), (t_xgb_l, t_lgb_l, t_cb_l) = get_oof_and_test_preds(
        Xl_comb, y_level_enc, Xl_test_comb, is_multiclass=True
    )
    wl = find_best_weights((o_xgb_l, o_lgb_l, o_cb_l), y_level_enc, is_multiclass=True)
    print("Level weights:", wl)
    blend_oof_l = wl[0]*o_xgb_l + wl[1]*o_lgb_l + wl[2]*o_cb_l
    print("Level OOF AUC:", roc_auc_score(y_level_enc, blend_oof_l, multi_class='ovr', average='micro'))
    blend_test_l = wl[0]*t_xgb_l + wl[1]*t_lgb_l + wl[2]*t_cb_l

    idx_l = np.where(np.max(blend_test_l, axis=1) > 0.95)[0]
    yl_pseudo = np.argmax(blend_test_l[idx_l], axis=1)
    Xl_pseudo = Xl_test_comb[idx_l]

    # --- (H) 加上偽標籤，以LGBM單模型訓練 ---
    print(">>> Pseudo-Labeling + Retrain LGBM ...")
    # Gender
    Xg_aug = np.vstack([Xg_comb, Xg_pseudo]); yg_aug = np.concatenate([y_gender, yg_pseudo])
    clf_g2 = LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, random_state=42)
    clf_g2.fit(Xg_aug, yg_aug)
    final_pred_g = clf_g2.predict_proba(Xg_test_comb)[:, list(clf_g2.classes_).index(1)]

    # Hand
    Xh_aug = np.vstack([Xh_comb, Xh_pseudo]); yh_aug = np.concatenate([y_hand, yh_pseudo])
    clf_h2 = LGBMClassifier(**clf_g2.get_params()); clf_h2.fit(Xh_aug, yh_aug)
    final_pred_h = clf_h2.predict_proba(Xh_test_comb)[:, list(clf_h2.classes_).index(1)]

    # Years
    Xy_aug = np.vstack([Xy_comb, Xy_pseudo]); yy_aug = np.concatenate([y_years_enc, yy_pseudo])
    clf_y2 = LGBMClassifier(**clf_g2.get_params()); clf_y2.fit(Xy_aug, yy_aug)
    final_pred_y = clf_y2.predict_proba(Xy_test_comb)

    # Level
    Xl_aug = np.vstack([Xl_comb, Xl_pseudo]); yl_aug = np.concatenate([y_level_enc, yl_pseudo])
    clf_l2 = LGBMClassifier(**clf_g2.get_params()); clf_l2.fit(Xl_aug, yl_aug)
    final_pred_l = clf_l2.predict_proba(Xl_test_comb)

    # --- (I) 產出最終 submission.csv ---
    print(">>> Generate submission CSV ...")
    sub = pd.DataFrame({'unique_id': test_info['unique_id']})
    sub['gender']             = np.round(final_pred_g, 6)
    sub['hold racket handed'] = np.round(final_pred_h, 6)

    for i, cls_idx in enumerate(clf_y2.classes_):
        orig = years_le.classes_[cls_idx]
        sub[f'play years_{orig}'] = np.round(final_pred_y[:, i], 6)

    for i, cls_idx in enumerate(clf_l2.classes_):
        orig = level_le.classes_[cls_idx]
        sub[f'level_{orig}'] = np.round(final_pred_l[:, i], 6)

    cols = [
        'unique_id',
        'gender', 'hold racket handed',
        'play years_0','play years_1','play years_2',
        'level_2','level_3','level_4','level_5'
    ]
    sub = sub[cols]
    sub.to_csv('/content/drive/MyDrive/work/tt/submission_advanced_cnn_blend_pseudo.csv', index=False, float_format='%.6f')
    print("✅ Finished: submission_advanced_cnn_blend_pseudo.csv")
# 在colab跑 怕存暫存會不見，絕對路徑存在雲端

if __name__ == "__main__":
    main()