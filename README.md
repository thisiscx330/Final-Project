# Final-Project
機器學習概論期末報告

AI CUP_桌球智慧球拍資料分析之選手預測系統


---------------------------------
檔案說明

Readme(test).txt：test data 說明

Readme(train).txt：train data 說明

aicup_advanced_cnn.py： 主要程式

aicup_final：執行程式(包含pip套建、tsfresh、aicup_advanced_cnn.py)

submission_advanced_cnn_blend_pseudo.csv：產生的預測結果

test_info.csv：測試資料

train_info.csv：訓練資料

tsf_cols_gender.json：性別特徵

tsf_cols_hand.json：持拍手特徵

tsf_cols_level.json：等級特徵

tsf_cols_years.json：球齡特徵

*aicup_final.py 中的 tsfesh 會產生四類別.json與X_tsf_full.pkl，X_tsf_full.pkl檔案太大，可至連結下載

*train_data、test_data資料夾 檔案太大，可至連結下載

*其他實驗過程與產出csv檔可至連結下載

https://drive.google.com/drive/folders/16TU6myqljwtWm0aBaYEk_PbcA2-p6BUh?usp=sharing

---------------------------------


實驗

I.	特徵工程

將原始六軸感測訊號切為 27 段，每段提取五類特徵：

(A) 時域統計（6 軸 × mean, var, RMS, max, min, peak-to-peak） = 36 維

(B) 頻域統計（加速度/角速度向量 × FFT、PSD、entropy、主頻率）= 8 維

(C) Wavelet (4層 × 平均、變異數、最大、最小 × 加速度/角速度合量)= 32 維

(D) Sliding Window（全段滑窗統計後取 min/max/mean/std）= 4 維

(E) 倒頻譜Cepstrum（取前 10 個係數）= 10 維

全局特徵（12 維）與 Dynamic Time Warping（1 維）

→總共 103 維

TSFresh特徵：對每段樣本抽取統計特徵並以 impute 處理缺值，select_features對四個任務選出不同特徵欄位。

分別建立專特徵欄位 JSON，儲存在 X_tsf_full.pkl

CNN：使用 1D-CNN 模型取中間層輸出作為 128 維嵌入向量。

特徵合併：將三種特徵（手工＋TSFresh＋CNN）合併，標準化後作為輸入。

---------------------------------



II.	5-Fold OOF
使用 XGBoost、LightGBM、CatBoost 建構三組 5-fold OOF 模型交叉訓練。
訓練資料分成五份4個訓練、1個驗證。

---------------------------------



III.	optimal weight ensemble
scipy.optimize.minimize最小化負 AUC 的方式尋找最適權重組合

---------------------------------



IV.	 Pseudo-labeling
根據融合後預測機率很大或很小的 高信心test 資料當偽標籤，加入訓練資料

---------------------------------



V.	LightGBM訓練，預測

---------------------------------



VI.	產出submission.csv結果


---------------------------------


實驗結果

輸出預測結果csv檔

安裝套件

設定路徑

Tsfresh特徵

訓練、預測結果

產出submission_advanced_cnn_blend_pseudo.csv

---------------------------------



AI CUP競賽排名

21/633

