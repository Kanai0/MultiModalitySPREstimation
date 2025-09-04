# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 16:35:48 2025

@author: Kanai
"""

import os
import numpy as np
import pydicom
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from viewer_3d_gui import SliceViewer3D
from ANNmodel import build_full_model, HybridLoss
import threading
import multiprocessing
import tensorflow as tf
from keras import layers, models, losses, optimizers, Input
from keras import backend as K
import numpy as np


def load_dicom_series(folder_path):
    dicom_files = [pydicom.dcmread(os.path.join(folder_path, f)) 
                   for f in os.listdir(folder_path) if f.endswith(".dcm")]

    # スライス位置でソート（ImagePositionPatientがない場合はInstanceNumberで）
    try:
        dicom_files.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except:
        dicom_files.sort(key=lambda x: int(x.InstanceNumber))

    # 画像サイズと枚数を取得して空の配列を作成
    image_shape = dicom_files[0].pixel_array.shape
    volume_shape = (len(dicom_files), image_shape[0], image_shape[1])
    volume = np.zeros(volume_shape, dtype=dicom_files[0].pixel_array.dtype)

    for i, dcm in enumerate(dicom_files):
        volume[i, :, :] = dcm.pixel_array

    return volume

def load_npz_as_array(folder_path):

    npz_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npz')])
    if not npz_files:
        print("npzファイルが見つかりません。")
        return

    array_list = []
    for fname in npz_files:
        path = os.path.join(folder_path, fname)
        data = np.load(path)
        array_list.append(data['arr_0'])
    return np.stack(array_list, axis=0)  # shape: (スライス数, H, W)

def launch_viewer(volume, title):
    root = tk.Tk()
    app = SliceViewer3D(root, volume, title=title)
    root.mainloop()

def launch_viewers(volumes_with_titles):
    processes = []
    for volume, title in volumes_with_titles:
        p = multiprocessing.Process(target=launch_viewer, args=(volume, title))
        p.start()
        processes.append(p)

    # 任意で終了待機（メインプロセスがすぐに終わらないように）
    for p in processes:
        p.join()
# 2つのGUIを同時に起動（必要に応じて増やせます）

dir_simCT = r'C:/Users/Kanai/Synology/TWMU/0_張先生_共同研究_MRI阻止能/FromIzo/phantoms/simCT_80kVp/'
dir_theoCT = r'C:/Users/Kanai/Synology/TWMU/0_張先生_共同研究_MRI阻止能/FromIzo/phantoms/theoCT_80kVp/'

print("DICOM読み込み中...")
simCT = load_dicom_series(dir_simCT)
theoCT = load_npz_as_array(dir_theoCT)

# volumes_with_titles = [
#     (simCT, "Simulated CT"),
#     (theoCT, "Theoretical CT")
# ]
# launch_viewers(volumes_with_titles)
    


## ファイルを読み込む
#data = np.load(file_path)
#
## 含まれる配列の名前を一覧表示
#print("含まれている配列の名前:")
#print(data.files)
#
#
#for name in data.files:
#    print(f"\n{name} の中身:")
#    print(data[name])
#
#
#arr = data['arr_0']
#
#

# --- ハイパーパラメータ ---
input_dim = (7,1)  # MRI+DECTチャネル数: T1DF, T1DW, T2-STIR, HighE, LowE, ρₑ, VMI
delta = 1e-6    # 1: physics-constrained loss, 0: standard loss



# --- モデルコンパイル ---
model = build_full_model(input_dim)
loss_fn = HybridLoss(delta=delta, z_3_62=7.0, z_1_86=6.5)  # 実験値で調整
model.compile(optimizer=optimizers.Adam(lr=1e-4), loss=loss_fn)

# --- ダミーデータでテスト ---
n_vox = 1000
X_dummy = np.random.rand(n_vox, input_dim[0], 1).astype(np.float32)
rho_gt = np.random.uniform(0.8, 2.0, size=(n_vox, 1)).astype(np.float32)
ct_gt = np.random.uniform(-100, 2000, size=(n_vox, 1)).astype(np.float32)
y_dummy = np.concatenate([rho_gt, ct_gt], axis=-1)

# --- 学習 ---
model.fit(X_dummy, y_dummy, epochs=5)

