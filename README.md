# MICHI-AI

MICHI-AI is an AGPL-licensed open-source AI system for road traffic analysis from roadside video data.
This project focuses on transparency and reproducibility for public infrastructure and research use.

MICHI-AIは、道路脇の映像データから道路交通状況を分析するための、AGPLライセンスに基づくオープンソースAIシステムです。
GUI から動画を読み込み、ライン通過数・時間帯別集計・車種別集計・CSV/JSON 出力を行います。

## 特徴

- YOLOv8 による車両検出
- ByteTrack による追跡と通過カウント
- 時間帯別集計 / 車種別集計
- CSV / JSON 出力
- 画像保存（オプション）
- GPU / TensorRT 対応（任意）

## 動作環境
確認できている範囲は記載のとおりです。他の環境でも動作する可能性はあります。
- Python 3.10+
- OS: Linux / Windows
- GPU 使用時: NVIDIA GPU + CUDA

## インストール

```bash
pip install -r requirements.txt
```

GPU 使用時（CUDA 11.8 の例）:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## フォルダ準備

```bash
python setup_folders.py
```

## YOLOv8モデルのダウンロード

[Ultralytics公式](https://docs.ultralytics.com/models/yolov8/)から任意のYOLOv8モデルをダウンロードし、`models/` フォルダに配置してください。

利用可能なモデル:
- `yolov8n.pt` (最軽量)
- `yolov8s.pt`
- `yolov8m.pt` (推奨)
- `yolov8l.pt`
- `yolov8x.pt` (最高精度)

## 使い方（GUI）

```bash
python main_gui.py
```

### GUI でできること

- 入力動画の選択
- 出力先の指定
- Count Line の設定
- GPU / TensorRT / バッチ推論の設定
- 車種判別の有効化

## 設定

主要な設定は `config.json` で行います。GUI から編集可能です。

例（抜粋）:

```json
{
  "model": {
    "model_file": "models/yolov8m.pt",
    "confidence_threshold": 0.3,
    "iou_threshold": 0.4,
    "image_size": 320
  },
  "performance": {
    "use_gpu": true,
    "use_tensorrt": false,
    "use_batch_inference": false,
    "batch_size": 4,
    "frame_skip": 0
  },
  "lines": {
    "mode": "dual",
    "up_line": {"start_x": 100, "start_y": 200, "end_x": 1400, "end_y": 200},
    "down_line": {"start_x": 100, "start_y": 300, "end_x": 1400, "end_y": 300}
  }
}
```

## 出力

- `results/` に CSV / JSON を出力
- 車両画像保存を有効にすると `vehicle_images/` 以下に保存

## 構成

主要モジュール:

- `main_gui.py`: GUI エントリ
- `video_processor.py`: 処理パイプライン
- `video_processing/`: 分割された処理モジュール
  - `writer.py`: 非同期動画出力
  - `image_saver.py`: 車両画像保存
  - `counts.py`: 時間帯/車種カウント
  - `recognition.py`: 認識結果管理
  - `detection.py`: 検出/トラッキング設定
  - `exporter.py`: CSV/JSON 出力

## ライセンス

AGPL-3.0

このソフトウェアは、GNU Affero General Public License v3.0 に基づいてライセンスされています。

This software is licensed under the GNU Affero General Public License v3.0.

If you modify and run this software as a network service,
you must provide the complete corresponding source code.

## Third-Party Libraries

MICHI-AI depends on the following third-party open-source libraries:

- PyTorch (torch, torchvision)  
  License: BSD-style License  
  https://pytorch.org/

- OpenCV (opencv-python)  
  License: Apache License 2.0  
  https://opencv.org/

- Ultralytics YOLOv8 (ultralytics)  
  License: AGPL-3.0  
  https://github.com/ultralytics/ultralytics

- Supervision  
  License: MIT License  
  https://github.com/roboflow/supervision

- Pillow  
  License: HPND License  
  https://python-pillow.org/

- NumPy  
  License: BSD License  
  https://numpy.org/

- PySide6 (Qt for Python)  
  License: LGPL-3.0  
  https://www.qt.io/qt-for-python

- tqdm  
  License: MIT License  
  https://github.com/tqdm/tqdm

Optional components:

- NVIDIA TensorRT  
  License: NVIDIA Proprietary License  
  https://developer.nvidia.com/tensorrt

Each third-party library is subject to its own license.
Please review the respective licenses before use.

## 免責事項

本ソフトウェアは「現状有姿」で提供され、明示または黙示を問わず、いかなる保証も行いません。
作者および貢献者は、本ソフトウェアの使用により生じたいかなる損害に対しても責任を負いません。

This software is provided "as is" without warranty of any kind, express or implied.
The authors and contributors shall not be liable for any damages arising from the use of this software.