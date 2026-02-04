"""
動画処理とYOLOベースの車両検出・カウント処理
"""

import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
import torch
import json
import csv
from datetime import datetime, timedelta, time
from pathlib import Path
import threading
import queue
import time as time_module
import warnings
import os

from video_processing import (
    AsyncVideoWriter,
    VehicleImageSaver,
    CountManager,
    RecognitionResultsManager,
    setup_detection_components,
    ResultsExporter,
)

# TensorRTの警告を抑制
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=UserWarning, module='torch')


class VideoProcessor:
    """動画処理とYOLOベースの車両検出・追跡を行うクラス"""
    
    def __init__(self, log_callback=None, progress_callback=None, count_callback=None, frame_callback=None):
        """
        Args:
            log_callback: ログ出力用のコールバック関数
            progress_callback: プログレス更新用のコールバック関数
            count_callback: カウント更新用のコールバック関数
            frame_callback: フレーム更新用のコールバック関数
        """
        self.log = log_callback if log_callback else print
        self.update_progress = progress_callback if progress_callback else lambda x: None
        self.update_counts = count_callback if count_callback else lambda x, y, z, v=None: None
        self.update_frame = frame_callback if frame_callback else lambda x: None
        self.is_processing = False
        self.is_paused = False
        
        # 車両画像保存用の変数
        self.input_video_name = None
        self.video_fps = None
        self.vehicle_line_history = {}  # 車両のライン通過履歴 {track_id: {'line_type': last_frame_idx}}
        self.image_saver = VehicleImageSaver(log_func=self.log)
        
        # 認識結果/時間帯/車種の管理
        self.recognition_manager = RecognitionResultsManager(log_func=self.log)
        self.count_manager = CountManager(log_func=self.log)
        self.results_exporter = ResultsExporter(log_func=self.log)
        
        # 動的ライン更新用
        self.current_line_zones = None
        self.current_line_annotators = None
        self.current_config = None
        self.current_fps = None
        
        # ラインフラッシュ用
        self.line_flash_frames = {}  # {line_type: flash_end_frame_idx}
        self.flash_duration_frames = 6  # フラッシュ継続フレーム数（約0.2秒@30fps）
        
        # モデル関連（TensorRT対応）
        self.model = None
        self.class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}  # デフォルト
        
        # 一時停止用イベント
        import threading
        self.pause_event = threading.Event()
        self.pause_event.set()  # 初期状態は処理続行
        
        # 停止フラグ
        self.stop_flag = False
        self.output_frame_stride = 1
        self.output_video_writer = None
    
    def _cleanup_gpu_memory(self):
        """GPUメモリを解放"""
        try:
            # モデルをクリア
            if self.model is not None:
                del self.model
                self.model = None
            
            # 車両分類器のモデルをクリア
            if self.count_manager.vehicle_classifier is not None and hasattr(self.count_manager.vehicle_classifier, 'model'):
                if self.count_manager.vehicle_classifier.model is not None:
                    del self.count_manager.vehicle_classifier.model
                    self.count_manager.vehicle_classifier.model = None
            
            # PyTorchのGPUキャッシュをクリア
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                self.log("✓ GPUメモリを解放しました")
        except Exception as e:
            self.log(f"⚠️ GPUメモリ解放中にエラー: {e}")
    
    def process_video(self, config):
        """動画処理のメイン処理"""
        try:
            self.is_processing = True
            self.stop_flag = False  # 処理開始時にリセット
            
            # 車両管理用変数を初期化
            self.vehicle_line_history = {}  # 車両のライン通過履歴
            self.recent_saved_positions = {}  # 最近保存した車両の位置記録（重複防止用）
            
            # 時間帯別集計/車種判別の初期化
            self.count_manager.setup_time_settings(config)
            self.count_manager.setup_vehicle_classification(config)
            
            # フラッシュ状態をリセット
            self.line_flash_frames = {}
            recognition_config = config.get('recognition_results', {})  # 認識結果出力設定
            
            self.log("=" * 60)
            self.log("MICHI-AI - 処理開始")
            self.log("=" * 60)
            
            # TensorRTとバッチ推論の設定（モデル読み込み前にチェック）
            use_tensorrt = config['performance'].get('use_tensorrt', False)
            use_batch_inference = config['performance'].get('use_batch_inference', False)
            batch_size = config['performance'].get('batch_size', 4)
            tensorrt_precision = config['performance'].get('tensorrt_precision', 'fp16')
            
            # TensorRT + バッチ推論の組み合わせ対応
            # 非同期推論とバッチ推論は排他的
            use_async_inference = config['performance'].get('use_async_inference', False)
            if use_batch_inference and use_async_inference:
                self.log("⚠️ 警告: バッチ推論と非同期推論は同時に使用できません")
                self.log("   バッチ推論を優先し、非同期推論を無効化します")
                use_async_inference = False
                config['performance']['use_async_inference'] = False
            
            # バッチ推論使用時のメモリ使用についての説明
            if use_batch_inference and use_tensorrt:
                self.log("ℹ️ バッチ推論モード:")
                self.log(f"   バッチサイズ: {batch_size}")
                self.log("   ※TensorRTはバッチ処理用に複数のコンテキストを作成します")
                self.log("   ※起動時のGPUメモリ使用量は通常の約2-3倍になりますが正常です")
            
            # INT8 + バッチ推論の組み合わせチェック
            if use_tensorrt and use_batch_inference and tensorrt_precision == 'int8':
                self.log("⚠️ 警告: INT8量子化とバッチ推論の組み合わせは制限があります")
                self.log("   INT8キャリブレーションにはバッチサイズ以上の画像が必要です")
                self.log("   推奨: tensorrt_precision='fp16' に変更してください")
                self.log(f"   現在の設定を続行します（バッチサイズ: {batch_size}）")
            
            # TensorRT + バッチ推論の場合は情報を表示
            if use_tensorrt and use_batch_inference:
                self.log(f"✓ TensorRT + バッチ推論モード（バッチサイズ: {batch_size}）")
                if tensorrt_precision == 'fp16':
                    self.log("  推奨設定: FP16 + バッチ推論で最適なパフォーマンス")
            
            # デバイス設定
            device = self._setup_device(config)
            
            # モデル読み込み
            model = self._load_model(config, device)
            
            # 動画読み込み
            cap, video_info = self._load_video(config)
            w, h, fps, total_frames = video_info

            # 入力動画情報を保存（ファイル名生成用）
            self.input_video_name = Path(config['video']['input_file']).stem
            self.video_fps = fps
            self.count_manager.set_current_fps(fps)
            
            # 出力動画設定
            out = self._setup_output_video(config, fps, w, h)
            
            # ライン・トラッカー・アノテーター設定
            line_zones, line_annotators, tracker, box_annotator, label_annotator = setup_detection_components(config, fps, log_func=self.log)
            
            # 現在の設定を保存（動的更新用）
            self.current_line_zones = line_zones
            self.current_line_annotators = line_annotators
            self.current_config = config
            self.current_fps = fps
            
            # 車両画像保存設定
            self._setup_vehicle_images(config)
            self.recognition_manager.configure(
                recognition_config=recognition_config,
                video_fps=self.video_fps,
                video_start_time=self.count_manager.video_start_time,
                class_names=self.class_names,
                input_video_name=self.input_video_name,
            )
            self.results_exporter.configure(
                count_manager=self.count_manager,
                image_saver=self.image_saver,
                recognition_manager=self.recognition_manager,
            )
            
            # 処理パラメータ
            vehicle_classes = set(config['detection']['vehicle_classes'])
            conf_thresh = config['model']['confidence_threshold']
            iou_thresh = config['model']['iou_threshold']
            img_size = config['model']['image_size']
            frame_skip = config['performance']['frame_skip']
            
            # 非同期推論の設定
            use_async_inference = config['performance'].get('use_async_inference', False)
            frame_queue_size = config['performance'].get('frame_queue_size', 8)
            
            # バッチ推論の設定は既に上で定義済み（use_batch_inference, batch_size）
            
            # 処理モードの選択（優先順位: 非同期 > バッチ > 通常）
            if use_async_inference and not use_batch_inference:
                self.log(f"✓ 非同期推論モード（キューサイズ: {frame_queue_size}）")
                self.log("  フレーム読み込みと推論を並列実行します")
            elif use_batch_inference:
                self.log(f"✓ バッチ推論モード（バッチサイズ: {batch_size}）")
            else:
                self.log("✓ 通常推論モード")
            
            self.log("-" * 60)
            self.log("処理中...")
            
            # 処理開始時刻を記録（完了予想計算用）
            self.processing_start_time = datetime.now()
            self.last_progress_time = self.processing_start_time
            self.last_frame_idx = 0
            
            # フレーム処理ループ（非同期 > バッチ > 通常の優先順位）
            if use_async_inference and not use_batch_inference:
                processing_time = self._process_frames_async(
                    cap, out, model, self.current_line_zones, self.current_line_annotators, tracker, box_annotator, label_annotator,
                    vehicle_classes, conf_thresh, iou_thresh, img_size, frame_skip, total_frames, frame_queue_size
                )
            elif use_batch_inference:
                processing_time = self._process_frames_batch(
                    cap, out, model, self.current_line_zones, self.current_line_annotators, tracker, box_annotator, label_annotator,
                    vehicle_classes, conf_thresh, iou_thresh, img_size, frame_skip, total_frames, batch_size
                )
            else:
                processing_time = self._process_frames(
                    cap, out, model, self.current_line_zones, self.current_line_annotators, tracker, box_annotator, label_annotator,
                    vehicle_classes, conf_thresh, iou_thresh, img_size, frame_skip, total_frames
                )
            
            cap.release()
            if out is not None:
                out.release()
            
            # モデルとGPUメモリをクリア（処理完了時）
            if not self.stop_flag:  # 正常終了の場合のみログ表示
                self.log("✓ 処理が完了しました")
            
            # 結果出力
            self.results_exporter.output_results(config, self.current_line_zones, total_frames, fps, processing_time)
            
            # 車両画像保存統計を出力
            if self.image_saver.vehicle_images_config.get('save_images', False):
                self.image_saver.output_statistics(self.current_line_zones, self.vehicle_line_history)
            
            # 画像保存ワーカースレッドを停止（すべての保存が完了するまで待機）
            self.image_saver.stop()
            
            return True
            
        except Exception as e:
            self.log(f"✗ エラー: {e}")
            import traceback
            self.log(traceback.format_exc())
            return False
        
        finally:
            self.is_processing = False
            # 画像保存ワーカースレッドを停止
            self.image_saver.stop()
            # GPUメモリを解放
            self._cleanup_gpu_memory()
    
    def stop_processing(self):
        """処理を停止"""
        self.stop_flag = True
        self.is_processing = False
        self.is_paused = False
        self.pause_event.set()  # 一時停止中でも確実に終了させる
        
        # GPUメモリを解放
        self._cleanup_gpu_memory()
    
    def pause_processing(self):
        """処理を一時停止"""
        self.is_paused = True
        self.pause_event.clear()  # イベントをクリアして一時停止
        self.log("⏸ 処理を一時停止しました")
    
    def resume_processing(self):
        """処理を再開"""
        self.is_paused = False
        self.pause_event.set()  # イベントをセットして再開
        self.log("▶ 処理を再開しました")
    
    def update_line_configuration(self, config, fps):
        """一時停止中にライン設定を更新"""
        if not self.is_paused:
            return
        
        # 現在のカウント情報をログに記録
        if self.current_line_zones:
            total_counts = []
            for line_type, line_zone in self.current_line_zones.items():
                current_count = line_zone.in_count + line_zone.out_count
                total_counts.append(f"{line_type}:{current_count}")
            self.log(f"ライン再設定前のカウント - {', '.join(total_counts)}")
        
        # 新しいライン設定を適用
        new_line_zones, new_line_annotators, _, _, _ = setup_detection_components(config, fps, log_func=self.log)
        
        # 現在の設定を更新
        self.current_line_zones = new_line_zones
        self.current_line_annotators = new_line_annotators
        self.current_config = config
        
        self.log("Count Line設定を更新しました（カウントはリセットされます）")
    
    def _setup_device(self, config):
        """デバイス設定"""
        if config['performance']['use_gpu'] and torch.cuda.is_available():
            device = 'cuda'
            self.log(f"✓ GPU を使用: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            self.log("✓ CPU を使用")
        return device
    
    def _load_model(self, config, device):
        """モデル読み込み（TensorRT対応、FP16/INT8量子化サポート、バッチ推論対応）"""
        model_file = config['model']['model_file']
        
        # 既に同じモデルがロードされているかチェック
        if self.model is not None:
            # 同じモデルファイルの場合は再利用
            try:
                if hasattr(self, '_loaded_model_file') and self._loaded_model_file == model_file:
                    self.log(f"✓ モデルは既にロード済み: {model_file}")
                    return self.model
            except:
                pass
            
            # 異なるモデルの場合はクリア
            self.log("既存のモデルをクリアしています...")
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 現在のモデルファイルを記録
        self._loaded_model_file = model_file
        use_tensorrt = config['performance'].get('use_tensorrt', False)
        img_size = config['model']['image_size']  # config.jsonから入力サイズを取得
        tensorrt_precision = config['performance'].get('tensorrt_precision', 'fp16')  # fp16 or int8
        tensorrt_workspace = config['performance'].get('tensorrt_workspace', 4)  # GB
        use_batch_inference = config['performance'].get('use_batch_inference', False)
        batch_size = config['performance'].get('batch_size', 4)
        
        self.log(f"モデル読み込み中: {model_file}")
        self.log(f"  入力画像サイズ: {img_size}")
        
        # TensorRTエンジンファイルのパスを生成（入力サイズ、精度、バッチサイズを含める）
        model_path = Path(model_file)
        if use_tensorrt and use_batch_inference:
            engine_file = str(model_path.parent / f"{model_path.stem}_imgsz{img_size}_{tensorrt_precision}_batch{batch_size}.engine")
        else:
            engine_file = str(model_path.parent / f"{model_path.stem}_imgsz{img_size}_{tensorrt_precision}.engine")
        
        # TensorRTが有効で、GPUが利用可能な場合
        if use_tensorrt and device == 'cuda':
            # 精度設定の検証
            if tensorrt_precision not in ['fp16', 'int8']:
                self.log(f"⚠️ 無効なTensorRT精度設定: {tensorrt_precision}")
                self.log("   デフォルトのFP16を使用します")
                tensorrt_precision = 'fp16'
                engine_file = str(model_path.parent / f"{model_path.stem}_imgsz{img_size}_fp16.engine")
            
            self.log(f"  TensorRT精度: {tensorrt_precision.upper()}")
            
            # 既存の異なる設定のエンジンファイルをチェック
            existing_engines = list(model_path.parent.glob(f"{model_path.stem}_imgsz*_*.engine"))
            if existing_engines:
                existing_configs = []
                for eng in existing_engines:
                    # ファイル名から設定を抽出
                    try:
                        stem_parts = eng.stem.split('_imgsz')
                        if len(stem_parts) > 1:
                            config_part = stem_parts[1]  # "640_fp16" or "512_int8"
                            if config_part != f"{img_size}_{tensorrt_precision}":
                                existing_configs.append(config_part)
                    except:
                        pass
                
                if existing_configs:
                    self.log(f"ℹ️ 他の設定のTensorRTエンジンが存在します: {existing_configs}")
                    self.log(f"   現在の設定: imgsz{img_size}_{tensorrt_precision}")
            
            # エンジンファイルが存在するかチェック
            if Path(engine_file).exists():
                self.log(f"✓ TensorRTエンジンを検出: {Path(engine_file).name}")
                try:
                    # TensorRTのverboseログを一時的に抑制
                    import logging
                    trt_logger = logging.getLogger('tensorrt')
                    original_level = trt_logger.level
                    trt_logger.setLevel(logging.ERROR)
                    
                    model = YOLO(engine_file, verbose=False)
                    
                    # ログレベルを元に戻す
                    trt_logger.setLevel(original_level)
                    
                    self.log(f"✓ TensorRTエンジン読み込み完了（{tensorrt_precision.upper()}モード、入力サイズ: {img_size}）")
                    self.model = model  # インスタンス変数に保存
                    
                    # クラス名を保存（TensorRT対応）
                    try:
                        if hasattr(model, 'names'):
                            self.class_names = model.names
                        elif hasattr(model, 'model') and hasattr(model.model, 'names'):
                            self.class_names = model.model.names
                        else:
                            self.class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
                    except:
                        self.class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
                    
                    return model
                except Exception as e:
                    self.log(f"⚠️ TensorRTエンジン読み込み失敗: {e}")
                    self.log("通常のモデルを使用します")
            else:
                # エンジンファイルが存在しない場合、変換を試みる
                self.log(f"TensorRTエンジンが見つかりません")
                self.log(f"  設定: 入力サイズ={img_size}, 精度={tensorrt_precision.upper()}")
                self.log("変換を開始します...")
                
                try:
                    # 通常モデルをロード
                    model = YOLO(model_file)
                    model.to(device)
                    
                    # バッチ推論の設定を取得
                    use_batch_inference = config['performance'].get('use_batch_inference', False)
                    batch_size = config['performance'].get('batch_size', 4)
                    
                    # TensorRTエンジンにエクスポート
                    if tensorrt_precision == 'int8':
                        # INT8量子化の場合
                        self.log(f"TensorRTエンジンに変換中（INT8量子化、入力サイズ: {img_size}）")
                        if use_batch_inference:
                            self.log(f"  バッチ推論対応（バッチサイズ: 1-{batch_size}）")
                        self.log("⚠️ INT8変換はFP16より時間がかかります（5-15分程度）...")
                        self.log(f"  ワークスペースサイズ: {tensorrt_workspace}GB")
                        
                        # INT8エクスポート（キャリブレーションデータは自動生成）
                        export_params = {
                            'format': 'engine',
                            'half': False,  # INT8の場合はhalfをFalseに
                            'int8': True,   # INT8量子化を有効化
                            'device': device,
                            'imgsz': img_size,
                            'workspace': tensorrt_workspace,
                            'verbose': False
                        }
                        
                        # バッチ推論を使用する場合
                        # 注意: INT8キャリブレーションではバッチサイズ以上の画像が必要
                        # バッチサイズ指定なしで変換し、推論時に動的バッチを使用
                        if use_batch_inference:
                            # dynamic=Trueでバッチサイズなしでエクスポートすると、
                            # 推論時に任意のバッチサイズを使用できる
                            export_params['dynamic'] = True
                            self.log(f"  動的バッチサイズ対応（推論時: 1-{batch_size}）")
                        
                        try:
                            model.export(**export_params)
                        except Exception as e:
                            # キャリブレーションエラーの場合、バッチサイズを1に制限して再試行
                            if 'calibration' in str(e).lower() or 'batch' in str(e).lower():
                                self.log(f"⚠️ キャリブレーションエラー: {e}")
                                self.log("  バッチサイズを1で再試行します...")
                                export_params.pop('batch', None)
                                export_params['dynamic'] = False
                                model.export(**export_params)
                                
                                # バッチ推論を無効化
                                if use_batch_inference:
                                    self.log("⚠️ INT8モードではバッチ推論を使用できません")
                                    self.log("  config.jsonでtensorrt_precision='fp16'に変更するか、")
                                    self.log("  use_batch_inference=falseに設定してください")
                                    config['performance']['use_batch_inference'] = False
                            else:
                                raise
                    else:
                        # FP16量子化の場合
                        self.log(f"TensorRTエンジンに変換中（FP16、入力サイズ: {img_size}）")
                        if use_batch_inference:
                            self.log(f"  バッチ推論対応（バッチサイズ: 1-{batch_size}）")
                        self.log("  数分かかる場合があります...")
                        self.log(f"  ワークスペースサイズ: {tensorrt_workspace}GB")
                        
                        export_params = {
                            'format': 'engine',
                            'half': True,   # FP16を有効化
                            'device': device,
                            'imgsz': img_size,
                            'workspace': tensorrt_workspace,
                            'verbose': False
                        }
                        
                        # バッチ推論を使用する場合は動的バッチサイズを設定
                        if use_batch_inference:
                            export_params['dynamic'] = True
                            export_params['batch'] = batch_size
                        
                        model.export(**export_params)
                    
                    # exportメソッドは元のファイル名に.engineを付けるので、リネームが必要
                    default_engine = str(Path(model_file).with_suffix('.engine'))
                    if Path(default_engine).exists() and default_engine != engine_file:
                        import shutil
                        shutil.move(default_engine, engine_file)
                        self.log(f"✓ エンジンファイルをリネーム: {Path(engine_file).name}")
                    
                    # 変換されたエンジンをロード
                    if Path(engine_file).exists():
                        # TensorRTのverboseログを抑制
                        import logging
                        trt_logger = logging.getLogger('tensorrt')
                        original_level = trt_logger.level
                        trt_logger.setLevel(logging.ERROR)
                        
                        model = YOLO(engine_file, verbose=False)
                        
                        trt_logger.setLevel(original_level)
                        self.log(f"✓ TensorRTエンジン変換・読み込み完了")
                        self.log(f"  ファイル: {Path(engine_file).name}")
                        self.log(f"  精度: {tensorrt_precision.upper()}")
                        self.log(f"  入力サイズ: {img_size}（固定）")
                        
                        # エンジンファイルサイズを表示
                        file_size_mb = Path(engine_file).stat().st_size / (1024 * 1024)
                        self.log(f"  ファイルサイズ: {file_size_mb:.1f} MB")
                        self.log("  次回起動時から高速化されます")
                        
                        self.model = model
                        
                        # クラス名を保存
                        try:
                            if hasattr(model, 'names'):
                                self.class_names = model.names
                            elif hasattr(model, 'model') and hasattr(model.model, 'names'):
                                self.class_names = model.model.names
                            else:
                                self.class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
                        except:
                            self.class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
                        
                        return model
                    else:
                        self.log("⚠️ エンジンファイルが生成されませんでした")
                except Exception as e:
                    self.log(f"⚠️ TensorRT変換失敗: {e}")
                    self.log("通常のモデルを使用します")
                    import traceback
                    self.log(traceback.format_exc())
        
        # 通常モード（TensorRT未使用またはCPU）
        model = YOLO(model_file, verbose=False)
        model.to(device)
        self.log("✓ モデル読み込み完了（通常モード）")
        self.model = model  # インスタンス変数に保存
        
        # クラス名を保存（TensorRT対応）
        try:
            if hasattr(model, 'names'):
                self.class_names = model.names
            elif hasattr(model, 'model') and hasattr(model.model, 'names'):
                self.class_names = model.model.names
            else:
                # デフォルトのCOCOクラス名
                self.class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        except:
            self.class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        
        return model
    
    def _load_video(self, config):
        """動画読み込み"""
        # CUDA版VideoCapture使用オプションを確認
        use_cuda = config['video'].get('use_cuda_videocapture', False)
        
        if use_cuda:
            try:
                # CUDA バックエンドで VideoCapture を初期化
                cap = cv2.VideoCapture(config['video']['input_file'], cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
                self.log("✓ CUDA版 VideoCapture を初期化中...")
                
                # 正常に開けたか確認
                if not cap.isOpened():
                    self.log("⚠ CUDA VideoCapture の初期化に失敗、CPU版にフォールバック")
                    cap = cv2.VideoCapture(config['video']['input_file'])
                else:
                    self.log("✓ CUDA版 VideoCapture を使用")
            except Exception as e:
                self.log(f"⚠ CUDA VideoCapture エラー: {e}, CPU版にフォールバック")
                cap = cv2.VideoCapture(config['video']['input_file'])
        else:
            cap = cv2.VideoCapture(config['video']['input_file'])
        
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.log(f"✓ 入力動画: {config['video']['input_file']}")
        self.log(f"  解像度: {w}x{h}, FPS: {fps:.2f}, フレーム数: {total_frames}")
        
        return cap, (w, h, fps, total_frames)
    
    def _setup_output_video(self, config, fps, w, h):
        """出力動画設定"""
        # 動画出力が有効かチェック
        if config['video'].get('enable_output', True):
            output_cfg = config['video']

            # 出力パスを正規化
            raw_output_path = output_cfg.get('output_file', 'videos/output/output.mp4')
            output_path = Path(raw_output_path)

            # ディレクトリ指定判定
            hinted_directory = str(raw_output_path).endswith(('/', '\\'))
            if output_path.exists() and output_path.is_dir() or hinted_directory:
                # ディレクトリの場合は末尾フォルダ名や動画名からファイル名を生成
                base_name = self.input_video_name or (output_path.name if output_path.name else 'output')
                # directory path should not include filename yet
                directory = output_path if output_path.exists() and output_path.is_dir() else output_path
                output_path = directory / f"{base_name}.mp4"
                self.log(f"⚠️ 指定された出力パスがフォルダのため '{base_name}.mp4' を作成します: {output_path}")

            if not output_path.suffix:
                output_path = output_path.with_suffix('.mp4')
                self.log(f"⚠️ 出力ファイルに拡張子が無いため '.mp4' を付与しました: {output_path}")

            output_path.parent.mkdir(parents=True, exist_ok=True)
            # コンフィグを書き換えて後段処理とログが一致するようにする
            config['video']['output_file'] = str(output_path)

            target_fps = output_cfg.get('output_fps')
            input_fps = fps if fps and fps > 0 else target_fps or 30

            # 出力FPSとフレーム間引き幅を計算
            if target_fps and target_fps > 0:
                stride = max(1, int(round(input_fps / target_fps)))
            else:
                stride = 1
                target_fps = input_fps

            effective_fps = input_fps / stride if stride > 0 else input_fps
            self.output_frame_stride = max(1, stride)

            default_queue = max(1, int(input_fps) * 4)
            max_queue_size = max(1, output_cfg.get('output_queue_size', default_queue))

            async_writer = AsyncVideoWriter(
                str(output_path),
                effective_fps,
                (w, h),
                codec=output_cfg.get('codec', 'mp4v'),
                max_queue_size=max_queue_size,
                log_func=self.log
            )
            self.output_video_writer = async_writer

            self.log(f"✓ 出力動画: {output_cfg['output_file']}")
            self.log(f"  FPS: {effective_fps:.2f}（入力{input_fps:.2f}fps → {self.output_frame_stride}フレームごとに記録）")
            self.log(f"  書き出しキューサイズ: {max_queue_size}")
            return async_writer
        else:
            self.log("✓ 動画出力は無効です")
            self.output_frame_stride = 1
            self.output_video_writer = None
            return None

    def _should_write_output_frame(self, frame_idx):
        stride = getattr(self, 'output_frame_stride', 1) or 1
        if stride <= 1 or frame_idx is None:
            return True
        return ((frame_idx - 1) % stride) == 0

    def _write_output_frame(self, writer, frame_idx, frame):
        if writer is None:
            return
        if self._should_write_output_frame(frame_idx):
            writer.write(frame)
    
    def _create_flash_annotator(self, line_type):
        """フラッシュ用のLineZoneAnnotatorを作成
        
        Args:
            line_type: ラインタイプ ('up', 'down', 'single')
            
        Returns:
            フラッシュ色のLineZoneAnnotator
        """
        config = self.current_config
        
        # フラッシュ色を定義（明るい黄色/オレンジ）
        flash_color = sv.Color(255, 200, 0)  # 明るいオレンジ色
        
        # ライン設定から太さを取得
        if 'lines' in config:
            line_mode = config['lines'].get('mode', 'dual')
            
            if line_mode == 'single' or line_type == 'single':
                thickness = config['lines']['up_line']['thickness']
            elif line_type == 'up':
                thickness = config['lines']['up_line']['thickness']
            elif line_type == 'down':
                thickness = config['lines']['down_line']['thickness']
            else:
                thickness = 2
        else:
            # 旧形式の設定
            thickness = config['line']['thickness']
        
        # フラッシュ用アノテーターを作成（太さを1.5倍にして目立たせる）
        return sv.LineZoneAnnotator(thickness=int(thickness * 1.5), color=flash_color)
    
    def _calculate_adaptive_frame_skip(self, detection_count, base_frame_skip, config):
        """動的フレームスキップを計算
        
        Args:
            detection_count: 現在のフレームでの検出数
            base_frame_skip: 基本フレームスキップ値
            config: 設定
            
        Returns:
            適用すべきフレームスキップ値
        """
        # 動的フレームスキップが無効の場合は基本値を返す
        if not config['performance'].get('use_adaptive_skip', False):
            return base_frame_skip
        
        # 検出数に応じてフレームスキップを調整
        # 車両が多い時は細かく処理、少ない時は大きくスキップ
        if detection_count == 0:
            # 車両なし → 大きくスキップ（5フレーム）
            return max(base_frame_skip, 5)
        elif detection_count <= 1:
            # 車両少ない → 中程度スキップ（3フレーム）
            return max(base_frame_skip, 3)
        elif detection_count <= 2:
            # 車両やや多い → 小スキップ（2フレーム）
            return max(base_frame_skip, 2)
        else:
            # 車両多い → 全フレーム処理（スキップなし）
            return 0
    
    def _process_frames_batch(self, cap, out, model, line_zones, line_annotators, tracker, box_annotator, 
                             label_annotator, vehicle_classes, conf_thresh, iou_thresh, 
                             img_size, frame_skip, total_frames, batch_size):
        """フレーム処理ループ（バッチ推論対応）"""
        frame_idx = 0
        start_time = datetime.now()
        
        # バッチ処理用のバッファ
        frame_buffer = []
        frame_info_buffer = []  # (frame_idx, frame)のタプルを保存
        
        # 動的フレームスキップ用の変数
        current_skip = frame_skip
        skip_counter = 0
        
        while self.is_processing:
            # 一時停止チェック
            self.pause_event.wait()
            
            # 停止フラグチェック
            if self.stop_flag or not self.is_processing:
                break
                
            ret, frame = cap.read()
            if not ret:
                # 残りのバッファを処理
                if frame_buffer:
                    self._process_batch(frame_buffer, frame_info_buffer, model, line_zones, 
                                      line_annotators, tracker, box_annotator, label_annotator,
                                      vehicle_classes, conf_thresh, iou_thresh, img_size, out)
                break
            
            frame_idx += 1
            
            # 経過時間とFPS、残り時間を計算
            elapsed_time = (datetime.now() - start_time).total_seconds()
            fps = frame_idx / elapsed_time if elapsed_time > 0 else 0
            
            # 残り時間の推定
            if fps > 0:
                remaining_frames = total_frames - frame_idx
                estimated_remaining = remaining_frames / fps
            else:
                estimated_remaining = None
            
            # プログレス更新（詳細情報付き）
            progress_info = {
                'percent': (frame_idx / total_frames) * 100,
                'elapsed': elapsed_time,
                'remaining': estimated_remaining,
                'fps': fps,
                'frame': frame_idx,
                'total_frames': total_frames
            }
            self.update_progress(progress_info)
            
            # 動的フレームスキップ判定
            skip_counter += 1
            if current_skip > 0 and skip_counter <= current_skip:
                # スキップするフレーム
                self._write_output_frame(out, frame_idx, frame)
                continue
            
            # スキップカウンターをリセット
            skip_counter = 0
            
            # フレームをバッファに追加
            frame_buffer.append(frame.copy())
            frame_info_buffer.append((frame_idx, frame))
            
            # バッファが満杯になったらバッチ処理
            if len(frame_buffer) >= batch_size:
                detection_count = self._process_batch(frame_buffer, frame_info_buffer, model, line_zones, 
                                                     line_annotators, tracker, box_annotator, label_annotator,
                                                     vehicle_classes, conf_thresh, iou_thresh, img_size, out)
                
                # 次のフレームスキップ値を動的に計算
                current_skip = self._calculate_adaptive_frame_skip(detection_count, frame_skip, self.current_config)
                
                # バッファをクリア
                frame_buffer = []
                frame_info_buffer = []
            
            # カウント更新（100フレームごと）
            if frame_idx % 100 == 0:
                self._log_progress(frame_idx, total_frames, progress_info['percent'])
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        return processing_time
    
    def _process_batch(self, frame_buffer, frame_info_buffer, model, line_zones, line_annotators,
                      tracker, box_annotator, label_annotator, vehicle_classes, 
                      conf_thresh, iou_thresh, img_size, out):
        """バッチ推論を実行"""
        if not frame_buffer:
            return 0
        
        # バッチ推論実行
        results = model(frame_buffer, conf=conf_thresh, iou=iou_thresh, imgsz=img_size, verbose=False)
        
        total_detections = 0
        
        # 各フレームの結果を処理
        for i, (result, (frame_idx, original_frame)) in enumerate(zip(results, frame_info_buffer)):
            # 検出結果を変換
            detections = sv.Detections.from_ultralytics(result)
            
            # 車両フィルタ
            if len(detections.class_id) > 0:
                mask = np.array([cls in vehicle_classes for cls in detections.class_id])
                detections = detections[mask]
            
            # 車両タイプを統一
            if len(detections.class_id) > 0:
                unified_class_id = np.copy(detections.class_id)
                unified_class_id[unified_class_id == 3] = 2
                unified_class_id[unified_class_id == 5] = 2
                unified_class_id[unified_class_id == 7] = 2
                detections.class_id = unified_class_id
            
            # 追跡
            tracks = tracker.update_with_detections(detections)
            total_detections += len(tracks.tracker_id)
            
            # ライン交差判定と画像保存（改善版：trigger戻り値を使用）
            for line_type, line_zone in line_zones.items():
                prev_in_count = line_zone.in_count
                prev_out_count = line_zone.out_count
                
                # トリガー実行して戻り値を取得
                crossed_in, crossed_out = line_zone.trigger(tracks)
                
                # カウント変化をチェック
                current_in_count = line_zone.in_count
                current_out_count = line_zone.out_count
                if current_in_count > prev_in_count or current_out_count > prev_out_count:
                    # フラッシュ開始（終了フレーム番号を記録）
                    self.line_flash_frames[line_type] = frame_idx + self.flash_duration_frames
                    
                    # 時間帯別カウント更新（詳細版）
                    if line_type == 'single':
                        # singleラインの場合、in/outカウントの変化を個別に処理
                        if current_in_count > prev_in_count:
                            self.count_manager.update_hourly_count('up', frame_idx, self.current_fps)
                        if current_out_count > prev_out_count:
                            self.count_manager.update_hourly_count('down', frame_idx, self.current_fps)
                    else:
                        # 2本ライン（up/down）の場合
                        self.count_manager.update_hourly_count(line_type, frame_idx, self.current_fps)
                    
                    # 実際に通過した車両を処理（改善版）
                    if self.image_saver.vehicle_images_config.get('save_images', False):
                        self._handle_vehicle_crossing(original_frame, tracks, line_type, 
                                                      crossed_in, crossed_out, frame_idx)
            
            # 認識結果記録
            if len(tracks.tracker_id) > 0:
                self.recognition_manager.record_all_detections(original_frame, tracks, line_zones, frame_idx)
            
            # 可視化
            processed_frame = original_frame.copy()
            if len(tracks.tracker_id) > 0:
                labels = [
                    f"ID:{tid} {self.class_names.get(int(cid), 'vehicle')} {conf:.2f}"
                    for conf, cid, tid in zip(tracks.confidence, tracks.class_id, tracks.tracker_id)
                ]
                processed_frame = box_annotator.annotate(scene=processed_frame, detections=tracks)
                processed_frame = label_annotator.annotate(scene=processed_frame, detections=tracks, labels=labels)
            
            # ライン描画（フラッシュ対応）
            for line_type, line_annotator in line_annotators.items():
                if line_type in line_zones:
                    # フラッシュ中かチェック
                    if line_type in self.line_flash_frames and frame_idx < self.line_flash_frames[line_type]:
                        # フラッシュ色で描画
                        flash_annotator = self._create_flash_annotator(line_type)
                        processed_frame = flash_annotator.annotate(frame=processed_frame, line_counter=line_zones[line_type])
                    else:
                        # 通常色で描画
                        processed_frame = line_annotator.annotate(frame=processed_frame, line_counter=line_zones[line_type])
                        # フラッシュ終了したら記録を削除
                        if line_type in self.line_flash_frames and frame_idx >= self.line_flash_frames[line_type]:
                            del self.line_flash_frames[line_type]
            
            # カウント表示
            self._draw_count_overlay(processed_frame, line_zones)
            
            # プレビュー更新（10フレームごとに更新頻度を下げて安定性向上）
            if self.current_config.get('performance', {}).get('show_preview', True) and frame_idx % 10 == 0:
                try:
                    # コピーせず参照を渡す（メモリ効率化）
                    self.update_frame(processed_frame)
                except Exception:
                    pass  # プレビューエラーは無視
            
            # 動画出力
            self._write_output_frame(out, frame_idx, processed_frame)
            
            # カウント更新
            self._update_counts_display(line_zones)
        
        # 平均検出数を返す
        return total_detections // len(frame_buffer) if frame_buffer else 0
    
    def _log_progress(self, frame_idx, total_frames, progress):
        """進捗ログを出力（完了予想時刻を含む）"""
        # 経過時間と残り時間を計算
        current_time = datetime.now()
        elapsed_time = (current_time - self.processing_start_time).total_seconds()
        
        # 処理速度を計算（FPS）
        if elapsed_time > 0:
            fps = frame_idx / elapsed_time
            remaining_frames = total_frames - frame_idx
            estimated_remaining_seconds = remaining_frames / fps if fps > 0 else 0
            estimated_completion_time = current_time + timedelta(seconds=estimated_remaining_seconds)
            
            # 経過時間と残り時間をフォーマット
            elapsed_str = self._format_time(elapsed_time)
            remaining_str = self._format_time(estimated_remaining_seconds)
            completion_time_str = estimated_completion_time.strftime("%H:%M:%S")
            
            # 進捗情報
            progress_info = f"処理中: {frame_idx}/{total_frames} ({progress:.1f}%) - {fps:.1f}fps"
            time_info = f"経過:{elapsed_str} 残り:{remaining_str} 完了予想:{completion_time_str}"
        else:
            progress_info = f"処理中: {frame_idx}/{total_frames} ({progress:.1f}%)"
            time_info = "計算中..."
        
        # カウント情報
        if 'up' in self.current_line_zones and 'down' in self.current_line_zones:
            up_count = self.current_line_zones['up'].in_count + self.current_line_zones['up'].out_count
            down_count = self.current_line_zones['down'].in_count + self.current_line_zones['down'].out_count
            up_saved = len(self.image_saver.saved_vehicle_ids.get('up', set()))
            down_saved = len(self.image_saver.saved_vehicle_ids.get('down', set()))
            count_info = f"上り:{up_count}(保存:{up_saved}) 下り:{down_count}(保存:{down_saved})"
        else:
            line_zone = self.current_line_zones['single']
            single_saved = len(self.image_saver.saved_vehicle_ids.get('single', set()))
            total_count = line_zone.in_count + line_zone.out_count
            count_info = f"上り:{line_zone.in_count} 下り:{line_zone.out_count} 合計:{total_count}(保存:{single_saved})"
        
        # ログ出力
        self.log(f"{progress_info} | {time_info}")
        self.log(f"  {count_info}")
    
    def _format_time(self, seconds):
        """秒数を時:分:秒形式にフォーマット"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}時間{minutes}分{secs}秒"
        elif minutes > 0:
            return f"{minutes}分{secs}秒"
        else:
            return f"{secs}秒"
    
    def _update_counts_display(self, line_zones):
        """カウント表示を更新"""
        # 車種別総合カウントを取得
        vehicle_counts = self._get_total_vehicle_counts()
        
        if 'up' in line_zones and 'down' in line_zones:
            up_count = line_zones['up'].in_count + line_zones['up'].out_count
            down_count = line_zones['down'].in_count + line_zones['down'].out_count
            total_count = up_count + down_count
            self.update_counts(up_count, down_count, total_count, vehicle_counts)
        else:
            line_zone = line_zones['single']
            self.update_counts(line_zone.in_count, line_zone.out_count, 
                             line_zone.in_count + line_zone.out_count, vehicle_counts)
    
    def _get_total_vehicle_counts(self):
        """総合的な車種別カウントを取得"""
        total_counts = {'small': 0, 'large': 0, 'unknown': 0}
        
        # 車種判別が有効な場合のみカウントを計算
        if self.count_manager.vehicle_classification_enabled:
            # 全方向の車種別カウントを合計
            for direction in ['up', 'down', 'single']:
                if direction in self.count_manager.total_vehicle_class_counts:
                    for vehicle_type in ['small', 'large', 'unknown']:
                        if vehicle_type in self.count_manager.total_vehicle_class_counts[direction]:
                            total_counts[vehicle_type] += self.count_manager.total_vehicle_class_counts[direction][vehicle_type]
        
        # 車種判別が無効でも、常にカウントオブジェクトを返す
        return total_counts
    
    def _setup_vehicle_images(self, config):
        """車両画像保存設定（改善版：階層構造対応）"""
        self.image_saver.configure(
            config=config,
            input_video_name=self.input_video_name,
            video_start_time=self.count_manager.video_start_time,
            video_fps=self.video_fps,
            vehicle_classifier=self.count_manager.vehicle_classifier,
            vehicle_classification_enabled=self.count_manager.vehicle_classification_enabled,
            update_vehicle_class_count_cb=self.count_manager.update_vehicle_class_count,
        )

        if self.image_saver.vehicle_images_config.get('save_images', False):
            self.image_saver.start()
    
    def _process_frames(self, cap, out, model, line_zones, line_annotators, tracker, box_annotator, 
                       label_annotator, vehicle_classes, conf_thresh, iou_thresh, 
                       img_size, frame_skip, total_frames):
        """フレーム処理ループ（動的フレームスキップ対応）"""
        frame_idx = 0
        start_time = datetime.now()
        
        # 動的フレームスキップ用の変数
        current_skip = frame_skip  # 現在のスキップ値
        last_detection_count = 0  # 前回の検出数
        skip_counter = 0  # スキップカウンター
        
        while self.is_processing:
            # 一時停止チェック（pause_eventを使用）
            self.pause_event.wait()
            
            # 停止フラグチェック（pause_event待機後にも確認）
            if self.stop_flag or not self.is_processing:
                break
                
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # 経過時間とFPS、残り時間を計算
            elapsed_time = (datetime.now() - start_time).total_seconds()
            fps = frame_idx / elapsed_time if elapsed_time > 0 else 0
            
            # 残り時間の推定
            if fps > 0:
                remaining_frames = total_frames - frame_idx
                estimated_remaining = remaining_frames / fps
            else:
                estimated_remaining = None
            
            # プログレス更新（詳細情報付き）
            progress_info = {
                'percent': (frame_idx / total_frames) * 100,
                'elapsed': elapsed_time,
                'remaining': estimated_remaining,
                'fps': fps,
                'frame': frame_idx,
                'total_frames': total_frames
            }
            self.update_progress(progress_info)
            
            # 動的フレームスキップ判定
            skip_counter += 1
            if current_skip > 0 and skip_counter <= current_skip:
                # スキップするフレーム
                self._write_output_frame(out, frame_idx, frame)
                continue
            
            # スキップカウンターをリセット
            skip_counter = 0
            
            # 検出・追跡・可視化（検出数も取得）
            frame, detection_count = self._process_single_frame(
                frame, model, self.current_line_zones, self.current_line_annotators, tracker, box_annotator, 
                label_annotator, vehicle_classes, conf_thresh, iou_thresh, img_size, frame_idx
            )
            
            # 次のフレームスキップ値を動的に計算
            current_skip = self._calculate_adaptive_frame_skip(detection_count, frame_skip, self.current_config)
            last_detection_count = detection_count
            
            # カウント更新（2本ライン対応）
            # カウント更新は統一されたメソッドを使用
            self._update_counts_display(self.current_line_zones)
            
            # フレーム更新（プレビュー表示が有効な場合のみ、10フレームごとに更新頻度を下げる）
            if self.current_config.get('performance', {}).get('show_preview', True) and frame_idx % 10 == 0:
                try:
                    self.update_frame(frame)
                except Exception:
                    pass  # プレビューエラーは無視
            
            # 動画出力（有効な場合のみ）
            self._write_output_frame(out, frame_idx, frame)
            
            # 定期的にログ出力（詳細情報付き）
            if frame_idx % 100 == 0:
                self._log_progress(frame_idx, total_frames, progress_info['percent'])
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        return processing_time
    
    def _process_frames_async(self, cap, out, model, line_zones, line_annotators, tracker, box_annotator, 
                              label_annotator, vehicle_classes, conf_thresh, iou_thresh, 
                              img_size, frame_skip, total_frames, queue_size):
        """フレーム処理ループ（非同期推論: フレーム読み込みと推論を並列実行）"""
        start_time = datetime.now()
        
        # 推論スレッド数を取得（デフォルト: 1）
        num_inference_threads = self.current_config['performance'].get('num_inference_threads', 1)
        num_inference_threads = max(1, min(num_inference_threads, 32))  # 1-32の範囲に制限
        
        # フレームキュー（読み込みスレッド → 推論スレッド）
        frame_queue = queue.Queue(maxsize=queue_size)
        # 結果キュー（推論スレッド → メインスレッド）
        result_queue = queue.Queue(maxsize=queue_size * 2)  # 複数スレッド対応で少し大きく
        
        # スレッド制御フラグ
        read_finished = threading.Event()
        process_finished = threading.Event()
        
        # 統計情報
        stats = {
            'frames_read': 0,
            'frames_processed': 0,
            'frames_skipped': 0
        }
        stats_lock = threading.Lock()
        
        def frame_reader_thread():
            """フレーム読み込みスレッド"""
            frame_idx = 0
            skip_counter = 0
            current_skip = frame_skip
            
            try:
                while self.is_processing and not self.stop_flag:
                    # 一時停止チェック
                    self.pause_event.wait()
                    
                    if self.stop_flag or not self.is_processing:
                        break
                    
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_idx += 1
                    with stats_lock:
                        stats['frames_read'] = frame_idx
                    
                    # 経過時間とFPS、残り時間を計算
                    elapsed_time = (datetime.now() - start_time).total_seconds()
                    fps = frame_idx / elapsed_time if elapsed_time > 0 else 0
                    
                    # 残り時間の推定
                    if fps > 0:
                        remaining_frames = total_frames - frame_idx
                        estimated_remaining = remaining_frames / fps
                    else:
                        estimated_remaining = None
                    
                    # プログレス更新（詳細情報付き）
                    progress_info = {
                        'percent': (frame_idx / total_frames) * 100,
                        'elapsed': elapsed_time,
                        'remaining': estimated_remaining,
                        'fps': fps,
                        'frame': frame_idx,
                        'total_frames': total_frames
                    }
                    self.update_progress(progress_info)
                    
                    # フレームスキップ判定
                    skip_counter += 1
                    if current_skip > 0 and skip_counter <= current_skip:
                        # スキップするフレーム（出力のみ）
                        if out is not None:
                            result_queue.put(('skip', frame_idx, frame, None))
                        with stats_lock:
                            stats['frames_skipped'] += 1
                        continue
                    
                    skip_counter = 0
                    
                    # フレームをキューに追加（ブロッキング）
                    frame_queue.put((frame_idx, frame.copy()), block=True)
                    
            except Exception as e:
                self.log(f"⚠️ フレーム読み込みエラー: {e}")
            finally:
                read_finished.set()
                # 終了シグナルを送る（推論スレッド数分）
                try:
                    for _ in range(num_inference_threads):
                        frame_queue.put(None, block=False)
                except:
                    pass
        
        def inference_thread(thread_id):
            """推論スレッド"""
            try:
                while self.is_processing and not self.stop_flag:
                    # 一時停止チェック
                    self.pause_event.wait()
                    
                    if self.stop_flag or not self.is_processing:
                        break
                    
                    try:
                        # フレームをキューから取得（タイムアウト付き）
                        item = frame_queue.get(timeout=1.0)
                        
                        if item is None:
                            # 終了シグナル - 他のスレッドにも伝播
                            frame_queue.put(None, block=False)
                            break
                        
                        frame_idx, frame = item
                        
                        # 推論・追跡・可視化
                        processed_frame, detection_count = self._process_single_frame(
                            frame, model, line_zones, line_annotators, tracker, box_annotator, 
                            label_annotator, vehicle_classes, conf_thresh, iou_thresh, img_size, frame_idx
                        )
                        
                        # 結果をキューに追加
                        result_queue.put(('process', frame_idx, processed_frame, detection_count), block=True)
                        
                        with stats_lock:
                            stats['frames_processed'] += 1
                        
                        frame_queue.task_done()
                        
                    except queue.Empty:
                        if read_finished.is_set() and frame_queue.empty():
                            break
                        continue
                    
            except Exception as e:
                self.log(f"⚠️ 推論スレッド[{thread_id}]エラー: {e}")
                import traceback
                self.log(traceback.format_exc())
            finally:
                with stats_lock:
                    if not hasattr(self, '_finished_inference_threads'):
                        self._finished_inference_threads = 0
                    self._finished_inference_threads += 1
                    
                    # 全ての推論スレッドが終了したら終了シグナルを送る
                    if self._finished_inference_threads >= num_inference_threads:
                        process_finished.set()
                        try:
                            result_queue.put(None, block=False)
                        except:
                            pass
        
        # スレッド開始
        self.log(f"✓ 非同期推論: 読込スレッド×1, 推論スレッド×{num_inference_threads}")
        
        reader = threading.Thread(target=frame_reader_thread, name="FrameReader", daemon=True)
        reader.start()
        
        # 推論スレッドを複数起動
        self._finished_inference_threads = 0  # カウンターを初期化
        processors = []
        for i in range(num_inference_threads):
            processor = threading.Thread(
                target=lambda tid=i: inference_thread(tid), 
                name=f"InferenceProcessor-{i}", 
                daemon=True
            )
            processors.append(processor)
            processor.start()
        
        # メインスレッド: 結果の処理と出力
        processed_count = 0
        last_log_time = time_module.time()
        
        try:
            while True:
                # 一時停止チェック
                self.pause_event.wait()
                
                if self.stop_flag or not self.is_processing:
                    break
                
                try:
                    # 結果をキューから取得
                    item = result_queue.get(timeout=1.0)
                    
                    if item is None:
                        # 終了シグナル
                        break
                    
                    action, frame_idx, frame, detection_count = item
                    
                    if action == 'skip':
                        # スキップフレーム: そのまま出力
                        self._write_output_frame(out, frame_idx, frame)
                    else:
                        # 処理済みフレーム
                        processed_count += 1
                        
                        # カウント更新
                        self._update_counts_display(line_zones)
                        
                        # フレーム更新（10フレームごとに更新頻度を下げる）
                        if self.current_config.get('performance', {}).get('show_preview', True) and processed_count % 10 == 0:
                            try:
                                self.update_frame(frame)
                            except Exception:
                                pass  # プレビューエラーは無視
                        
                        # 動画出力
                        self._write_output_frame(out, frame_idx, frame)
                        
                        # 定期的にログ出力（1秒ごと）
                        current_time = time_module.time()
                        if current_time - last_log_time >= 1.0:
                            with stats_lock:
                                read_count = stats['frames_read']
                                proc_count = stats['frames_processed']
                                skip_count = stats['frames_skipped']
                            
                            progress = (read_count / total_frames) * 100 if total_frames > 0 else 0
                            queue_len = frame_queue.qsize()
                            
                            # 経過時間と完了予想を計算
                            elapsed_time = (datetime.now() - self.processing_start_time).total_seconds()
                            if elapsed_time > 0 and read_count > 0:
                                fps = read_count / elapsed_time
                                remaining_frames = total_frames - read_count
                                estimated_remaining_seconds = remaining_frames / fps if fps > 0 else 0
                                estimated_completion_time = datetime.now() + timedelta(seconds=estimated_remaining_seconds)
                                
                                elapsed_str = self._format_time(elapsed_time)
                                remaining_str = self._format_time(estimated_remaining_seconds)
                                completion_time_str = estimated_completion_time.strftime("%H:%M:%S")
                                time_info = f"{fps:.1f}fps | 経過:{elapsed_str} 残り:{remaining_str} 完了予想:{completion_time_str}"
                            else:
                                time_info = "計算中..."
                            
                            if 'up' in line_zones and 'down' in line_zones:
                                up_count = line_zones['up'].in_count + line_zones['up'].out_count
                                down_count = line_zones['down'].in_count + line_zones['down'].out_count
                                up_saved = len(self.image_saver.saved_vehicle_ids.get('up', set()))
                                down_saved = len(self.image_saver.saved_vehicle_ids.get('down', set()))
                                self.log(f"処理中: {read_count}/{total_frames} ({progress:.1f}%) [キュー:{queue_len}] | {time_info}")
                                self.log(f"  上り:{up_count}(保存:{up_saved}) 下り:{down_count}(保存:{down_saved})")
                            else:
                                line_zone = line_zones['single']
                                single_saved = len(self.image_saver.saved_vehicle_ids.get('single', set()))
                                total_count = line_zone.in_count + line_zone.out_count
                                self.log(f"処理中: {read_count}/{total_frames} ({progress:.1f}%) [キュー:{queue_len}] | {time_info}")
                                self.log(f"  合計:{total_count}(保存:{single_saved})")
                            
                            last_log_time = current_time
                    
                    result_queue.task_done()
                    
                except queue.Empty:
                    if process_finished.is_set() and result_queue.empty():
                        break
                    continue
                    
        except Exception as e:
            self.log(f"⚠️ メイン処理エラー: {e}")
            import traceback
            self.log(traceback.format_exc())
        finally:
            # スレッドの終了を待つ
            self.stop_flag = True
            reader.join(timeout=5.0)
            processor.join(timeout=5.0)
            
            # 統計情報を表示
            with stats_lock:
                self.log(f"✓ 非同期処理完了: 読込={stats['frames_read']}, 処理={stats['frames_processed']}, スキップ={stats['frames_skipped']}")
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        return processing_time
    
    def _process_single_frame(self, frame, model, line_zones, line_annotators, tracker, box_annotator, 
                             label_annotator, vehicle_classes, conf_thresh, iou_thresh, img_size, frame_idx):
        """単一フレームの処理"""
        # YOLO推論
        results = model(frame, conf=conf_thresh, iou=iou_thresh, imgsz=img_size, verbose=False)
        
        # 検出結果を変換
        detections = sv.Detections.from_ultralytics(results[0])
        
        # 車両フィルタ（Car/Truck統一処理）
        if len(detections.class_id) > 0:
            mask = np.array([cls in vehicle_classes for cls in detections.class_id])
            detections = detections[mask]
            
        # 全車両タイプを統一クラスに変更（追跡安定化のため）
        if len(detections.class_id) > 0:
            unified_class_id = np.copy(detections.class_id)
            # Car(2), Motorcycle(3), Bus(5), Truck(7) をすべて Car(2) に統一
            unified_class_id[unified_class_id == 3] = 2  # motorcycle -> car
            unified_class_id[unified_class_id == 5] = 2  # bus -> car
            unified_class_id[unified_class_id == 7] = 2  # truck -> car
            detections.class_id = unified_class_id
        
        # 追跡
        tracks = tracker.update_with_detections(detections)
        
        # ライン交差判定（改善版：trigger戻り値を使用）
        for line_type, line_zone in line_zones.items():
            # 通過前のカウントを記録
            prev_in_count = line_zone.in_count
            prev_out_count = line_zone.out_count
            
            # トリガー実行して戻り値を取得
            crossed_in, crossed_out = line_zone.trigger(tracks)
            
            # カウント変化をチェック
            current_in_count = line_zone.in_count
            current_out_count = line_zone.out_count
            
            # カウントが増加した場合
            if current_in_count > prev_in_count or current_out_count > prev_out_count:
                # フラッシュ開始（終了フレーム番号を記録）
                self.line_flash_frames[line_type] = frame_idx + self.flash_duration_frames
                
                # 時間帯別カウント更新（詳細版）
                if line_type == 'single':
                    # singleラインの場合、in/outカウントの変化を個別に処理
                    if current_in_count > prev_in_count:
                        self.count_manager.update_hourly_count('up', frame_idx, self.current_fps)
                    if current_out_count > prev_out_count:
                        self.count_manager.update_hourly_count('down', frame_idx, self.current_fps)
                else:
                    # 2本ライン（up/down）の場合
                    self.count_manager.update_hourly_count(line_type, frame_idx, self.current_fps)
                
                # 実際に通過した車両を処理（改善版）
                if self.image_saver.vehicle_images_config.get('save_images', False):
                    self._handle_vehicle_crossing(frame, tracks, line_type, 
                                                  crossed_in, crossed_out, frame_idx)
        
        # 全認識結果を記録（通過・非通過関係なく）
        if len(tracks.tracker_id) > 0:
            self.recognition_manager.record_all_detections(frame, tracks, line_zones, frame_idx)
        
        # 可視化
        if len(tracks.tracker_id) > 0:
            # インスタンス変数のクラス名を使用
            labels = [
                f"ID:{tid} {self.class_names.get(int(cid), 'vehicle')} {conf:.2f}"
                for conf, cid, tid in zip(tracks.confidence, tracks.class_id, tracks.tracker_id)
            ]
            
            frame = box_annotator.annotate(scene=frame, detections=tracks)
            frame = label_annotator.annotate(scene=frame, detections=tracks, labels=labels)
        
        # ライン描画（2本ライン対応、フラッシュ機能付き）
        for line_type, line_annotator in line_annotators.items():
            if line_type in line_zones:
                # フラッシュ中かチェック
                if line_type in self.line_flash_frames and frame_idx < self.line_flash_frames[line_type]:
                    # フラッシュ色で描画
                    flash_annotator = self._create_flash_annotator(line_type)
                    frame = flash_annotator.annotate(frame=frame, line_counter=line_zones[line_type])
                else:
                    # 通常色で描画
                    frame = line_annotator.annotate(frame=frame, line_counter=line_zones[line_type])
                    # フラッシュ終了したら記録を削除
                    if line_type in self.line_flash_frames and frame_idx >= self.line_flash_frames[line_type]:
                        del self.line_flash_frames[line_type]
        
        # カウント表示
        self._draw_count_overlay(frame, line_zones)
        
        # 検出数を返す（動的フレームスキップ用）
        detection_count = len(tracks.tracker_id) if len(tracks.tracker_id) > 0 else 0
        
        return frame, detection_count
    
    def _draw_count_overlay(self, frame, line_zones):
        """フレームにカウント情報を描画"""
        cv2.rectangle(frame, (20, 20), (400, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (20, 20), (400, 120), (255, 255, 255), 2)
        
        if 'up' in line_zones and 'down' in line_zones:
            # 2本ライン表示
            up_count = line_zones['up'].in_count + line_zones['up'].out_count
            down_count = line_zones['down'].in_count + line_zones['down'].out_count
            total_count = up_count + down_count
            
            cv2.putText(frame, f"Up Line: {up_count}", 
                       (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Down Line: {down_count}", 
                       (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"Total: {total_count}", 
                       (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            # 旧形式（1本ライン）
            line_zone = line_zones['single']
            cv2.putText(frame, f"In: {line_zone.in_count} Out: {line_zone.out_count}", 
                       (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Total: {line_zone.in_count + line_zone.out_count}", 
                       (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    def _handle_vehicle_crossing(self, frame, tracks, line_type, crossed_in, crossed_out, frame_idx):
        """実際に通過した車両のイベント処理
        
        Args:
            frame: 現在のフレーム
            tracks: 追跡結果
            line_type: ラインタイプ ('up', 'down', 'single')
            crossed_in: 内側に通過した検出のブール配列
            crossed_out: 外側に通過した検出のブール配列
            frame_idx: フレーム番号
        """
        self.image_saver.handle_vehicle_crossing(
            frame=frame,
            tracks=tracks,
            line_type=line_type,
            crossed_in=crossed_in,
            crossed_out=crossed_out,
            frame_idx=frame_idx,
            record_result_cb=self.recognition_manager.record_recognition_result,
        )
    
    def _point_to_line_distance_simple(self, px, py, line_start, line_end):
        """点からライン（線分）までの距離を計算（シンプル版）"""
        x1, y1 = line_start.x, line_start.y
        x2, y2 = line_end.x, line_end.y
        
        # 線分の長さの二乗
        line_length_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
        
        if line_length_sq == 0:
            # 線分が点の場合
            return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5
        
        # 点から線分への投影パラメータ
        t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_length_sq))
        
        # 投影点の座標
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)
        
        # 点と投影点の距離
        return ((px - proj_x) ** 2 + (py - proj_y) ** 2) ** 0.5
    
    def _bbox_to_line_distance(self, bbox, line_start, line_end):
        """バウンディングボックスからラインまでの最短距離を計算
        
        バウンディングボックスの4隅、4辺の中点、中心点の計9点のうち、
        ラインに最も近い点までの距離を返す。
        これにより、ラインをまたいでいる車両を正確に検出できる。
        """
        x1, y1, x2, y2 = bbox
        
        # 検証する点のリスト
        points = [
            # 4隅
            (x1, y1),  # 左上
            (x2, y1),  # 右上
            (x1, y2),  # 左下
            (x2, y2),  # 右下
            # 4辺の中点
            ((x1 + x2) / 2, y1),  # 上辺中点
            ((x1 + x2) / 2, y2),  # 下辺中点
            (x1, (y1 + y2) / 2),  # 左辺中点
            (x2, (y1 + y2) / 2),  # 右辺中点
            # 中心点
            ((x1 + x2) / 2, (y1 + y2) / 2)
        ]
        
        # 各点からラインまでの距離を計算し、最小値を返す
        min_distance = float('inf')
        for px, py in points:
            distance = self._point_to_line_distance_simple(px, py, line_start, line_end)
            min_distance = min(min_distance, distance)
        
        return min_distance
    
