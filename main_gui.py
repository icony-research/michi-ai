"""
MICHI-AI - PySide6 GUI版

"""

import sys
from pathlib import Path
from datetime import datetime
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QTextEdit, QProgressBar,
    QFileDialog, QMessageBox, QTabWidget, QGroupBox, QGridLayout,
    QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox, QListWidget,
    QSplitter, QFrame, QDialog
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSize, QPoint, QObject
from PySide6.QtGui import QPixmap, QImage, QFont, QPainter, QPen, QColor
import queue
import cv2
import numpy as np

# 既存のモジュールをインポート
from translations import get_text
from config_manager import ConfigManager
from video_processor import VideoProcessor


class LineDrawerDialog(QDialog):
    """カウントライン描画ダイアログ"""
    
    def __init__(self, frame, is_dual_mode=True, parent=None):
        super().__init__(parent)
        self.setWindowTitle("カウントライン設定")
        self.setModal(True)
        
        self.frame = frame.copy()
        self.display_frame = frame.copy()
        self.drawing = False
        self.current_line = None
        self.lines = {'up': None, 'down': None}
        self.current_mode = 'up'  # 'up' or 'down'
        self.is_dual_mode = is_dual_mode
        
        self.setup_ui()
        
    def setup_ui(self):
        """UI構築"""
        layout = QVBoxLayout(self)
        
        # 説明ラベル
        if self.is_dual_mode:
            instruction_text = "上りライン(青)を引いてください: 始点をクリック → 終点をクリック"
        else:
            instruction_text = "カウントライン(青)を引いてください: 始点をクリック → 終点をクリック"
        self.instruction_label = QLabel(instruction_text)
        self.instruction_label.setStyleSheet("font-size: 14px; padding: 10px; background: #e3f2fd;")
        layout.addWidget(self.instruction_label)
        
        # 画像表示ラベル
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMouseTracking(True)
        layout.addWidget(self.image_label)
        
        # ボタン
        btn_layout = QHBoxLayout()
        
        self.reset_btn = QPushButton("🔄 リセット")
        self.reset_btn.clicked.connect(self.reset_current_line)
        btn_layout.addWidget(self.reset_btn)
        
        self.ok_btn = QPushButton("✓ 完了")
        self.ok_btn.clicked.connect(self.accept)
        self.ok_btn.setEnabled(False)
        btn_layout.addWidget(self.ok_btn)
        
        self.cancel_btn = QPushButton("✗ キャンセル")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(btn_layout)
        
        # 初期画像表示
        self.update_display()
        
    def mousePressEvent(self, event):
        """マウスクリックイベント"""
        if event.button() == Qt.LeftButton and self.image_label.underMouse():
            # ラベル内の座標を取得
            label_pos = self.image_label.mapFromGlobal(event.globalPosition().toPoint())
            
            # 画像のスケールを考慮した実際の座標を計算
            pixmap = self.image_label.pixmap()
            if pixmap:
                label_size = self.image_label.size()
                pixmap_size = pixmap.size()
                
                # スケール比率を計算
                scale_x = self.frame.shape[1] / pixmap_size.width()
                scale_y = self.frame.shape[0] / pixmap_size.height()
                
                # ラベル内での画像の開始位置を計算（中央配置を考慮）
                offset_x = (label_size.width() - pixmap_size.width()) / 2
                offset_y = (label_size.height() - pixmap_size.height()) / 2
                
                # 実際の画像座標に変換
                img_x = int((label_pos.x() - offset_x) * scale_x)
                img_y = int((label_pos.y() - offset_y) * scale_y)
                
                # 範囲チェック
                if 0 <= img_x < self.frame.shape[1] and 0 <= img_y < self.frame.shape[0]:
                    self.add_point(img_x, img_y)
    
    def add_point(self, x, y):
        """ポイント追加"""
        if self.current_line is None:
            # 始点
            self.current_line = [(x, y)]
        elif len(self.current_line) == 1:
            # 終点
            self.current_line.append((x, y))
            self.lines[self.current_mode] = self.current_line.copy()
            self.current_line = None
            
            # 次のラインへ
            if self.current_mode == 'up' and self.is_dual_mode:
                self.current_mode = 'down'
                self.instruction_label.setText("下りライン(緑)を引いてください: 始点をクリック → 終点をクリック")
                self.instruction_label.setStyleSheet("font-size: 14px; padding: 10px; background: #e8f5e9;")
            else:
                if self.is_dual_mode:
                    self.instruction_label.setText("✓ 両方のラインが設定されました。「完了」をクリックしてください")
                else:
                    self.instruction_label.setText("✓ カウントラインが設定されました。「完了」をクリックしてください")
                self.instruction_label.setStyleSheet("font-size: 14px; padding: 10px; background: #c8e6c9;")
                self.ok_btn.setEnabled(True)
        
        self.update_display()
    
    def reset_current_line(self):
        """現在のラインをリセット"""
        self.current_line = None
        if self.current_mode == 'up':
            self.lines['up'] = None
        else:
            self.lines['down'] = None
        self.update_display()
    
    def update_display(self):
        """表示更新"""
        display = self.frame.copy()
        
        # 確定した上りライン（青）
        if self.lines['up']:
            cv2.line(display, self.lines['up'][0], self.lines['up'][1], (255, 0, 0), 3)
            cv2.circle(display, self.lines['up'][0], 5, (255, 0, 0), -1)
            cv2.circle(display, self.lines['up'][1], 5, (255, 0, 0), -1)
        
        # 確定した下りライン（緑）
        if self.lines['down']:
            cv2.line(display, self.lines['down'][0], self.lines['down'][1], (0, 255, 0), 3)
            cv2.circle(display, self.lines['down'][0], 5, (0, 255, 0), -1)
            cv2.circle(display, self.lines['down'][1], 5, (0, 255, 0), -1)
        
        # 現在描画中のライン
        if self.current_line and len(self.current_line) == 1:
            color = (255, 0, 0) if self.current_mode == 'up' else (0, 255, 0)
            cv2.circle(display, self.current_line[0], 5, color, -1)
        
        # QImageに変換
        rgb_frame = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        
        # 画面サイズに合わせてスケール（最大1200x800）
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(1200, 800, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.image_label.setPixmap(scaled_pixmap)
    
    def get_lines(self):
        """設定されたラインを取得"""
        return self.lines


class ProcessingThread(QThread):
    """動画処理用スレッド"""
    finished = Signal(bool)
    error = Signal(str)
    
    def __init__(self, video_processor, config):
        super().__init__()
        self.video_processor = video_processor
        self.config = config
        
    def run(self):
        """処理実行"""
        try:
            success = self.video_processor.process_video(self.config)
            self.finished.emit(success)
        except Exception as e:
            self.error.emit(str(e))


class UiDispatcher(QObject):
    """ワーカースレッドからUIスレッドへイベントを配送するためのブリッジ"""
    log = Signal(str)
    progress = Signal(object)
    counts = Signal(int, int, int, object)
    frame = Signal(object)


class TrafficCounterMainWindow(QMainWindow):
    """メインウィンドウ"""
    
    def __init__(self):
        super().__init__()
        
        # 基本設定
        self.current_language = 'ja'
        self.is_processing = False
        self.is_batch_processing = False
        self.batch_video_list = []
        self.current_batch_index = 0
        self.batch_stop_requested = False
        self.processing_thread = None
        
        # ログキュー
        self.log_queue = queue.Queue()
        self.ui_dispatcher = UiDispatcher()
        self.ui_dispatcher.log.connect(self.log, Qt.QueuedConnection)
        self.ui_dispatcher.progress.connect(self.update_progress, Qt.QueuedConnection)
        self.ui_dispatcher.counts.connect(self.update_counts, Qt.QueuedConnection)
        self.ui_dispatcher.frame.connect(self.update_frame, Qt.QueuedConnection)
        
        # 設定マネージャー
        self.config_manager = ConfigManager(get_text_func=self.get_text)
        self.config_manager.create_required_folders()
        self.config = self.config_manager.load_default_config()
        self.current_language = self.config.get('language', 'ja')
        
        # 動画処理器
        self.video_processor = VideoProcessor(
            log_callback=self.ui_dispatcher.log.emit,
            progress_callback=self.ui_dispatcher.progress.emit,
            count_callback=lambda up, down, total, vehicle_counts=None: self.ui_dispatcher.counts.emit(
                up, down, total, vehicle_counts
            ),
            frame_callback=self.ui_dispatcher.frame.emit
        )
        
        # UI構築
        self.init_ui()
        
        # タイマー設定
        self.log_timer = QTimer()
        self.log_timer.timeout.connect(self.update_log)
        self.log_timer.start(100)
        
        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self.update_video_preview)
        # プレビュー更新頻度を下げる（安定性向上）
        self.frame_timer.start(100)  # 約10fps
        
        # フレームキュー（サイズを1に制限して最新フレームのみ保持）
        self.frame_queue = queue.Queue(maxsize=1)
        
        # プレビュー有効フラグ
        self.preview_enabled = True
        
        # プレビュー用のフレーム参照を保持（ガベージコレクション対策）
        self._last_frame = None
        
    def init_ui(self):
        """UI初期化"""
        self.setWindowTitle(self.get_text('title'))
        self.setGeometry(100, 100, 1400, 900)
        
        # 中央ウィジェット
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # メインレイアウト
        main_layout = QHBoxLayout(central_widget)
        
        # 左側：設定パネル
        left_panel = self.create_left_panel()
        
        # 右側：プレビュー・ログパネル
        right_panel = self.create_right_panel()
        
        # スプリッター
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter)
        
        # 設定をGUIに反映
        self.load_config_to_gui()
        
    def create_left_panel(self):
        """左側パネル作成"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # タブウィジェット
        tabs = QTabWidget()
        
        # 基本設定タブ
        tabs.addTab(self.create_basic_tab(), self.get_text('basic_settings'))
        
        # 詳細設定タブ
        tabs.addTab(self.create_advanced_tab(), self.get_text('advanced_settings'))
        
        # バッチ処理タブ
        tabs.addTab(self.create_batch_tab(), "バッチ処理")
        
        layout.addWidget(tabs)
        
        # 制御ボタン
        control_layout = self.create_control_buttons()
        layout.addLayout(control_layout)
        
        return panel
        
    def create_basic_tab(self):
        """基本設定タブ"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 言語選択
        lang_group = QGroupBox("言語 / Language")
        lang_layout = QHBoxLayout(lang_group)
        self.language_combo = QComboBox()
        self.language_combo.addItems(["ja (日本語)", "en (English)"])
        self.language_combo.currentTextChanged.connect(self.on_language_change)
        lang_layout.addWidget(QLabel("言語:"))
        lang_layout.addWidget(self.language_combo)
        lang_layout.addStretch()
        layout.addWidget(lang_group)
        
        # 入力動画
        video_group = QGroupBox(self.get_text('video_settings'))
        video_layout = QGridLayout(video_group)
        
        video_layout.addWidget(QLabel(self.get_text('input_file')), 0, 0)
        self.input_file_edit = QLineEdit()
        video_layout.addWidget(self.input_file_edit, 0, 1)
        self.input_browse_btn = QPushButton(self.get_text('browse'))
        self.input_browse_btn.clicked.connect(self.browse_input_file)
        video_layout.addWidget(self.input_browse_btn, 0, 2)
        
        # 出力動画
        video_layout.addWidget(QLabel(self.get_text('output_file')), 1, 0)
        self.output_file_edit = QLineEdit()
        video_layout.addWidget(self.output_file_edit, 1, 1)
        self.output_browse_btn = QPushButton(self.get_text('browse'))
        self.output_browse_btn.clicked.connect(self.browse_output_file)
        video_layout.addWidget(self.output_browse_btn, 1, 2)

        # 入力ベースフォルダ
        video_layout.addWidget(QLabel(self.get_text('input_base_folder')), 2, 0)
        self.input_base_edit = QLineEdit()
        video_layout.addWidget(self.input_base_edit, 2, 1)
        self.input_base_browse_btn = QPushButton(self.get_text('browse'))
        self.input_base_browse_btn.clicked.connect(self.browse_input_base_folder)
        video_layout.addWidget(self.input_base_browse_btn, 2, 2)

        # 出力ベースフォルダ
        video_layout.addWidget(QLabel(self.get_text('output_base_folder')), 3, 0)
        self.output_base_edit = QLineEdit()
        video_layout.addWidget(self.output_base_edit, 3, 1)
        self.output_base_browse_btn = QPushButton(self.get_text('browse'))
        self.output_base_browse_btn.clicked.connect(self.browse_output_base_folder)
        video_layout.addWidget(self.output_base_browse_btn, 3, 2)
        
        self.enable_output_check = QCheckBox(self.get_text('enable_video_output'))
        self.enable_output_check.setChecked(True)
        video_layout.addWidget(self.enable_output_check, 4, 0, 1, 3)
        
        layout.addWidget(video_group)
        
        # モデル設定
        model_group = QGroupBox(self.get_text('model_settings'))
        model_layout = QGridLayout(model_group)
        
        model_layout.addWidget(QLabel(self.get_text('model_file')), 0, 0)
        self.model_file_edit = QLineEdit()
        model_layout.addWidget(self.model_file_edit, 0, 1)
        self.model_browse_btn = QPushButton(self.get_text('browse'))
        self.model_browse_btn.clicked.connect(self.browse_model_file)
        model_layout.addWidget(self.model_browse_btn, 0, 2)
        
        model_layout.addWidget(QLabel(self.get_text('confidence_threshold')), 1, 0)
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setValue(0.25)
        model_layout.addWidget(self.confidence_spin, 1, 1, 1, 2)
        
        layout.addWidget(model_group)
        
        # ライン設定
        line_group = QGroupBox(self.get_text('line_settings'))
        line_layout = QGridLayout(line_group)
        
        # ラインモード選択
        line_layout.addWidget(QLabel(self.get_text('line_mode')), 0, 0)
        self.line_mode_combo = QComboBox()
        self.line_mode_combo.addItems([
            self.get_text('single_line'),
            self.get_text('dual_line')
        ])
        self.line_mode_combo.setCurrentIndex(1)  # デフォルトは2本
        self.line_mode_combo.currentIndexChanged.connect(self.on_line_mode_changed)
        line_layout.addWidget(self.line_mode_combo, 0, 1, 1, 3)
        
        # ライン1 (上り/単一)
        self.up_line_label = QLabel("上りライン")
        line_layout.addWidget(self.up_line_label, 1, 0)
        line_layout.addWidget(QLabel("始点X:"), 2, 0)
        self.up_start_x_spin = QSpinBox()
        self.up_start_x_spin.setRange(0, 3840)
        self.up_start_x_spin.setValue(100)
        line_layout.addWidget(self.up_start_x_spin, 2, 1)
        
        line_layout.addWidget(QLabel("始点Y:"), 2, 2)
        self.up_start_y_spin = QSpinBox()
        self.up_start_y_spin.setRange(0, 2160)
        self.up_start_y_spin.setValue(200)
        line_layout.addWidget(self.up_start_y_spin, 2, 3)
        
        line_layout.addWidget(QLabel("終点X:"), 3, 0)
        self.up_end_x_spin = QSpinBox()
        self.up_end_x_spin.setRange(0, 3840)
        self.up_end_x_spin.setValue(1400)
        line_layout.addWidget(self.up_end_x_spin, 3, 1)
        
        line_layout.addWidget(QLabel("終点Y:"), 3, 2)
        self.up_end_y_spin = QSpinBox()
        self.up_end_y_spin.setRange(0, 2160)
        self.up_end_y_spin.setValue(200)
        line_layout.addWidget(self.up_end_y_spin, 3, 3)
        
        # ライン2 (下り)
        self.down_line_label = QLabel("下りライン")
        line_layout.addWidget(self.down_line_label, 4, 0)
        
        self.down_start_x_label = QLabel("始点X:")
        line_layout.addWidget(self.down_start_x_label, 5, 0)
        self.down_start_x_spin = QSpinBox()
        self.down_start_x_spin.setRange(0, 3840)
        self.down_start_x_spin.setValue(100)
        line_layout.addWidget(self.down_start_x_spin, 5, 1)
        
        self.down_start_y_label = QLabel("始点Y:")
        line_layout.addWidget(self.down_start_y_label, 5, 2)
        self.down_start_y_spin = QSpinBox()
        self.down_start_y_spin.setRange(0, 2160)
        self.down_start_y_spin.setValue(300)
        line_layout.addWidget(self.down_start_y_spin, 5, 3)
        
        self.down_end_x_label = QLabel("終点X:")
        line_layout.addWidget(self.down_end_x_label, 6, 0)
        self.down_end_x_spin = QSpinBox()
        self.down_end_x_spin.setRange(0, 3840)
        self.down_end_x_spin.setValue(1400)
        line_layout.addWidget(self.down_end_x_spin, 6, 1)
        
        self.down_end_y_label = QLabel("終点Y:")
        line_layout.addWidget(self.down_end_y_label, 6, 2)
        self.down_end_y_spin = QSpinBox()
        self.down_end_y_spin.setRange(0, 2160)
        self.down_end_y_spin.setValue(300)
        line_layout.addWidget(self.down_end_y_spin, 6, 3)
        
        # 下りライン関連ウィジェットをリストに保存
        self.down_line_widgets = [
            self.down_line_label,
            self.down_start_x_label, self.down_start_x_spin,
            self.down_start_y_label, self.down_start_y_spin,
            self.down_end_x_label, self.down_end_x_spin,
            self.down_end_y_label, self.down_end_y_spin
        ]
        
        # 線を引くボタン
        self.draw_lines_btn = QPushButton(self.get_text('draw_lines'))
        self.draw_lines_btn.clicked.connect(self.open_line_drawer)
        line_layout.addWidget(self.draw_lines_btn, 7, 0, 1, 4)
        
        layout.addWidget(line_group)
        
        # 結果保存設定
        results_group = QGroupBox("結果保存設定")
        results_layout = QGridLayout(results_group)
        
        results_layout.addWidget(QLabel(self.get_text('results_folder')), 0, 0)
        self.results_folder_edit = QLineEdit()
        self.results_folder_edit.setText("results")
        results_layout.addWidget(self.results_folder_edit, 0, 1)
        self.results_browse_btn = QPushButton(self.get_text('browse'))
        self.results_browse_btn.clicked.connect(self.browse_results_folder)
        results_layout.addWidget(self.results_browse_btn, 0, 2)
        
        # 動画開始時刻設定
        results_layout.addWidget(QLabel(self.get_text('video_start_time')), 1, 0)
        self.video_start_time_edit = QLineEdit()
        self.video_start_time_edit.setPlaceholderText("HH:MM:SS (例: 12:30:00)")
        self.video_start_time_edit.setText("")
        results_layout.addWidget(self.video_start_time_edit, 1, 1, 1, 2)
        
        layout.addWidget(results_group)
        
        layout.addStretch()
        return widget
        
    def create_advanced_tab(self):
        """詳細設定タブ"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # パフォーマンス設定
        perf_group = QGroupBox(self.get_text('performance_settings'))
        perf_layout = QGridLayout(perf_group)
        
        self.use_gpu_check = QCheckBox(self.get_text('use_gpu'))
        self.use_gpu_check.setChecked(True)
        perf_layout.addWidget(self.use_gpu_check, 0, 0)
        
        self.use_tensorrt_check = QCheckBox("TensorRT使用")
        self.use_tensorrt_check.setChecked(False)
        perf_layout.addWidget(self.use_tensorrt_check, 0, 1)
        
        perf_layout.addWidget(QLabel("フレームスキップ:"), 1, 0)
        self.frame_skip_spin = QSpinBox()
        self.frame_skip_spin.setRange(0, 10)
        self.frame_skip_spin.setValue(0)
        perf_layout.addWidget(self.frame_skip_spin, 1, 1)
        
        self.use_batch_check = QCheckBox("バッチ推論")
        perf_layout.addWidget(self.use_batch_check, 2, 0)
        
        perf_layout.addWidget(QLabel("バッチサイズ:"), 2, 1)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 32)
        self.batch_size_spin.setValue(8)
        perf_layout.addWidget(self.batch_size_spin, 2, 2)
        
        self.show_preview_check = QCheckBox("プレビュー表示")
        self.show_preview_check.setChecked(True)
        self.show_preview_check.stateChanged.connect(self.on_preview_toggle)
        perf_layout.addWidget(self.show_preview_check, 3, 0)
        
        layout.addWidget(perf_group)
        
        # 出力設定
        output_group = QGroupBox(self.get_text('output_settings'))
        output_layout = QVBoxLayout(output_group)
        
        self.save_csv_check = QCheckBox("CSV保存")
        self.save_csv_check.setChecked(True)
        output_layout.addWidget(self.save_csv_check)
        
        self.save_json_check = QCheckBox("JSON保存")
        self.save_json_check.setChecked(True)
        output_layout.addWidget(self.save_json_check)
        
        self.save_vehicle_images_check = QCheckBox("車両画像保存")
        output_layout.addWidget(self.save_vehicle_images_check)
        
        layout.addWidget(output_group)
        
        # 車種判別設定
        vehicle_group = QGroupBox("車種判別")
        vehicle_layout = QGridLayout(vehicle_group)
        
        self.enable_classification_check = QCheckBox("車種判別を有効化")
        vehicle_layout.addWidget(self.enable_classification_check, 0, 0, 1, 3)
        
        vehicle_layout.addWidget(QLabel("モデル:"), 1, 0)
        self.vehicle_model_edit = QLineEdit()
        self.vehicle_model_edit.setText("car_classfier/vehicle_model.pt")
        vehicle_layout.addWidget(self.vehicle_model_edit, 1, 1)
        
        self.vehicle_model_browse_btn = QPushButton("参照")
        self.vehicle_model_browse_btn.clicked.connect(self.browse_vehicle_model)
        vehicle_layout.addWidget(self.vehicle_model_browse_btn, 1, 2)
        
        vehicle_layout.addWidget(QLabel(self.get_text('classification_threshold')), 2, 0)
        self.classification_threshold_spin = QDoubleSpinBox()
        self.classification_threshold_spin.setRange(0.0, 1.0)
        self.classification_threshold_spin.setSingleStep(0.05)
        self.classification_threshold_spin.setValue(0.5)
        self.classification_threshold_spin.setToolTip("車種判別の信頼度しきい値 (0.0-1.0)")
        vehicle_layout.addWidget(self.classification_threshold_spin, 2, 1, 1, 2)
        
        layout.addWidget(vehicle_group)
        
        layout.addStretch()
        return widget
        
    def create_batch_tab(self):
        """バッチ処理タブ"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 説明
        info_label = QLabel("複数の動画を連続で処理します")
        layout.addWidget(info_label)
        
        # 動画リスト
        self.batch_list = QListWidget()
        self.batch_list.setSelectionMode(QListWidget.ExtendedSelection)
        layout.addWidget(self.batch_list)
        
        # ボタン（2行に分ける）
        btn_layout1 = QHBoxLayout()
        
        self.add_video_btn = QPushButton("📹 動画を1つ追加")
        self.add_video_btn.clicked.connect(self.browse_single_video)
        btn_layout1.addWidget(self.add_video_btn)
        
        self.add_videos_btn = QPushButton("📋 動画を複数追加")
        self.add_videos_btn.clicked.connect(self.browse_multiple_videos)
        btn_layout1.addWidget(self.add_videos_btn)
        
        self.remove_video_btn = QPushButton("🗑 選択項目を削除")
        self.remove_video_btn.clicked.connect(self.remove_selected_videos)
        btn_layout1.addWidget(self.remove_video_btn)
        
        layout.addLayout(btn_layout1)
        
        btn_layout2 = QHBoxLayout()
        
        self.clear_list_btn = QPushButton("🧹 リストクリア")
        self.clear_list_btn.clicked.connect(self.clear_batch_list)
        btn_layout2.addWidget(self.clear_list_btn)
        
        self.start_batch_btn = QPushButton("▶ バッチ開始")
        self.start_batch_btn.clicked.connect(self.start_batch_processing)
        btn_layout2.addWidget(self.start_batch_btn)
        
        layout.addLayout(btn_layout2)
        
        return widget
        
    def create_control_buttons(self):
        """制御ボタン作成"""
        layout = QHBoxLayout()
        
        self.start_btn = QPushButton(self.get_text('start_processing'))
        self.start_btn.clicked.connect(self.start_processing)
        self.start_btn.setMinimumHeight(40)
        layout.addWidget(self.start_btn)
        
        self.pause_btn = QPushButton(self.get_text('pause_processing'))
        self.pause_btn.clicked.connect(self.pause_processing)
        self.pause_btn.setMinimumHeight(40)
        self.pause_btn.setEnabled(False)
        layout.addWidget(self.pause_btn)
        
        self.stop_btn = QPushButton(self.get_text('stop_processing'))
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setEnabled(False)
        layout.addWidget(self.stop_btn)
        
        # 設定の保存/読み込みボタンを横並びに
        config_buttons_layout = QHBoxLayout()
        
        self.save_config_btn = QPushButton(self.get_text('save_config'))
        self.save_config_btn.clicked.connect(self.save_config)
        self.save_config_btn.setMinimumHeight(40)
        config_buttons_layout.addWidget(self.save_config_btn)
        
        self.load_config_btn = QPushButton(self.get_text('load_config'))
        self.load_config_btn.clicked.connect(self.load_config_file)
        self.load_config_btn.setMinimumHeight(40)
        config_buttons_layout.addWidget(self.load_config_btn)
        
        layout.addLayout(config_buttons_layout)
        
        return layout
        
    def create_right_panel(self):
        """右側パネル作成"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # プレビュー
        preview_group = QGroupBox(self.get_text('video_preview'))
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(640, 360)
        self.preview_label.setStyleSheet("border: 1px solid #ccc; background: #000;")
        self.preview_label.setText("プレビュー")
        preview_layout.addWidget(self.preview_label)
        
        layout.addWidget(preview_group)
        
        # カウント表示
        count_group = QGroupBox("カウント")
        count_layout = QGridLayout(count_group)
        
        # 基本カウント（上り/下り/合計）
        count_layout.addWidget(QLabel("上り:"), 0, 0)
        self.up_count_label = QLabel("0")
        self.up_count_label.setFont(QFont("Arial", 16, QFont.Bold))
        count_layout.addWidget(self.up_count_label, 0, 1)
        
        count_layout.addWidget(QLabel("下り:"), 0, 2)
        self.down_count_label = QLabel("0")
        self.down_count_label.setFont(QFont("Arial", 16, QFont.Bold))
        count_layout.addWidget(self.down_count_label, 0, 3)
        
        count_layout.addWidget(QLabel("合計:"), 0, 4)
        self.total_count_label = QLabel("0")
        self.total_count_label.setFont(QFont("Arial", 16, QFont.Bold))
        count_layout.addWidget(self.total_count_label, 0, 5)
        
        # 車種判別カウント
        count_layout.addWidget(QLabel(self.get_text('large_vehicles')), 1, 0)
        self.large_count_label = QLabel("0")
        self.large_count_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.large_count_label.setStyleSheet("color: #d32f2f;")
        count_layout.addWidget(self.large_count_label, 1, 1)
        
        count_layout.addWidget(QLabel(self.get_text('small_vehicles')), 1, 2)
        self.small_count_label = QLabel("0")
        self.small_count_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.small_count_label.setStyleSheet("color: #1976d2;")
        count_layout.addWidget(self.small_count_label, 1, 3)
        
        count_layout.addWidget(QLabel(self.get_text('unknown_vehicles')), 1, 4)
        self.unknown_count_label = QLabel("0")
        self.unknown_count_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.unknown_count_label.setStyleSheet("color: #757575;")
        count_layout.addWidget(self.unknown_count_label, 1, 5)
        
        layout.addWidget(count_group)
        
        # 処理情報表示
        info_group = QGroupBox("処理情報")
        info_layout = QGridLayout(info_group)
        
        info_layout.addWidget(QLabel(self.get_text('elapsed_time')), 0, 0)
        self.elapsed_time_label = QLabel("00:00:00")
        self.elapsed_time_label.setFont(QFont("Arial", 12))
        info_layout.addWidget(self.elapsed_time_label, 0, 1)
        
        info_layout.addWidget(QLabel(self.get_text('estimated_time')), 0, 2)
        self.estimated_time_label = QLabel("--:--:--")
        self.estimated_time_label.setFont(QFont("Arial", 12))
        info_layout.addWidget(self.estimated_time_label, 0, 3)
        
        info_layout.addWidget(QLabel(self.get_text('current_time')), 1, 0)
        self.current_time_label = QLabel("--:--:--")
        self.current_time_label.setFont(QFont("Arial", 12))
        info_layout.addWidget(self.current_time_label, 1, 1)
        
        info_layout.addWidget(QLabel(self.get_text('fps')), 1, 2)
        self.fps_label = QLabel("0.0")
        self.fps_label.setFont(QFont("Arial", 12))
        info_layout.addWidget(self.fps_label, 1, 3)
        
        info_layout.addWidget(QLabel(self.get_text('frame_info')), 2, 0)
        self.frame_info_label = QLabel("0 / 0")
        self.frame_info_label.setFont(QFont("Arial", 12))
        info_layout.addWidget(self.frame_info_label, 2, 1, 1, 3)
        
        layout.addWidget(info_group)
        
        # プログレスバー
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # ログ
        log_group = QGroupBox(self.get_text('processing_log'))
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(200)
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_group)
        
        return panel
        
    def get_text(self, key, *args):
        """翻訳テキスト取得"""
        return get_text(self.current_language, key, *args)
        
    def on_language_change(self, text):
        """言語変更"""
        new_language = text.split(' ')[0]
        if new_language != self.current_language:
            self.current_language = new_language
            self.config['language'] = new_language
            self.config_manager.save_config_silently(self.config)
            # TODO: UI更新
    
    def on_line_mode_changed(self, index):
        """ラインモード変更時の処理"""
        # index 0: 1本（単一ライン）, index 1: 2本（上り/下り）
        is_dual = (index == 1)
        
        # 下りライン関連のウィジェットの表示/非表示
        for widget in self.down_line_widgets:
            widget.setVisible(is_dual)
        
        # 上りラインのラベルを変更
        if is_dual:
            self.up_line_label.setText("上りライン")
        else:
            self.up_line_label.setText("カウントライン")
            
    def log(self, message):
        """ログ追加"""
        self.log_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        
    def update_log(self):
        """ログ更新"""
        try:
            while True:
                message = self.log_queue.get_nowait()
                self.log_text.append(message)
        except queue.Empty:
            pass
            
    def update_progress(self, progress):
        """プログレス更新"""
        if isinstance(progress, dict):
            # 辞書形式の詳細情報
            self.progress_bar.setValue(int(progress.get('percent', 0)))
            
            # 経過時間
            if 'elapsed' in progress:
                elapsed = progress['elapsed']
                hours, remainder = divmod(int(elapsed), 3600)
                minutes, seconds = divmod(remainder, 60)
                self.elapsed_time_label.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
            
            # 推定残り時間
            if 'remaining' in progress:
                remaining = progress['remaining']
                if remaining is not None and remaining >= 0:
                    hours, remainder = divmod(int(remaining), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    self.estimated_time_label.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
                else:
                    self.estimated_time_label.setText("--:--:--")
            
            # FPS
            if 'fps' in progress:
                self.fps_label.setText(f"{progress['fps']:.1f}")
            
            # フレーム情報
            if 'frame' in progress and 'total_frames' in progress:
                self.frame_info_label.setText(f"{progress['frame']} / {progress['total_frames']}")
        else:
            # 従来の数値形式（後方互換性）
            self.progress_bar.setValue(int(progress))
        
        # 現在時刻を常に更新
        from datetime import datetime
        self.current_time_label.setText(datetime.now().strftime("%H:%M:%S"))
        
    def update_counts(self, up_count, down_count, total_count, vehicle_counts=None):
        """カウント更新"""
        self.up_count_label.setText(str(up_count))
        self.down_count_label.setText(str(down_count))
        self.total_count_label.setText(str(total_count))
        
        # 車種判別カウントを更新
        if vehicle_counts:
            large_count = vehicle_counts.get('large', 0)
            small_count = vehicle_counts.get('small', 0)
            unknown_count = vehicle_counts.get('unknown', 0)
            self.large_count_label.setText(str(large_count))
            self.small_count_label.setText(str(small_count))
            self.unknown_count_label.setText(str(unknown_count))
        
    def on_preview_toggle(self, state):
        """プレビュー表示のオン/オフ切り替え時の処理"""
        if not state:
            # プレビューオフ時はキューとラベルをクリア
            try:
                while not self.frame_queue.empty():
                    self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            self.preview_label.clear()
            self.log("📹 プレビュー表示を無効化しました（安定性向上）")
        else:
            self.log("📹 プレビュー表示を有効化しました")
    
    def update_frame(self, frame):
        """フレーム更新（プレビューが無効の場合はスキップ）"""
        # プレビューが無効またはチェックボックスがオフの場合はスキップ
        if not self.preview_enabled or not self.show_preview_check.isChecked():
            return
            
        try:
            # キューが満杯の場合、古いフレームを破棄
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            # 最新フレームのみ追加（参照渡しでコピーしない - メモリ効率化）
            self.frame_queue.put(frame, block=False)
        except queue.Full:
            pass
        except Exception:
            pass
            
    def update_video_preview(self):
        """プレビュー更新（プレビュー無効時は何もしない）"""
        # プレビューが無効の場合は処理をスキップ
        if not self.preview_enabled or not self.show_preview_check.isChecked():
            return
            
        try:
            # キュー内の古いフレームをすべてスキップして最新のみ取得
            frame = None
            frame_count = 0
            while not self.frame_queue.empty() and frame_count < 10:  # 無限ループ防止
                try:
                    frame = self.frame_queue.get_nowait()
                    frame_count += 1
                except queue.Empty:
                    break
            
            if frame is None or frame.size == 0:
                return
            
            # フレームの検証
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                return
                
            h, w, ch = frame.shape
            if h <= 0 or w <= 0:
                return
            
            # OpenCV (BGR) -> Qt (RGB) - 独立したコピーを作成
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except Exception:
                return
            
            # contiguous配列として確実にコピー
            rgb_frame = np.ascontiguousarray(rgb_frame)
            
            # QImageを作成（データのコピーを使用）
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
            
            # 明示的にデータを保持（QImageが有効な間）
            del rgb_frame
            
            # QPixmapに変換
            if qt_image.isNull():
                return
                
            pixmap = QPixmap.fromImage(qt_image)
            if pixmap.isNull():
                return
            
            # ラベルサイズに合わせてスケール
            scaled_pixmap = pixmap.scaled(
                self.preview_label.size(),
                Qt.KeepAspectRatio,
                Qt.FastTransformation  # 高速変換に変更
            )
            
            # GUIスレッドで安全に更新
            if not scaled_pixmap.isNull():
                self.preview_label.setPixmap(scaled_pixmap)
            
            # メモリをクリア
            del qt_image, pixmap, scaled_pixmap
            
        except queue.Empty:
            pass
        except Exception as e:
            # エラーをログに記録（デバッグ用）
            # print(f"Preview update error: {e}")
            pass
            
    def load_config_to_gui(self):
        """設定をGUIに反映"""
        paths = self._ensure_paths_config()
        self.input_base_edit.setText(paths.get('input_base', ''))
        self.output_base_edit.setText(paths.get('output_base', ''))
        self.input_file_edit.setText(self.config['video']['input_file'])
        self.output_file_edit.setText(self.config['video']['output_file'])
        self.enable_output_check.setChecked(self.config['video'].get('enable_output', True))
        self.model_file_edit.setText(self.config['model']['model_file'])
        self.confidence_spin.setValue(self.config['model']['confidence_threshold'])
        
        # ライン設定
        if 'lines' in self.config:
            # ラインモード
            line_mode = self.config['lines'].get('mode', 'dual')
            self.line_mode_combo.setCurrentIndex(0 if line_mode == 'single' else 1)
            
            self.up_start_x_spin.setValue(self.config['lines']['up_line']['start_x'])
            self.up_start_y_spin.setValue(self.config['lines']['up_line']['start_y'])
            self.up_end_x_spin.setValue(self.config['lines']['up_line']['end_x'])
            self.up_end_y_spin.setValue(self.config['lines']['up_line']['end_y'])
            
            self.down_start_x_spin.setValue(self.config['lines']['down_line']['start_x'])
            self.down_start_y_spin.setValue(self.config['lines']['down_line']['start_y'])
            self.down_end_x_spin.setValue(self.config['lines']['down_line']['end_x'])
            self.down_end_y_spin.setValue(self.config['lines']['down_line']['end_y'])
        
        # パフォーマンス設定
        self.use_gpu_check.setChecked(self.config['performance']['use_gpu'])
        self.use_tensorrt_check.setChecked(self.config['performance'].get('use_tensorrt', False))
        self.frame_skip_spin.setValue(self.config['performance']['frame_skip'])
        self.use_batch_check.setChecked(self.config['performance'].get('use_batch_inference', False))
        self.batch_size_spin.setValue(self.config['performance'].get('batch_size', 8))
        self.show_preview_check.setChecked(self.config['performance'].get('show_preview', True))
        
        # 出力設定
        self.save_csv_check.setChecked(self.config['output']['save_csv'])
        self.save_json_check.setChecked(self.config['output']['save_json'])
        self.results_folder_edit.setText(self.config['output'].get('results_folder', 'results'))
        self.save_vehicle_images_check.setChecked(self.config.get('vehicle_images', {}).get('save_images', False))
        
        # 動画開始時刻設定
        if 'time_settings' in self.config and 'video_start_time' in self.config['time_settings']:
            self.video_start_time_edit.setText(self.config['time_settings']['video_start_time'])
        else:
            self.video_start_time_edit.setText("")
        
        # 車種判別
        if 'vehicle_classification' in self.config:
            self.enable_classification_check.setChecked(self.config['vehicle_classification'].get('enabled', False))
            self.vehicle_model_edit.setText(self.config['vehicle_classification'].get('model_path', 'car_classfier/vehicle_model.pt'))
            self.classification_threshold_spin.setValue(self.config['vehicle_classification'].get('threshold', 0.5))
            
    def get_config_from_gui(self):
        """GUIから設定取得"""
        self.config['video']['input_file'] = self.input_file_edit.text()
        self.config['video']['output_file'] = self.output_file_edit.text()
        self.config['video']['enable_output'] = self.enable_output_check.isChecked()
        paths = self._ensure_paths_config()
        paths['input_base'] = self.input_base_edit.text().strip()
        paths['output_base'] = self.output_base_edit.text().strip()
        self.config['model']['model_file'] = self.model_file_edit.text()
        self.config['model']['confidence_threshold'] = self.confidence_spin.value()
        
        # ライン設定
        if 'lines' not in self.config:
            self.config['lines'] = {
                'mode': 'dual',
                'up_line': {},
                'down_line': {}
            }
        
        # ラインモードを保存 (0: single, 1: dual)
        self.config['lines']['mode'] = 'single' if self.line_mode_combo.currentIndex() == 0 else 'dual'
            
        self.config['lines']['up_line']['start_x'] = self.up_start_x_spin.value()
        self.config['lines']['up_line']['start_y'] = self.up_start_y_spin.value()
        self.config['lines']['up_line']['end_x'] = self.up_end_x_spin.value()
        self.config['lines']['up_line']['end_y'] = self.up_end_y_spin.value()
        
        self.config['lines']['down_line']['start_x'] = self.down_start_x_spin.value()
        self.config['lines']['down_line']['start_y'] = self.down_start_y_spin.value()
        self.config['lines']['down_line']['end_x'] = self.down_end_x_spin.value()
        self.config['lines']['down_line']['end_y'] = self.down_end_y_spin.value()
        
        # パフォーマンス
        self.config['performance']['use_gpu'] = self.use_gpu_check.isChecked()
        self.config['performance']['use_tensorrt'] = self.use_tensorrt_check.isChecked()
        self.config['performance']['frame_skip'] = self.frame_skip_spin.value()
        self.config['performance']['use_batch_inference'] = self.use_batch_check.isChecked()
        self.config['performance']['batch_size'] = self.batch_size_spin.value()
        self.config['performance']['show_preview'] = self.show_preview_check.isChecked()
        
        # 出力
        self.config['output']['save_csv'] = self.save_csv_check.isChecked()
        self.config['output']['save_json'] = self.save_json_check.isChecked()
        self.config['output']['results_folder'] = self.results_folder_edit.text()
        
        if 'vehicle_images' not in self.config:
            self.config['vehicle_images'] = {}
        self.config['vehicle_images']['save_images'] = self.save_vehicle_images_check.isChecked()
        # 車両画像の保存先を統一
        self.config['vehicle_images']['output_folder'] = self.results_folder_edit.text()
        
        # 認識結果CSVの保存先も統一
        if 'recognition_results' not in self.config:
            self.config['recognition_results'] = {}
        self.config['recognition_results']['output_folder'] = self.results_folder_edit.text()
        
        # 動画開始時刻設定
        if 'time_settings' not in self.config:
            self.config['time_settings'] = {}
        video_start_time_text = self.video_start_time_edit.text().strip()
        if video_start_time_text:
            self.config['time_settings']['video_start_time'] = video_start_time_text
        elif 'video_start_time' in self.config['time_settings']:
            # 空の場合は削除
            del self.config['time_settings']['video_start_time']
        
        # 車種判別
        if 'vehicle_classification' not in self.config:
            self.config['vehicle_classification'] = {}
        self.config['vehicle_classification']['enabled'] = self.enable_classification_check.isChecked()
        self.config['vehicle_classification']['model_path'] = self.vehicle_model_edit.text()
        self.config['vehicle_classification']['threshold'] = self.classification_threshold_spin.value()
        
        return self.config

    def _ensure_paths_config(self):
        """pathsセクションの存在を保証"""
        paths = self.config.get('paths')
        if not isinstance(paths, dict):
            paths = {'input_base': '', 'output_base': ''}
            self.config['paths'] = paths
        else:
            paths.setdefault('input_base', '')
            paths.setdefault('output_base', '')
        return paths

    def get_input_base_dir(self):
        """入力ベースフォルダを取得"""
        return self._ensure_paths_config().get('input_base', '')

    def get_output_base_dir(self):
        """出力ベースフォルダを取得"""
        return self._ensure_paths_config().get('output_base', '')

    def resolve_input_path(self, path_value):
        """入力パスをベースフォルダ込みで解決"""
        return self.config_manager.resolve_with_base(path_value, self.get_input_base_dir())

    def resolve_output_path(self, path_value):
        """出力パスをベースフォルダ込みで解決"""
        return self.config_manager.resolve_with_base(path_value, self.get_output_base_dir())

    def to_relative_input_path(self, path_value):
        """入力パスを可能ならベースフォルダ相対に変換"""
        if not path_value:
            return path_value
        return self.config_manager.make_relative_to_base(path_value, self.get_input_base_dir())

    def to_relative_output_path(self, path_value):
        """出力パスを可能ならベースフォルダ相対に変換"""
        if not path_value:
            return path_value
        return self.config_manager.make_relative_to_base(path_value, self.get_output_base_dir())

    def _suggest_output_path(self, video_file: Path):
        """ベースフォルダ付きのデフォルト出力パスを生成"""
        output_base = self.get_output_base_dir()
        if output_base:
            base_dir = Path(output_base)
        else:
            parent = video_file.parent
            if parent == Path('.'):
                base_dir = Path('videos/output')
            else:
                base_dir = parent.parent / 'output'
        suffix = video_file.suffix or '.mp4'
        return base_dir / f"{video_file.stem}_output{suffix}"
        
    def browse_input_file(self):
        """入力ファイル選択"""
        start_dir = self.get_input_base_dir() or ""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            self.get_text('select_input_file'),
            start_dir,
            "動画ファイル (*.mp4 *.avi *.mov *.mkv);;すべてのファイル (*.*)"
        )
        if filename:
            display_value = self.to_relative_input_path(filename)
            self.input_file_edit.setText(display_value)
            
    def browse_output_file(self):
        """出力ファイル選択"""
        start_dir = self.get_output_base_dir() or ""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            self.get_text('select_output_file'),
            start_dir,
            "動画ファイル (*.mp4 *.avi *.mov);;すべてのファイル (*.*)"
        )
        if filename:
            display_value = self.to_relative_output_path(filename)
            self.output_file_edit.setText(display_value)

    def browse_input_base_folder(self):
        """入力ベースフォルダ選択"""
        folder = QFileDialog.getExistingDirectory(
            self,
            self.get_text('select_input_base_folder'),
            self.get_input_base_dir() or ""
        )
        if folder:
            self.input_base_edit.setText(folder)

    def browse_output_base_folder(self):
        """出力ベースフォルダ選択"""
        folder = QFileDialog.getExistingDirectory(
            self,
            self.get_text('select_output_base_folder'),
            self.get_output_base_dir() or ""
        )
        if folder:
            self.output_base_edit.setText(folder)
            
    def browse_model_file(self):
        """モデルファイル選択"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            self.get_text('select_model_file'),
            "models",
            "モデルファイル (*.pt *.engine);;すべてのファイル (*.*)"
        )
        if filename:
            self.model_file_edit.setText(filename)
            
    def browse_vehicle_model(self):
        """車種判別モデル選択"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "車種判別モデルを選択",
            "car_classfier",
            "PyTorch models (*.pt);;すべてのファイル (*.*)"
        )
        if filename:
            self.vehicle_model_edit.setText(filename)
    
    def browse_results_folder(self):
        """結果保存フォルダ選択"""
        folder = QFileDialog.getExistingDirectory(
            self,
            self.get_text('select_results_folder'),
            "results"
        )
        if folder:
            self.results_folder_edit.setText(folder)
    
    def open_line_drawer(self):
        """線描画ダイアログを開く"""
        # 入力動画を確認
        input_file_value = self.input_file_edit.text().strip()
        resolved_input_file = self.resolve_input_path(input_file_value)
        if not input_file_value or not Path(resolved_input_file).exists():
            QMessageBox.warning(
                self,
                self.get_text('warning'),
                self.get_text('input_file_not_found')
            )
            return
        
        # 動画の最初のフレームを取得
        cap = cv2.VideoCapture(resolved_input_file)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            QMessageBox.critical(
                self,
                self.get_text('error'),
                "動画の読み込みに失敗しました"
            )
            return
        
        # 線描画ダイアログを表示
        is_dual = (self.line_mode_combo.currentIndex() == 1)
        dialog = LineDrawerDialog(frame, is_dual, self)
        if dialog.exec() == QDialog.Accepted:
            lines = dialog.get_lines()
            
            # 上りライン
            if lines['up']:
                self.up_start_x_spin.setValue(lines['up'][0][0])
                self.up_start_y_spin.setValue(lines['up'][0][1])
                self.up_end_x_spin.setValue(lines['up'][1][0])
                self.up_end_y_spin.setValue(lines['up'][1][1])
            
            # 下りライン
            if lines['down']:
                self.down_start_x_spin.setValue(lines['down'][0][0])
                self.down_start_y_spin.setValue(lines['down'][0][1])
                self.down_end_x_spin.setValue(lines['down'][1][0])
                self.down_end_y_spin.setValue(lines['down'][1][1])
            
            QMessageBox.information(
                self,
                self.get_text('info'),
                self.get_text('line_draw_complete')
            )
            
    def browse_single_video(self):
        """動画を1つ追加"""
        start_dir = self.get_input_base_dir() or ""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "バッチ処理に追加する動画を選択",
            start_dir,
            "動画ファイル (*.mp4 *.avi *.mov *.mkv);;すべてのファイル (*.*)"
        )
        if filename:
            stored_path = self.to_relative_input_path(filename)
            # 既に追加されていないかチェック
            if stored_path not in self.batch_video_list:
                self.batch_video_list.append(stored_path)
                self.update_batch_list_display()
                video_name = Path(filename).name
                self.log(f"📹 バッチリストに追加: {video_name}")
            else:
                QMessageBox.information(
                    self,
                    self.get_text('info'),
                    "この動画は既にリストに追加されています"
                )
    
    def browse_multiple_videos(self):
        """複数動画選択"""
        start_dir = self.get_input_base_dir() or ""
        filenames, _ = QFileDialog.getOpenFileNames(
            self,
            "バッチ処理する動画を選択",
            start_dir,
            "動画ファイル (*.mp4 *.avi *.mov *.mkv);;すべてのファイル (*.*)"
        )
        if filenames:
            # 重複を除いて追加
            added_count = 0
            for filename in filenames:
                stored_path = self.to_relative_input_path(filename)
                if stored_path not in self.batch_video_list:
                    self.batch_video_list.append(stored_path)
                    added_count += 1
            
            if added_count > 0:
                self.update_batch_list_display()
                self.log(f"📋 バッチリストに追加: {added_count}個の動画")
            
            if added_count < len(filenames):
                QMessageBox.information(
                    self,
                    self.get_text('info'),
                    f"{len(filenames) - added_count}個の動画は既にリストに含まれています"
                )
            
    def update_batch_list_display(self):
        """バッチリスト表示更新"""
        self.batch_list.clear()
        for i, video in enumerate(self.batch_video_list, 1):
            video_name = Path(video).name
            if self.is_batch_processing and i - 1 == self.current_batch_index:
                self.batch_list.addItem(f"▶ {i}. {video_name}")
            elif self.is_batch_processing and i - 1 < self.current_batch_index:
                self.batch_list.addItem(f"✓ {i}. {video_name}")
            else:
                self.batch_list.addItem(f"  {i}. {video_name}")
                
    def clear_batch_list(self):
        """バッチリストクリア"""
        if self.is_processing:
            QMessageBox.warning(self, "警告", "処理実行中はリストをクリアできません")
            return
        self.batch_video_list = []
        self.batch_list.clear()
        self.log("📋 バッチリストをクリアしました")
    
    def remove_selected_videos(self):
        """選択された動画をリストから削除"""
        if self.is_processing:
            QMessageBox.warning(self, "警告", "処理実行中は動画を削除できません")
            return
            
        selected_items = self.batch_list.selectedItems()
        if not selected_items:
            QMessageBox.information(
                self,
                self.get_text('info'),
                "削除する動画を選択してください"
            )
            return
        
        # 選択されたインデックスを取得（降順でソート）
        selected_indices = sorted([self.batch_list.row(item) for item in selected_items], reverse=True)
        
        # リストから削除（後ろから削除することでインデックスのずれを防ぐ）
        for index in selected_indices:
            if 0 <= index < len(self.batch_video_list):
                video_name = Path(self.batch_video_list[index]).name
                del self.batch_video_list[index]
                self.log(f"🗑 削除: {video_name}")
        
        # 表示を更新
        self.update_batch_list_display()
        self.log(f"📋 リストに{len(self.batch_video_list)}個の動画が残っています")
        
    def start_batch_processing(self):
        """バッチ処理開始"""
        if not self.batch_video_list:
            QMessageBox.warning(self, "警告", "バッチ処理する動画が選択されていません")
            return
        if self.is_processing:
            QMessageBox.warning(self, "警告", "既に処理が実行中です")
            return
            
        self.is_batch_processing = True
        self.batch_stop_requested = False
        self.current_batch_index = 0
        self.log("=" * 60)
        self.log(f"🎬 バッチ処理開始: {len(self.batch_video_list)}個の動画")
        self.log("=" * 60)
        self.process_next_batch_video()
        
    def process_next_batch_video(self):
        """次の動画を処理"""
        if not self.is_batch_processing:
            return
        if self.current_batch_index >= len(self.batch_video_list):
            self.log("=" * 60)
            self.log(f"✅ バッチ処理完了: {len(self.batch_video_list)}個の動画を処理")
            self.log("=" * 60)
            self.is_batch_processing = False
            QMessageBox.information(self, "完了", f"バッチ処理が完了しました\n{len(self.batch_video_list)}個の動画を処理")
            return
            
        current_video = self.batch_video_list[self.current_batch_index]
        video_name = Path(current_video).name
        self.log(f"\n📹 [{self.current_batch_index + 1}/{len(self.batch_video_list)}] 処理中: {video_name}")
        
        self.update_batch_list_display()
        self.input_file_edit.setText(current_video)

        resolved_video_path = self.resolve_input_path(current_video)
        
        # 動画専用の設定ファイルが存在するか確認
        video_file = Path(resolved_video_path)
        base_dir = Path(__file__).resolve().parent
        config_dir = base_dir / 'configs'
        candidate_paths = [
            config_dir / f"{video_file.stem}_config.json",
            config_dir / f"{video_file.stem}.json",
            video_file.with_name(f"{video_file.stem}_config.json"),
            video_file.with_suffix('.json'),
        ]
        seen = set()
        unique_candidates = []
        for candidate in candidate_paths:
            try:
                key = candidate.resolve()
            except FileNotFoundError:
                key = candidate
            if key in seen:
                continue
            seen.add(key)
            unique_candidates.append(candidate)
        video_config_path = next((path for path in unique_candidates if path.exists()), None)
        
        if video_config_path:
            try:
                loaded_config = self.config_manager.load_config_from_path(video_config_path)
                # 設定を更新
                self.config = loaded_config
                
                # 必要な設定の補完（互換性対応）
                if 'output' not in self.config:
                    self.config['output'] = {}
                if 'results_folder' not in self.config['output']:
                    self.config['output']['results_folder'] = 'results'
                
                if 'recognition_results' not in self.config:
                    self.config['recognition_results'] = {}
                if 'output_folder' not in self.config['recognition_results']:
                    self.config['recognition_results']['output_folder'] = self.config['output']['results_folder']
                
                if 'vehicle_images' not in self.config:
                    self.config['vehicle_images'] = {}
                if 'output_folder' not in self.config['vehicle_images']:
                    self.config['vehicle_images']['output_folder'] = self.config['output']['results_folder']
                
                # GUIに反映
                self.load_config_to_gui()
                
                if video_config_path.parent == config_dir:
                    display_name = video_config_path.name
                else:
                    display_name = str(video_config_path)
                self.log(f"✓ 設定ファイル読み込み: {display_name}")
            except Exception as e:
                self.log(f"⚠ 設定ファイル読み込みエラー（デフォルト設定を使用）: {e}")
        else:
            self.log(f"ℹ️ 専用設定ファイルなし（現在の設定を使用）")
            fallback_candidates = ", ".join(str(path) for path in unique_candidates)
            if fallback_candidates:
                self.log(f"   試行パス: {fallback_candidates}")
        
        # 出力ファイル名をベースフォルダ込みで自動設定
        suggested_output = self._suggest_output_path(video_file)
        display_output = self.to_relative_output_path(str(suggested_output))
        self.output_file_edit.setText(display_output)
        
        self.current_batch_index += 1
        self.start_processing()
        
    def start_processing(self):
        """処理開始"""
        if self.is_processing:
            return
            
        gui_config = self.get_config_from_gui()
        config = self.config_manager.prepare_runtime_config(gui_config)
        
        # ファイル存在チェック
        if not Path(config['video']['input_file']).exists():
            QMessageBox.critical(self, "エラー", self.get_text('input_file_not_found'))
            return
        if not Path(config['model']['model_file']).exists():
            QMessageBox.critical(self, "エラー", self.get_text('model_file_not_found'))
            return
            
        self.log(self.get_text('starting_processing'))
        self.is_processing = True
        
        # ボタン状態変更
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        
        # 処理スレッド開始
        self.processing_thread = ProcessingThread(self.video_processor, config)
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.error.connect(self.on_processing_error)
        self.processing_thread.start()
        
    def pause_processing(self):
        """処理一時停止/再開"""
        if not self.is_processing:
            return
            
        if self.video_processor.is_paused:
            config = self.get_config_from_gui()
            fps = getattr(self.video_processor, 'current_fps', 30)
            self.video_processor.update_line_configuration(config, fps)
            self.video_processor.resume_processing()
            self.pause_btn.setText(self.get_text('pause_processing'))
        else:
            self.video_processor.pause_processing()
            self.pause_btn.setText(self.get_text('resume_processing'))
            
    def stop_processing(self):
        """処理停止"""
        if self.is_processing:
            self.log(self.get_text('stopping_processing'))
            self.video_processor.stop_processing()
            self.is_processing = False
            if self.is_batch_processing:
                self.log("⏹ バッチ処理を中断要求しました")
                self.batch_stop_requested = True
                self.is_batch_processing = False
                self.update_batch_list_display()
            
            self.start_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            
    def on_processing_finished(self, success):
        """処理完了（メモリクリーンアップを含む）"""
        self.is_processing = False
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # フレームキューをクリア
        try:
            while not self.frame_queue.empty():
                self.frame_queue.get_nowait()
        except queue.Empty:
            pass
        
        # プレビューラベルをクリア
        self.preview_label.clear()
        
        # ガベージコレクションを強制実行
        import gc
        gc.collect()
        
        if success:
            if self.is_batch_processing:
                self.log(f"✅ 動画処理完了 [{self.current_batch_index}/{len(self.batch_video_list)}]")
                QTimer.singleShot(1000, self.process_next_batch_video)
            else:
                if self.batch_stop_requested:
                    self.log("⏹ バッチ処理を中断しました")
                    QMessageBox.information(self, "中断", "バッチ処理を停止しました")
                    self.batch_stop_requested = False
                    self.update_batch_list_display()
                else:
                    QMessageBox.information(self, "完了", self.get_text('processing_completed'))
                
    def on_processing_error(self, error):
        """処理エラー"""
        self.is_processing = False
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        
        if self.is_batch_processing:
            reply = QMessageBox.question(
                self, "エラー",
                f"動画処理中にエラーが発生しました:\n{error}\n\n残りの動画の処理を続行しますか？",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                self.is_batch_processing = False
        else:
            QMessageBox.critical(self, "エラー", self.get_text('processing_error', error))
            
    def save_config(self):
        """設定保存"""
        config = self.get_config_from_gui()
        video_path = config['video']['input_file']
        
        if not video_path:
            self.config_manager.save_config(config)
            return
            
        video_file = Path(video_path)
        config_dir = Path('configs')
        config_dir.mkdir(exist_ok=True)
        video_config_path = config_dir / f"{video_file.stem}_config.json"
        
        try:
            import json
            with open(video_config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            QMessageBox.information(self, "保存完了", f"動画専用設定を保存しました:\n{video_config_path}")
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"設定の保存に失敗しました:\n{e}")
    
    def load_config_file(self):
        """設定ファイルを読み込む"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            self.get_text('select_config_file'),
            "configs",
            "JSON Files (*.json);;All Files (*.*)"
        )
        
        if not filename:
            return
        
        try:
            loaded_config = self.config_manager.load_config_from_path(filename)
            
            # 設定を更新
            self.config = loaded_config
            
            # 必要な設定の補完（互換性対応）
            if 'output' not in self.config:
                self.config['output'] = {}
            if 'results_folder' not in self.config['output']:
                self.config['output']['results_folder'] = 'results'
            
            if 'recognition_results' not in self.config:
                self.config['recognition_results'] = {}
            if 'output_folder' not in self.config['recognition_results']:
                self.config['recognition_results']['output_folder'] = self.config['output']['results_folder']
            
            if 'vehicle_images' not in self.config:
                self.config['vehicle_images'] = {}
            if 'output_folder' not in self.config['vehicle_images']:
                self.config['vehicle_images']['output_folder'] = self.config['output']['results_folder']
            
            # GUIに反映
            self.load_config_to_gui()
            
            QMessageBox.information(
                self,
                self.get_text('info'),
                f"設定を読み込みました:\n{Path(filename).name}"
            )
            self.log(f"✓ 設定ファイル読み込み: {filename}")
        except Exception as e:
            QMessageBox.critical(
                self,
                self.get_text('error'),
                f"設定の読み込みに失敗しました:\n{e}"
            )
            self.log(f"✗ 設定読み込みエラー: {e}")


def main():
    """メイン関数"""
    app = QApplication(sys.argv)
    
    # アプリケーション情報
    app.setApplicationName("MICHI-AI")
    app.setOrganizationName("Traffic Analysis")
    
    # メインウィンドウ
    window = TrafficCounterMainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
