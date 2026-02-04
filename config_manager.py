"""
設定管理とファイル操作のユーティリティ
"""

import json
import platform
import os
import glob
from copy import deepcopy
from pathlib import Path
from tkinter import filedialog, messagebox


class ConfigManager:
    """設定の保存・読み込みを管理するクラス"""
    
    def __init__(self, get_text_func=None):
        """
        Args:
            get_text_func: 翻訳関数（翻訳が必要な場合）
        """
        self.get_text = get_text_func if get_text_func else lambda x, *args: x
    
    def load_default_config(self):
        """デフォルト設定を読み込む"""
        try:
            config = self.load_config_from_path("config.json")
            return config
        except Exception:
            # OS別にフォントパスを設定
            font_path = self._get_pil_font_path()
            if font_path is None:
                # フォントが見つからない場合はOS別のデフォルト
                system = platform.system()
                if system == "Windows":
                    font_path = "C:/Windows/Fonts/msgothic.ttc"
                elif system == "Darwin":  # macOS
                    font_path = "/System/Library/Fonts/Hiragino Sans GB.ttc"
                else:  # Linux
                    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            
            default_config = {
                "paths": {
                    "input_base": "",
                    "output_base": ""
                },
                "video": {"input_file": "videos/input/input.mp4", "output_file": "videos/output/output.mp4", "codec": "mp4v"},
                "model": {"model_file": "models/yolov8x.pt", "confidence_threshold": 0.25, "iou_threshold": 0.45, "image_size": 640, "device": "auto"},
                "detection": {"vehicle_classes": [2, 3, 5, 7], "class_names": {"2": "car", "3": "motorcycle", "5": "bus", "7": "truck"}},
                "tracking": {"track_activation_threshold": 0.25, "lost_track_buffer": 60, "minimum_matching_threshold": 0.8, "frame_rate": 30},
                "lines": {
                    "mode": "dual",  # "single" または "dual"
                    "up_line": {"start_x": 100, "start_y": 200, "end_x": 1400, "end_y": 200, "thickness": 2},
                    "down_line": {"start_x": 100, "start_y": 300, "end_x": 1400, "end_y": 300, "thickness": 2}
                },
                # 下位互換性のため旧形式も保持
                "line": {"start_x": 100, "start_y": 250, "end_x": 1400, "end_y": 250, "thickness": 2},
                "visualization": {"box_thickness": 2, "label_text_scale": 0.5, "label_text_thickness": 1, "label_text_padding": 5, "font_path": font_path, "font_size": 32},
                "output": {"save_csv": True, "save_json": True, "csv_file": "results/traffic_count.csv", "json_file": "results/traffic_count.json", "results_folder": "results"},
                "vehicle_images": {"save_images": False, "output_folder": "results"},
                "recognition_results": {"output_folder": "results"},
                "performance": {"use_gpu": True, "frame_skip": 0, "show_preview": False},
                "language": "ja"
            }
            return self._ensure_paths_section(default_config)
    
    def save_config(self, config, filename="config.json"):
        """設定をJSONファイルに保存"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            messagebox.showinfo(
                self.get_text('info'),
                self.get_text('config_saved')
            )
        except Exception as e:
            messagebox.showerror(
                self.get_text('error'),
                self.get_text('config_save_failed', str(e))
            )
    
    def load_config(self):
        """設定ファイルを読み込む"""
        filename = filedialog.askopenfilename(
            title=self.get_text('select_config_file'),
            filetypes=[
                (self.get_text('json_files'), "*.json"),
                (self.get_text('all_files'), "*.*")
            ]
        )
        
        if filename:
            try:
                config = self.load_config_from_path(filename)
                messagebox.showinfo(
                    self.get_text('info'),
                    self.get_text('config_loaded', filename)
                )
                return config
            except Exception as e:
                messagebox.showerror(
                    self.get_text('error'),
                    self.get_text('config_load_failed', str(e))
                )
        return None
    
    def save_config_silently(self, config, filename="config.json"):
        """設定をメッセージなしで保存（内部使用）"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"設定の保存に失敗しました: {e}")
    
    def load_config_from_path(self, config_path, visited=None):
        """指定されたパスから設定を読み込み、base_configがあればマージ"""
        config_path = Path(config_path)
        if visited is None:
            visited = set()
        resolved = config_path.resolve()
        if resolved in visited:
            raise ValueError(f"循環参照が検出されました: {resolved}")
        visited.add(resolved)

        with open(resolved, 'r', encoding='utf-8') as f:
            config = json.load(f)

        base_config_path = config.pop('base_config', None)
        if base_config_path:
            base_path = Path(base_config_path)
            if not base_path.is_absolute():
                base_path = (resolved.parent / base_path).resolve()
            base_config = self.load_config_from_path(base_path, visited)
        else:
            base_config = {}

        merged = self._deep_merge_dicts(base_config, config)

        # 言語設定のデフォルト
        if 'language' not in merged:
            merged['language'] = 'ja'
        
        # 下位互換性のためライン設定を統一
        merged = self._convert_to_dual_lines(merged)
        return self._ensure_paths_section(merged)

    def _deep_merge_dicts(self, base, override):
        """dict同士を再帰的にマージ（override優先）"""
        if not isinstance(base, dict):
            base = {}
        if not isinstance(override, dict):
            return deepcopy(override)

        merged = deepcopy(base)
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._deep_merge_dicts(merged[key], value)
            else:
                merged[key] = deepcopy(value)
        return merged

    def merge_configs(self, base_config, override_config):
        """外部から利用できるマージラッパー"""
        return self._deep_merge_dicts(base_config, override_config)

    def prepare_runtime_config(self, config):
        """ベースフォルダ設定を反映させた設定のコピーを返す"""
        runtime_config = deepcopy(config)
        paths = self._ensure_paths_section(runtime_config).get('paths', {})
        video_cfg = runtime_config.get('video', {})

        if 'input_file' in video_cfg:
            video_cfg['input_file'] = self.resolve_with_base(
                video_cfg['input_file'],
                paths.get('input_base', "")
            )
        if 'output_file' in video_cfg:
            video_cfg['output_file'] = self.resolve_with_base(
                video_cfg['output_file'],
                paths.get('output_base', "")
            )
        return runtime_config

    def resolve_with_base(self, path_value, base_dir):
        """ベースフォルダを考慮してパスを絶対化"""
        if not path_value:
            return path_value
        path_obj = Path(path_value)
        if path_obj.is_absolute():
            return str(path_obj)
        if base_dir:
            return str(Path(base_dir) / path_obj)
        return str(path_obj)

    def make_relative_to_base(self, path_value, base_dir):
        """パスがベースフォルダ配下なら相対パスに変換"""
        if not path_value or not base_dir:
            return path_value
        try:
            base_path = Path(base_dir).resolve()
            target_path = Path(path_value).resolve()
            relative = target_path.relative_to(base_path)
            return str(relative)
        except Exception:
            return path_value

    def _ensure_paths_section(self, config):
        """pathsセクションが存在しデフォルト値を持つよう保証"""
        if not isinstance(config, dict):
            return config
        paths = config.get('paths')
        if not isinstance(paths, dict):
            paths = {}
            config['paths'] = paths
        paths.setdefault('input_base', "")
        paths.setdefault('output_base', "")
        return config
    
    def create_required_folders(self):
        """必要なフォルダを自動作成（改善版：階層構造対応）"""
        folders = [
            "models",
            "videos/input",
            "videos/output",
            "results",  # 基本resultsフォルダのみ（動画名・時間帯は保存時に自動作成）
            "docs"
        ]
        
        for folder in folders:
            folder_path = Path(folder)
            if not folder_path.exists():
                folder_path.mkdir(parents=True, exist_ok=True)
                print(f"📁 フォルダを作成しました: {folder}")
    
    def _get_pil_font_path(self):
        """PIL用の日本語フォントパスを取得"""
        system = platform.system()
        
        if system == "Windows":
            # Windows用フォントパス候補
            font_paths = [
                "C:/Windows/Fonts/msgothic.ttc",
                "C:/Windows/Fonts/meiryo.ttc",
                "C:/Windows/Fonts/YuGothR.ttc"
            ]
        elif system == "Darwin":  # macOS
            # macOS用フォントパス候補
            font_paths = [
                "/System/Library/Fonts/Hiragino Sans GB.ttc",
                "/Library/Fonts/Arial Unicode.ttf",
                "/System/Library/Fonts/Geneva.ttf"
            ]
        else:  # Linux
            # Linux用フォントパス候補
            font_paths = []
            # Noto フォントを検索
            noto_paths = glob.glob("/usr/share/fonts/*/Noto*CJK*JP*.ttf") + \
                        glob.glob("/usr/share/fonts/*/Noto*CJK*JP*.otf") + \
                        glob.glob("/usr/share/fonts/truetype/noto/Noto*CJK*JP*.ttf") + \
                        glob.glob("/usr/share/fonts/opentype/noto/Noto*CJK*JP*.otf")
            font_paths.extend(noto_paths)
            
            # その他のフォント
            font_paths.extend([
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf"
            ])
        
        # 存在するフォントパスを返す
        for font_path in font_paths:
            if os.path.exists(font_path):
                return font_path
        
        return None
    
    def browse_input_file(self):
        """入力動画ファイルを選択"""
        filename = filedialog.askopenfilename(
            title=self.get_text('select_input_video'),
            filetypes=[
                (self.get_text('video_files'), "*.mp4 *.avi *.mov *.mkv"),
                (self.get_text('mp4_files'), "*.mp4"),
                (self.get_text('all_files'), "*.*")
            ]
        )
        return filename
    
    def browse_output_file(self):
        """出力動画ファイルを選択"""
        filename = filedialog.asksaveasfilename(
            title=self.get_text('save_output_video'),
            defaultextension=".mp4",
            filetypes=[
                (self.get_text('mp4_files'), "*.mp4"),
                (self.get_text('video_files'), "*.avi *.mov *.mkv"),
                (self.get_text('all_files'), "*.*")
            ]
        )
        return filename
    
    def browse_model_file(self):
        """モデルファイルを選択"""
        filename = filedialog.askopenfilename(
            title=self.get_text('select_yolo_model'),
            filetypes=[
                (self.get_text('yolo_models'), "*.pt"),
                (self.get_text('all_files'), "*.*")
            ]
        )
        return filename
    
    def _convert_to_dual_lines(self, config):
        """旧形式の設定を2本ライン形式に変換"""
        # 新形式のlines設定がない場合、旧形式のlineから作成
        if 'lines' not in config and 'line' in config:
            old_line = config['line']
            config['lines'] = {
                'up_line': {
                    'start_x': old_line['start_x'],
                    'start_y': old_line['start_y'] - 50,  # 上ライン
                    'end_x': old_line['end_x'],
                    'end_y': old_line['end_y'] - 50,
                    'thickness': old_line.get('thickness', 2)
                },
                'down_line': {
                    'start_x': old_line['start_x'],
                    'start_y': old_line['start_y'] + 50,  # 下ライン
                    'end_x': old_line['end_x'],
                    'end_y': old_line['end_y'] + 50,
                    'thickness': old_line.get('thickness', 2)
                }
            }
        
        # lines設定がない場合はデフォルト作成
        if 'lines' not in config:
            config['lines'] = {
                'mode': 'dual',
                'up_line': {'start_x': 100, 'start_y': 200, 'end_x': 1400, 'end_y': 200, 'thickness': 2},
                'down_line': {'start_x': 100, 'start_y': 300, 'end_x': 1400, 'end_y': 300, 'thickness': 2}
            }
        
        # lines設定にmodeがない場合は追加
        if 'mode' not in config['lines']:
            config['lines']['mode'] = 'dual'
        
        # 車両画像設定がない場合はデフォルト作成
        if 'vehicle_images' not in config:
            config['vehicle_images'] = {
                'save_images': False,
                'output_folder': 'results'
            }
        
        # output設定にresults_folderがない場合は追加
        if 'output' not in config:
            config['output'] = {}
        if 'results_folder' not in config['output']:
            config['output']['results_folder'] = 'results'
        
        # 認識結果の保存先設定がない場合は追加
        if 'recognition_results' not in config:
            config['recognition_results'] = {}
        if 'output_folder' not in config['recognition_results']:
            # output.results_folderと統一
            config['recognition_results']['output_folder'] = config['output'].get('results_folder', 'results')
        
        # vehicle_imagesのoutput_folderがない場合も統一
        if 'output_folder' not in config['vehicle_images']:
            config['vehicle_images']['output_folder'] = config['output'].get('results_folder', 'results')
        
        return config
