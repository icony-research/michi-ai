from datetime import datetime, timedelta
from pathlib import Path
import csv


class RecognitionResultsManager:
    """認識結果の記録と保存を管理する"""

    def __init__(self, log_func=None):
        self._log = log_func if callable(log_func) else (lambda *_, **__: None)
        self.recognition_results = []
        self.recognition_config = {}
        self.video_fps = None
        self.video_start_time = None
        self.class_names = {}
        self.input_video_name = None

    def configure(self, recognition_config, video_fps, video_start_time, class_names, input_video_name):
        self.recognition_config = recognition_config or {}
        self.video_fps = video_fps
        self.video_start_time = video_start_time
        self.class_names = class_names or {}
        self.input_video_name = input_video_name
        self.recognition_results = []

    def record_all_detections(self, frame, tracks, line_zones, frame_idx):
        if not self.recognition_config.get('record_all_detections', False):
            return

        video_time_seconds = frame_idx / self.video_fps if self.video_fps > 0 else 0
        hours = int(video_time_seconds // 3600)
        minutes = int((video_time_seconds % 3600) // 60)
        seconds = int(video_time_seconds % 60)
        milliseconds = int((video_time_seconds % 1) * 1000)
        video_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

        for bbox, conf, class_id, track_id in zip(
            tracks.xyxy, tracks.confidence, tracks.class_id, tracks.tracker_id
        ):
            yolo_class_name = self.class_names.get(int(class_id), f'class_{class_id}')

            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1

            closest_line_type = None
            closest_distance = float('inf')
            crossed = False
            crossing_direction = None

            for line_type, line_zone in line_zones.items():
                distance = abs(center_y - line_zone.vector.start.y)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_line_type = line_type

            result = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                'frame_number': frame_idx,
                'vehicle_id': track_id,
                'yolo_class_name': yolo_class_name,
                'confidence': round(float(conf), 6),
                'bbox_x1': round(float(x1), 2),
                'bbox_y1': round(float(y1), 2),
                'bbox_x2': round(float(x2), 2),
                'bbox_y2': round(float(y2), 2),
                'center_x': round(float(center_x), 2),
                'center_y': round(float(center_y), 2),
                'width': round(float(width), 2),
                'height': round(float(height), 2),
                'line_type': closest_line_type or 'unknown',
                'crossing_direction': crossing_direction or 'none',
                'line_distance': round(float(closest_distance), 2),
                'crossed': crossed,
                'image_saved': False,
                'image_filename': '',
                'video_time': video_time,
                'processing_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }
            self.recognition_results.append(result)

    def record_recognition_result(
        self,
        frame_idx,
        vehicle_id,
        yolo_class_name,
        confidence,
        bbox,
        line_type=None,
        crossing_direction=None,
        line_distance=None,
        crossed=False,
        image_saved=False,
        image_filename=None,
        vehicle_class=None,
    ):
        if not self.recognition_config.get('save_results', True):
            return

        elapsed_seconds = frame_idx / self.video_fps if self.video_fps > 0 else 0

        if self.video_start_time:
            start_datetime = datetime.combine(datetime.today(), self.video_start_time)
            actual_datetime = start_datetime + timedelta(seconds=elapsed_seconds)
            hours = actual_datetime.hour
            minutes = actual_datetime.minute
            seconds = actual_datetime.second
            milliseconds = int((elapsed_seconds % 1) * 1000)
        else:
            hours = int(elapsed_seconds // 3600)
            minutes = int((elapsed_seconds % 3600) // 60)
            seconds = int(elapsed_seconds % 60)
            milliseconds = int((elapsed_seconds % 1) * 1000)

        video_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1

        result = {
            'timestamp': elapsed_seconds,
            'frame_number': frame_idx,
            'vehicle_id': vehicle_id,
            'yolo_class_name': yolo_class_name,
            'vehicle_class': vehicle_class or '',
            'confidence': confidence,
            'bbox_x1': int(x1),
            'bbox_y1': int(y1),
            'bbox_x2': int(x2),
            'bbox_y2': int(y2),
            'center_x': int(center_x),
            'center_y': int(center_y),
            'width': int(width),
            'height': int(height),
            'line_type': line_type or '',
            'crossing_direction': crossing_direction or '',
            'line_distance': round(line_distance, 2) if line_distance is not None else '',
            'crossed': crossed,
            'image_saved': image_saved,
            'image_filename': image_filename or '',
            'video_time': video_time,
            'processing_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        self.recognition_results.append(result)

    def save_recognition_results_csv(self, custom_filename=None):
        try:
            if not self.recognition_results:
                self._log("⚠️ 保存する認識結果がありません")
                return False

            if custom_filename:
                csv_path = Path(custom_filename)
            else:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                video_name = self.input_video_name or 'unknown'
                safe_video_name = "".join(c for c in video_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                csv_filename = f"{safe_video_name}_recognition_results_{timestamp}.csv"
                output_folder = self.recognition_config.get('output_folder', 'results')
                Path(output_folder).mkdir(parents=True, exist_ok=True)
                csv_path = Path(output_folder) / csv_filename

            csv_path.parent.mkdir(parents=True, exist_ok=True)

            headers = [
                'timestamp', 'frame_number', 'vehicle_id', 'yolo_class_name', 'vehicle_class', 'confidence',
                'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'center_x', 'center_y',
                'width', 'height', 'line_type', 'crossing_direction', 'line_distance',
                'crossed', 'image_saved', 'image_filename', 'video_time', 'processing_time',
            ]

            with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                for i, result in enumerate(self.recognition_results):
                    try:
                        writer.writerow(result)
                    except Exception as row_error:
                        self._log(f"⚠️ 行 {i+1} の書き込みエラー: {row_error}")
                        self._log(f"   問題のあるデータ: {result}")
                        raise

            self._log(f"✓ 認識結果CSV保存: {csv_path} ({len(self.recognition_results)}件のレコード)")
            return str(csv_path)

        except Exception as exc:
            self._log(f"⚠️ 認識結果CSV保存エラー: {exc}")
            import traceback
            self._log(traceback.format_exc())
            return False

    def get_recognition_results_summary(self):
        if not self.recognition_results:
            return "認識結果なし"

        total_detections = len(self.recognition_results)
        unique_vehicles = len(set(r['vehicle_id'] for r in self.recognition_results))
        crossed_vehicles = len([r for r in self.recognition_results if r['crossed']])
        saved_images = len([r for r in self.recognition_results if r['image_saved']])

        class_counts = {}
        for result in self.recognition_results:
            yolo_class_name = result['yolo_class_name']
            class_counts[yolo_class_name] = class_counts.get(yolo_class_name, 0) + 1

        summary = (
            "📊 認識結果統計:\n"
            f"• 総検出数: {total_detections}件\n"
            f"• ユニーク車両数: {unique_vehicles}台\n"
            f"• ライン通過車両: {crossed_vehicles}台\n"
            f"• 画像保存数: {saved_images}枚\n"
            f"• クラス別: {', '.join([f'{k}:{v}' for k, v in class_counts.items()])}"
        )

        return summary
