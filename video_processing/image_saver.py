import queue
import threading
from pathlib import Path
from datetime import datetime, timedelta

import cv2
import numpy as np


class VehicleImageSaver:
    """車両画像の保存処理を担当するクラス（非同期ワーカー込み）"""

    def __init__(self, log_func=None, queue_size=100):
        self._log = log_func if callable(log_func) else (lambda *_, **__: None)
        self._queue = queue.Queue(maxsize=queue_size)
        self._thread = None
        self._stop_flag = False

        self.vehicle_images_config = {}
        self.vehicle_image_counter = {'up': 0, 'down': 0}
        self.saved_vehicle_ids = {}

        self.input_video_name = None
        self.video_start_time = None
        self.video_fps = None

        self.vehicle_classifier = None
        self.vehicle_classification_enabled = False
        self._update_vehicle_class_count = None

    def configure(
        self,
        config,
        input_video_name,
        video_start_time,
        video_fps,
        vehicle_classifier,
        vehicle_classification_enabled,
        update_vehicle_class_count_cb,
    ):
        self.vehicle_images_config = config.get('vehicle_images', {})
        self.input_video_name = input_video_name
        self.video_start_time = video_start_time
        self.video_fps = video_fps
        self.vehicle_classifier = vehicle_classifier
        self.vehicle_classification_enabled = vehicle_classification_enabled
        self._update_vehicle_class_count = update_vehicle_class_count_cb

        if self.vehicle_images_config.get('save_images', False):
            output_folder = self.vehicle_images_config.get('output_folder', 'results')
            Path(output_folder).mkdir(parents=True, exist_ok=True)
            self.vehicle_image_counter = {'up': 0, 'down': 0}
            self.saved_vehicle_ids = {}
            self._log(f"✓ 車両画像保存が有効です: {output_folder}/")
            self._log(f"  保存形式: {output_folder}/{{動画名}}/{{時間帯}}/HHmmss_class_idXXXX.jpg")
        else:
            self._log("✓ 車両画像保存は無効です")

    def start(self):
        self._stop_flag = False
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self._log("✓ 画像保存ワーカースレッド開始")

    def stop(self):
        if self._thread and self._thread.is_alive():
            self._stop_flag = True
            try:
                self._queue.put(None, timeout=1.0)
            except queue.Full:
                pass
            self._thread.join(timeout=5.0)
            self._log("✓ 画像保存ワーカースレッド停止")

    def handle_vehicle_crossing(self, frame, tracks, line_type, crossed_in, crossed_out, frame_idx, record_result_cb):
        if len(tracks.tracker_id) == 0:
            return 0

        if line_type not in self.saved_vehicle_ids:
            self.saved_vehicle_ids[line_type] = set()

        crossing_count = 0
        for i, tracker_id in enumerate(tracks.tracker_id):
            if crossed_in[i] or crossed_out[i]:
                crossing_count += 1
                direction = 'in' if crossed_in[i] else 'out'

                if tracker_id in self.saved_vehicle_ids[line_type]:
                    continue

                image_saved, image_filename, vehicle_class, yolo_class_name, confidence, bbox = (
                    self._save_single_vehicle_image(frame, tracks, i, line_type, direction, frame_idx)
                )

                record_result_cb(
                    frame_idx=frame_idx,
                    vehicle_id=tracker_id,
                    yolo_class_name=yolo_class_name,
                    confidence=confidence,
                    bbox=bbox,
                    line_type=line_type,
                    crossing_direction=direction,
                    line_distance=None,
                    crossed=True,
                    image_saved=image_saved,
                    image_filename=image_filename,
                    vehicle_class=vehicle_class,
                )

                if image_saved:
                    self.saved_vehicle_ids[line_type].add(tracker_id)

        if crossing_count > 0:
            self._log(f"✓ {line_type}ライン通過: {crossing_count}台検出")

        return crossing_count

    def _save_single_vehicle_image(self, frame, tracks, vehicle_index, line_type, direction, frame_idx):
        tracker_id = tracks.tracker_id[vehicle_index]
        bbox = tracks.xyxy[vehicle_index]
        class_id = tracks.class_id[vehicle_index] if hasattr(tracks, 'class_id') and len(tracks.class_id) > vehicle_index else 2

        yolo_class_names = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
        yolo_class_name = yolo_class_names.get(class_id, "vehicle")

        confidence = float(tracks.confidence[vehicle_index]) if hasattr(tracks, 'confidence') and len(tracks.confidence) > vehicle_index else 0.0

        x1, y1, x2, y2 = map(int, bbox)
        vehicle_crop = frame[y1:y2, x1:x2].copy()
        if vehicle_crop.size == 0:
            return False, None, '', yolo_class_name, confidence, bbox

        save_line_type = f"{line_type}_{direction}"
        image_saved, image_filename, vehicle_class = self._save_vehicle_image(
            vehicle_crop, save_line_type, tracker_id, frame_idx, class_id
        )

        return image_saved, image_filename, vehicle_class, yolo_class_name, confidence, bbox

    def _worker(self):
        while not self._stop_flag:
            try:
                item = self._queue.get(timeout=0.5)
                if item is None:
                    self._queue.task_done()
                    break

                vehicle_crop, metadata = item
                self._write_vehicle_image_to_disk(vehicle_crop, metadata)
                self._queue.task_done()

            except queue.Empty:
                continue
            except Exception as exc:
                self._log(f"⚠ 画像保存ワーカーエラー: {exc}")

    def _save_vehicle_image(self, vehicle_crop, line_type, track_id, frame_idx, class_id=2):
        try:
            vehicle_crop_copy = np.ascontiguousarray(vehicle_crop.copy())
            metadata = self._prepare_vehicle_image_metadata(
                vehicle_crop=vehicle_crop,
                line_type=line_type,
                track_id=track_id,
                frame_idx=frame_idx,
                class_id=class_id,
            )
            if not metadata:
                return False, None, ''

            self._queue.put((vehicle_crop_copy, metadata), block=False)
            return True, metadata['relative_path'], metadata.get('vehicle_class', '') or ''
        except queue.Full:
            self._log(f"⚠ 画像保存キューが満杯 (ID:{track_id})")
            return False, None, ''
        except Exception as exc:
            self._log(f"⚠ 画像保存キュー追加エラー (ID:{track_id}): {exc}")
            return False, None, ''

    def _prepare_vehicle_image_metadata(self, vehicle_crop, line_type, track_id, frame_idx, class_id):
        base_output_folder = self.vehicle_images_config.get('output_folder', 'results')

        class_names = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
        yolo_class_name = class_names.get(class_id, "vehicle")

        parts = line_type.split('_')
        base_type = parts[0] if parts else line_type
        direction = parts[1] if len(parts) > 1 else ''

        if base_type not in self.vehicle_image_counter:
            self.vehicle_image_counter[base_type] = 0

        elapsed_seconds = frame_idx / self.video_fps if self.video_fps > 0 else 0
        if self.video_start_time:
            start_datetime = datetime.combine(datetime.today(), self.video_start_time)
            actual_datetime = start_datetime + timedelta(seconds=elapsed_seconds)
            hours = actual_datetime.hour
            minutes = actual_datetime.minute
            seconds = actual_datetime.second
        else:
            hours = int(elapsed_seconds // 3600)
            minutes = int((elapsed_seconds % 3600) // 60)
            seconds = int(elapsed_seconds % 60)

        safe_video_name = "".join(
            c for c in (self.input_video_name or 'unknown')
            if c.isalnum() or c in (' ', '-', '_')
        ).rstrip() or 'unknown'

        vehicle_class = ''
        if self.vehicle_classification_enabled and self.vehicle_classifier:
            try:
                classification_result = self.vehicle_classifier.classify_image(vehicle_crop)
                vehicle_class = classification_result.get('class_name', '') or ''
            except Exception as exc:
                self._log(f"⚠ 車種判別エラー (ID:{track_id}): {exc}")
                vehicle_class = 'unknown'

        if self.vehicle_classification_enabled and self._update_vehicle_class_count:
            tracked_class = vehicle_class if vehicle_class else 'unknown'
            self._update_vehicle_class_count(base_type, tracked_class, frame_idx, direction)

        time_str = f"{hours:02d}{minutes:02d}{seconds:02d}"
        if vehicle_class:
            filename = f"{time_str}_{vehicle_class}_id{track_id:04d}.jpg"
        else:
            filename = f"{time_str}_{yolo_class_name}_id{track_id:04d}.jpg"

        hour_folder = f"{hours:02d}"
        save_dir = Path(base_output_folder) / safe_video_name / hour_folder
        relative_path = f"{safe_video_name}/{hour_folder}/{filename}"

        return {
            'save_dir': save_dir,
            'filename': filename,
            'relative_path': relative_path,
            'base_type': base_type,
            'direction': direction,
            'vehicle_class': vehicle_class,
            'yolo_class_name': yolo_class_name,
            'track_id': track_id,
            'line_type': line_type,
        }

    def _write_vehicle_image_to_disk(self, vehicle_crop, metadata):
        try:
            save_dir = metadata['save_dir']
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / metadata['filename']

            vehicle_crop_copy = np.ascontiguousarray(vehicle_crop)
            success = cv2.imwrite(str(save_path), vehicle_crop_copy)

            track_id = metadata.get('track_id')
            base_type = metadata.get('base_type')

            if success:
                if base_type not in self.vehicle_image_counter:
                    self.vehicle_image_counter[base_type] = 0
                self.vehicle_image_counter[base_type] += 1

                direction = metadata.get('direction') or ''
                vehicle_label = metadata.get('vehicle_class') or metadata.get('yolo_class_name')
                relative_path = metadata.get('relative_path')

                direction_label = f"({direction})" if direction else ""
                self._log(f"📸 {base_type}{direction_label} 車両保存 [{vehicle_label}] (ID:{track_id}): {relative_path}")
            else:
                self._log(f"⚠ 車両画像保存失敗 (ID:{track_id}): cv2.imwrite failed")
        except Exception as exc:
            line_type = metadata.get('line_type')
            track_id = metadata.get('track_id')
            self._log(f"⚠ 車両画像保存エラー ({line_type}, ID:{track_id}): {exc}")
            import traceback
            self._log(traceback.format_exc())

    def save_vehicle_image_sync(self, vehicle_crop, line_type, track_id, frame_idx, class_id=2):
        metadata = self._prepare_vehicle_image_metadata(
            vehicle_crop=vehicle_crop,
            line_type=line_type,
            track_id=track_id,
            frame_idx=frame_idx,
            class_id=class_id,
        )
        if not metadata:
            return False, None, ''
        self._write_vehicle_image_to_disk(vehicle_crop, metadata)
        return True, metadata['relative_path'], metadata.get('vehicle_class', '') or ''

    def output_statistics(self, current_line_zones, vehicle_line_history=None):
        self._log("=" * 60)
        self._log("🚗 車両画像保存統計")
        self._log("=" * 60)

        base_folder = self.vehicle_images_config.get('output_folder', 'results')
        safe_video_name = "".join(
            c for c in (self.input_video_name or 'unknown')
            if c.isalnum() or c in (' ', '-', '_')
        ).rstrip() or 'unknown'
        video_folder = Path(base_folder) / safe_video_name

        if video_folder.exists():
            hour_stats = {}
            total_files = 0

            for hour_folder in sorted(video_folder.iterdir()):
                if hour_folder.is_dir():
                    hour = hour_folder.name
                    files = list(hour_folder.glob("*.jpg")) + list(hour_folder.glob("*.jpeg")) + list(hour_folder.glob("*.png"))
                    file_count = len(files)
                    hour_stats[hour] = file_count
                    total_files += file_count

            self._log(f"保存先: {video_folder}")
            self._log(f"保存された画像ファイル総数: {total_files}枚")

            if hour_stats:
                self._log("\n時間帯別保存統計:")
                for hour in sorted(hour_stats.keys()):
                    count = hour_stats[hour]
                    if count > 0:
                        self._log(f"  {hour}時台: {count}枚")

            if self.vehicle_classification_enabled:
                vehicle_class_stats = {'small': 0, 'large': 0, 'unknown': 0, 'other': 0}

                for hour_folder in video_folder.iterdir():
                    if hour_folder.is_dir():
                        for img_file in hour_folder.iterdir():
                            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                                filename = img_file.name
                                if '_small_' in filename:
                                    vehicle_class_stats['small'] += 1
                                elif '_large_' in filename:
                                    vehicle_class_stats['large'] += 1
                                elif '_unknown_' in filename:
                                    vehicle_class_stats['unknown'] += 1
                                else:
                                    vehicle_class_stats['other'] += 1

                if sum(vehicle_class_stats.values()) > 0:
                    self._log("\n車種別保存統計:")
                    if vehicle_class_stats['small'] > 0:
                        self._log(f"  小型車: {vehicle_class_stats['small']}枚")
                    if vehicle_class_stats['large'] > 0:
                        self._log(f"  大型車: {vehicle_class_stats['large']}枚")
                    if vehicle_class_stats['unknown'] > 0:
                        self._log(f"  不明: {vehicle_class_stats['unknown']}枚")
                    if vehicle_class_stats['other'] > 0:
                        self._log(f"  その他: {vehicle_class_stats['other']}枚")
        else:
            self._log(f"⚠ 保存フォルダが見つかりません: {video_folder}")
            self._log("車両画像は保存されませんでした")

        if current_line_zones:
            self._log("\nライン通過統計:")
            for line_type, line_zone in current_line_zones.items():
                total_count = line_zone.in_count + line_zone.out_count
                saved_count = len(self.saved_vehicle_ids.get(line_type, set()))
                save_rate = (saved_count / total_count * 100) if total_count > 0 else 0
                self._log(f"  {line_type}ライン: 通過{total_count}台 → 保存{saved_count}枚 ({save_rate:.1f}%)")

        total_tracked_vehicles = len(vehicle_line_history) if vehicle_line_history is not None else 0
        self._log(f"\n追跡された車両総数: {total_tracked_vehicles}台")

        self._log("=" * 60)
