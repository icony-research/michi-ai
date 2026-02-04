from datetime import datetime, timedelta, time


class CountManager:
    """時間帯別カウントと車種別カウントを管理する"""

    def __init__(self, log_func=None):
        self._log = log_func if callable(log_func) else (lambda *_, **__: None)
        self.hourly_counts = {}
        self.video_start_time = None
        self.current_fps = None

        self.vehicle_classifier = None
        self.vehicle_classification_enabled = False
        self.vehicle_class_counts = {}
        self.total_vehicle_class_counts = {
            'up': {'small': 0, 'large': 0, 'unknown': 0},
            'down': {'small': 0, 'large': 0, 'unknown': 0},
            'single': {'small': 0, 'large': 0, 'unknown': 0},
        }

    def set_current_fps(self, fps):
        self.current_fps = fps

    def setup_time_settings(self, config):
        start_time_str = config.get('time_settings', {}).get('video_start_time', '07:00:00')
        try:
            if start_time_str.count(':') == 2:
                hour, minute, second = map(int, start_time_str.split(':'))
                self.video_start_time = time(hour, minute, second)
                self._log(f"✓ 動画開始時刻: {start_time_str}")
            elif start_time_str.count(':') == 1:
                hour, minute = map(int, start_time_str.split(':'))
                self.video_start_time = time(hour, minute, 0)
                self._log(f"✓ 動画開始時刻: {start_time_str}:00")
            else:
                raise ValueError("Invalid time format")
        except (ValueError, AttributeError):
            self.video_start_time = time(7, 0, 0)
            self._log(f"⚠ 時刻形式エラー、デフォルト値（07:00:00）を使用: {start_time_str}")

        self.hourly_counts = {}
        for i in range(12):
            hour = (self.video_start_time.hour + i) % 24
            self.hourly_counts[hour] = {'up': 0, 'down': 0}

    def get_current_hour_from_frame(self, frame_idx, fps):
        elapsed_seconds = frame_idx / fps
        start_datetime = datetime.combine(datetime.today(), self.video_start_time)
        current_datetime = start_datetime + timedelta(seconds=elapsed_seconds)
        return current_datetime.hour

    def update_hourly_count(self, line_type, frame_idx, fps):
        current_hour = self.get_current_hour_from_frame(frame_idx, fps)
        if current_hour in self.hourly_counts:
            self.hourly_counts[current_hour][line_type] += 1

    def get_hourly_summary(self):
        summary = []
        for hour in sorted(self.hourly_counts.keys()):
            up_count = self.hourly_counts[hour]['up']
            down_count = self.hourly_counts[hour]['down']
            total = up_count + down_count
            time_period = f"{hour:02d}:00:00-{(hour+1)%24:02d}:00:00"
            summary.append({
                'time_period': time_period,
                'hour': hour,
                'up': up_count,
                'down': down_count,
                'total': total,
            })
        return summary

    def setup_vehicle_classification(self, config):
        classification_config = config.get('vehicle_classification', {})
        self.vehicle_classification_enabled = classification_config.get('enabled', False)

        if self.vehicle_classification_enabled:
            try:
                from vehicle_classifier import VehicleClassifier

                model_path = classification_config.get('model_path', 'car_classfier/vehicle_model.pt')
                confidence_threshold = classification_config.get('confidence_threshold', 0.6)

                self.vehicle_classifier = VehicleClassifier(model_path=model_path)
                self.vehicle_classifier.set_confidence_threshold(confidence_threshold)

                if self.vehicle_classifier.load_model():
                    self._log(f"✓ 車種判別モデル読み込み完了: {model_path}")
                    self._log(f"  信頼度閾値: {confidence_threshold}")
                else:
                    self._log(f"⚠ 車種判別モデル読み込み失敗: {model_path}")
                    self.vehicle_classification_enabled = False
            except ImportError as exc:
                self._log(f"⚠ 車種判別機能の読み込みに失敗: {exc}")
                self.vehicle_classification_enabled = False
        else:
            self._log("車種判別機能: 無効")

        self.total_vehicle_class_counts = {
            'up': {'small': 0, 'large': 0, 'unknown': 0},
            'down': {'small': 0, 'large': 0, 'unknown': 0},
            'single': {'small': 0, 'large': 0, 'unknown': 0},
        }

        self.vehicle_class_counts = {}
        for hour in self.hourly_counts.keys():
            self.vehicle_class_counts[hour] = {
                'up': {'small': 0, 'large': 0, 'unknown': 0},
                'down': {'small': 0, 'large': 0, 'unknown': 0},
                'single': {'small': 0, 'large': 0, 'unknown': 0},
            }

    def update_vehicle_class_count(self, line_type, vehicle_class, frame_idx=None, direction='in'):
        if line_type == 'single':
            count_direction = 'up' if direction == 'in' else 'down'
        else:
            count_direction = line_type

        if count_direction in self.total_vehicle_class_counts and vehicle_class in self.total_vehicle_class_counts[count_direction]:
            self.total_vehicle_class_counts[count_direction][vehicle_class] += 1

        if frame_idx is not None and self.current_fps:
            current_hour = self.get_current_hour_from_frame(frame_idx, self.current_fps)
            if current_hour in self.vehicle_class_counts:
                if count_direction in self.vehicle_class_counts[current_hour] and vehicle_class in self.vehicle_class_counts[current_hour][count_direction]:
                    self.vehicle_class_counts[current_hour][count_direction][vehicle_class] += 1

    def get_vehicle_class_summary(self):
        if not self.vehicle_classification_enabled:
            return None

        summary = {
            'total': self.total_vehicle_class_counts,
            'hourly': [],
        }

        for hour in sorted(self.vehicle_class_counts.keys()):
            hour_data = {
                'hour': hour,
                'time_period': f"{hour:02d}:00:00-{(hour+1)%24:02d}:00:00",
                'up': self.vehicle_class_counts[hour]['up'],
                'down': self.vehicle_class_counts[hour]['down'],
            }
            summary['hourly'].append(hour_data)

        return summary
