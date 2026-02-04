import csv
import json
from datetime import datetime
from pathlib import Path


class ResultsExporter:
    """CSV/JSON出力とサマリー表示を管理する"""

    def __init__(self, log_func=None):
        self._log = log_func if callable(log_func) else (lambda *_, **__: None)
        self.count_manager = None
        self.image_saver = None
        self.recognition_manager = None

    def configure(self, count_manager, image_saver, recognition_manager):
        self.count_manager = count_manager
        self.image_saver = image_saver
        self.recognition_manager = recognition_manager

    def output_results(self, config, line_zones, total_frames, fps, processing_time):
        self._log("-" * 60)
        self._log("✓ 処理完了！")
        self._log(f"総フレーム数: {total_frames}")

        if 'up' in line_zones and 'down' in line_zones:
            up_count = line_zones['up'].in_count + line_zones['up'].out_count
            down_count = line_zones['down'].in_count + line_zones['down'].out_count
            total_count = up_count + down_count

            self._log(f"上りライン通過: {up_count}")
            self._log(f"下りライン通過: {down_count}")
            self._log(f"総通過台数: {total_count}")
        else:
            line_zone = line_zones['single']
            self._log(f"上り方向: {line_zone.in_count}")
            self._log(f"下り方向: {line_zone.out_count}")
            self._log(f"総通過台数: {line_zone.in_count + line_zone.out_count}")

        self._log(f"処理時間: {processing_time:.2f}秒")
        if config['video'].get('enable_output', True):
            self._log(f"出力動画: {config['video']['output_file']}")
        else:
            self._log("出力動画: 無効（動画出力OFF）")

        hourly_summary = self.count_manager.get_hourly_summary()
        if hourly_summary:
            self._log("-" * 40)
            self._log("時間帯別交通量サマリー:")
            for hour_data in hourly_summary:
                time_period = hour_data['time_period']
                up_count = hour_data['up']
                down_count = hour_data['down']
                total = hour_data['total']
                if total > 0:
                    self._log(f"  {time_period}: 上り{up_count}台, 下り{down_count}台, 計{total}台")
            self._log("-" * 40)

        if self.count_manager.vehicle_classification_enabled:
            vehicle_summary = self.count_manager.get_vehicle_class_summary()
            if vehicle_summary:
                self._log("-" * 40)
                self._log("車種別交通量サマリー:")
                total_counts = vehicle_summary['total']
                for direction in ['up', 'down']:
                    direction_name = '上り' if direction == 'up' else '下り'
                    counts = total_counts[direction]
                    small_count = counts['small']
                    large_count = counts['large']
                    unknown_count = counts['unknown']
                    total_dir = small_count + large_count + unknown_count
                    if total_dir > 0:
                        self._log(f"  {direction_name}: 小型車{small_count}台, 大型車{large_count}台, 不明{unknown_count}台, 計{total_dir}台")
                self._log("-" * 40)

        if self.image_saver.vehicle_images_config.get('save_images', False):
            total_saved_images = sum(self.image_saver.vehicle_image_counter.values())
            unique_vehicles = len(self.image_saver.saved_vehicle_ids)
            self._log(f"保存車両画像数: {total_saved_images}")
            self._log(f"ユニーク車両数: {unique_vehicles}")

        if config['output']['save_csv']:
            self.save_csv(config, line_zones, processing_time)

        if config['output']['save_json']:
            self.save_json(config, line_zones, total_frames, fps, processing_time)

        if self.recognition_manager.recognition_config.get('auto_save_on_completion', True):
            csv_path = self.recognition_manager.save_recognition_results_csv()
            if csv_path:
                summary = self.recognition_manager.get_recognition_results_summary()
                self._log(summary)

        self._log("=" * 60)

    def save_csv(self, config, line_zones, processing_time):
        try:
            video_basename = Path(config['video']['input_file']).stem
            safe_video_name = "".join(c for c in video_basename if c.isalnum() or c in (' ', '-', '_')).rstrip()

            results_folder = Path(config['output'].get('results_folder', 'results'))

            original_csv_path = Path(config['output']['csv_file'])
            new_csv_filename = f"{safe_video_name}_{original_csv_path.name}"
            new_csv_path = results_folder / new_csv_filename

            new_csv_path.parent.mkdir(parents=True, exist_ok=True)

            with open(new_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)

                if 'up' in line_zones and 'down' in line_zones:
                    up_count = line_zones['up'].in_count + line_zones['up'].out_count
                    down_count = line_zones['down'].in_count + line_zones['down'].out_count
                else:
                    line_zone = line_zones['single']
                    up_count = line_zone.in_count
                    down_count = line_zone.out_count

                writer.writerow(['タイムスタンプ', '上り', '下り', '合計', '処理時間(秒)'])
                writer.writerow([
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    up_count,
                    down_count,
                    up_count + down_count,
                    f'{processing_time:.2f}',
                ])

                hourly_summary = self.count_manager.get_hourly_summary()
                if hourly_summary:
                    writer.writerow([])
                    writer.writerow(['時間帯別交通量サマリー'])
                    writer.writerow(['時間帯', '上り', '下り', '合計'])
                    for hour_data in hourly_summary:
                        if hour_data['total'] > 0:
                            writer.writerow([
                                hour_data['time_period'],
                                hour_data['up'],
                                hour_data['down'],
                                hour_data['total'],
                            ])

                if self.count_manager.vehicle_classification_enabled:
                    vehicle_summary = self.count_manager.get_vehicle_class_summary()
                    if vehicle_summary:
                        writer.writerow([])
                        writer.writerow(['車種別交通量サマリー（全体）'])
                        writer.writerow(['方向', '小型車', '大型車', '不明', '合計'])

                        total_counts = vehicle_summary['total']
                        for direction in ['up', 'down']:
                            direction_name = '上り' if direction == 'up' else '下り'
                            counts = total_counts[direction]
                            small = counts['small']
                            large = counts['large']
                            unknown = counts['unknown']
                            total_dir = small + large + unknown

                            if total_dir > 0:
                                writer.writerow([direction_name, small, large, unknown, total_dir])

                        if vehicle_summary['hourly']:
                            writer.writerow([])
                            writer.writerow(['時間帯別車種別交通量サマリー'])
                            writer.writerow(['時間帯', '上り-小型車', '上り-大型車', '上り-不明', '上り-合計',
                                           '下り-小型車', '下り-大型車', '下り-不明', '下り-合計', '合計'])

                            for hour_data in vehicle_summary['hourly']:
                                up_counts = hour_data['up']
                                down_counts = hour_data['down']

                                up_small = up_counts['small']
                                up_large = up_counts['large']
                                up_unknown = up_counts['unknown']
                                up_total = up_small + up_large + up_unknown

                                down_small = down_counts['small']
                                down_large = down_counts['large']
                                down_unknown = down_counts['unknown']
                                down_total = down_small + down_large + down_unknown

                                hour_total = up_total + down_total

                                if hour_total > 0:
                                    writer.writerow([
                                        hour_data['time_period'],
                                        up_small, up_large, up_unknown, up_total,
                                        down_small, down_large, down_unknown, down_total,
                                        hour_total,
                                    ])
            self._log(f"✓ CSV保存: {new_csv_path}")
        except Exception as exc:
            self._log(f"⚠ CSV保存エラー: {exc}")

    def save_json(self, config, line_zones, total_frames, fps, processing_time):
        try:
            video_basename = Path(config['video']['input_file']).stem
            safe_video_name = "".join(c for c in video_basename if c.isalnum() or c in (' ', '-', '_')).rstrip()

            results_folder = Path(config['output'].get('results_folder', 'results'))

            original_json_path = Path(config['output']['json_file'])
            new_json_filename = f"{safe_video_name}_{original_json_path.name}"
            new_json_path = results_folder / new_json_filename

            new_json_path.parent.mkdir(parents=True, exist_ok=True)

            if 'up' in line_zones and 'down' in line_zones:
                up_count = line_zones['up'].in_count + line_zones['up'].out_count
                down_count = line_zones['down'].in_count + line_zones['down'].out_count
                total_count = up_count + down_count

                results = {
                    'timestamp': datetime.now().isoformat(),
                    'video': {
                        'input': config['video']['input_file'],
                        'output': config['video']['output_file'],
                        'total_frames': total_frames,
                        'fps': fps,
                        'processing_time_seconds': processing_time,
                    },
                    'counts': {
                        'up_line': up_count,
                        'down_line': down_count,
                        'total': total_count,
                    },
                    'line_details': {
                        'up_line': {
                            'in_count': line_zones['up'].in_count,
                            'out_count': line_zones['up'].out_count,
                        },
                        'down_line': {
                            'in_count': line_zones['down'].in_count,
                            'out_count': line_zones['down'].out_count,
                        },
                    },
                    'hourly_counts': self.count_manager.get_hourly_summary(),
                    'vehicle_classification': self.count_manager.get_vehicle_class_summary() if self.count_manager.vehicle_classification_enabled else None,
                }
            else:
                line_zone = line_zones['single']
                results = {
                    'timestamp': datetime.now().isoformat(),
                    'video': {
                        'input': config['video']['input_file'],
                        'output': config['video']['output_file'],
                        'total_frames': total_frames,
                        'fps': fps,
                        'processing_time_seconds': processing_time,
                    },
                    'counts': {
                        'up': line_zone.in_count,
                        'down': line_zone.out_count,
                        'total': line_zone.in_count + line_zone.out_count,
                    },
                    'hourly_counts': self.count_manager.get_hourly_summary(),
                    'vehicle_classification': self.count_manager.get_vehicle_class_summary() if self.count_manager.vehicle_classification_enabled else None,
                }

            with open(new_json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            self._log(f"✓ JSON保存: {new_json_path}")
        except Exception as exc:
            self._log(f"⚠ JSON保存エラー: {exc}")
