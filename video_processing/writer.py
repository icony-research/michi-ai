import cv2
import threading
import queue


class AsyncVideoWriter:
    """OpenCV VideoWriter をバックグラウンドスレッドで駆動するラッパー"""

    def __init__(self, output_path, fps, frame_size, codec="mp4v", max_queue_size=240, log_func=None):
        self._output_path = str(output_path)
        self._fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(self._output_path, self._fourcc, fps, frame_size)
        if not self._writer.isOpened():
            raise RuntimeError(f"✗ 動画出力の初期化に失敗しました: {self._output_path}")

        self._queue = queue.Queue(maxsize=max_queue_size)
        self._thread = threading.Thread(target=self._run, name="AsyncVideoWriter", daemon=True)
        self._log = log_func if callable(log_func) else (lambda *_, **__: None)
        self._closed = False
        self._dropped_frames = 0
        self._overflow_warned = False
        self._thread.start()

    def _run(self):
        try:
            while True:
                frame = self._queue.get()
                if frame is None:
                    break
                self._writer.write(frame)
        finally:
            self._writer.release()
            if self._dropped_frames > 0:
                self._log(f"⚠️ 動画出力キューが飽和したため {self._dropped_frames} フレームを破棄しました")

    def write(self, frame):
        if self._closed:
            return

        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            self._drop_oldest()
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                self._dropped_frames += 1
                if not self._overflow_warned:
                    self._log("⚠️ 動画出力キューが満杯です。フレームを破棄します")
                    self._overflow_warned = True

    def _drop_oldest(self):
        try:
            _ = self._queue.get_nowait()
            self._dropped_frames += 1
        except queue.Empty:
            pass

    def release(self):
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join()

    def __del__(self):
        self.release()
