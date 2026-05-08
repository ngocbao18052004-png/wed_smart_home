import re
import json
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from threading import Event, Lock, Thread
import traceback
from uuid import uuid4

import cv2
import face_recognition
import numpy as np


class FaceRecognitionCamera:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.project_root = self.base_dir
        self.dataset_dir = self.project_root / "DATASET" / "Trainining"

        self.known_names = []
        self.known_encodings = []
        self.capture = None
        self.capture_lock = Lock()
        self.is_running = False
        self.current_camera_index = None
        self.last_status = "Chưa mở camera."
        self.last_frame_time = None
        self.current_fps = 0.0
        self.processing_scale = 0.5
        self.process_every_n_frames = 3
        self.frame_counter = 0
        self.cached_face_results = []
        self.recognition_lock = Lock()
        self.latest_frame_for_recognition = None
        self.last_recognition_time = 0.0
        self.last_recognition_snapshot_time = 0.0
        self.recognition_interval = 0.05
        self.recognition_thread = None
        self.recognition_stop_event = Event()
        self.recognition_timeout = 45.0
        self.face_match_tolerance = 0.45
        self.runtime_dir = self.base_dir / "_runtime"
        self.recognition_worker_path = self.base_dir / "recognize_worker.py"
        self.use_subprocess_recognition = False
        self.current_alert_level = "idle"
        self.current_alert_text = "Chưa khởi động camera."
        self.last_recognized_names = []
        self.last_unknown_count = 0
        self.supported_suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self._load_dataset()

    def _load_dataset(self):
        if not self.dataset_dir.exists():
            raise FileNotFoundError(
                f"Không tìm thấy dataset tại đường link: {self.dataset_dir}"
            )

        self.known_names = []
        self.known_encodings = []

        for image_path in sorted(self.dataset_dir.iterdir()):
            if image_path.suffix.lower() not in self.supported_suffixes:
                continue

            try:
                image = face_recognition.load_image_file(str(image_path))
            except Exception as exc:
                print(f"Skipping invalid dataset file {image_path}: {exc}")
                continue

            if image is None or not isinstance(image, np.ndarray):
                print(f"Skipping invalid dataset file {image_path}: not a valid image")
                continue

            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.ndim == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            elif image.ndim != 3 or image.shape[2] != 3:
                print(f"Skipping unsupported dataset image format {image_path}: shape={image.shape}")
                continue

            try:
                encodings = face_recognition.face_encodings(image)
            except Exception as exc:
                print(f"Skipping dataset file {image_path}: cannot encode face: {exc}")
                continue

            if not encodings:
                continue

            self.known_encodings.append(encodings[0])
            self.known_names.append(self._normalize_name(image_path.stem))

    @staticmethod
    def _normalize_name(file_stem):
        match = re.search(r"^(.*?)(_\d+)?$", file_stem)
        if match and match.group(1):
            return match.group(1).strip()
        return file_stem.strip()

    @staticmethod
    def _slugify_name(name):
        # Giữ nguyên tên người dùng nhập để làm tên file
        return name.strip()

    @staticmethod
    def _face_center(points):
        point_array = np.array(points, dtype=np.float32)
        return point_array.mean(axis=0)

    def _next_dataset_path(self, name, suffix=".jpg"):
        base_name = self._slugify_name(name)
        # Tìm xem đã có file nào bắt đầu bằng tên này chưa
        existing = sorted(self.dataset_dir.glob(f"{base_name}*{suffix}"))
        if not existing:
            return self.dataset_dir / f"{base_name}{suffix}"
        
        next_index = len(existing) + 1
        return self.dataset_dir / f"{base_name}_{next_index}{suffix}"

    def _crop_face_image(self, frame_bgr, face_location, padding_ratio=0.32):
        top, right, bottom, left = face_location
        face_height = bottom - top
        face_width = right - left
        pad_y = int(face_height * padding_ratio)
        pad_x = int(face_width * padding_ratio)

        crop_top = max(top - pad_y, 0)
        crop_bottom = min(bottom + pad_y, frame_bgr.shape[0])
        crop_left = max(left - pad_x, 0)
        crop_right = min(right + pad_x, frame_bgr.shape[1])
        return frame_bgr[crop_top:crop_bottom, crop_left:crop_right]

    def _validate_face_pose(self, landmarks):
        required_keys = {"left_eye", "right_eye", "nose_tip"}
        if not required_keys.issubset(landmarks):
            return False, "Không đọc được đặc điểm khuôn mặt."

        left_eye_center = self._face_center(landmarks["left_eye"])
        right_eye_center = self._face_center(landmarks["right_eye"])
        nose_center = self._face_center(landmarks["nose_tip"])

        eye_dx = abs(right_eye_center[0] - left_eye_center[0])
        if eye_dx < 1:
            return False, "Không xác định được vị trí khuôn mặt."

        eye_slope = abs(right_eye_center[1] - left_eye_center[1]) / eye_dx
        eye_mid_x = (left_eye_center[0] + right_eye_center[0]) / 2
        nose_offset = abs(nose_center[0] - eye_mid_x) / eye_dx

        if eye_slope > 0.22:
            return False, "Vui lòng giữ thẳng, không nghiêng đầu."

        if nose_offset > 0.28:
            return False, "Vui lòng hướng mặt vào camera."

        return True, None

    def _prepare_registration(self, frame_bgr):
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")

        if len(face_locations) != 1:
            return False, "Cần 1 khuôn mặt rõ ràng trong ảnh.", None, None

        face_location = face_locations[0]
        top, right, bottom, left = face_location
        face_width = right - left
        face_height = bottom - top
        if face_width < 120 or face_height < 120:
            return False, "Khuôn mặt quá nhỏ, vui lòng tiến gần lại.", None, None

        landmarks_list = face_recognition.face_landmarks(rgb_frame, [face_location])
        if not landmarks_list:
            return False, "Không đọc được khuôn mặt, vui lòng thử lại.", None, None

        is_pose_ok, pose_message = self._validate_face_pose(landmarks_list[0])
        if not is_pose_ok:
            return False, pose_message, None, None

        face_encodings = face_recognition.face_encodings(rgb_frame, [face_location])
        if not face_encodings:
            return False, "Không mã hóa được khuôn mặt, vui lòng thử ảnh khác.", None, None

        cropped_face = self._crop_face_image(frame_bgr, face_location)
        if cropped_face.size == 0:
            return False, "Không cắt được vùng khuôn mặt hợp lệ.", None, None

        return True, "Hợp lệ", cropped_face, face_encodings[0]

    def _find_existing_face_name(self, face_encoding, tolerance=0.45):
        if not self.known_encodings:
            return None

        distances = face_recognition.face_distance(self.known_encodings, face_encoding)
        if len(distances) == 0:
            return None

        best_match_index = int(np.argmin(distances))
        if distances[best_match_index] <= tolerance:
            return self.known_names[best_match_index]

        return None

    def _save_registered_face(self, name, face_image_bgr, face_encoding=None):
        file_path = self._next_dataset_path(name)
        success, encoded_img = cv2.imencode(file_path.suffix, face_image_bgr)
        if not success:
            return False, "Không thể mã hóa ảnh để lưu vào dataset."
            
        try:
            with open(file_path, "wb") as f:
                f.write(encoded_img.tobytes())
        except Exception as e:
            return False, f"Không thể lưu vào dataset: {str(e)}"

        if face_encoding is not None:
            self.known_names.append(self._normalize_name(file_path.stem))
            self.known_encodings.append(face_encoding)
        else:
            try:
                self._load_dataset()
            except Exception:
                if file_path.exists():
                    file_path.unlink()
                raise

        return True, f"Đã lưu {self._normalize_name(name)} vào dữ liệu."

    def delete_face(self, base_name):
        deleted_count = 0
        target = self._slugify_name(base_name)
        for filepath in self.dataset_dir.iterdir():
            if filepath.suffix.lower() in self.supported_suffixes:
                if self._normalize_name(filepath.stem) == target:
                    filepath.unlink()
                    deleted_count += 1
        
        if deleted_count > 0:
            self._load_dataset()
            return True, f"Đã xóa thành công toàn bộ ảnh của {base_name}."
        return False, "Không tìm thấy dữ liệu để xóa."

    def rename_face(self, old_name, new_name):
        old_target = self._slugify_name(old_name)
        new_target = self._slugify_name(new_name)
        if not new_target:
            return False, "Tên mới không hợp lệ."
        
        renamed_count = 0
        for file in self.dataset_dir.iterdir():
            if file.suffix.lower() in self.supported_suffixes:
                if self._normalize_name(file.stem) == new_target:
                    return False, "Tên mới đã trùng với một dữ liệu có sẵn khác."

        for filepath in self.dataset_dir.iterdir():
            if filepath.suffix.lower() in self.supported_suffixes:
                if self._normalize_name(filepath.stem) == old_target:
                    import re
                    match = re.search(r"^.*?(_\d+)?$", filepath.stem)
                    suffix_part = match.group(1) if match and match.group(1) else ""
                    new_filename = f"{new_target}{suffix_part}{filepath.suffix}"
                    new_filepath = self.dataset_dir / new_filename
                    filepath.rename(new_filepath)
                    renamed_count += 1
                    
        if renamed_count > 0:
            self._load_dataset()
            return True, f"Đã đổi tên dữ liệu thành {new_name}."
        return False, "Không tìm thấy dữ liệu để đổi tên."

    def register_face_from_upload(self, name, image_bytes):
        clean_name = (name or "").strip()
        if not clean_name:
            return False, "Vui lòng nhập trên trước khi upload."

        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return False, "Ảnh upload không hợp lệ."

        is_valid, message, cropped_face, face_encoding = self._prepare_registration(frame_bgr)
        if not is_valid:
            return False, message, "invalid"

        existing_name = self._find_existing_face_name(face_encoding)
        if existing_name:
            return (
                False,
                f"Dữ liệu {existing_name} đã tồn tại. Thêm dữ liệu mới.",
                "duplicate",
            )

        ok, message = self._save_registered_face(clean_name, cropped_face, face_encoding)
        return ok, message, "created" if ok else "invalid"

    def start_camera(self, camera_index):
        self.stop_camera()

        with self.capture_lock:
            self.is_running = False
            self.current_camera_index = None
            self.last_frame_time = None
            self.current_fps = 0.0
            self.frame_counter = 0
            self.cached_face_results = []
            self.current_alert_level = "idle"
            self.current_alert_text = "Đang khởi động camera."
            self.last_recognized_names = []
            self.last_unknown_count = 0

            capture, backend_name, attempted_backends = self._open_camera(camera_index)
            if capture is None:
                self.last_status = (
                    f"Không mở được camera index {camera_index}. Vui lòng nhập index khác!"
                )
                self.current_alert_level = "error"
                self.current_alert_text = "Không thể kết nối camera."
                return False, self.last_status

            self.capture = capture
            self.is_running = True
            self.current_camera_index = camera_index
            self.last_status = "Camera đang hoạt động."
            self.current_alert_level = "monitoring"
            self.current_alert_text = "Hệ thống đang giám sát khuôn mặt."
            self._start_recognition_worker()
            return True, self.last_status

    def _start_recognition_worker(self):
        self.recognition_stop_event.clear()
        if self.recognition_thread is not None and self.recognition_thread.is_alive():
            return

        self.recognition_thread = Thread(
            target=self._recognition_loop,
            name="face-recognition-worker",
            daemon=True,
        )
        self.recognition_thread.start()

    def _stop_recognition_worker(self):
        self.recognition_stop_event.set()
        if self.recognition_thread is not None and self.recognition_thread.is_alive():
            self.recognition_thread.join(timeout=1.5)
        self.recognition_thread = None

    def _open_camera(self, camera_index):
        backend_candidates = []

        if hasattr(cv2, "CAP_DSHOW"):
            backend_candidates.append(("DSHOW", cv2.CAP_DSHOW))
        if hasattr(cv2, "CAP_MSMF"):
            backend_candidates.append(("MSMF", cv2.CAP_MSMF))
        backend_candidates.append(("AUTO", None))

        attempted_backends = []

        for backend_name, backend_flag in backend_candidates:
            attempted_backends.append(backend_name)

            try:
                if backend_flag is None:
                    capture = cv2.VideoCapture(camera_index)
                else:
                    capture = cv2.VideoCapture(camera_index, backend_flag)
            except Exception:
                traceback.print_exc()
                continue

            if capture is not None and capture.isOpened():
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                for _ in range(8):
                    success, frame = capture.read()
                    if success and frame is not None:
                        return capture, backend_name, attempted_backends
                    time.sleep(0.08)

            if capture is not None:
                capture.release()

        return None, None, attempted_backends

    def stop_camera(self):
        self._stop_recognition_worker()
        capture_to_release = None
        with self.capture_lock:
            if self.capture is not None:
                capture_to_release = self.capture
                self.capture = None

            self.is_running = False
            self.current_camera_index = None
            self.last_frame_time = None
            self.current_fps = 0.0
            self.frame_counter = 0
            with self.recognition_lock:
                self.cached_face_results = []
                self.latest_frame_for_recognition = None
                self.last_recognition_time = 0.0
                self.last_recognition_snapshot_time = 0.0
            self.last_status = "Đã tắt camera."
            self.current_alert_level = "idle"
            self.current_alert_text = "Camera đã được tắt."
            self.last_recognized_names = []
            self.last_unknown_count = 0

        if capture_to_release is not None:
            try:
                capture_to_release.release()
            except Exception:
                traceback.print_exc()
            time.sleep(0.25)

        self.last_status = "Đã tắt camera."
        self.current_alert_level = "idle"
        self.current_alert_text = "Camera đã được tắt."

    def cleanup_camera(self):
        try:
            self.stop_camera()
            cv2.destroyAllWindows()
        except Exception:
            traceback.print_exc()

    def _put_text_vietnamese(self, frame_bgr, text, position, font_size=20, color=(255, 255, 255)):
        try:
            from PIL import Image, ImageDraw, ImageFont
            img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(pil_img)
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()
            
            x, y = position
            y = y - int(font_size * 0.8)
            
            b, g, r = color
            draw.text((x, y), text, font=font, fill=(r, g, b))
            
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            cv2.putText(frame_bgr, str(text), position, cv2.FONT_HERSHEY_SIMPLEX, font_size/25.0, color, 1)
            return frame_bgr

    def _annotate_frame(self, frame):
        now = time.time()
        if self.last_frame_time is not None:
            delta = now - self.last_frame_time
            if delta > 0:
                instant_fps = 1.0 / delta
                self.current_fps = (
                    instant_fps
                    if self.current_fps == 0.0
                    else (self.current_fps * 0.85) + (instant_fps * 0.15)
                )
        self.last_frame_time = now

        with self.recognition_lock:
            face_data = list(self.cached_face_results)

        if face_data:
            frame = self._draw_overlay(frame, face_data)
            cv2.putText(
                frame,
                f"FPS: {self.current_fps:.1f}",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )
            return frame

        face_boxes = []
        try:
            if not self.face_cascade.empty():
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_boxes = self.face_cascade.detectMultiScale(
                    gray_frame,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(70, 70),
                )
        except Exception:
            traceback.print_exc()
            face_boxes = []

        for x, y, width, height in face_boxes:
            label = "Kiểm tra dữ liệu"
            color = (0, 180, 255)
            cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
            cv2.rectangle(
                frame,
                (x, y + height - 30),
                (x + width, y + height),
                color,
                cv2.FILLED,
            )
            frame = self._put_text_vietnamese(
                frame,
                label,
                (x + 6, y + height - 9),
                font_size=18,
                color=(20, 20, 20)
            )

        self.last_recognized_names = []
        self.last_unknown_count = 0
        self.current_alert_level = "monitoring"
        self.current_alert_text = (
            f"Phát hiện {len(face_boxes)} khuôn mặt, đang kiểm tra dữ liệu."
            if len(face_boxes)
            else "Không phát hiện khuôn mặt nào trong hình."
        )
        self.last_status = "Camera đang hoạt động."

        cv2.putText(
            frame,
            f"FPS: {self.current_fps:.1f}",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )
        return frame

    def _recognition_loop(self):
        while not self.recognition_stop_event.is_set():
            try:
                if not self.is_running:
                    self.recognition_stop_event.wait(0.2)
                    continue

                with self.recognition_lock:
                    if self.latest_frame_for_recognition is None:
                        frame = None
                    else:
                        frame = self.latest_frame_for_recognition
                        self.latest_frame_for_recognition = None

                if frame is None:
                    self.recognition_stop_event.wait(0.2)
                    continue

                face_results = (
                    self._run_recognition_subprocess(frame)
                    if self.use_subprocess_recognition
                    else self._process_recognition_frame(frame)
                )
                if self.recognition_stop_event.is_set() or not self.is_running:
                    continue

                with self.recognition_lock:
                    self.cached_face_results = face_results
                    self.last_recognition_time = time.time()

                self.recognition_stop_event.wait(self.recognition_interval)
            except Exception:
                if self.recognition_stop_event.is_set() or not self.is_running:
                    break

                self.last_status = "Lỗi nhận diện. Camera vẫn đang chạy."
                self.current_alert_level = "error"
                self.current_alert_text = "Lỗi nhận diện. Hệ thống sẽ thử lại."
                traceback.print_exc()
                self.recognition_stop_event.wait(self.recognition_interval)

    def _run_recognition_subprocess(self, frame):
        self.runtime_dir.mkdir(exist_ok=True)
        frame_path = self.runtime_dir / "recognition_frame.jpg"
        if not cv2.imwrite(str(frame_path), frame):
            return []

        completed = subprocess.run(
            [
                sys.executable,
                str(self.recognition_worker_path),
                str(frame_path),
                str(self.dataset_dir),
                str(self.processing_scale),
                str(self.face_match_tolerance),
            ],
            capture_output=True,
            text=True,
            timeout=self.recognition_timeout,
        )

        if self.recognition_stop_event.is_set() or not self.is_running:
            return []

        if completed.returncode != 0:
            if self.recognition_stop_event.is_set() or not self.is_running:
                return []

            self.last_status = "Lỗi nhận diện. Camera vẫn đang chạy."
            self.current_alert_level = "monitoring"
            self.current_alert_text = "Lỗi nhận diện. Hệ thống sẽ thử lại."
            if completed.stderr:
                print(completed.stderr)
            return []

        payload = json.loads(completed.stdout or "[]")
        return [
            (tuple(item["location"]), item["label"], item["status"])
            for item in payload
        ]

    def _process_recognition_frame(self, frame):
        small_frame = cv2.resize(
            frame,
            (0, 0),
            fx=self.processing_scale,
            fy=self.processing_scale,
        )
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        scale_back = int(round(1 / self.processing_scale))
        results = []

        for face_location, face_encoding in zip(face_locations, face_encodings):
            top, right, bottom, left = face_location
            label = "CẢNH BÁO: NGƯỜI LẠ"
            status = "unknown"

            if self.known_encodings:
                matches = face_recognition.compare_faces(
                    self.known_encodings,
                    face_encoding,
                    tolerance=self.face_match_tolerance,
                )
                distances = face_recognition.face_distance(
                    self.known_encodings,
                    face_encoding,
                )
                best_match_index = int(np.argmin(distances))
                if matches[best_match_index]:
                    label = self.known_names[best_match_index]
                    status = "known"

            results.append(
                (
                    (
                        top * scale_back,
                        right * scale_back,
                        bottom * scale_back,
                        left * scale_back,
                    ),
                    label,
                    status,
                )
            )

        return results

    def _annotate_frame_with_recognition(self, frame):
        return self._annotate_frame(frame)

    def _draw_overlay(self, frame, face_data):
        recognized_names = []
        unknown_count = 0

        for (top, right, bottom, left), label, status in face_data:
            if status == "known":
                color = (0, 180, 0)
                recognized_names.append(label)
            else:
                color = (0, 0, 255)
                unknown_count += 1

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            frame = self._put_text_vietnamese(
                frame,
                label,
                (left + 6, bottom - 10),
                font_size=20,
                color=(255, 255, 255)
            )

        self.last_recognized_names = recognized_names
        self.last_unknown_count = unknown_count
        self._update_alert_state(face_data, recognized_names, unknown_count)

        return frame

    def _update_alert_state(self, face_data, recognized_names, unknown_count):
        if not self.is_running:
            self.current_alert_level = "idle"
            self.current_alert_text = "Camera chưa hoạt động."
            return

        if unknown_count > 0:
            self.current_alert_level = "danger"
            self.current_alert_text = (
                f"Phát hiện {unknown_count} người lạ trong vùng giám sát."
            )
            return

        if recognized_names:
            unique_names = sorted(set(recognized_names))
            self.current_alert_level = "safe"
            self.current_alert_text = (
                "An toàn. Đã nhận diện: " + ", ".join(unique_names)
            )
            return

        if face_data:
            self.current_alert_level = "monitoring"
            self.current_alert_text = "Đang phân tích khuôn mặt."
            return

        self.current_alert_level = "monitoring"
        self.current_alert_text = "Không phát hiện khuôn mặt nào trong hình."

    def get_status_payload(self):
        unique_known_names = sorted(set(self.known_names))
        with self.recognition_lock:
            recognition_count = len(self.cached_face_results)
            last_recognition_age = (
                None
                if self.last_recognition_time == 0
                else round(time.time() - self.last_recognition_time, 1)
            )
        return {
            "camera_running": self.is_running,
            "camera_index": self.current_camera_index,
            "fps": round(self.current_fps, 1),
            "alert_level": self.current_alert_level,
            "alert_text": self.current_alert_text,
            "recognized_names": self.last_recognized_names,
            "unknown_count": self.last_unknown_count,
            "status_text": self.last_status,
            "known_faces": len(unique_known_names),
            "known_names": unique_known_names,
            "recognition_count": recognition_count,
            "last_recognition_age": last_recognition_age,
        }

    def generate_frames(self):
        while True:
            try:
                with self.capture_lock:
                    capture = self.capture

                if capture is None or not self.is_running:
                    placeholder = np.zeros((480, 800, 3), dtype=np.uint8)
                    placeholder = self._put_text_vietnamese(
                        placeholder,
                        "Nhập camera index và bấm Mở Camera",
                        (40, 220),
                        font_size=28,
                        color=(255, 255, 255)
                    )
                    placeholder = self._put_text_vietnamese(
                        placeholder,
                        self.last_status,
                        (40, 270),
                        font_size=24,
                        color=(0, 255, 255)
                    )
                    ok, buffer = cv2.imencode(".jpg", placeholder)
                    if ok:
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                        )
                    time.sleep(0.1)
                    continue

                with self.capture_lock:
                    capture = self.capture
                    if capture is None or not self.is_running:
                        continue
                    success, frame = capture.read()

                if not success:
                    self.last_status = (
                        f"Mất kết nối camera index {self.current_camera_index}."
                    )
                    self.current_alert_level = "error"
                    self.current_alert_text = "Mất kết nối camera."
                    self.stop_camera()
                    continue

                frame = cv2.flip(frame, 1)

                with self.recognition_lock:
                    now = time.time()
                    if (
                        now - self.last_recognition_snapshot_time
                        >= self.recognition_interval
                    ):
                        self.latest_frame_for_recognition = frame.copy()
                        self.last_recognition_snapshot_time = now

                frame = self._annotate_frame(frame)
                ok, buffer = cv2.imencode(".jpg", frame)
                if not ok:
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )
            except Exception:
                self.last_status = "Lỗi stream camera. Hệ thống vẫn đang chạy."
                self.current_alert_level = "error"
                self.current_alert_text = "Lỗi stream camera. Hệ thống đang giữ kết nối."
                traceback.print_exc()
                time.sleep(0.2)
