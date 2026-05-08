#& "C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel --url http://localhost:5000
#(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& d:\HOC_TAP\DO_AN_TOT_NGHIEP\wed\Flask_FaceRecoginition-main\.venv\Scripts\Activate.ps1)
#py app.py

import base64
import binascii
import atexit
import signal
import sys
import traceback

from flask import Flask, Response, jsonify, render_template, request, send_from_directory
from werkzeug.exceptions import HTTPException

from face_service import FaceRecognitionCamera
from flask_cors import CORS # Thêm dòng này ở phần import phía trên


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
camera_service = FaceRecognitionCamera()
atexit.register(camera_service.cleanup_camera)


def cleanup_and_exit(signum, frame):
    camera_service.cleanup_camera()
    sys.exit(0)


signal.signal(signal.SIGINT, cleanup_and_exit)
signal.signal(signal.SIGTERM, cleanup_and_exit)


@app.route("/")
def index():
    camera_service.cleanup_camera()
    return render_template("dashboard.html")


@app.get("/favicon.ico")
def favicon():
    return send_from_directory(
        app.static_folder,
        "favicon.svg",
        mimetype="image/svg+xml",
    )


@app.post("/start_camera")
def start_camera():
    data = request.get_json(silent=True) or {}
    camera_index = data.get("camera_index", 0)

    try:
        camera_index = int(camera_index)
    except (TypeError, ValueError, binascii.Error):
        return jsonify({"ok": False, "message": "Camera index phải là số nguyên."}), 400

    ok, message = camera_service.start_camera(camera_index)
    status_code = 200 if ok else 400
    return jsonify({"ok": ok, "message": message}), status_code


@app.post("/stop_camera")
def stop_camera():
    camera_service.cleanup_camera()
    return jsonify({"ok": True, "message": "Đã tắt camera."})


@app.post("/register/upload")
def register_upload():
    person_name = (request.form.get("person_name") or "").strip()
    file = request.files.get("face_image")

    if file is None or not file.filename:
        return jsonify({"ok": False, "message": "Vui lòng chọn ảnh khuôn mặt để upload."}), 400

    ok, message, result_code = camera_service.register_face_from_upload(person_name, file.read())
    status_code = 200 if ok else 409 if result_code == "duplicate" else 400
    return jsonify({"ok": ok, "message": message, "code": result_code}), status_code


@app.post("/register/frame")
def register_frame():
    data = request.get_json(silent=True) or {}
    person_name = (data.get("person_name") or "").strip()
    image_data = data.get("image_data") or ""

    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(image_data, validate=True)
    except (TypeError, ValueError, binascii.Error):
        return jsonify({"ok": False, "message": "Dữ liệu ảnh quét không hợp lệ."}), 400

    ok, message, result_code = camera_service.register_face_from_upload(person_name, image_bytes)
    status_code = 200 if ok else 409 if result_code == "duplicate" else 400
    return jsonify({"ok": ok, "message": message, "code": result_code}), status_code


@app.route("/data/delete", methods=["POST"])
def delete_data():
    data = request.get_json(silent=True) or {}
    person_name = (data.get("person_name") or "").strip()
    if not person_name:
        return jsonify({"ok": False, "message": "Tên không hợp lệ."}), 400
    
    ok, msg = camera_service.delete_face(person_name)
    return jsonify({"ok": ok, "message": msg}), 200 if ok else 404


@app.route("/data/rename", methods=["POST"])
def rename_data():
    data = request.get_json(silent=True) or {}
    old_name = (data.get("old_name") or "").strip()
    new_name = (data.get("new_name") or "").strip()
    if not old_name or not new_name:
        return jsonify({"ok": False, "message": "Tên không hợp lệ."}), 400
        
    ok, msg = camera_service.rename_face(old_name, new_name)
    return jsonify({"ok": ok, "message": msg}), 200 if ok else 400


@app.get("/video_feed")
def video_feed():
    return Response(
        camera_service.generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/health")
def health():
    payload = camera_service.get_status_payload()
    payload.update(
        {
            "ok": True,
            "dataset_dir": str(camera_service.dataset_dir),
        }
    )
    return jsonify(payload)


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    if isinstance(error, HTTPException):
        return error

    traceback.print_exc()
    return (
        jsonify(
            {
                "ok": False,
                "message": "Server gặp lỗi nội bộ. Vui lòng xem log Flask để kiểm tra chi tiết.",
            }
        ),
        500,
    )

# app_2.py
@app.after_request
def add_headers(response):
    # Cho phép mọi nguồn truy cập (Quan trọng cho GitHub Pages)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS, PUT, DELETE'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    # Cho phép nhúng vào iframe mà không bị chặn bảo mật
    response.headers['Content-Security-Policy'] = "frame-ancestors *" 
    return response


if __name__ == "__main__":
    print("Starting Flask server on http://127.0.0.1:5000")
    app.run(debug=False, host="0.0.0.0", port=5000, use_reloader=False, threaded=True)
