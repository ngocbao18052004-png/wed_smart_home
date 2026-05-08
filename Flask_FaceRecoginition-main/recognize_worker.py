import json
import re
import sys
from pathlib import Path

import cv2
import face_recognition
import numpy as np


SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def normalize_name(file_stem):
    clean_name = re.sub(r"\d+", "", file_stem).strip("_- ")
    if not clean_name:
        return "Unknown"
    return clean_name.replace("_", " ").replace("-", " ").title()


def load_dataset(dataset_dir):
    known_names = []
    known_encodings = []

    for image_path in sorted(Path(dataset_dir).iterdir()):
        if image_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        image = face_recognition.load_image_file(str(image_path))
        encodings = face_recognition.face_encodings(image)
        if not encodings:
            continue

        known_encodings.append(encodings[0])
        known_names.append(normalize_name(image_path.stem))

    return known_names, known_encodings


def recognize(frame_path, dataset_dir, processing_scale, tolerance):
    known_names, known_encodings = load_dataset(dataset_dir)
    frame = cv2.imread(str(frame_path))
    if frame is None:
        return []

    small_frame = cv2.resize(
        frame,
        (0, 0),
        fx=processing_scale,
        fy=processing_scale,
    )
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    scale_back = int(round(1 / processing_scale))
    results = []

    for face_location, face_encoding in zip(face_locations, face_encodings):
        top, right, bottom, left = face_location
        label = "CANH BAO: NGUOI LA"
        status = "unknown"

        if known_encodings:
            matches = face_recognition.compare_faces(
                known_encodings, face_encoding, tolerance=tolerance
            )
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = int(np.argmin(distances))

            if matches[best_match_index]:
                label = known_names[best_match_index]
                status = "known"

        results.append(
            {
                "location": [
                    top * scale_back,
                    right * scale_back,
                    bottom * scale_back,
                    left * scale_back,
                ],
                "label": label,
                "status": status,
            }
        )

    return results


def main():
    frame_path = Path(sys.argv[1])
    dataset_dir = Path(sys.argv[2])
    processing_scale = float(sys.argv[3])
    tolerance = float(sys.argv[4])
    print(json.dumps(recognize(frame_path, dataset_dir, processing_scale, tolerance)))


if __name__ == "__main__":
    main()
