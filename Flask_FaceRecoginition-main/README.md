# Flask Face Recognition App

## Chuc nang

- Nhap `camera index` tren giao dien web.
- Bam `Mo Camera` de ket noi webcam theo index.
- Neu khuon mat trung voi anh trong `DATASET/Trainining` thi hien ten.
- Neu khong trung dataset thi hien `CANH BAO: KHUON MAT LA`.

## Cau truc

- `app.py`: Flask app va cac API mo/tat camera.
- `face_service.py`: Xu ly dataset, webcam, nhan dien khuon mat.
- `templates/index.html`: Giao dien nhap index va xem stream.
- `static/style.css`: Giao dien co ban.

## Cach chay

1. Cai dependencies:

```bash
pip install -r FLASK_FACE_APP/requirements.txt
```

2. Chay app:

```bash
python FLASK_FACE_APP/app.py
```

3. Mo trinh duyet:

```text
http://127.0.0.1:5000
```

## Luu y

- Dataset dang duoc doc tu `DATASET/Trainining`.
- Moi file anh trong dataset nen chi chua 1 khuon mat ro rang.
- Neu camera chinh khong mo duoc voi `0`, hay thu `1`, `2`, ...
