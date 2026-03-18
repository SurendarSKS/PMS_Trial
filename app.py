import os
import re
import io
import cv2
import time
import base64
import warnings
import datetime
import urllib.request
import numpy as np
import pandas as pd

from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
import easyocr
import torch

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
CAPTURE_DIR = os.path.join(DATA_DIR, "captures")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CAPTURE_DIR, exist_ok=True)

PLATE_MODEL_PATH = os.path.join(MODEL_DIR, "plate_detector.pt")
SPREADSHEET_FILE = os.path.join(BASE_DIR, "plates.xlsx")
RESULTS_FILE = os.path.join(BASE_DIR, "capture_results.csv")

if not os.path.exists(RESULTS_FILE):
    pd.DataFrame(columns=[
        "username", "timestamp", "image_name",
        "car_number", "car_confidence",
        "slot_number", "slot_confidence"
    ]).to_csv(RESULTS_FILE, index=False)

# ═══════════════════════════════════════════
# DOWNLOAD MODEL IF MISSING
# ═══════════════════════════════════════════
if not os.path.exists(PLATE_MODEL_PATH):
    print("⏳ Downloading YOLO plate model...")
    urllib.request.urlretrieve(
        "https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8/raw/main/license_plate_detector.pt",
        PLATE_MODEL_PATH
    )

# ═══════════════════════════════════════════
# LOAD MODELS
# ═══════════════════════════════════════════
print("⏳ Loading YOLO...")
plate_model = YOLO(PLATE_MODEL_PATH)
print("✅ YOLO loaded!")

print("⏳ Loading EasyOCR...")
HAS_GPU = torch.cuda.is_available()
reader = easyocr.Reader(['en'], gpu=HAS_GPU)
print(f"✅ EasyOCR ({'GPU' if HAS_GPU else 'CPU'})")

# ═══════════════════════════════════════════
# GLOBALS
# ═══════════════════════════════════════════
PLATE_DB = set()
PLATE_DB_LIST = []
PLATE_FULL_DATA = {}
LAST_PLATE_BBOX = None
CURRENT_USERNAME = "default_user"

VALID_STATES = {
    'AN','AP','AR','AS','BR','CG','CH','DD','DL','GA',
    'GJ','HP','HR','JH','JK','KA','KL','LA','LD','MH',
    'ML','MN','MP','MZ','NL','OD','PB','PY','RJ','SK',
    'TN','TR','TS','UK','UP','WB'
}
TO_LETTER = {'0':'O','1':'I','2':'Z','3':'E','4':'A','5':'S','6':'G','7':'T','8':'B','9':'P'}
TO_DIGIT = {'O':'0','Q':'0','D':'0','I':'1','L':'1','Z':'2','E':'3','A':'4','S':'5','G':'6','B':'8','P':'9','R':'9','C':'0','U':'0','J':'1','Y':'7','F':'7'}

# ═══════════════════════════════════════════
# SPREADSHEET
# ═══════════════════════════════════════════
def load_plates_from_file(filename=SPREADSHEET_FILE):
    global PLATE_DB, PLATE_DB_LIST, PLATE_FULL_DATA

    if not os.path.exists(filename):
        print("⚠️ Spreadsheet not found.")
        PLATE_DB, PLATE_DB_LIST, PLATE_FULL_DATA = set(), [], {}
        return 0

    if filename.endswith(".csv"):
        df = pd.read_csv(filename)
    else:
        df = pd.read_excel(filename)

    col_name = df.columns[0]

    PLATE_DB = set()
    PLATE_DB_LIST = []
    PLATE_FULL_DATA = {}

    for _, row in df.iterrows():
        val = row[col_name]
        if pd.isna(val):
            continue
        cleaned = re.sub(r'[^A-Za-z0-9]', '', str(val).upper())
        if len(cleaned) >= 7:
            PLATE_DB.add(cleaned)
            PLATE_DB_LIST.append(cleaned)
            PLATE_FULL_DATA[cleaned] = row.to_dict()

    print(f"✅ Loaded {len(PLATE_DB)} plates")
    return len(PLATE_DB)

def search_plate_in_sheet(plate_number):
    cleaned = re.sub(r'[^A-Za-z0-9]', '', str(plate_number).upper())
    if cleaned in PLATE_FULL_DATA:
        return True, PLATE_FULL_DATA[cleaned]
    return False, None

load_plates_from_file()

# ═══════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════
def clean_plate_text(t):
    return re.sub(r'[^A-Za-z0-9]', '', str(t).upper())

def validate_plate(t):
    if len(t) == 10:
        return t[0:2].isalpha() and t[2:4].isdigit() and t[4:6].isalpha() and t[6:10].isdigit()
    elif len(t) == 9:
        return t[0:2].isalpha() and t[2:4].isdigit() and t[4].isalpha() and t[5:9].isdigit()
    return False

def format_plate(t):
    if len(t) == 10 and validate_plate(t):
        return f"{t[0:2]} {t[2:4]} {t[4:6]} {t[6:10]}"
    elif len(t) == 9 and validate_plate(t):
        return f"{t[0:2]} {t[2:4]} {t[4]} {t[5:9]}"
    return t

def fix_plate(raw):
    t = clean_plate_text(raw)
    if len(t) < 7:
        return t, 0.0, 99

    # strict formats
    formats = ['LLDDLLDDDD', 'LLDDLDDDD']
    best = t
    best_score = 0.0
    best_corr = 99

    for fmt in formats:
        fl = len(fmt)
        if len(t) < fl:
            continue

        ch = list(t[:fl])
        corr = 0
        sc = 0.0

        for i, e in enumerate(fmt):
            c = ch[i]
            if e == 'L':
                if c.isalpha():
                    sc += 1.0
                elif c in TO_LETTER:
                    ch[i] = TO_LETTER[c]
                    sc += 0.35
                    corr += 1
            else:
                if c.isdigit():
                    sc += 1.0
                elif c in TO_DIGIT:
                    ch[i] = TO_DIGIT[c]
                    sc += 0.35
                    corr += 1

        out = ''.join(ch)
        n = sc / fl
        if validate_plate(out):
            n += 0.20

        if n > best_score or (n == best_score and corr < best_corr):
            best = out
            best_score = n
            best_corr = corr

    return best, best_score, best_corr

# ═══════════════════════════════════════════
# PLATE DETECTOR
# ═══════════════════════════════════════════
def find_plate_yolo(image, model):
    results = model(image, verbose=False)
    cands = []
    h, w = image.shape[:2]

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < 0.3:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1=max(0,x1); y1=max(0,y1); x2=min(w,x2); y2=min(h,y2)
            bw=x2-x1; bh=y2-y1
            if bw<20 or bh<10:
                continue

            px=max(15,int(bw*0.08))
            py=max(10,int(bh*0.15))
            cands.append({
                "x1": max(0, x1-px),
                "y1": max(0, y1-py),
                "x2": min(w, x2+px),
                "y2": min(h, y2+py),
                "score": conf*20,
                "method": f"yolo({conf:.0%})",
                "yolo_conf": conf
            })
    return cands

def preprocess_plate(plate_img):
    h,w=plate_img.shape[:2]
    if h < 5 or w < 10:
        return {}

    th = 400
    s = th/h
    nw = int(w*s)
    big = cv2.resize(plate_img, (nw, th), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    sharp = clahe.apply(gray)
    blur = cv2.GaussianBlur(sharp,(0,0),2)
    sharp = cv2.addWeighted(sharp,2.3,blur,-1.3,0)

    smooth = cv2.bilateralFilter(gray,9,75,75)
    _, otsu = cv2.threshold(smooth,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return {
        "raw_crop": big,
        "sharp_gray": sharp,
        "otsu_bilateral": otsu
    }

def ocr_easy(img, label):
    try:
        if len(img.shape)==2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        results = reader.readtext(
            img,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            paragraph=False,
            min_size=5,
            text_threshold=0.35,
            low_text=0.2,
            width_ths=0.5,
            decoder='greedy'
        )
        out = []
        for (_, t, c) in results:
            cleaned = clean_plate_text(t)
            if len(cleaned) >= 2:
                out.append((cleaned, c, f'easy_{label}'))
        return out
    except:
        return []

def pick_best_ocr(ocr_reads):
    if not ocr_reads:
        return "", 0.0, []

    candidates = []
    for raw, conf, source in ocr_reads:
        fixed, fmt_score, corr = fix_plate(raw)
        raw_valid = validate_plate(raw)
        chosen = raw if raw_valid else fixed
        valid = validate_plate(chosen)

        score = (
            conf * 0.40 +
            fmt_score * 0.20 +
            (0.30 if valid else 0.0) +
            (0.10 if source.endswith("raw_crop") else 0.0)
        )
        candidates.append((chosen, score, conf, source, raw))

    valid_candidates = [c for c in candidates if validate_plate(c[0])]
    if valid_candidates:
        candidates = valid_candidates

    candidates.sort(key=lambda x: x[1], reverse=True)
    best = candidates[0]
    return best[0], min(best[1], 1.0), ocr_reads

def detect_plate(image_path=None, image_array=None):
    global LAST_PLATE_BBOX

    image = image_array.copy() if image_array is not None else cv2.imread(image_path)
    if image is None:
        return None

    original = image.copy()
    if image.shape[1] > 1024:
        s = 1024 / image.shape[1]
        image = cv2.resize(image, None, fx=s, fy=s)

    yolo_cands = find_plate_yolo(image, plate_model)
    if not yolo_cands:
        return "", 0.0, original, ""

    yolo_cands.sort(key=lambda x: x["score"], reverse=True)
    best = yolo_cands[0]
    LAST_PLATE_BBOX = (best["x1"], best["y1"], best["x2"], best["y2"])

    crop = image[best["y1"]:best["y2"], best["x1"]:best["x2"]].copy()
    variants = preprocess_plate(crop)

    ocr_reads = []
    if "raw_crop" in variants:
        ocr_reads.extend(ocr_easy(variants["raw_crop"], "raw_crop"))
    if "sharp_gray" in variants:
        ocr_reads.extend(ocr_easy(variants["sharp_gray"], "sharp_gray"))
    if "otsu_bilateral" in variants:
        ocr_reads.extend(ocr_easy(variants["otsu_bilateral"], "otsu_bilateral"))

    ocr_plate, ocr_conf, all_combined = pick_best_ocr(ocr_reads)

    final_plate = ocr_plate
    final_conf = ocr_conf

    # spreadsheet correction
    if PLATE_DB and len(PLATE_DB) > 0 and ocr_plate:
        cleaned = clean_plate_text(ocr_plate)
        if cleaned in PLATE_DB:
            final_plate = cleaned
            final_conf = max(final_conf, 0.95)

    formatted = format_plate(final_plate) if final_plate else "Not found"

    annotated = original.copy()
    cv2.rectangle(annotated, (best["x1"], best["y1"]), (best["x2"], best["y2"]), (0,255,0), 3)
    cv2.putText(
        annotated, formatted,
        (best["x1"], max(25, best["y1"]-10)),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2
    )

    return formatted, final_conf, annotated, final_plate

# ═══════════════════════════════════════════
# SLOT DETECTOR
# ═══════════════════════════════════════════
reader_slot = easyocr.Reader(['en'], gpu=False)

def clean_slot_text(t):
    if not t:
        return ""
    t = str(t).upper().strip()
    t = t.replace('S', '5').replace('O', '0')
    match = re.search(r'([A-Z0-9])\s*[-]?\s*(\d{1,2})', t)
    if match:
        char_part = match.group(1)
        num_part = match.group(2)
        if char_part in ['6', '0']:
            char_part = 'G'
        return f"{char_part}{num_part}"
    return ""

def get_slot_roi(image, plate_bbox=None):
    h, w = image.shape[:2]
    if plate_bbox is not None:
        x1, y1, x2, y2 = plate_bbox
        pw, ph = (x2 - x1), (y2 - y1)
        rx1 = max(0, x1 - int(pw * 1.5))
        rx2 = min(w, x2 + int(pw * 1.5))
        ry1 = min(h, y2 + int(ph * 1.0))
        ry2 = min(h, y2 + int(ph * 9.0))
        return image[ry1:ry2, rx1:rx2].copy()
    return image[int(h*0.7):h, int(w*0.1):int(w*0.9)].copy()

def detect_floor_slot(image_path=None, image_array=None, show=False, plate_bbox=None):
    img = image_array.copy() if image_array is not None else cv2.imread(image_path)
    if img is None:
        return "", 0.0, None

    roi = get_slot_roi(img, plate_bbox=plate_bbox if plate_bbox is not None else LAST_PLATE_BBOX)

    enhanced = cv2.resize(roi, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    b = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)
    enhanced_final = cv2.merge([l, a, b])
    enhanced_final = cv2.cvtColor(enhanced_final, cv2.COLOR_LAB2BGR)
    enhanced_final = cv2.GaussianBlur(enhanced_final, (3,3), 0)

    results = reader_slot.readtext(enhanced_final, detail=0, contrast_ths=0.1, low_text=0.3)

    best_slot = ""
    for res in results:
        cleaned = clean_slot_text(res)
        if len(cleaned) >= 2:
            best_slot = cleaned
            break

    conf = 1.0 if best_slot else 0.0
    return best_slot, conf, roi

# ═══════════════════════════════════════════
# STORAGE + WEB APP
# ═══════════════════════════════════════════

if not os.path.exists(RESULTS_FILE):
    pd.DataFrame(columns=[
        "username","timestamp","image_name","car_number","car_confidence","slot_number","slot_confidence"
    ]).to_csv(RESULTS_FILE, index=False)

CURRENT_USERNAME = "default_user"

def save_uploaded_file(file_storage, save_path):
    file_storage.save(save_path)

def save_base64_image(data_url, save_path):
    encoded = data_url.split(",")[1]
    binary = base64.b64decode(encoded)
    with open(save_path, "wb") as f:
        f.write(binary)

def image_to_data_url_bgr(img_bgr):
    _, buffer = cv2.imencode(".jpg", img_bgr)
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

def store_result(username, timestamp, image_name, car_number, car_conf, slot_number, slot_conf):
    row = pd.DataFrame([{
        "username": username,
        "timestamp": timestamp,
        "image_name": image_name,
        "car_number": car_number,
        "car_confidence": round(float(car_conf), 4),
        "slot_number": slot_number,
        "slot_confidence": round(float(slot_conf), 4)
    }])
    old_df = pd.read_csv(RESULTS_FILE)
    new_df = pd.concat([old_df, row], ignore_index=True)
    new_df.to_csv(RESULTS_FILE, index=False)

def run_detection_on_file(image_path):
    plate_result = detect_plate(image_path=image_path)
    if plate_result is not None:
        car_number = plate_result[0] if plate_result[0] else "Not found"
        car_conf = float(plate_result[1]) if plate_result[1] is not None else 0.0
        annotated = plate_result[2] if plate_result[2] is not None else cv2.imread(image_path)
    else:
        car_number = "Not found"
        car_conf = 0.0
        annotated = cv2.imread(image_path)

    slot_result = detect_floor_slot(image_path=image_path, show=False, plate_bbox=LAST_PLATE_BBOX)
    if slot_result is not None:
        slot_number = slot_result[0] if slot_result[0] else "Not found"
        slot_conf = float(slot_result[1]) if slot_result[1] is not None else 0.0
    else:
        slot_number = "Not found"
        slot_conf = 0.0

    return car_number, car_conf, slot_number, slot_conf, annotated

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/set_user", methods=["POST"])
def set_user():
    global CURRENT_USERNAME
    CURRENT_USERNAME = request.form.get("username", "default_user").strip() or "default_user"
    return redirect(url_for("home"))

@app.route("/camera", methods=["GET"])
def camera():
    return render_template("camera_fullscreen.html")

@app.route("/upload_capture", methods=["POST"])
def upload_capture():
    if "image" not in request.files:
        return "No image uploaded", 400

    file = request.files["image"]
    if file.filename == "":
        return "Empty filename", 400

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = f"{CURRENT_USERNAME}_{timestamp}_{file.filename}"
    save_path = os.path.join(CAPTURE_DIR, image_name)

    save_uploaded_file(file, save_path)

    car_number, car_conf, slot_number, slot_conf, annotated = run_detection_on_file(save_path)
    store_result(CURRENT_USERNAME, timestamp, image_name, car_number, car_conf, slot_number, slot_conf)

    image_url = image_to_data_url_bgr(annotated)

    return render_template(
        "result.html",
        image_url=image_url,
        username=CURRENT_USERNAME,
        timestamp=timestamp,
        car_number=car_number,
        car_confidence=f"{car_conf:.0%}",
        slot_number=slot_number,
        slot_confidence=f"{slot_conf:.0%}"
    )

@app.route("/capture_fullscreen", methods=["POST"])
def capture_fullscreen():
    image_data = request.form.get("image_data", None)
    if not image_data:
        return "No captured image data", 400

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = f"{CURRENT_USERNAME}_{timestamp}_camera.jpg"
    save_path = os.path.join(CAPTURE_DIR, image_name)

    save_base64_image(image_data, save_path)

    car_number, car_conf, slot_number, slot_conf, annotated = run_detection_on_file(save_path)
    store_result(CURRENT_USERNAME, timestamp, image_name, car_number, car_conf, slot_number, slot_conf)

    image_url = image_to_data_url_bgr(annotated)

    return render_template(
        "result.html",
        image_url=image_url,
        username=CURRENT_USERNAME,
        timestamp=timestamp,
        car_number=car_number,
        car_confidence=f"{car_conf:.0%}",
        slot_number=slot_number,
        slot_confidence=f"{slot_conf:.0%}"
    )

@app.route("/history", methods=["GET"])
def history():
    df = pd.read_csv(RESULTS_FILE)
    rows = df.sort_values("timestamp", ascending=False).to_dict(orient="records")
    return render_template("history.html", rows=rows)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
