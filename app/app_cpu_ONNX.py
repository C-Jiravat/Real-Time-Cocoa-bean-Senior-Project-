import logging, time, cv2, numpy as np, torch, os, threading, shutil, sys
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
from flask import Flask, render_template, Response, jsonify, request
import timm
from torchvision import transforms
from werkzeug.utils import secure_filename
import torch.nn.functional as F

# Allow importing benchmark_engine from same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark_engine as bm

# ------------------------------------------------------------------------------
# 0) DEVICE SETUP & THREAD TUNING
# ------------------------------------------------------------------------------
import os as _os

device = torch.device('cpu')

# กำหนด Thread สำหรับ AI อย่างชาญฉลาด
# เหลือ 2 Core ให้ Flask I/O threads + OS ใช้
# เพราะเมื่อรัน Flask threaded=True จะมีหลาย Thread แย่งกัน
_cpu_count  = _os.cpu_count() or 4
_ai_threads = max(2, _cpu_count - 2)  # เช่น 8 core → ให้ AI 6, Flask 2
torch.set_num_threads(_ai_threads)

# บอก ONNX Runtime และ OpenMP Backend ให้ใช้ Thread เท่ากัน
_os.environ["OMP_NUM_THREADS"]        = str(_ai_threads)
_os.environ["MKL_NUM_THREADS"]        = str(_ai_threads)
_os.environ["OPENBLAS_NUM_THREADS"]   = str(_ai_threads)
_os.environ["VECLIB_MAXIMUM_THREADS"] = str(_ai_threads)

print(f"🚀 Processing Device   : CPU")
print(f"⚙️  Total CPU Cores     : {_cpu_count}")
print(f"🧵 AI Threads (ONNX)   : {_ai_threads}  (2 cores reserved for Flask)")
# ------------------------------------------------------------------------------

# 1) CONFIG 
# ------------------------------------------------------------------------------
logging.getLogger("ultralytics").setLevel(logging.ERROR)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER): os.makedirs(UPLOAD_FOLDER)

class_names_weight1 = {0: "เมล็ดสีม่วง", 1: "เมล็ดสีน้ำตาล"}
class_names_vit_defect = {0: "เมล็ดปกติ", 1: "เมล็ดงอก", 2: "เมล็ดสีเทาหินชนวน", 3: "เมล็ดขึ้นรา"}
class_names_vit_color = class_names_weight1.copy()

color_map_weight1 = { 0: (128, 0, 128), 1: (96, 164, 244) }
color_map_weight2 = { 0: (0, 255, 0), 1: (0, 255, 255), 2: (255, 0, 0), 3: (0, 0, 255) }

DESIRED_WIDTH, DESIRED_HEIGHT = 1280, 720

# ------------------------------------------------------------------------------
# 2) LOAD MODELS 
# ------------------------------------------------------------------------------
print("Loading Models...")

model_yolo_seed = None
model_convnext_color = None
model_convnext_defect = None


YOLO_SEED_PATH = r"D:\Chula\Senior_Project\yolo11n_run\weights\best.onnx"

try:
    model_yolo_seed = YOLO(YOLO_SEED_PATH)
except Exception as e: print(f"Error YOLO: {e}")

import onnxruntime as ort

VIT_COLOR_ONNX_PATH  = r"D:\Chula\Senior_Project\Phase2_batch128_color_best.onnx"
VIT_DEFECT_ONNX_PATH = r"D:\Chula\Senior_Project\Phase3_WD0.15_best.onnx"

try:
    sess_options = ort.SessionOptions()
    # ใช้ Thread เท่ากับ AI threads ที่กำหนด เพื่อให้สอดคล้องกับ environment
    sess_options.intra_op_num_threads = _ai_threads
    sess_options.inter_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
    model_convnext_color  = ort.InferenceSession(VIT_COLOR_ONNX_PATH,  sess_options=sess_options, providers=providers)
    model_convnext_defect = ort.InferenceSession(VIT_DEFECT_ONNX_PATH, sess_options=sess_options, providers=providers)
    active_provider = model_convnext_color.get_providers()[0]
    print(f"✅ ONNX Models Loaded | Provider: {active_provider} | Threads: {_ai_threads}")

    # --- Model Warm-up (Optimization #3) ---
    # ONNX Runtime JIT-compiles the graph on the FIRST inference call.
    def warm_up_model(session):
        input_name = session.get_inputs()[0].name
        input_type = session.get_inputs()[0].type
        dtype = np.float16 if "float16" in input_type else np.float32
        dummy = np.zeros((1, 3, 224, 224), dtype=dtype)
        session.run(None, {input_name: dummy})

    warm_up_model(model_convnext_color)
    warm_up_model(model_convnext_defect)
    print("✅ Model warm-up complete (dtype-aware)")
except Exception as e: print(f"Error ONNX: {e}")

# ------------------------------------------------------------------------------
# 3) UTILITIES  
# ------------------------------------------------------------------------------
def detect_objects_single_frame(model, frame, conf_thres=0.25, iou_thres=0.35):
    result = model.predict(source=frame, conf=conf_thres, iou=iou_thres, verbose=False, device=device)[0]
    detections = []
    for box in result.boxes:
        detections.append({
            "class_id": int(box.cls.item()),
            "confidence": float(box.conf.item()),
            "bbox": box.xyxy[0].tolist()
        })
    return detections

# ── Constants for Vectorized Normalize ──────────────────────────────────────
_NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # ImageNet mean
_NORM_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)  # ImageNet std

# Pre-compute FP16 constants for sessions that require it
_NORM_MEAN_FP16 = _NORM_MEAN.astype(np.float16)
_NORM_STD_FP16  = _NORM_STD.astype(np.float16)

def vit_predict_batch(session, crops, class_dict, batch_size=32, threshold=20.0):
    """
    [FIX #1 - Vectorized Preprocessing + FP16 Optimized]
    - threshold: minimum probability % to include a label (Multi-label support)
    """
    if not crops:
        return []

    all_results = []
    input_name = session.get_inputs()[0].name
    
    # Check FP16 requirement ONCE before loop to eliminate per-batch overhead
    is_fp16 = "float16" in session.get_inputs()[0].type

    for i in range(0, len(crops), batch_size):
        batch_crops = crops[i:i + batch_size]

        # Step A: resize all crops to (224,224) RGB—still uint8
        resized_list = [
            cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2RGB), (224, 224))
            for c in batch_crops
        ]

        if is_fp16:
            # FP16 path: normalize directly in float16 to avoid double allocation
            batch_nhwc = np.stack(resized_list).astype(np.float16) / np.float16(255.0)
            batch_nhwc = (batch_nhwc - _NORM_MEAN_FP16) / _NORM_STD_FP16
        else:
            # FP32 path: original optimized path
            batch_nhwc = np.stack(resized_list).astype(np.float32) / 255.0
            batch_nhwc = (batch_nhwc - _NORM_MEAN) / _NORM_STD

        # NHWC → NCHW, contiguous for ONNX — 1 transpose
        batch_tensor = np.ascontiguousarray(batch_nhwc.transpose(0, 3, 1, 2))

        logits = session.run(None, {input_name: batch_tensor})[0]
        logits_f32 = logits.astype(np.float32)

        # Numerically-stable softmax (in float32)
        exp_L = np.exp(logits_f32 - np.max(logits_f32, axis=1, keepdims=True))
        probs = exp_L / np.sum(exp_L, axis=1, keepdims=True)

        for j in range(len(batch_crops)):
            res = []
            # Check all classes against threshold
            for k in range(logits_f32.shape[1]):
                p_val = float(probs[j, k]) * 100
                if p_val >= threshold:
                    res.append((k, class_dict.get(k, str(k)), p_val))
            
            # Fallback: if none above threshold (rare), take the max
            if not res:
                max_idx = int(np.argmax(probs[j]))
                res.append((max_idx, class_dict.get(max_idx, str(max_idx)), float(np.max(probs[j])) * 100))
            
            all_results.append(res)

    return all_results

def draw_rect(img, bbox, color, thick=2, text=None):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
    if text:
        font_scale = 0.4; thickness = 1
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h + 5), color, -1)
        text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
        cv2.putText(img, text, (x1, y1 + h + 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)

def check_grade(c1, c2, c3):
    th = [("พิเศษ", 3, 3, 2.5), ("ชั้น 1", 3, 5, 3), ("ชั้น 2", 4, 8, 5)]
    if c1 > 4 or c2 > 8 or c3 > 5: return "ตกเกรด"
    for g, m1, m2, m3 in th:
        if c1 <= m1 and c2 <= m2 and c3 <= m3: return g
    return "ตกเกรด"

# ------------------------------------------------------------------------------
# 4) FLASK & STATE
# ------------------------------------------------------------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

latest_detection = {"live1": {}, "live2": {}, "grade": {}, "profiling": {}}
prediction_mode = False
current_model_mode = 'combo'

cap = None
source_type = None
static_frame = None
is_processing = False
processed_frame = None

# YOLO Confidence Threshold (Adjustable from UI)
yolo_conf_threshold = 0.5

# [FIX #3 - Thread-safe Lock]
# ป้องกัน Race Condition ระหว่าง AI background thread กับ Flask request threads
# ที่ต่างกันอ่าน/เขียน latest_detection พร้อมกัน
_state_lock = threading.Lock()

def init_source():
    global cap, source_type, static_frame
    if cap is not None: cap.release()
    if source_type == 'webcam':
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened(): cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    elif source_type == 'image':
        cap = None

# ... [KEEP YOUR EXISTING ROUTES: /, /use_webcam, /upload_image, /toggle_prediction, /exit_prediction, /set_model_mode] ...
# (ละ route เดิมไว้เพื่อให้โค้ดกระชับ คุณสามารถใช้ของเดิมในส่วนนี้ได้เลยครับ)

@app.route('/')
def index(): return render_template('index_video.html')

@app.route('/use_webcam', methods=['POST'])
def use_webcam():
    global source_type, prediction_mode
    source_type = 'webcam'; prediction_mode = False
    init_source()
    return jsonify({"status": "success"})

@app.route('/upload_image', methods=['POST'])
def upload_image():
    global source_type, static_frame, prediction_mode
    file = request.files.get('file')
    if file and file.filename != '':
        try:
            filename = secure_filename(file.filename)
            if not filename: filename = f"img_{int(time.time())}.jpg"
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if img is not None:
                static_frame = cv2.resize(img, (DESIRED_WIDTH, DESIRED_HEIGHT))
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                cv2.imwrite(path, static_frame) 
                
                source_type = 'image'; prediction_mode = False
                latest_detection["grade"] = {}
                latest_detection["live2"] = {}
                latest_detection["profiling"] = {}
                processed_frame = None; is_processing = False
                init_source()
                return jsonify({"status": "success", "file": filename})
            else:
                return jsonify({"status": "error", "message": "ไม่สามารถแปลงไฟล์รูปภาพได้"})
        except Exception as e:
            return jsonify({"status": "error", "message": f"Server Error: {str(e)}"})
    return jsonify({"status": "error", "message": "ไม่พบไฟล์ที่ส่งมา"})

@app.route('/toggle_prediction', methods=['POST'])
def toggle_prediction():
    global prediction_mode, is_processing, processed_frame
    if source_type is None: return jsonify({"status": "error", "message": "No source"})
    prediction_mode = True; is_processing = False; processed_frame = None
    return jsonify({"status": "active"})

@app.route('/exit_prediction', methods=['POST'])
def exit_prediction():
    global prediction_mode, is_processing, processed_frame
    prediction_mode = False; is_processing = False; processed_frame = None
    return jsonify({"status": "live mode resumed"})

@app.route('/set_model_mode', methods=['POST'])
def set_model_mode():
    global current_model_mode
    current_model_mode = request.json.get('mode', 'combo')
    return jsonify({"status": "success", "mode": current_model_mode})

@app.route('/set_yolo_conf', methods=['POST'])
def set_yolo_conf():
    global yolo_conf_threshold
    val = float(request.json.get('value', 0.25))
    yolo_conf_threshold = max(0.01, min(0.99, val))
    return jsonify({"status": "success", "value": yolo_conf_threshold})

@app.route('/detection_data')
def detection_data():
    # [FIX #3] ใช้ Lock เวลาอ่านเพื่อป้องกัน partial-write จาก AI thread
    with _state_lock:
        data = {k: (v.copy() if isinstance(v, dict) else v)
                for k, v in latest_detection.items()}
    data["mode"] = "Prediction" if prediction_mode else "Live"
    return jsonify(data)

# ------------------------------------------------------------------------------
# 7) BENCHMARK ROUTES
# ------------------------------------------------------------------------------
@app.route('/benchmark')
def benchmark_page():
    return render_template('benchmark.html')


@app.route('/run_batch_benchmark', methods=['POST'])
def run_batch_benchmark_route():
    global model_yolo_seed, model_convnext_color, model_convnext_defect
    """
    Accepts a ZIP file (images/ + labels/), runs batch inference, returns JSON metrics.
    Form fields:
        zip_file     : the ZIP file
        task         : 'color' | 'defect'
        model_mode   : 'yolo' | 'combo'
        iou_thresh   : float (default 0.5)
    """
    task       = request.form.get('task', 'color')
    model_mode = request.form.get('model_mode', 'combo')
    iou_thresh = float(request.form.get('iou_thresh', 0.5))
    conf_thres = float(request.form.get('conf_thres', yolo_conf_threshold))

    zip_file = request.files.get('zip_file')
    if not zip_file or zip_file.filename == '':
        return jsonify({"error": "ไม่พบไฟล์ ZIP"}), 400

    if model_yolo_seed is None or model_convnext_color is None or model_convnext_defect is None:
        return jsonify({"error": "โมเดลยังไม่ถูกโหลดหรือโหลดไม่สำเร็จ กรุณาตรวจสอบ Console ของ Server"}), 500

    tmp_dir = None
    try:
        zip_bytes = zip_file.read()
        tmp_dir, images_dir, labels_dir = bm.extract_zip_to_temp(zip_bytes)

        res_viz_session_id = f"batch_{int(time.time())}"
        viz_dir = os.path.join(app.root_path, 'static', 'benchmark_results', res_viz_session_id)
        os.makedirs(viz_dir, exist_ok=True)

        result = bm.run_batch_benchmark(
            images_dir           = images_dir,
            labels_dir           = labels_dir,
            task                 = task,
            model_mode           = 'combo',
            model_yolo_seed      = model_yolo_seed,
            model_convnext_color  = model_convnext_color,
            model_convnext_defect = model_convnext_defect,
            detect_fn            = detect_objects_single_frame,
            batch_predict_fn     = vit_predict_batch,
            iou_thresh           = iou_thresh,
            conf_thres           = conf_thres,
            save_annotated_dir   = viz_dir
        )
        result['viz_session_id'] = res_viz_session_id
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


@app.route('/run_single_benchmark', methods=['POST'])
def run_single_benchmark_route():
    """
    Accepts one image + label file(s), runs inference on that image, returns JSON metrics.
    Form fields:
        image_file        : image
        color_label_file  : .txt label for color task (optional)
        defect_label_file : .txt label for defect task (optional)
        task              : 'color' | 'defect' | 'both'
        model_mode        : 'yolo' | 'combo'
        iou_thresh        : float (default 0.5)
    """
    task       = request.form.get('task', 'both')
    model_mode = request.form.get('model_mode', 'combo')
    iou_thresh = float(request.form.get('iou_thresh', 0.5))
    conf_thres = float(request.form.get('conf_thres', yolo_conf_threshold))

    image_file = request.files.get('image_file')
    if not image_file or image_file.filename == '':
        return jsonify({"error": "ไม่พบไฟล์ภาพ"}), 400

    if model_yolo_seed is None or model_convnext_color is None or model_convnext_defect is None:
        return jsonify({"error": "โมเดลยังไม่ถูกโหลดหรือโหลดไม่สำเร็จ กรุณาตรวจสอบ Console ของ Server"}), 500

    tmp_dir = None
    try:
        tmp_dir = __import__('tempfile').mkdtemp(prefix='cocoa_single_')

        # Decode image
        t_load_start = time.time()
        img_bytes = np.frombuffer(image_file.read(), np.uint8)
        frame = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        data_loader_sec = time.time() - t_load_start
        if frame is None:
            return jsonify({"error": "ไม่สามารถอ่านไฟล์ภาพได้"}), 400

        # Save label files to temp
        color_label_path = defect_label_path = None
        color_lf  = request.files.get('color_label_file')
        defect_lf = request.files.get('defect_label_file')

        if color_lf and color_lf.filename != '':
            color_label_path = os.path.join(tmp_dir, 'color.txt')
            color_lf.save(color_label_path)

        if defect_lf and defect_lf.filename != '':
            defect_label_path = os.path.join(tmp_dir, 'defect.txt')
            defect_lf.save(defect_label_path)

        result = bm.run_single_image_benchmark(
            frame                = frame,
            color_label_path     = color_label_path,
            defect_label_path    = defect_label_path,
            task                 = task,
            model_mode           = 'combo',
            model_yolo_seed      = model_yolo_seed,
            model_convnext_color  = model_convnext_color,
            model_convnext_defect = model_convnext_defect,
            detect_fn              = detect_objects_single_frame,
            batch_predict_fn       = vit_predict_batch,
            iou_thresh             = iou_thresh,
            conf_thres             = conf_thres,
            data_loader_sec        = data_loader_sec,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ------------------------------------------------------------------------------
# 6) MAIN LOGIC (เพิ่มการจับเวลาใน Pipeline)
# ------------------------------------------------------------------------------
def process_ai_background(frame_big, mode):
    global is_processing, processed_frame, latest_detection
    global model_yolo_seed, model_convnext_color, model_convnext_defect

    try:
        t_start_total = time.time()

        # --- 1. Initialization & Prep ---
        t0 = time.time()
        frame1, cnt_color  = frame_big.copy(), {}
        frame2, cnt_defect = frame_big.copy(), {}
        t_prep = time.time() - t0

        t_yolo = t_crop = t_vit = t_logic = t_render = 0.0
        combo = np.hstack((frame1, frame2))  # fallback

        # COMBO MODE (Only mode)
        # --- 2. YOLO Detection (locate beans only) ---
        t1 = time.time()
        dets = detect_objects_single_frame(model_yolo_seed, frame_big, conf_thres=yolo_conf_threshold)
        t_yolo = time.time() - t1

        # --- 3. Image Cropping (NumPy slice, near-free) ---
        t2 = time.time()
        valid_indices, crops_list = [], []
        for i, d in enumerate(dets):
            x1, y1, x2, y2 = map(int, d["bbox"])
            crop = frame_big[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue
            valid_indices.append(i)
            crops_list.append(crop)
        t_crop = time.time() - t2

        # --- 4. ConvNeXt Batch Inference — PARALLEL (Optimization #1) ---
        # Color and Defect models are completely independent.
        # Run both simultaneously on separate threads to halve ConvNeXt time.
        t3 = time.time()
        with ThreadPoolExecutor(max_workers=2) as executor:
            fut_color  = executor.submit(vit_predict_batch, model_convnext_color,  crops_list, class_names_vit_color)
            fut_defect = executor.submit(vit_predict_batch, model_convnext_defect, crops_list, class_names_vit_defect)
            color_results  = fut_color.result()
            defect_results = fut_defect.result()
        t_vit = time.time() - t3

        # --- 5. Rendering: draw BBox + hstack (Combo mode) ---
        # NOTE: นับ cnt_color/cnt_defect ที่นี่ เพื่อใช้ใน Grading ขั้นต่อไป
        t_render_start = time.time()
        for j, vi in enumerate(valid_indices):
            # Process ALL predicted labels for this crop (Multi-label support)
            # Color model
            for idx_c, lbl_c, conf_c in color_results[j]:
                cnt_color[lbl_c] = cnt_color.get(lbl_c, 0) + 1
                # Only draw the first one if we want to avoid messy UI, or just stack them
                draw_rect(frame1, dets[vi]["bbox"], color_map_weight1.get(idx_c, (255,255,255)), 2, f"C:{int(conf_c)}%")

            # Defect model
            for idx_d, lbl_d, conf_d in defect_results[j]:
                cnt_defect[lbl_d] = cnt_defect.get(lbl_d, 0) + 1
                draw_rect(frame2, dets[vi]["bbox"], color_map_weight2.get(idx_d, (255,255,255)), 2, f"D:{int(conf_d)}%")
        
        combo = np.hstack((frame1, frame2))
        t_render = time.time() - t_render_start

        # --- 6. Grading Logic (pure arithmetic, separate from render) ---
        t_logic_start = time.time()
        total1 = sum(cnt_color.values())
        total2 = sum(cnt_defect.values())
        if total2 > 0:
            c1 = 100 * cnt_defect.get("เมล็ดขึ้นรา", 0) / total2
            c3 = 100 * cnt_defect.get("เมล็ดงอก",   0) / total2
            c2 = 100 * (cnt_color.get("เมล็ดสีม่วง", 0) +
                        cnt_defect.get("เมล็ดสีเทาหินชนวน", 0)) / total2
        else:
            c1, c2, c3 = 0.0, 0.0, 0.0
        grade = check_grade(c1, c2, c3)
        t_logic = time.time() - t_logic_start

        total_time = time.time() - t_start_total

        # --- 7. Profiling overlay on image ---
        overlay_y = 40
        cv2.putText(combo, "DETAILED PROFILING (CPU)",         (20, overlay_y),     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(combo, f"1. Prep:      {t_prep:.3f}s",     (20, overlay_y+30),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
        cv2.putText(combo, f"2. YOLO:      {t_yolo:.3f}s",     (20, overlay_y+55),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
        cv2.putText(combo, f"3. Crop:      {t_crop:.3f}s",     (20, overlay_y+80),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
        cv2.putText(combo, f"4. ConvNeXt:  {t_vit:.3f}s",     (20, overlay_y+105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
        cv2.putText(combo, f"5. Render:    {t_render:.3f}s",   (20, overlay_y+130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
        cv2.putText(combo, f"6. Logic:     {t_logic:.3f}s",    (20, overlay_y+155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
        cv2.putText(combo, f"TOTAL TIME:   {total_time:.3f}s", (20, overlay_y+190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),  2)

        # --- 8. Thread-safe state update [FIX #3] ---
        with _state_lock:
            latest_detection["profiling"] = {
                "prep_sec":   round(t_prep,      4),
                "yolo_sec":   round(t_yolo,      4),
                "crop_sec":   round(t_crop,      4),
                "vit_sec":    round(t_vit,       4),
                "render_sec": round(t_render,    4),
                "logic_sec":  round(t_logic,     4),
                "total_sec":  round(total_time,  4),
            }
            latest_detection["live1"]  = {"Total": total1, "Counts": cnt_color}
            latest_detection["live2"]  = {"Total": total2, "Counts": cnt_defect}
            latest_detection["grade"]  = {
                "c1": round(c1, 2), "c2": round(c2, 2),
                "c3": round(c3, 2), "grade": grade,
                "time_sec": round(total_time, 2)
            }

        processed_frame = combo

    except Exception as e:
        import traceback
        print(f"❌ Error in AI background: {e}")
        traceback.print_exc()
        processed_frame = np.hstack((frame_big, frame_big))
    finally:
        is_processing = False

# ... [KEEP YOUR EXISTING generate_frames, empty_frame, encode_frame, if __name__ == "__main__":] ...
# (ส่วนล่างสุดคงเดิมได้เลยครับ)

def generate_frames():
    global prediction_mode, latest_detection, cap, source_type, current_model_mode, static_frame
    global is_processing, processed_frame
    
    prev_time = time.time()

    while True:
        frame = None
        if source_type == 'webcam':
            if cap is None or not cap.isOpened():
                time.sleep(0.01); yield empty_frame(); continue
            ret, f = cap.read()
            if not ret: cap.release(); time.sleep(1); continue
            frame = cv2.rotate(f, cv2.ROTATE_180)
            frame = cv2.resize(frame, (DESIRED_WIDTH, DESIRED_HEIGHT))
        elif source_type == 'image':
            if static_frame is None:
                time.sleep(0.1); yield empty_frame(); continue
            frame = static_frame.copy()
        else:
            time.sleep(0.1); yield empty_frame("Please Select Source"); continue

        if not prediction_mode:
            # LIVE VIEW
            # [FIX #4] np.hstack creates a NEW array, frame.copy() inside is redundant
            # saves ~2.7MB allocation per frame at 30fps = ~81MB/sec of unnecessary GC pressure
            combo = np.hstack((frame, frame))
            mode_txt = "COMBO"
            cv2.putText(combo, f"Mode: {mode_txt} | Src: {source_type.upper()}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            yield encode_frame(combo)
            time.sleep(0.03)

        else:
            # PREDICTION MODE
            if not is_processing and processed_frame is None:
                # เริ่มให้ AI ทำงานเบื้องหลัง (ทำงานแค่ครั้งเดียว)
                is_processing = True
                latest_detection["grade"] = {"status": "processing"}
                frame_to_process = frame.copy()
                threading.Thread(target=process_ai_background, args=(frame_to_process, current_model_mode)).start()

            if is_processing:
                # ระหว่างรอ AI คิด ส่งภาพนี้ไปเลี้ยง Browser ไม่ให้หลุด
                temp = frame.copy()
                cv2.putText(temp, "AI is Analyzing... Please Wait", (DESIRED_WIDTH//2-200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 3)
                yield encode_frame(np.hstack((temp, temp.copy())))
                time.sleep(0.1) # ส่งภาพทุกๆ 0.1 วินาที
                
            elif processed_frame is not None:
                # เมื่อ AI คิดเสร็จแล้ว ให้ส่งภาพผลลัพธ์โชว์ค้างไว้
                yield encode_frame(processed_frame)
                time.sleep(0.1)
def empty_frame(text="Waiting for Source..."):
    blank = np.zeros((DESIRED_HEIGHT, DESIRED_WIDTH, 3), dtype=np.uint8)
    cv2.putText(blank, text, (100, DESIRED_HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return encode_frame(blank)

def encode_frame(img):
    ok, buf = cv2.imencode(".jpg", img)
    return (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    