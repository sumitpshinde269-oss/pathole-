import io
import os
import csv
import time
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# -------------------------------
# CONFIGURATION
# -------------------------------

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "best.pt"
IMG_SIZE = 800
MAX_HISTORY = 50

st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🛣",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------
# STYLING
# -------------------------------

st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
    }
    .block-container { padding-top: 2rem; padding-bottom: 2.5rem; max-width: 1250px; }

    .rd-header { padding-bottom: 1rem; margin-bottom: 1.25rem;
        border-bottom: 1px solid rgba(120,130,150,0.22); }
    .rd-header h1 { font-size: 1.65rem; font-weight: 650; margin: 0 0 .25rem 0; }
    .rd-header p { margin: 0; font-size: .9rem; opacity: .72; }

    .rd-brand { padding: .2rem 0 1rem 0; }
    .rd-brand .name { font-size: 1rem; font-weight: 650; }
    .rd-brand .tag { font-size: .74rem; opacity: .6; text-transform: uppercase; letter-spacing: .08em; }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(120,130,150,0.22);
        border-radius: 10px;
        padding: .85rem 1rem;
        background: rgba(120,130,150,0.04);
    }
    div[data-testid="stMetricLabel"] p { font-size: .78rem !important; opacity: .75; }
    div[data-testid="stMetricValue"] { font-size: 1.55rem; font-weight: 650; }

    .rd-card-title { font-size: .82rem; font-weight: 650; text-transform: uppercase;
        letter-spacing: .07em; opacity: .62; margin-bottom: .6rem; }
    .rd-empty { border: 1px dashed rgba(120,130,150,0.35); border-radius: 10px;
        padding: 1.6rem; text-align: center; font-size: .9rem; opacity: .7; }
    .rd-kv { font-size: .86rem; padding: .22rem 0; }
    .rd-kv span { opacity: .66; }
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        padding: .42rem .55rem; border-radius: 7px; }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(120,130,150,0.10); }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------
# MODEL
# -------------------------------


@st.cache_resource(show_spinner=False)
def load_model():
    return YOLO(str(MODEL_PATH))


@st.cache_resource(show_spinner=False)
def detect_device():
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0", torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "cpu", "CPU"


# -------------------------------
# SESSION STATE
# -------------------------------


def init_state():
    ss = st.session_state
    ss.setdefault("nav", "Dashboard")
    ss.setdefault("history", [])
    ss.setdefault("image_result", None)
    ss.setdefault("video_result", None)
    ss.setdefault("run_id", 0)


def add_history(kind, source, detections, avg_conf, frames=None):
    ss = st.session_state
    ss.run_id += 1
    ss.history.insert(
        0,
        {
            "id": ss.run_id,
            "time": datetime.now().strftime("%H:%M:%S"),
            "type": kind,
            "source": source,
            "detections": int(detections),
            "avg_conf": float(avg_conf) if avg_conf is not None else 0.0,
            "frames": frames,
        },
    )
    ss.history = ss.history[:MAX_HISTORY]


def counts():
    hist = st.session_state.history
    images = sum(1 for h in hist if h["type"] == "Image")
    videos = sum(1 for h in hist if h["type"] == "Video")
    defects = sum(h["detections"] for h in hist)
    confs = [h["avg_conf"] for h in hist if h["detections"] > 0]
    avg = float(np.mean(confs)) if confs else 0.0
    return images, videos, defects, avg


# -------------------------------
# DETECTION HELPERS
# -------------------------------


def encode_jpeg(rgb_array, quality=92):
    bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else b""


def detection_rows(boxes, names):
    rows = []
    for i, box in enumerate(boxes):
        cls_id = int(box.cls.item())
        rows.append(
            {
                "Detection": i + 1,
                "Class": names.get(cls_id, str(cls_id)),
                "Confidence": "{:.2%}".format(box.conf.item()),
                "Box (x1, y1, x2, y2)": ", ".join(
                    str(int(v)) for v in box.xyxy[0].tolist()
                ),
            }
        )
    return rows


def run_image_detection(image, conf, iou, device):
    model = load_model()
    started = time.perf_counter()
    result = model.predict(
        source=image,
        conf=conf,
        iou=iou,
        imgsz=IMG_SIZE,
        device=device,
        verbose=False,
    )[0]
    elapsed = time.perf_counter() - started

    boxes = result.boxes
    ann_rgb = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
    confs = [float(c) for c in boxes.conf.tolist()] if len(boxes) else []

    return {
        "annotated": ann_rgb,
        "jpeg": encode_jpeg(ann_rgb),
        "count": len(boxes),
        "avg_conf": float(np.mean(confs)) if confs else 0.0,
        "rows": detection_rows(boxes, result.names),
        "elapsed": elapsed,
        "size": image.size,
        "conf": conf,
        "iou": iou,
    }


def make_writer(path_stem, fps, size):
    for tag in ("avc1", "mp4v"):
        path = path_stem + ".mp4"
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*tag), fps, size)
        if writer.isOpened():
            return writer, path, tag
        writer.release()
    return None, None, None


def run_video_detection(uploaded, conf, iou, device):
    model = load_model()

    suffix = Path(uploaded.name).suffix or ".mp4"
    tmp_dir = tempfile.mkdtemp(prefix="rd_video_")
    in_path = os.path.join(tmp_dir, "input" + suffix)
    with open(in_path, "wb") as fh:
        fh.write(uploaded.getbuffer())

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        return {"error": "Could not open the uploaded video file."}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    width -= width % 2
    height -= height % 2
    if width < 2 or height < 2:
        cap.release()
        return {"error": "Video has invalid dimensions."}

    writer, out_path, codec = make_writer(
        os.path.join(tmp_dir, "annotated"), fps, (width, height)
    )
    if writer is None:
        cap.release()
        return {"error": "No available video encoder for the output file."}

    progress = st.progress(0.0)
    status = st.empty()

    frame_count = 0
    total_detections = 0
    confs = []
    class_counts = {}
    samples = []
    started = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))

        result = model.predict(
            frame,
            conf=conf,
            iou=iou,
            imgsz=IMG_SIZE,
            device=device,
            verbose=False,
        )[0]

        annotated = result.plot()
        writer.write(annotated)

        boxes = result.boxes
        total_detections += len(boxes)
        for c in boxes.conf.tolist():
            confs.append(float(c))
        for c in boxes.cls.tolist():
            name = result.names.get(int(c), str(int(c)))
            class_counts[name] = class_counts.get(name, 0) + 1

        if frame_count < 6:
            samples.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

        frame_count += 1
        if total:
            progress.progress(min(frame_count / total, 1.0))
            status.caption("Processing frame {} of {}".format(frame_count, total))
        else:
            status.caption("Processing frame {}".format(frame_count))

    cap.release()
    writer.release()
    elapsed = time.perf_counter() - started
    progress.progress(1.0)
    status.caption("Processed {} frames in {:.1f}s".format(frame_count, elapsed))

    try:
        os.remove(in_path)
    except OSError:
        pass

    with open(out_path, "rb") as fh:
        video_bytes = fh.read()

    return {
        "video_bytes": video_bytes,
        "path": out_path,
        "frames": frame_count,
        "total": total_detections,
        "avg_conf": float(np.mean(confs)) if confs else 0.0,
        "class_counts": class_counts,
        "samples": samples,
        "fps": fps,
        "size": (width, height),
        "elapsed": elapsed,
        "codec": codec,
        "conf": conf,
        "iou": iou,
    }


# -------------------------------
# UI PIECES
# -------------------------------


def header(title, subtitle):
    st.markdown(
        '<div class="rd-header"><h1>{}</h1><p>{}</p></div>'.format(title, subtitle),
        unsafe_allow_html=True,
    )


def card_title(text):
    st.markdown('<div class="rd-card-title">' + text + '</div>', unsafe_allow_html=True)


def empty_state(text):
    st.markdown('<div class="rd-empty">' + text + '</div>', unsafe_allow_html=True)


def settings_row(key_prefix):
    col1, col2 = st.columns(2)
    with col1:
        conf = st.slider(
            "Confidence threshold", 0.10, 1.00, 0.40, 0.05, key=key_prefix + "_conf"
        )
    with col2:
        iou = st.slider(
            "NMS IoU threshold", 0.10, 0.90, 0.45, 0.05, key=key_prefix + "_iou"
        )
    return conf, iou


def results_block(result, filename, key_suffix):
    c1, c2, c3 = st.columns(3)
    c1.metric("Total detections", result["count"])
    c2.metric("Average confidence", "{:.2%}".format(result["avg_conf"]))
    c3.metric("Inference time", "{:.0f} ms".format(result["elapsed"] * 1000))

    if result["rows"]:
        card_title("Detections")
        st.dataframe(result["rows"], width="stretch", hide_index=True)
    else:
        st.info("No road damage detected above the confidence threshold.")

    st.download_button(
        "Download annotated image",
        data=result["jpeg"],
        file_name=filename,
        mime="image/jpeg",
        key="dl_img_" + key_suffix,
    )


def video_summary(result, file_name, key):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Frames processed", result["frames"])
    c2.metric("Total detections", result["total"])
    c3.metric("Average confidence", "{:.2%}".format(result["avg_conf"]))
    c4.metric("Processing time", "{:.1f} s".format(result["elapsed"]))

    if result["class_counts"]:
        card_title("Detections by class")
        st.dataframe(
            [
                {"Class": k, "Detections": v}
                for k, v in sorted(result["class_counts"].items(), key=lambda kv: -kv[1])
            ],
            width="stretch",
            hide_index=True,
        )

    st.video(result["video_bytes"])

    if result["samples"]:
        card_title("Annotated frames")
        cols = st.columns(len(result["samples"]))
        for col, frame in zip(cols, result["samples"]):
            with col:
                st.image(frame, width="stretch")

    st.download_button(
        "Download annotated video",
        data=result["video_bytes"],
        file_name=file_name,
        mime="video/mp4",
        key="dl_vid_" + key,
    )


# -------------------------------
# SECTIONS
# -------------------------------


def section_dashboard():
    header("Dashboard", "Monitoring overview for road damage inspections.")

    images, videos, defects, avg_conf = counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Images processed", images)
    c2.metric("Videos processed", videos)
    c3.metric("Total defects detected", defects)
    c4.metric("Average confidence", "{:.2%}".format(avg_conf))

    left, right = st.columns([1.55, 1])

    with left:
        with st.container(border=True):
            card_title("Recent inspections")
            hist = st.session_state.history[:6]
            if hist:
                st.dataframe(
                    [
                        {
                            "Time": h["time"],
                            "Type": h["type"],
                            "Source": h["source"],
                            "Detections": h["detections"],
                            "Avg confidence": "{:.2%}".format(h["avg_conf"])
                            if h["detections"]
                            else "-",
                        }
                        for h in hist
                    ],
                    width="stretch",
                    hide_index=True,
                )
            else:
                empty_state(
                    "No inspections yet. Run an image or video detection to populate the dashboard."
                )

    with right:
        with st.container(border=True):
            card_title("System")
            device, device_name = detect_device()
            model = load_model()
            st.markdown(
                '<div class="rd-kv"><span>Compute device</span><br><b>{}</b></div>'
                '<div class="rd-kv"><span>Inference resolution</span><br><b>{} x {}</b></div>'
                '<div class="rd-kv"><span>Classes</span><br><b>{}</b></div>'.format(
                    device_name, IMG_SIZE, IMG_SIZE, ", ".join(model.names.values())
                ),
                unsafe_allow_html=True,
            )
            if device == "cuda:0":
                st.caption("GPU acceleration is active.")

    with st.container(border=True):
        card_title("Quick actions")
        a, b, c = st.columns(3)
        if a.button("Run image detection", width="stretch"):
            st.session_state.nav_target = "Image Detection"
            st.rerun()
        if b.button("Run video detection", width="stretch"):
            st.session_state.nav_target = "Video Detection"
            st.rerun()
        if c.button("View inspection history", width="stretch"):
            st.session_state.nav_target = "Inspection History"
            st.rerun()


def section_image():
    header("Image Detection", "Upload a road image to detect potholes with YOLO11n.")
    device, _ = detect_device()
    conf, iou = settings_row("img")

    uploaded = st.file_uploader(
        "Road image", type=["jpg", "jpeg", "png"], key="img_upload"
    )

    if uploaded is None:
        return

    try:
        image = Image.open(uploaded).convert("RGB")
    except Exception:
        st.error("This file could not be read as an image.")
        return

    current = st.session_state.image_result
    has_result = current is not None and current.get("source") == uploaded.name

    left, right = st.columns(2)
    with left:
        card_title("Original image")
        st.image(image, width="stretch")
    with right:
        card_title("Detection results")
        if has_result:
            st.image(current["annotated"], width="stretch")
        else:
            empty_state("Run detection to generate the annotated image.")

    if st.button("Detect potholes", type="primary"):
        with st.spinner("Analyzing road image..."):
            try:
                result = run_image_detection(image, conf, iou, device)
            except Exception as exc:
                st.error("Detection failed: {}".format(exc))
                return
        result["source"] = uploaded.name
        st.session_state.image_result = result
        add_history(
            "Image", uploaded.name, result["count"], result["avg_conf"]
        )
        st.rerun()

    current = st.session_state.image_result
    if current is not None and current.get("source") == uploaded.name:
        st.divider()
        results_block(
            current,
            "road_damage_" + Path(uploaded.name).stem + ".jpg",
            "img_" + uploaded.name,
        )


def section_video():
    header(
        "Video Detection",
        "Process a road video frame by frame and export the annotated result.",
    )
    device, _ = detect_device()
    conf, iou = settings_row("vid")

    uploaded = st.file_uploader(
        "Road video", type=["mp4", "avi", "mov"], key="vid_upload"
    )

    if uploaded is None:
        return

    st.caption("Selected file: " + uploaded.name)

    if st.button("Start video detection", type="primary"):
        try:
            with st.spinner("Processing video..."):
                result = run_video_detection(uploaded, conf, iou, device)
        except Exception as exc:
            st.error("Video processing failed: {}".format(exc))
            return

        if result.get("error"):
            st.error(result["error"])
            return

        result["source"] = uploaded.name
        st.session_state.video_result = result
        add_history(
            "Video",
            uploaded.name,
            result["total"],
            result["avg_conf"],
            result["frames"],
        )
        st.rerun()

    result = st.session_state.video_result
    if result is not None and result.get("source") == uploaded.name:
        st.divider()
        video_summary(
            result,
            "road_damage_" + Path(uploaded.name).stem + ".mp4",
            "vid_" + uploaded.name,
        )


def section_results():
    header("Detection Results", "Annotated output from the most recent inspections.")
    img = st.session_state.image_result
    vid = st.session_state.video_result

    if img is None and vid is None:
        empty_state(
            "No results available yet. Run image or video detection first."
        )
        return

    tabs = st.tabs(["Image result", "Video result"])
    with tabs[0]:
        if img is None:
            empty_state("No image result in this session.")
        else:
            st.caption(
                "{} | {}x{} px | conf {:.2f} | IoU {:.2f}".format(
                    img["source"],
                    img["size"][0],
                    img["size"][1],
                    img["conf"],
                    img["iou"],
                )
            )
            st.image(img["annotated"], width="stretch")
            results_block(
                img,
                "road_damage_" + Path(img["source"]).stem + ".jpg",
                "res_" + img["source"],
            )

    with tabs[1]:
        if vid is None:
            empty_state("No video result in this session.")
        else:
            st.caption(
                "{} | {} frames | {}x{} px".format(
                    vid["source"], vid["frames"], vid["size"][0], vid["size"][1]
                )
            )
            video_summary(vid, "road_damage_video.mp4", "results_" + vid["source"])


def section_history():
    header("Inspection History", "All detections performed in this session.")
    hist = st.session_state.history

    if not hist:
        empty_state(
            "No inspections recorded yet. History is kept for the current session only."
        )
        return

    images, videos, defects, avg_conf = counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Inspection runs", len(hist))
    c2.metric("Images / videos", "{} / {}".format(images, videos))
    c3.metric("Total detections", defects)
    c4.metric("Average confidence", "{:.2%}".format(avg_conf))

    rows = [
        {
            "ID": h["id"],
            "Time": h["time"],
            "Type": h["type"],
            "Source": h["source"],
            "Detections": h["detections"],
            "Avg confidence": "{:.2%}".format(h["avg_conf"])
            if h["detections"]
            else "-",
            "Frames": str(h["frames"]) if h["frames"] else "-",
        }
        for h in hist
    ]
    st.dataframe(rows, width="stretch", hide_index=True)

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

    a, b = st.columns(2)
    with a:
        st.download_button(
            "Export history as CSV",
            data=buffer.getvalue().encode("utf-8"),
            file_name="inspection_history.csv",
            mime="text/csv",
        )
    with b:
        if st.button("Clear history"):
            st.session_state.history = []
            st.session_state.image_result = None
            st.session_state.video_result = None
            st.rerun()


def section_about():
    header("About Model", "Model, hardware and reported performance.")

    model = load_model()
    device, device_name = detect_device()
    params = sum(p.numel() for p in model.model.parameters())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Parameters", "{:.1f} M".format(params / 1e6))
    c2.metric("Input resolution", "{} px".format(IMG_SIZE))
    c3.metric("Classes", len(model.names))
    c4.metric("Weights", "{:.1f} MB".format(MODEL_PATH.stat().st_size / 1e6))

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            card_title("Model")
            st.markdown(
                '<div class="rd-kv"><span>Architecture</span><br><b>YOLO11n (Ultralytics)</b></div>'
                '<div class="rd-kv"><span>Task</span><br><b>{}</b></div>'
                '<div class="rd-kv"><span>Classes</span><br><b>{}</b></div>'
                '<div class="rd-kv"><span>Weights file</span><br><b>{}</b></div>'.format(
                    model.task, ", ".join(model.names.values()), MODEL_PATH.name
                ),
                unsafe_allow_html=True,
            )
    with right:
        with st.container(border=True):
            card_title("Runtime")
            gpu_note = (
                "GPU acceleration active" if device == "cuda:0" else "CPU inference"
            )
            st.markdown(
                '<div class="rd-kv"><span>Device</span><br><b>{}</b></div>'
                '<div class="rd-kv"><span>Mode</span><br><b>{}</b></div>'
                '<div class="rd-kv"><span>Inference size</span><br><b>{} x {}</b></div>'
                '<div class="rd-kv"><span>Task type</span><br><b>Object detection</b></div>'.format(
                    device_name, gpu_note, IMG_SIZE, IMG_SIZE
                ),
                unsafe_allow_html=True,
            )

    with st.container(border=True):
        card_title("Reported validation metrics")
        st.caption(
            "Values documented in the project README and training notebook (800 px run)."
        )
        st.dataframe(
            [
                {"Metric": "Precision", "Score": "89.5%"},
                {"Metric": "Recall", "Score": "83.9%"},
                {"Metric": "mAP@50", "Score": "91.0%"},
                {"Metric": "mAP@50-95", "Score": "57.9%"},
            ],
            width="stretch",
            hide_index=True,
        )


# -------------------------------
# MAIN
# -------------------------------

SECTIONS = {
    "Dashboard": section_dashboard,
    "Image Detection": section_image,
    "Video Detection": section_video,
    "Detection Results": section_results,
    "Inspection History": section_history,
    "About Model": section_about,
}


def main():
    init_state()

    target = st.session_state.pop("nav_target", None)
    if target in SECTIONS:
        st.session_state.nav = target

    with st.sidebar:
        st.markdown(
            '<div class="rd-brand"><div class="name">Road Damage Detection</div>'
            '<div class="tag">YOLO11n monitoring</div></div>',
            unsafe_allow_html=True,
        )
        st.radio(
            "Navigation",
            list(SECTIONS.keys()),
            key="nav",
            label_visibility="collapsed",
        )
        st.divider()
        device, device_name = detect_device()
        st.caption("Model: " + MODEL_PATH.name)
        st.caption("Device: " + device_name)
        images, videos, defects, _ = counts()
        st.caption(
            "Session: {} images / {} videos / {} detections".format(
                images, videos, defects
            )
        )

    SECTIONS[st.session_state.nav]()


if __name__ == "__main__":
    main()
