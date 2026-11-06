"""Smart CrowdFlow - AI Crowd Monitoring System (YOLOv8 + Streamlit)

Run:  python -m streamlit run app.py --server.port 8502
Keep temple.mp4, yolov8n.pt, auth_sms.py, sms_alerts.py (and optionally my_botsort.yaml) beside this file.
Any extra *.mp4 files beside app.py automatically appear in the camera dropdown.
"""

import html
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ultralytics import YOLO

import auth_sms

# ==================== CONFIGURATION ====================

BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = BASE_DIR / "temple.mp4"
MODEL_PATH = BASE_DIR / "yolov8n.pt"
TRACKER_PATH = BASE_DIR / "my_botsort.yaml"
SNAP_DIR = BASE_DIR / "snapshots"

st.set_page_config(page_title="Smart CrowdFlow", page_icon="🟢", layout="wide")

ZONE_TO_LEVEL = {"GREEN": "LOW", "YELLOW": "MEDIUM", "RED": "HIGH"}
LEVEL_COLOR = {"LOW": "#42c87b", "MEDIUM": "#ffb83f", "HIGH": "#ff514d"}
ZONE_COLOR = {"GREEN": "#37c779", "YELLOW": "#ffb62f", "RED": "#f0443e"}
# BGR colours for detection boxes
BOX_BGR = {"low": (113, 191, 47), "mid": (46, 184, 245), "high": (45, 59, 239)}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def discover_cameras():
    cams = {"Camera 01 - Main Entrance": VIDEO_PATH}
    n = 2
    for p in sorted(BASE_DIR.glob("*.mp4")):
        if p.name != VIDEO_PATH.name:
            cams[f"Camera {n:02d} - {p.stem.replace('_', ' ').title()}"] = p
            n += 1
    return cams


CAMERAS = discover_cameras()

# ==================== SESSION STATE ====================

ss = st.session_state

defaults = {
    "page": "Overview",
    "focus": False,
    "camera_sel": list(CAMERAS)[0],
    "people_count": 0,
    "density_pct": 0,
    "flow_rate": 0,
    "video_status": "Stopped",
    "video_cap": None,
    "video_frame_rgb": None,
    "video_error": "",
    "video_processed_frames": 0,
    "last_zone": None,
    "current_zone": "GREEN",
    "current_alert": "NORMAL",
    "crowd_density": "LOW",
    "alert_history": [],
    "sms_log": [],
    "history": [],
    "heat": None,
    "bg_small": None,
    "seen_ids": {},
    "last_hist_vt": -99.0,
    "last_alert_vt": -99.0,
    "preview": None,
    "trend_range": "Session",
    "thr_low": 20,
    "thr_high": 50,
    "capacity": 50,
    "user": None,
    "sms_auto": True,
    "sms_levels": "HIGH only",
    "sms_cooldown": 300,
}

for key, value in defaults.items():
    if key not in ss:
        ss[key] = value


@st.cache_resource
def load_yolo_model():
    return YOLO(str(MODEL_PATH))


def current_video() -> Path:
    return CAMERAS.get(ss.camera_sel, VIDEO_PATH)


def release_video():
    cap = ss.get("video_cap")
    if cap is not None:
        cap.release()
    ss.video_cap = None


def reset_stats():
    ss.video_frame_rgb = None
    ss.video_processed_frames = 0
    ss.people_count = 0
    ss.density_pct = 0
    ss.flow_rate = 0
    ss.last_zone = None
    ss.current_zone = "GREEN"
    ss.current_alert = "NORMAL"
    ss.crowd_density = "LOW"
    ss.history = []
    ss.heat = None
    ss.bg_small = None
    ss.seen_ids = {}
    ss.last_hist_vt = -99.0
    ss.last_alert_vt = -99.0


def start_video():
    path = current_video()
    if not path.exists():
        ss.video_error = f"Video file not found: {path}"
        return

    release_video()
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        cap.release()
        ss.video_error = f"Could not open {path.name}. Check the video file."
        ss.video_status = "Stopped"
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ss.video_cap = cap
    ss.video_status = "Running"
    ss.video_error = ""
    reset_stats()

    # Reset YOLO tracker state when a new session starts.
    model = load_yolo_model()
    if hasattr(model, "predictor"):
        model.predictor = None


def restart_video():
    release_video()
    start_video()


# ==================== CUSTOM CSS ====================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: Inter, sans-serif; }

.stApp {
    background: radial-gradient(ellipse at 70% 0%, #0b1b2c 0%, #050d17 55%, #030a12 100%);
    color: #edf5fc;
}
header[data-testid="stHeader"] { background: transparent; height: 0; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding: .8rem 1.25rem 1rem; max-width: 1900px; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #061321, #030b14);
    border-right: 1px solid #14283b;
    min-width: 280px; max-width: 280px;
}
[data-testid="stSidebar"] * { color: #dce7f1; }
[data-testid="stSidebar"] .stButton > button {
    justify-content: flex-start; text-align: left; width: 100%;
    background: transparent; border: 1px solid transparent;
    padding: .72rem 1rem; font-size: 14px; border-radius: 9px; color: #c9d6e3;
}
[data-testid="stSidebar"] .stButton > button p { text-align: left; }
[data-testid="stSidebar"] .stButton > button:hover { background: #0f2538; border-color: #14283b; }
[data-testid="stSidebar"] button[kind="primary"],
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
    background: #14704a !important; border-color: #1f8f5f !important; color: #fff !important;
}
.brand { display: flex; gap: 10px; align-items: center; padding: 4px 2px 22px; }
.brandicon {
    width: 44px; height: 48px; border: 2px solid #35c57b; border-radius: 12px 12px 22px 22px;
    display: grid; place-items: center; color: #45d78c; font-size: 22px;
}
.brandname { font-size: 20px; font-weight: 800; }
.brandname span { color: #48d48b; }
.sub { font-size: 10px; color: #9aabba; margin-top: 4px; }
.zonebox { border: 1px solid #16304a; border-radius: 10px; padding: 14px; margin-top: 26px; background: #050f1b; }
.zonebox h4 { color: #48d48b; font-size: 13px; margin: 0 0 8px; font-weight: 600; }
.zone { display: flex; align-items: center; gap: 9px; margin: 13px 0; font-size: 12px; }
.dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
.zbadge { margin-left: auto; padding: 4px 8px; border-radius: 5px; font-size: 9px; font-weight: 700; }
.high { background: #431b24; color: #ff5b59; }
.medium { background: #3a2a14; color: #ffc04c; }
.low { background: #10352b; color: #51d994; }

/* ---------- top bar ---------- */
.topline {
    display: flex; justify-content: flex-end; align-items: center; gap: 26px;
    color: #a6b5c5; font-size: 12px; padding: 2px 6px 12px;
}
.online { color: #42d38a; }
.bell { position: relative; display: inline-block; font-size: 15px; }
.bell b {
    position: absolute; top: -7px; right: -9px; background: #e5352d; color: #fff;
    font-size: 9px; border-radius: 50%; min-width: 15px; height: 15px;
    display: flex; align-items: center; justify-content: center;
}
.avatar { display: inline-block; width: 24px; height: 24px; border-radius: 50%;
    background: #cfd6e2; vertical-align: middle; margin-right: 6px; }

/* ---------- text / cards ---------- */
.pagetitle { font-size: 22px; font-weight: 700; padding-top: 6px; }
.muted { color: #91a4b8; font-size: 12px; line-height: 1.5; }
.paneltitle { font-size: 15px; font-weight: 600; margin-bottom: 8px; color: #eaf2fa; }
.metric {
    min-height: 92px; background: linear-gradient(140deg, #0a1c2b, #071523);
    border: 1px solid #18334a; border-radius: 12px; padding: 14px;
    display: flex; align-items: center; gap: 14px;
}
.mi { width: 52px; height: 52px; border-radius: 12px; display: grid; place-items: center;
    font-size: 24px; flex-shrink: 0; }
.ml { font-size: 12px; color: #c4d2e0; margin-bottom: 2px; }
.mv { font-size: 26px; font-weight: 700; line-height: 1.2; }
.mv small { font-size: 11px; font-weight: 600; margin-left: 6px; }
.mn { font-size: 10px; color: #8da2b5; margin-top: 3px; }
.pill { display: inline-block; border: 1px solid #28445a; background: #0b1d2d;
    border-radius: 8px; padding: 8px 10px; font-size: 11px; color: #b8c9d9; }
.live { display: inline-block; color: #fff; background: #a31f24; border-radius: 7px;
    padding: 8px 12px; font-size: 11px; font-weight: 700; margin-top: 2px; }
.live.off { background: #1b2c3b; color: #9aabba; }
.density { font-size: 40px; font-weight: 800; text-align: center; margin-top: 2px; line-height: 1.1; }
.percent { text-align: center; font-size: 15px; font-weight: 600; }
.risk { font-size: 24px; font-weight: 800; margin: 8px 0; }
.alertrow { display: flex; gap: 10px; padding: 9px 0; border-bottom: 1px solid #142b3c; font-size: 11px; }
.alertrow:last-child { border-bottom: none; }
.atime { color: #b6c4d2; min-width: 42px; }
.alertmsg { color: #e6eef6; line-height: 1.5; }

/* ---------- widgets ---------- */
div[data-testid="stButton"] > button {
    border-radius: 8px; border: 1px solid #214057; background: #0b1e2e; color: #e8f1f9; font-size: 12px;
}
div[data-testid="stButton"] > button:hover { border-color: #36c77b; color: #fff; background: #103229; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color: #18334a !important; background: rgba(7,21,34,.78); border-radius: 12px; }
div[data-baseweb="select"] > div { background: #0b1d2d !important; border-color: #28445a !important; color: #e8f1f9 !important; }
div[data-baseweb="select"] * { color: #e8f1f9 !important; }
ul[role="listbox"] { background: #0b1d2d !important; }
.st-key-view_all button { background: transparent !important; border: none !important; color: #48d48b !important; font-size: 12px; }
[data-testid="stImage"] img { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# Login / register screen. Nothing below runs until the user is logged in.
auth_sms.require_login()


# ==================== VIDEO DRAWING HELPERS ====================

def _blend_rect(img, x0, y0, x1, y1, color=(30, 20, 10), alpha=0.7):
    x0, y0 = max(0, x0), max(0, y0)
    roi = img[y0:y1, x0:x1]
    if roi.size == 0:
        return
    img[y0:y1, x0:x1] = cv2.addWeighted(roi, 1 - alpha, np.full_like(roi, color), alpha, 0)


def draw_overlay(frame, count):
    """Clock (top-left) and People Count box (bottom-left) burned into the frame, like the target UI."""
    h, w = frame.shape[:2]
    s = max(0.5, w / 1280)
    pad = int(10 * s)

    txt = datetime.now().strftime("%H:%M:%S")
    scale, th = 0.85 * s, max(1, int(2 * s))
    (tw, tht), base = cv2.getTextSize(txt, FONT, scale, th)
    x0 = y0 = int(12 * s)
    _blend_rect(frame, x0, y0, x0 + tw + 2 * pad, y0 + tht + 2 * pad + base)
    cv2.putText(frame, txt, (x0 + pad, y0 + pad + tht), FONT, scale, (255, 255, 255), th, cv2.LINE_AA)

    label, ls, lth = "People Count", 0.5 * s, max(1, int(s))
    (lw, lh), _ = cv2.getTextSize(label, FONT, ls, lth)
    num, ns, nth = str(count), 1.4 * s, max(2, int(3 * s))
    (nw, nh), nb = cv2.getTextSize(num, FONT, ns, nth)
    bw, bh = max(lw, nw) + 2 * pad, lh + nh + 3 * pad + nb
    bx0, by1 = int(12 * s), h - int(12 * s)
    by0 = by1 - bh
    _blend_rect(frame, bx0, by0, bx0 + bw, by1)
    cv2.putText(frame, label, (bx0 + pad, by0 + pad + lh), FONT, ls, (235, 225, 215), lth, cv2.LINE_AA)
    cv2.putText(frame, num, (bx0 + pad, by0 + 2 * pad + lh + nh), FONT, ns, (137, 215, 75), nth, cv2.LINE_AA)
    return frame


def draw_detections(frame, xyxy):
    """Boxes coloured by local crowding: green (sparse) / yellow / red (packed)."""
    if len(xyxy) == 0:
        return
    h, w = frame.shape[:2]
    centers = np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2], axis=1)
    dist = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    neighbours = (dist < 0.07 * w).sum(axis=1) - 1
    thick = max(1, w // 900)
    for (x1, y1, x2, y2), n in zip(xyxy.astype(int), neighbours):
        col = BOX_BGR["low"] if n <= 3 else BOX_BGR["mid"] if n <= 8 else BOX_BGR["high"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, thick)


def process_frame():
    """Read + track one frame, update all session metrics. Called by the video fragment while Running."""
    cap = ss.video_cap
    need_rerun = False

    if cap is None or not cap.isOpened():
        ss.video_status = "Stopped"
        ss.video_error = "Video capture is not open. Press Start."
        st.rerun()

    try:
        ok, frame = cap.read()
        if not ok:
            release_video()
            ss.video_status = "Completed"
            need_rerun = True
        else:
            h, w = frame.shape[:2]
            model = load_yolo_model()
            args = {"source": frame, "persist": True, "classes": [0],
                    "conf": 0.20, "imgsz": 1280, "verbose": False}
            if TRACKER_PATH.exists():
                args["tracker"] = str(TRACKER_PATH)

            result = model.track(**args)[0]

            ids, xyxy = [], np.empty((0, 4))
            if result.boxes is not None and result.boxes.id is not None:
                ids = result.boxes.id.int().cpu().tolist()
                xyxy = result.boxes.xyxy.cpu().numpy()
            count = len(set(ids))

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            ss.video_processed_frames += 1
            vt = ss.video_processed_frames / fps  # video time in seconds

            # --- density / zone (thresholds adjustable in Settings) ---
            if count <= ss.thr_low:
                label, zone, alert = "LOW", "GREEN", "NORMAL"
            elif count <= ss.thr_high:
                label, zone, alert = "MEDIUM", "YELLOW", "MONITOR CROWD"
            else:
                label, zone, alert = "HIGH", "RED", "HIGH CROWD ALERT"
            density = min(100, round(count / max(1, ss.capacity) * 100))

            # --- flow rate: new people first seen in the last 60 s of video ---
            for i in ids:
                ss.seen_ids.setdefault(i, vt if vt >= 2 else -1e9)  # people present at start aren't "flow"
            flow = sum(1 for t in ss.seen_ids.values() if vt - t <= 60)

            # --- heat accumulation (feet position of every tracked person) ---
            if ss.heat is None:
                ss.heat = np.zeros((36, 64), np.float32)
            ss.heat *= 0.995
            for x1, y1, x2, y2 in xyxy:
                gx = int(np.clip((x1 + x2) / 2 / w * 64, 0, 63))
                gy = int(np.clip(y2 / h * 36, 0, 35))
                ss.heat[gy, gx] += 1.0
            ss.bg_small = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2RGB)

            # --- draw ---
            draw_detections(frame, xyxy)
            draw_overlay(frame, count)

            ss.people_count = count
            ss.density_pct = density
            ss.flow_rate = flow
            ss.crowd_density = label
            ss.current_zone = zone
            ss.current_alert = alert
            ss.video_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # --- history sample once per video second ---
            if vt - ss.last_hist_vt >= 1:
                ss.last_hist_vt = vt
                ss.history.append({"t": datetime.now(), "density": density, "count": count,
                                   "flow": flow, "zone": zone})
                ss.history = ss.history[-600:]

            # --- alerts on zone change (rate limited to one per 3 video seconds) ---
            prev = ss.last_zone
            if zone != prev and (vt - ss.last_alert_vt >= 3 or zone == "RED"):
                entry = None
                if zone == "RED":
                    entry = ("HIGH", "High density detected")
                elif zone == "YELLOW":
                    entry = ("MEDIUM", "Crowd increasing rapidly" if prev == "GREEN" else "Medium density")
                elif zone == "GREEN" and prev is not None:
                    entry = ("LOW", "Low density")
                if entry:
                    ss.alert_history.insert(0, {
                        "Time": datetime.now().strftime("%H:%M"),
                        "Level": entry[0], "Message": entry[1],
                        "Area": f"at {ss.camera_sel.split(' - ', 1)[-1]}",
                    })
                    ss.last_alert_vt = vt

                    # --- automatic SMS to registered numbers, then logged-in numbers ---
                                       # --- automatic SMS to registered numbers, then logged-in numbers ---
                    send_high = entry[0] == "HIGH"
                    # MEDIUM only when rising (not when dropping from HIGH back to MEDIUM)
                    send_medium = (entry[0] == "MEDIUM" and ss.sms_levels == "HIGH and MEDIUM"
                                   and prev in (None, "GREEN"))
                    if ss.sms_auto and (send_high or send_medium):
                        auth_sms.send_alert_async(
                            "RED_ZONE" if send_high else "YELLOW_ZONE",
                            count,
                            "RED" if send_high else "YELLOW",
                            ss.sms_cooldown,
                        )

    except Exception as exc:
        ss.video_error = f"Video processing failed: {exc}"
        release_video()
        ss.video_status = "Stopped"
        need_rerun = True

    if need_rerun:
        st.rerun()  # refresh run_every settings + status badge


# ==================== CHARTS / COMPONENTS ====================

def metric(icon, label, value_html, note_html, color):
    st.markdown(f"""
    <div class="metric">
        <div class="mi" style="background:{color}22;color:{color}">{icon}</div>
        <div>
            <div class="ml">{label}</div>
            <div class="mv">{value_html}</div>
            <div class="mn">{note_html}</div>
        </div>
    </div>""", unsafe_allow_html=True)


def delta_vs_15min(key):
    hist = ss.history
    if len(hist) < 2:
        return None
    now, cur = hist[-1]["t"], hist[-1][key]
    old = next((h for h in hist if now - h["t"] <= timedelta(minutes=15)), hist[0])[key]
    if not old:
        return None
    return (cur - old) / old * 100


def delta_html(d, up_color, down_color):
    if d is None:
        return ""
    arrow, col = ("▲", up_color) if d >= 0 else ("▼", down_color)
    return f'<small style="color:{col}">{arrow} {abs(d):.0f}%</small>'


def render_gauge(value, key="crowd_density_gauge"):
    fig = go.Figure(go.Indicator(
        mode="gauge", value=value,
        gauge={
            "axis": {"range": [0, 100], "tickvals": [0, 50, 100],
                     "ticktext": ["0%", "50%", "100%"], "tickcolor": "#6c8196",
                     "tickfont": {"color": "#c4d2e0", "size": 11}},
            "bar": {"color": "rgba(0,0,0,0)"},
            "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
            "steps": [{"range": [0, 33], "color": "#37c779"},
                      {"range": [33, 66], "color": "#ffb62f"},
                      {"range": [66, 100], "color": "#f0443e"}],
            "threshold": {"line": {"color": "#f4f6fa", "width": 6}, "thickness": 0.9, "value": value},
        }))
    fig.update_layout(height=165, margin=dict(l=22, r=22, t=8, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", font={"color": "#dce7f1"})
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=key)


def render_trend(window="Session", height=215, key="density_trend_chart"):
    hist = ss.history
    layout = dict(height=height, margin=dict(l=8, r=8, t=5, b=5),
                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
                  font={"color": "#9cafc1", "size": 11})
    yaxis = {"range": [0, 100], "ticksuffix": "%", "gridcolor": "#12283a",
             "title": "Density (%)", "title_font": {"size": 9}}
    fig = go.Figure()

    if len(hist) < 2:  # nothing recorded yet -> demo curve
        values = [12, 16, 20, 23, 27, 32, 35, 39, 42, 46, 49, 51, 48, 54, 58,
                  61, 66, 63, 71, 78, 88, 76, 64, 55, 50]
        for i in range(len(values) - 1):
            v = values[i]
            c = ZONE_COLOR["GREEN"] if v < 35 else ZONE_COLOR["YELLOW"] if v < 66 else ZONE_COLOR["RED"]
            fig.add_trace(go.Scatter(x=[i, i + 1], y=values[i:i + 2], mode="lines",
                                     line=dict(color=c, width=3), hoverinfo="skip"))
        fig.update_layout(**layout, yaxis=yaxis, xaxis={
            "tickmode": "array", "tickvals": [0, 4, 8, 12, 16, 20, 24],
            "ticktext": ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "24:00"],
            "gridcolor": "#12283a"})
    else:
        now = hist[-1]["t"]
        mins = {"Last 5 min": 5, "Last 1 min": 1}.get(window)
        data = [h for h in hist if mins is None or now - h["t"] <= timedelta(minutes=mins)]
        if len(data) < 2:
            data = hist[-2:]
        x = [h["t"] for h in data]
        y = [h["density"] for h in data]
        z = [h["zone"] for h in data]
        start = 0
        for i in range(1, len(data) + 1):
            if i == len(data) or z[i] != z[start]:
                end = min(i + 1, len(data))
                fig.add_trace(go.Scatter(x=x[start:end], y=y[start:end], mode="lines",
                                         line=dict(color=ZONE_COLOR[z[start]], width=3),
                                         hoverinfo="skip"))
                start = i
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", marker=dict(size=8, opacity=0),
                                 hovertemplate="%{x|%H:%M:%S}<br>%{y}%<extra></extra>"))
        pk = int(np.argmax(y))
        fig.add_trace(go.Scatter(x=[x[pk]], y=[y[pk]], mode="markers",
                                 marker=dict(size=10, color=ZONE_COLOR[z[pk]],
                                             line=dict(color="#fff", width=2)),
                                 hovertemplate="Peak %{x|%H:%M:%S}<br>%{y}%<extra></extra>"))
        fig.update_layout(**layout, yaxis=yaxis,
                          xaxis={"tickformat": "%H:%M:%S", "gridcolor": "#12283a"})

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=key)
    return len(hist) < 2


def heat_image(width=320, height=180):
    """Heat overlay built from real tracked positions; falls back to a demo pattern before data exists."""
    demo = ss.heat is None or ss.bg_small is None or float(ss.heat.max()) <= 0
    if demo:
        bg = np.full((180, 320, 3), (12, 28, 38), np.uint8)
        yy, xx = np.mgrid[0:36, 0:64]
        heat = np.zeros((36, 64), np.float32)
        for cy, cx, st_ in [(10, 16, 1), (20, 30, .9), (27, 46, .75), (8, 40, .55)]:
            heat += st_ * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 4.4 ** 2))
    else:
        bg, heat = ss.bg_small, ss.heat
    h = cv2.GaussianBlur(heat, (0, 0), 1.6)
    n = cv2.resize(h / max(float(h.max()), 1e-6), (320, 180))
    colored = cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1].astype(np.float32)
    alpha = np.clip(n * 1.6, 0, 0.8)[..., None]
    out = bg.astype(np.float32) * 0.55 * (1 - alpha) + colored * alpha
    return out.astype(np.uint8), demo


def heat_legend():
    st.markdown("""
    <div style="height:8px;background:linear-gradient(90deg,#164c43,#2fb96d,#f0df38,#ff7c2d,#d92125);border-radius:4px"></div>
    <div style="display:flex;justify-content:space-between;color:#9aabba;font-size:10px;margin-top:4px">
        <span>Low</span><span>High</span></div>""", unsafe_allow_html=True)


def alert_rows(items):
    if not items:
        st.markdown('<div class="muted" style="padding:8px 0">No alerts yet.</div>', unsafe_allow_html=True)
        return
    for item in items:
        color = {"HIGH": "#ff5b55", "MEDIUM": "#ffb83f"}.get(item["Level"], "#48d48b")
        symbol = {"HIGH": "▲", "MEDIUM": "⚠"}.get(item["Level"], "▲")
        st.markdown(f"""
        <div class="alertrow">
            <div class="atime">{item["Time"]}</div>
            <div style="color:{color};font-size:14px">{symbol}</div>
            <div class="alertmsg">{item["Message"]}<br>
                <span style="color:#8fa3b6">{item["Area"]}</span></div>
        </div>""", unsafe_allow_html=True)


# ==================== LIVE (AUTO-REFRESHING) PANELS ====================

RUNNING = ss.video_status == "Running"
EVERY_FAST = 0.2 if RUNNING else None   # video + detection
EVERY_SLOW = 1 if RUNNING else None     # cards, charts, sidebar


NAV = [("Overview", "🏠"), ("Live Monitoring", "◎"), ("Density Map", "🗺"),
       ("Analytics", "📈"), ("Alerts", "🔔"), ("Reports", "📄"), ("Settings", "⚙️")]


@st.fragment(run_every=EVERY_SLOW)
def sidebar_panel():
    n = len(ss.alert_history)
    for label, icon in NAV:
        text = f"{icon}  {label}"
        if label == "Alerts" and n:
            text += f"   :red[● {min(n, 9)}{'+' if n > 9 else ''}]"
        if st.button(text, key=f"nav_{label}", use_container_width=True,
                     type="primary" if ss.page == label else "secondary"):
            ss.page = label
            ss.focus = False
            st.rerun()

    main_level = ZONE_TO_LEVEL[ss.current_zone]
    zones = [("Main Entrance", main_level),   # live from the detector
             ("Queue Area", "MEDIUM"), ("Area - 2", "LOW"), ("Exit Gate", "HIGH")]  # demo zones
    rows = "".join(
        f'<div class="zone"><span class="dot" style="background:{LEVEL_COLOR[lv]}"></span>'
        f'<span>{name}</span><span class="zbadge {lv.lower()}">{lv}</span></div>'
        for name, lv in zones)
    st.markdown(f'<div class="zonebox"><h4>Monitored Zones</h4>{rows}</div>', unsafe_allow_html=True)


@st.fragment(run_every=EVERY_SLOW)
def top_bar():
    auth_sms.heartbeat(ss.user["phone"])  # keeps this number counted as "logged in"
    n = len(ss.alert_history)
    badge = f"<b>{min(n, 9)}</b>" if n else ""
    st.markdown(f"""
    <div class="topline">
        <span>📅 &nbsp;{datetime.now().strftime("%d %b %Y, %I:%M %p")}</span>
        <span class="online">● System {"Online" if RUNNING or ss.video_status != "Stopped" else "Idle"}</span>
        <span class="bell">🔔{badge}</span>
        <span><i class="avatar"></i><b style="color:#e8f1f9">{html.escape(ss.user["name"])}</b> ▾</span>
    </div>""", unsafe_allow_html=True)


def header_row():
    c1, c2, c3 = st.columns([2.1, 2.3, 0.8])
    c1.markdown('<div class="pagetitle">Live Crowd Monitoring</div>', unsafe_allow_html=True)
    names = list(CAMERAS)
    with c2:
        choice = st.selectbox("Camera", names, index=names.index(ss.camera_sel),
                              label_visibility="collapsed")
    if choice != ss.camera_sel:  # new camera -> stop and reset
        ss.camera_sel = choice
        release_video()
        ss.video_status = "Stopped"
        ss.video_error = ""
        ss.preview = None
        reset_stats()
        st.rerun()
    status = ss.video_status
    badge = {"Running": '<span class="live">● LIVE</span>',
             "Paused": '<span class="live off">⏸ PAUSED</span>',
             "Completed": '<span class="live off">✔ DONE</span>'}.get(
        status, '<span class="live off">■ STOPPED</span>')
    c3.markdown(badge, unsafe_allow_html=True)


@st.fragment(run_every=EVERY_FAST)
def video_panel():
    if ss.video_status == "Running":
        process_frame()

    if ss.video_error:
        st.error(ss.video_error)

    frame_rgb = ss.video_frame_rgb
    path = current_video()
    if frame_rgb is None and path.exists():
        if ss.preview is None or ss.preview[0] != str(path):
            pc = cv2.VideoCapture(str(path))
            ok, pf = pc.read()
            pc.release()
            ss.preview = (str(path), cv2.cvtColor(draw_overlay(pf, 0), cv2.COLOR_BGR2RGB) if ok else None)
        frame_rgb = ss.preview[1]
        if frame_rgb is None:
            st.warning(f"{path.name} exists but could not be read.")

    if frame_rgb is not None:
        st.image(frame_rgb, use_container_width=True)
    elif not path.exists():
        st.markdown(f"""
        <div style="height:300px;display:grid;place-items:center;background:#06101b;
        border:1px solid #18334a;border-radius:12px;color:#91a4b8">
            Place {path.name} beside app.py to show the CCTV preview</div>""", unsafe_allow_html=True)

    b = st.columns(6)
    if b[0].button("▶ Start", key="video_start", use_container_width=True):
        if ss.video_status in ("Stopped", "Completed"):
            start_video()
        elif ss.video_status == "Paused":
            ss.video_status = "Running"
        st.rerun()
    if b[1].button("⏸ Pause", key="video_pause_resume", use_container_width=True):
        if ss.video_status == "Running":
            ss.video_status = "Paused"
        elif ss.video_status == "Paused":
            ss.video_status = "Running"
        st.rerun()
    if b[2].button("⏹ Stop", key="video_stop", use_container_width=True):
        release_video()
        ss.video_status = "Stopped"
        st.rerun()
    if b[3].button("↻ Restart", key="video_restart", use_container_width=True):
        restart_video()
        st.rerun()
    if b[4].button("⛶ Exit" if ss.focus else "⛶ Full", key="video_full", use_container_width=True):
        ss.focus = not ss.focus
        st.rerun()
    if b[5].button("📷 Snap", key="video_snap", use_container_width=True):
        if ss.video_frame_rgb is None:
            st.toast("Start the video first.")
        else:
            SNAP_DIR.mkdir(exist_ok=True)
            name = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(str(SNAP_DIR / name), cv2.cvtColor(ss.video_frame_rgb, cv2.COLOR_RGB2BGR))
            st.toast(f"Saved snapshots/{name}")

    color = {"Running": "#47d88c", "Paused": "#ffb83f"}.get(ss.video_status, "#9aabba")
    st.markdown(f'<div style="font-size:11px;color:{color}">● Video status: {ss.video_status}</div>',
                unsafe_allow_html=True)


@st.fragment(run_every=EVERY_SLOW)
def side_panel():
    level = ss.crowd_density
    color = LEVEL_COLOR[level]
    with st.container(border=True):
        st.markdown('<div class="paneltitle">Crowd Density</div>'
                    f'<div class="density" style="color:{color}">{level}</div>'
                    f'<div class="percent" style="color:{color}">({ss.density_pct}%)</div>',
                    unsafe_allow_html=True)
        render_gauge(ss.density_pct)
    with st.container(border=True):
        text = {"HIGH": ("HIGH RISK", "Crowd density is high.<br>Take necessary action."),
                "MEDIUM": ("MEDIUM RISK", "Crowd density is moderate.<br>Monitor the area."),
                "LOW": ("LOW RISK", "Crowd density is normal.<br>No action needed.")}[level]
        st.markdown(f'<div class="paneltitle">Risk Level ⓘ</div>'
                    f'<div class="risk" style="color:{color}">⚠ {text[0]}</div>'
                    f'<div class="muted">{text[1]}</div>', unsafe_allow_html=True)


@st.fragment(run_every=EVERY_SLOW)
def kpi_row():
    cards = st.columns(4, gap="small")
    total_cams = len(CAMERAS)
    active = 1 if ss.video_status in ("Running", "Paused") else 0
    with cards[0]:
        metric("👥", "Total People", f'{ss.people_count}{delta_html(delta_vs_15min("count"), "#48d48b", "#ff5b59")}',
               "vs last 15 min", "#43d6a0")
    with cards[1]:
        metric("📈", "Density", f'{ss.density_pct}%{delta_html(delta_vs_15min("density"), "#ffb83f", "#48d48b")}',
               "vs last 15 min", "#ffb83f")
    with cards[2]:
        metric("🚶", "Flow Rate", f'{ss.flow_rate}<small style="color:#a9bacb">people/min</small>'
               f'{delta_html(delta_vs_15min("flow"), "#48d48b", "#c6a5ff")}',
               "new people seen in last 60 s", "#c6a5ff")
    with cards[3]:
        metric("📹", "Active Cameras", f'{active}<small style="color:#a9bacb">/ {total_cams}</small>',
               "Online" if active else "Offline", "#57a8ff")


@st.fragment(run_every=EVERY_SLOW)
def bottom_row():
    trend_col, map_col, alerts_col = st.columns([1.55, 0.9, 0.95], gap="small")

    with trend_col:
        with st.container(border=True):
            t1, t2 = st.columns([3, 1.3])
            t1.markdown(f'<div class="paneltitle">Density Trend '
                        f'({"Demo" if len(ss.history) < 2 else ss.trend_range})</div>',
                        unsafe_allow_html=True)
            opts = ["Session", "Last 5 min", "Last 1 min"]
            ss.trend_range = t2.selectbox("Range", opts, index=opts.index(ss.trend_range),
                                          label_visibility="collapsed", key="trend_range_widget")
            render_trend(ss.trend_range)

    with map_col:
        with st.container(border=True):
            img, demo = heat_image()
            st.markdown(f'<div class="paneltitle">Density Map{" (Demo)" if demo else ""}</div>',
                        unsafe_allow_html=True)
            st.image(img, use_container_width=True)
            heat_legend()

    with alerts_col:
        with st.container(border=True):
            a1, a2 = st.columns([2, 1])
            a1.markdown('<div class="paneltitle">Recent Alerts</div>', unsafe_allow_html=True)
            if a2.button("View All", key="view_all"):
                ss.page = "Alerts"
                st.rerun()
            alert_rows(ss.alert_history[:4])


@st.fragment(run_every=EVERY_SLOW)
def big_heatmap():
    img, demo = heat_image()
    st.image(img, use_container_width=True)
    heat_legend()
    if demo:
        st.caption("Demo pattern — start the video to build a live density map from tracked people.")


@st.fragment(run_every=EVERY_SLOW)
def analytics_panel():
    render_trend(ss.trend_range, height=320, key="analytics_trend")
    c = st.columns(5)
    c[0].metric("People detected", ss.people_count)
    c[1].metric("Density", f"{ss.density_pct}%")
    c[2].metric("Current zone", ss.current_zone)
    c[3].metric("Video status", ss.video_status)
    c[4].metric("Frames processed", ss.video_processed_frames)


# ==================== PAGES ====================

with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div class="brandicon">👥</div>
        <div><div class="brandname">Smart <span>CrowdFlow</span></div>
        <div class="sub">AI Crowd Monitoring System</div></div>
    </div>""", unsafe_allow_html=True)
    sidebar_panel()
    if st.button("⏻  Log out", use_container_width=True, key="logout"):
        auth_sms.logout(ss.user["phone"])
        release_video()
        ss.video_status = "Stopped"
        ss.user = None
        ss.page = "Overview"
        st.rerun()

top_bar()

page = ss.page

if page in ("Overview", "Live Monitoring"):
    if ss.focus:
        header_row()
        video_panel()
    else:
        left, right = st.columns([2.55, 0.95], gap="medium")
        with left:
            header_row()
            video_panel()
        with right:
            side_panel()
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        kpi_row()
        if page == "Overview":
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            bottom_row()

elif page == "Density Map":
    st.markdown('<div class="pagetitle">Density Map</div>', unsafe_allow_html=True)
    with st.container(border=True):
        big_heatmap()

elif page == "Analytics":
    st.markdown('<div class="pagetitle">Analytics</div>', unsafe_allow_html=True)
    with st.container(border=True):
        opts = ["Session", "Last 5 min", "Last 1 min"]
        ss.trend_range = st.selectbox("Range", opts, index=opts.index(ss.trend_range), key="analytics_range")
        analytics_panel()

elif page == "Alerts":
    st.markdown('<div class="pagetitle">Alerts</div>', unsafe_allow_html=True)
    with st.container(border=True):
        alert_rows(ss.alert_history)
        if ss.alert_history and st.button("Clear all alerts"):
            ss.alert_history = []
            st.rerun()

elif page == "Reports":
    st.markdown('<div class="pagetitle">Reports</div>', unsafe_allow_html=True)
    with st.container(border=True):
        report_df = pd.DataFrame(ss.alert_history)
        if not report_df.empty:
            st.dataframe(report_df, use_container_width=True, hide_index=True)
        else:
            st.info("No alerts recorded in this session yet.")
        st.download_button("Download report CSV", data=report_df.to_csv(index=False),
                           file_name="crowdflow_report.csv", mime="text/csv", key="download_report")
        if ss.history:
            hist_df = pd.DataFrame(ss.history)
            st.download_button("Download density history CSV", data=hist_df.to_csv(index=False),
                               file_name="crowdflow_density_history.csv", mime="text/csv",
                               key="download_history")

else:  # Settings
    st.markdown('<div class="pagetitle">Settings</div>', unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    with s1:
        with st.container(border=True):
            st.markdown("#### Detection thresholds")
            ss.thr_low = st.slider("LOW up to (people)", 1, 200, int(ss.thr_low))
            ss.thr_high = st.slider("MEDIUM up to (people) — above is HIGH",
                                    int(ss.thr_low) + 1, 500, max(int(ss.thr_high), int(ss.thr_low) + 1))
            ss.capacity = st.slider("People count that equals 100% density", 10, 1000, int(ss.capacity))
            st.caption("Applies from the next processed frame.")
        with st.container(border=True):
            st.markdown("#### System")
            st.write("Video input:", "Found" if current_video().exists() else "Not found")
            st.write("Video path:", str(current_video()))
            st.write("YOLO weights:", "Found" if MODEL_PATH.exists() else "Not found")
            st.write("Tracker configuration:", "Found" if TRACKER_PATH.exists() else "Default tracker")
    with s2:
        with st.container(border=True):
            st.markdown("#### SMS alerts")
            ss.sms_auto = st.toggle("Send SMS automatically on risk alerts", value=ss.sms_auto)
            lv_opts = ["HIGH only", "HIGH and MEDIUM"]
            ss.sms_levels = st.selectbox("Send for", lv_opts, index=lv_opts.index(ss.sms_levels))
            ss.sms_cooldown = st.slider("Minimum seconds between SMS batches", 30, 3600, int(ss.sms_cooldown))

            me = auth_sms.get_user(ss.user["phone"])
            mine_on = st.toggle(f"Alerts to my number ({auth_sms.mask(ss.user['phone'])})",
                                value=bool(me and me.get("alerts", True)))
            if me and mine_on != me.get("alerts", True):
                auth_sms.set_alerts(ss.user["phone"], mine_on)

            recipients, registered, logged = auth_sms.build_recipients()
            st.caption(f"{len(registered)} registered · {len(logged)} logged in · "
                       f"{len(recipients)} will receive alerts (registered first, then logged-in).")
            st.dataframe([{"Order": i + 1, "Number": auth_sms.mask(p),
                           "Source": "Registered" if p in registered else "Logged in"}
                          for i, p in enumerate(recipients)],
                         use_container_width=True, hide_index=True)

            if st.button("Send Test SMS", key="send_sms_test"):
                ok, msg = auth_sms.send_test(ss.people_count)
                (st.success if ok else st.error)(msg)

            log = auth_sms.get_log()
            if log:
                st.markdown("##### SMS log")
                st.dataframe(log, use_container_width=True, hide_index=True)

st.markdown("""
<div style="text-align:center;color:#51677a;font-size:10px;padding-top:14px">
    Smart CrowdFlow · AI Crowd Monitoring System
</div>""", unsafe_allow_html=True)