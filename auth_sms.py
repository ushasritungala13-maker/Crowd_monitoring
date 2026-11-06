"""Registration, login and automatic SMS dispatch for Smart CrowdFlow."""

import hashlib
import hmac
import inspect
import json
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

USERS_FILE = Path(__file__).resolve().parent / "crowdflow_users.json"
SESSION_TTL = 1800  # seconds without activity before a user no longer counts as "logged in"
_lock = threading.Lock()


@st.cache_resource
def _shared():
    """State shared by every browser session on this server."""
    return {"logged_in": {}, "sms_log": [], "last_ts": 0.0}


# ---------------------------------------------------------------- phone / users

def normalize_phone(raw: str):
    p = re.sub(r"[^\d+]", "", (raw or "").strip())
    if not p:
        return None
    if not p.startswith("+"):
        p = "+91" + p if len(p) == 10 else "+" + p
    return p if re.fullmatch(r"\+\d{10,15}", p) else None


def mask(phone: str) -> str:
    return phone[:3] + "•" * max(0, len(phone) - 7) + phone[-4:]


def load_users():
    with _lock:
        if not USERS_FILE.exists():
            return []
        try:
            return json.loads(USERS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []


def _save_users(users):
    with _lock:
        tmp = USERS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(users, indent=2), encoding="utf-8")
        tmp.replace(USERS_FILE)


def _hash(password: str, salt_hex: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), 200_000).hex()


def register(name, phone_raw, password, wants_alerts=True):
    phone = normalize_phone(phone_raw)
    if not name.strip():
        return False, "Enter your name."
    if not phone:
        return False, "Enter a valid mobile number, e.g. +919876543210."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    users = load_users()
    if any(u["phone"] == phone for u in users):
        return False, "This number is already registered. Please log in."
    salt = os.urandom(16).hex()
    users.append({"name": name.strip(), "phone": phone, "salt": salt,
                  "hash": _hash(password, salt), "alerts": bool(wants_alerts),
                  "registered_at": datetime.now().isoformat(timespec="seconds")})
    _save_users(users)
    return True, "Registered. You can log in now."


def authenticate(phone_raw, password):
    phone = normalize_phone(phone_raw)
    for u in load_users():
        if u["phone"] == phone and hmac.compare_digest(u["hash"], _hash(password, u["salt"])):
            return u
    return None


def set_alerts(phone, enabled: bool):
    users = load_users()
    for u in users:
        if u["phone"] == phone:
            u["alerts"] = bool(enabled)
    _save_users(users)


def get_user(phone):
    return next((u for u in load_users() if u["phone"] == phone), None)


# ---------------------------------------------------------------- logged-in tracking

def heartbeat(phone):
    _shared()["logged_in"][phone] = time.time()


def logout(phone):
    _shared()["logged_in"].pop(phone, None)


def logged_in_numbers():
    now = time.time()
    return [p for p, t in _shared()["logged_in"].items() if now - t <= SESSION_TTL]


def build_recipients():
    """All registered numbers (alerts on) first, then logged-in numbers not already included."""
    registered = [u["phone"] for u in load_users() if u.get("alerts", True)]
    logged = logged_in_numbers()
    seen, ordered = set(), []
    for p in registered + logged:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return ordered, registered, logged


# ---------------------------------------------------------------- login screen

def require_login():
    ss = st.session_state
    if ss.get("user"):
        heartbeat(ss.user["phone"])
        return

    _, mid, _ = st.columns([1, 1.3, 1])
    with mid:
        st.markdown(
            '<div style="text-align:center;font-size:30px;font-weight:800;margin-top:60px">'
            'Smart <span style="color:#48d48b">CrowdFlow</span></div>'
            '<div style="text-align:center;color:#91a4b8;margin-bottom:18px">'
            'Log in to monitor crowds and receive risk alerts by SMS</div>',
            unsafe_allow_html=True)
        tab_in, tab_up = st.tabs(["Log in", "Register"])

        with tab_in:
            with st.form("login_form"):
                phone = st.text_input("Mobile number", placeholder="+919876543210")
                pw = st.text_input("Password", type="password")
                if st.form_submit_button("Log in", use_container_width=True):
                    user = authenticate(phone, pw)
                    if user:
                        ss.user = {"name": user["name"], "phone": user["phone"]}
                        heartbeat(user["phone"])
                        st.rerun()
                    else:
                        st.error("Invalid number or password.")

        with tab_up:
            with st.form("register_form"):
                name = st.text_input("Full name")
                phone = st.text_input("Mobile number", placeholder="+919876543210", key="reg_phone")
                pw = st.text_input("Password (min 6 characters)", type="password", key="reg_pw")
                pw2 = st.text_input("Confirm password", type="password")
                consent = st.checkbox("I agree to receive crowd-risk alert SMS on this number", value=True)
                if st.form_submit_button("Register", use_container_width=True):
                    if pw != pw2:
                        st.error("Passwords do not match.")
                    else:
                        ok, msg = register(name, phone, pw, consent)
                        (st.success if ok else st.error)(msg)
    st.stop()


# ---------------------------------------------------------------- SMS dispatch

def dispatch(zone, count, level, recipients):
    """Calls your sms_alerts.send_crowdflow_sms, passing the recipient list."""
    from sms_alerts import send_crowdflow_sms

    params = inspect.signature(send_crowdflow_sms).parameters
    kw = next((n for n in ("recipients", "numbers", "phone_numbers", "to_numbers") if n in params), None)
    if kw is None:
        raise RuntimeError("send_crowdflow_sms() must accept a 'recipients' list parameter.")
    res = send_crowdflow_sms(zone, count, level, **{kw: recipients})
    if not isinstance(res, list):
        res = [res]
    return [r if isinstance(r, dict) else {"result": str(r)} for r in res]


def _log(level, n, status, detail):
    sh = _shared()
    with _lock:
        sh["sms_log"].insert(0, {"Time": datetime.now().strftime("%H:%M:%S"), "Level": level,
                                 "Recipients": n, "Status": status, "Detail": str(detail)[:200]})
        del sh["sms_log"][100:]


def get_log():
    return list(_shared()["sms_log"])

def _summarize(results):
    if not isinstance(results, list):
        return "SUBMITTED", str(results)
    failed = [r for r in results if str(r.get("status", "")).upper() == "FAILED"]
    if failed:
        return "FAILED", "; ".join(
            f'{r.get("recipient", "?")}: {r.get("detail", "")}' for r in failed)[:300]
    return "SUBMITTED", ", ".join(
        f'{r.get("recipient", "?")} sid={r.get("message_sid", "")}' for r in results)[:300]

def send_alert_async(zone, count, level, cooldown_s):
    """Background send so video processing never blocks. Cooldown is tracked per level."""
    sh = _shared()
    recipients, _, _ = build_recipients()
    if not recipients:
        return "no recipients"
    with _lock:
        if not isinstance(sh.get("last_ts"), dict):
            sh["last_ts"] = {}
        if time.time() - sh["last_ts"].get(level, 0.0) < cooldown_s:
            return "cooldown"
        sh["last_ts"][level] = time.time()

    def work():
        try:
            status, detail = _summarize(dispatch(zone, count, level, recipients))
            _log(level, len(recipients), status, detail)
        except Exception as exc:
            _log(level, len(recipients), "FAILED", exc)

    threading.Thread(target=work, daemon=True).start()
    return "queued"

def send_test(count):
    recipients, _, _ = build_recipients()
    if not recipients:
        return False, "No recipients registered yet."
    try:
        status, detail = _summarize(dispatch("RED_ZONE", count, "RED", recipients))
        _log("TEST", len(recipients), status, detail)
        if status == "FAILED":
            return False, f"Twilio rejected the message: {detail}"
        return True, f"Accepted by Twilio for {len(recipients)} number(s). Check Twilio logs for delivery."
    except Exception as exc:
        _log("TEST", len(recipients), "FAILED", exc)
        return False, f"SMS test failed: {exc}"