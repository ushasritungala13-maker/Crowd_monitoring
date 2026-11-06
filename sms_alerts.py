import os
import re
from datetime import datetime

import requests

API_URL = "https://www.fast2sms.com/dev/bulkV2"


def _ten_digit(number):
    digits = re.sub(r"\D", "", number or "")
    return digits[-10:] if len(digits) >= 10 else None


def send_sms_to_recipients(message, recipients=None):
    api_key = os.getenv("FAST2SMS_API_KEY", "").strip()

    if recipients is None:
        recipients = os.getenv("CROWDFLOW_RECIPIENTS", "").split(",")

    numbers = list(dict.fromkeys(
        n for n in (_ten_digit(r) for r in recipients) if n
    ))

    if not api_key:
        return [{"recipient": "CONFIGURATION", "status": "FAILED",
                 "detail": "FAST2SMS_API_KEY is not set."}]
    if not numbers:
        return [{"recipient": "CONFIGURATION", "status": "FAILED",
                 "detail": "No valid recipient numbers."}]

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        resp = requests.post(
            API_URL,
            headers={"authorization": api_key, "Content-Type": "application/json"},
            json={"route": "q", "message": message,
                  "numbers": ",".join(numbers), "flash": 0},
            timeout=20,
        )
        data = resp.json()
        ok = bool(data.get("return"))
        detail = str(data.get("message", data))[:200]
        sid = str(data.get("request_id", ""))
    except Exception as error:
        ok, detail, sid = False, str(error)[:200], ""

    return [{
        "time": now,
        "recipient": n,
        "status": "SUBMITTED" if ok else "FAILED",
        "message_sid": sid,
        "detail": detail,
    } for n in numbers]


def send_crowdflow_sms(reason, people_count, zone, confidence=None, recipients=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if reason == "RED_ZONE":
        message = (f"SMART CROWDFLOW: HIGH CROWD ALERT\nTime: {timestamp}\n"
                   f"People count: {people_count}\nZone: {zone}\n"
                   "Please verify CCTV and follow safety procedures.")
    elif reason == "YELLOW_ZONE":
        message = (f"SMART CROWDFLOW: CROWD BUILDING UP\nTime: {timestamp}\n"
                   f"People count: {people_count}\nZone: {zone}\n"
                   "Please monitor the CCTV.")
    elif reason == "POSSIBLE_FIGHT":
        if confidence is None:
            return []
        message = (f"SMART CROWDFLOW: POSSIBLE FIGHT ALERT\nTime: {timestamp}\n"
                   f"People count: {people_count}\nConfidence: {confidence:.0%}\n"
                   "Please verify the CCTV footage.")
    else:
        return []

    return send_sms_to_recipients(message, recipients)