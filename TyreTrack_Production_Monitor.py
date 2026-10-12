# TYRETRACK_YELLOW_LINE_SMALL_MP4_NO_YARD_V2
# TYRECYCLE | TYRETRACK
# Windows 11 conveyor tyre counter
# OAK stream/count update 2026-10-08: independent MP4 writer; matched-frame
# AI view; confidence-gated track continuation; stable-side crossings.
# Reduced OAK stream: 640x360 at 20 FPS; MP4 and preview match 20 FPS.
# Requires on-site validation; does not change detector weights.

import configparser
import csv
import math
import numpy as np
import os
import shutil
import smtplib
import sqlite3
import sys
import threading
import queue
import time
import traceback
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_PORTABLE_ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
os.environ["YOLO_CONFIG_DIR"] = str(_PORTABLE_ROOT / "data" / "detector-settings")
(_PORTABLE_ROOT / "data" / "detector-settings").mkdir(parents=True, exist_ok=True)



# Keep DLL search-directory handles alive for the process lifetime.
_DLL_DIR_HANDLES = []
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    _site_libraries = Path(sys.executable).resolve().parent / "Lib" / "site-packages"
    if _site_libraries.is_dir():
        _DLL_DIR_HANDLES.append(os.add_dll_directory(str(_site_libraries)))


def bootstrap_frozen_dlls():
    if sys.platform != "win32" or not getattr(sys, "frozen", False):
        return

    base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    exe_dir = Path(sys.executable).parent
    candidates = (
        base,
        base / "torch" / "lib",
        base / "cv2",
        exe_dir / "_internal",
        exe_dir / "_internal" / "torch" / "lib",
    )

    seen = set()
    for directory in candidates:
        try:
            directory = directory.resolve()
            key = str(directory).lower()
            if directory.exists() and key not in seen:
                seen.add(key)
                os.environ["PATH"] = (
                    str(directory)
                    + os.pathsep
                    + os.environ.get("PATH", "")
                )
                if hasattr(os, "add_dll_directory"):
                    _DLL_DIR_HANDLES.append(
                        os.add_dll_directory(str(directory))
                    )
        except Exception:
            pass


bootstrap_frozen_dlls()

import cv2
from ultralytics import YOLO
from count_line import segment_bounds, crosses_segment
from hourly_dashboard import hourly_snapshot, brisbane_now

from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

try:
    import depthai as dai
    DEPTHAI_OK = True
except Exception:
    dai = None
    DEPTHAI_OK = False


APP_NAME = "TYRECYCLE | TYRETRACK"
MODEL_NAME = "tyre_detector.pt"
OAK_NAME = "OAK-D S2 PoE / CAT5"

IOU_DEFAULT = 0.50
INFERENCE_SIZE = 480

# V4.8 counting configuration
V48_HYSTERESIS_PX = 5
V48_MAX_TRACK_GAP = 20
V48_AGNOSTIC_NMS = True


# Target up to 20 live-preview updates/second, independently of inference.
OAK_PREVIEW_INTERVAL_MS = 50

# Draw the detector geometry faithfully; cosmetic shrinking does not improve detection.
VISUAL_BOX_INSET = 0.0

# Existing conservative conveyor-sized false-positive guard.
MAX_BOX_WIDTH_RATIO = 0.68
MAX_BOX_HEIGHT_RATIO = 0.70
MAX_BOX_AREA_RATIO = 0.32
MAX_BOX_ASPECT_RATIO = 3.60

RECORD_OAK_VIDEO = False
_EMAIL_LOCK = threading.Lock()
_OAK_IDLE_LOCK = threading.Lock()
_OAK_IDLE_SECONDS = 10 * 60
_OAK_IDLE = {
    "last_frame": None,
    "since": None,
    "last_count_id": None,
    "sent": False,
    "retry_after": 0.0,
}


def app_dir():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def resource_path(name):
    bundled = Path(getattr(sys, "_MEIPASS", app_dir())) / name
    return bundled if bundled.exists() else app_dir() / name


DATA = app_dir() / "data"
REPORTS = app_dir() / "reports"
OAK_RECORDINGS = DATA / "oak_recordings"
PRIVATE_CONFIG_DIR = DATA / "private_config"

for directory in (
    DATA,
    REPORTS,
    OAK_RECORDINGS,
    PRIVATE_CONFIG_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)

DB = DATA / "tyretrack.db"
LOG = DATA / "tyretrack.log"
EMAIL_INI = PRIVATE_CONFIG_DIR / "email_config.ini"


def log(message):
    try:
        with LOG.open("a", encoding="utf-8") as file:
            file.write(
                f"{datetime.now().isoformat(timespec='seconds')} "
                f"{message}\n"
            )
    except Exception:
        pass


def init_db():
    with sqlite3.connect(DB, timeout=15) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS counts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                source TEXT,
                track_id INTEGER,
                confidence REAL
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS state (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )

        columns = {
            row[1] for row in con.execute("PRAGMA table_info(counts)")
        }
        if "metric_utc" not in columns:
            con.execute("ALTER TABLE counts ADD COLUMN metric_utc REAL")
        if "tyre_type" not in columns:
            con.execute(
                "ALTER TABLE counts "
                "ADD COLUMN tyre_type TEXT DEFAULT 'unknown'"
            )
        if "class_confidence" not in columns:
            con.execute(
                "ALTER TABLE counts "
                "ADD COLUMN class_confidence REAL DEFAULT 0"
            )


def get_state(key, default=""):
    with sqlite3.connect(DB, timeout=15) as con:
        row = con.execute(
            "SELECT value FROM state WHERE key=?",
            (key,),
        ).fetchone()
    return row[0] if row else default


def set_state(key, value):
    with sqlite3.connect(DB, timeout=15) as con:
        con.execute(
            """
            INSERT INTO state(key,value) VALUES(?,?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            (key, str(value)),
        )


def db_count_since(start):
    with sqlite3.connect(DB, timeout=15) as con:
        return con.execute(
            "SELECT COUNT(*) FROM counts WHERE ts>=?",
            (start.isoformat(),),
        ).fetchone()[0]


def db_count_between(start, end):
    """Count persisted events in the clock hour, without clearing history."""
    with sqlite3.connect(DB, timeout=15) as con:
        return con.execute(
            "SELECT COUNT(*) FROM counts WHERE ts>=? AND ts<?",
            (start.isoformat(), end.isoformat()),
        ).fetchone()[0]


def db_total():
    with sqlite3.connect(DB, timeout=15) as con:
        return con.execute(
            "SELECT COUNT(*) FROM counts"
        ).fetchone()[0]



from collections import defaultdict

def tyre_class_name(name):
    name = str(name).strip().lower()

    if name == "passenger":
        return "passenger"

    if name in ("4x4", "suv", "suv/4x4", "suv / 4x4"):
        return "4x4"

    return "unknown"


def tyre_class_vote(votes):
    if not votes:
        return "unknown", 0.0

    passenger = float(votes.get("passenger", 0))
    suv = float(votes.get("4x4", 0))

    total = passenger + suv

    if total <= 0:
        return "unknown", 0.0

    if passenger >= suv:
        return "passenger", passenger / total

    return "4x4", suv / total


def classification_counts(start=None, end=None, source=None):
    conditions = []
    params = []

    if start is not None:
        conditions.append("ts >= ?")
        params.append(start.isoformat())

    if end is not None:
        conditions.append("ts < ?")
        params.append(end.isoformat())

    if source is not None:
        conditions.append("source = ?")
        params.append(source)

    where = (
        " WHERE " + " AND ".join(conditions)
        if conditions else ""
    )

    with sqlite3.connect(DB, timeout=15) as con:
        rows = con.execute(
            "SELECT tyre_type, COUNT(*) FROM counts"
            + where
            + " GROUP BY tyre_type",
            params
        ).fetchall()

    result = {
        "passenger": 0,
        "4x4": 0,
        "unknown": 0
    }

    for name, count in rows:
        result[tyre_class_name(name)] += count

    return result


def log_count(
    source,
    track_id,
    confidence,
    tyre_type="unknown",
    class_confidence=0.0
):
    tyre_type = tyre_class_name(tyre_type)

    with sqlite3.connect(DB, timeout=15) as con:
        con.execute(
            """
            INSERT INTO counts(
                ts,source,track_id,confidence,
                tyre_type,class_confidence,metric_utc
            )
            VALUES(?,?,?,?,?,?,?)
            """,
            (
                datetime.now().isoformat(),
                source,
                int(track_id),
                float(confidence),
                tyre_type,
                float(class_confidence),
                time.time()
            )
        )



def export_classification_csv(day=None):
    """Export hourly Passenger / SUV / Unknown totals."""

    if day is None:
        day = datetime.now().date()

    start = datetime.combine(
        day,
        datetime.min.time()
    )

    end = start + timedelta(days=1)

    output = REPORTS / (
        f"TyreTrack_Classification_{day:%Y%m%d}.csv"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with sqlite3.connect(DB, timeout=15) as con:
        rows = con.execute(
            """
            SELECT
                substr(ts, 12, 2) AS hour,
                tyre_type,
                COUNT(*)
            FROM counts
            WHERE ts >= ? AND ts < ?
            GROUP BY hour, tyre_type
            ORDER BY hour, tyre_type
            """,
            (
                start.isoformat(),
                end.isoformat()
            )
        ).fetchall()

    hours = {}

    for hour, tyre_type, count in rows:
        if hour not in hours:
            hours[hour] = {
                "passenger": 0,
                "4x4": 0,
                "unknown": 0
            }

        category = tyre_class_name(tyre_type)
        hours[hour][category] += count

    with open(
        output,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Date",
            "Hour",
            "Passenger",
            "SUV / 4x4",
            "Unknown",
            "Total"
        ])

        for hour, totals in sorted(hours.items()):
            passenger = totals["passenger"]
            suv = totals["4x4"]
            unknown = totals["unknown"]

            writer.writerow([
                str(day),
                hour,
                passenger,
                suv,
                unknown,
                passenger + suv + unknown
            ])

    return output


def export_daily_csv(day=None):
    """Total-only daily report; retain all stored count records."""
    if day is None:
        day = datetime.now().date()
    elif isinstance(day, str):
        day = datetime.fromisoformat(day).date()

    start = datetime.combine(day, datetime.min.time())
    end = start + timedelta(days=1)

    with sqlite3.connect(DB, timeout=15) as con:
        rows = con.execute(
            "SELECT ts FROM counts WHERE ts>=? AND ts<?",
            (start.isoformat(), end.isoformat()),
        ).fetchall()

    hourly = [0] * 24
    for (timestamp,) in rows:
        try:
            hourly[datetime.fromisoformat(timestamp).hour] += 1
        except ValueError:
            log(f"Invalid count timestamp: {timestamp!r}")

    path = REPORTS / f"TyreTrack_{day.isoformat()}.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["Date", "Hour", "Total"])
        for hour, count in enumerate(hourly):
            if count:
                writer.writerow(
                    [
                        day.strftime("%d/%m/%Y"),
                        f"{hour:02d}:00-{hour:02d}:59",
                        count,
                    ]
                )
        writer.writerow([])
        writer.writerow(["TOTAL", "", sum(hourly)])

    return path


def ensure_email_config():
    if not EMAIL_INI.exists():
        EMAIL_INI.write_text(
            "[EMAIL]\n"
            "sender_email = your_gmail@gmail.com\n"
            "sender_app_password = your_app_password\n"
            "recipient_email = recipient@example.com\n",
            encoding="utf-8",
        )


def email_settings():
    ensure_email_config()
    config = configparser.ConfigParser(interpolation=None)
    config.read(EMAIL_INI, encoding="utf-8")

    sender = config.get(
        "EMAIL", "sender_email", fallback=""
    ).strip()
    password = config.get(
        "EMAIL", "sender_app_password", fallback=""
    ).replace(" ", "")
    recipient = config.get(
        "EMAIL", "recipient_email", fallback=""
    ).strip()

    if (
        not sender
        or "your_" in sender
        or not password
        or "your_" in password
        or not recipient
        or "example.com" in recipient
    ):
        return None

    return sender, password, recipient


def send_message(subject, body, snapshot=None):
    settings = email_settings()
    if settings is None:
        log(f"Email waiting for configuration: {EMAIL_INI}")
        return False

    sender, password, recipient = settings
    message = MIMEMultipart() if snapshot and Path(snapshot).is_file() else MIMEText(body)
    if message.is_multipart():
        message.attach(MIMEText(body))
        with open(snapshot, "rb") as picture:
            attachment = MIMEImage(picture.read(), _subtype="png")
        attachment.add_header("Content-Disposition", "attachment", filename="TyreTrack_Dashboard.png")
        message.attach(attachment)
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = recipient

    try:
        with smtplib.SMTP_SSL(
            "smtp.gmail.com", 465, timeout=20
        ) as smtp:
            smtp.login(sender, password)
            smtp.send_message(message)
        log(f"Gmail accepted: {subject}")
        return True
    except Exception as exc:
        log(
            f"Gmail failed: {type(exc).__name__}: {exc}"
        )
        return False


DASHBOARD_SNAPSHOT = DATA / "dashboard_snapshot.png"

def production_snapshot(now=None):
    """Single Brisbane-time source for dashboard and reporting. OAK events only."""
    now = (now or brisbane_now()).astimezone(brisbane_now().tzinfo)
    day_start = now.replace(hour=5, minute=0, second=0, microsecond=0)
    cutoff = day_start.replace(hour=17, minute=5)
    effective = min(now, cutoff)
    with sqlite3.connect(DB, timeout=15) as con:
        slots, best, _ = hourly_snapshot(con, source=OAK_NAME, now=effective)
        rows = con.execute("SELECT ts, metric_utc FROM counts WHERE source=? ORDER BY id", (OAK_NAME,)).fetchall()
        all_time = con.execute("SELECT COUNT(*) FROM counts WHERE source=?", (OAK_NAME,)).fetchone()[0]
    events = []
    for ts, epoch in rows:
        try:
            stamp = (datetime.fromtimestamp(epoch, now.tzinfo) if epoch is not None
                     else datetime.fromisoformat(ts).replace(tzinfo=now.tzinfo))
            if day_start <= stamp < cutoff and stamp <= effective:
                events.append(stamp)
        except (ValueError, TypeError, OverflowError):
            continue
    events.sort()
    incidents = 0
    no_flow = 0.0
    if events:
        for previous, following in zip(events, events[1:]):
            gap = (following - previous).total_seconds()
            if gap >= 600:
                incidents += 1
                no_flow += gap
        trailing = max(0, (effective - events[-1]).total_seconds())
        if trailing >= 600:
            incidents += 1
            no_flow += trailing
        status = "CLOSED" if now >= cutoff else "NOT OPEN" if now < day_start else "NO COUNT" if trailing >= 600 else "RUNNING"
    else:
        status = "CLOSED" if now >= cutoff else "NOT OPEN" if now < day_start else "WAITING"
    current = next((slot for slot in slots if slot['start'] and slot['start'] <= effective < slot['end']), None)
    recent = sum(1 for event in events if effective - timedelta(minutes=5) <= event <= effective)
    return dict(day_start=day_start, day_end=cutoff, daily=len(events), all_time=all_time,
                current_hour=current['count'] if current else 0,
                best_count=best['count'] if best else 0,
                no_count_incidents=incidents, no_flow_minutes=no_flow / 60,
                no_count_status=status, flow_per_minute=recent / 5)


def production_report_counts(start, end):
    """One OAK-only database query for hourly, daily and classification counts."""
    brisbane = timezone(timedelta(hours=10))
    def as_brisbane(value):
        return value.replace(tzinfo=brisbane) if value.tzinfo is None else value.astimezone(brisbane)
    start, end = as_brisbane(start), as_brisbane(end)
    day_start = start.replace(hour=5, minute=0, second=0, microsecond=0)
    with sqlite3.connect(DB, timeout=15) as con:
        rows = con.execute(
            "SELECT metric_utc, tyre_type FROM counts "
            "WHERE source=? AND metric_utc>=? AND metric_utc<? ORDER BY metric_utc",
            (OAK_NAME, day_start.timestamp(), end.timestamp())
        ).fetchall()
        all_time = con.execute("SELECT COUNT(*) FROM counts WHERE source=?", (OAK_NAME,)).fetchone()[0]
    hourly_types = {"passenger": 0, "4x4": 0, "unknown": 0}
    daily_types = {"passenger": 0, "4x4": 0, "unknown": 0}
    hourly = 0
    for epoch, tyre_type in rows:
        kind = tyre_class_name(tyre_type)
        daily_types[kind] += 1
        if start.timestamp() <= epoch < end.timestamp():
            hourly += 1
            hourly_types[kind] += 1
    return dict(hourly=hourly, daily=len(rows), all_time=all_time,
                hourly_types=hourly_types, daily_types=daily_types,
                start=start, end=end)


def build_hourly_report(start, end):
    """Pure report formatter; used by live SMTP and the offline validation suite."""
    r = production_report_counts(start, end)
    start, end = r['start'], r['end']
    h, d = r['hourly_types'], r['daily_types']
    period_minutes = (end-start).total_seconds()/60
    flow = r['hourly'] / period_minutes if period_minutes > 0 else 0.0
    body = (
        "TYRECYCLE | TYRETRACK\n\n"
        f"Period: {start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M} (Brisbane time)\n\n"
        f"OAK HOURLY TOTAL: {r['hourly']}\n"
        f"Passenger: {h['passenger']}\n"
        f"SUV / 4x4: {h['4x4']}\n"
        f"Unknown: {h['unknown']}\n"
        f"AVERAGE FLOW: {flow:.2f} tyres/min\n\n"
        f"OAK DAILY TOTAL: {r['daily']}\n"
        f"Passenger: {d['passenger']}\n"
        f"SUV / 4x4: {d['4x4']}\n"
        f"Unknown: {d['unknown']}\n\n"
        f"ALL-TIME TOTAL: {r['all_time']}\n"
        "Video-file and webcam tests are excluded.\n"
        "A zero count does not prove that the camera was online.\n"
    )
    subject = f"TYRETRACK OAK Hourly Count - {start:%Y-%m-%d %H:%M} - {r['hourly']} tyres"
    return subject, body


def send_hourly_email(start, end):
    subject, body = build_hourly_report(start, end)
    return send_message(subject, body, DASHBOARD_SNAPSHOT)


def oak_in_operating_hours():
    now = datetime.now(timezone(timedelta(hours=10)))
    return 5 <= now.hour and (now.hour < 17 or (now.hour == 17 and now.minute < 5))


def oak_stream_stopped():
    with _OAK_IDLE_LOCK:
        _OAK_IDLE.update(
            last_frame=None,
            since=None,
            last_count_id=None,
            sent=False,
            retry_after=0.0,
        )


def oak_frame_received():
    now = time.monotonic()

    if not oak_in_operating_hours():
        oak_stream_stopped()
        return

    with _OAK_IDLE_LOCK:
        if (
            _OAK_IDLE["last_frame"] is None
            or now - _OAK_IDLE["last_frame"] > 30
        ):
            with sqlite3.connect(DB, timeout=15) as con:
                newest_id = con.execute(
                    "SELECT COALESCE(MAX(id),0) FROM counts "
                    "WHERE source=?",
                    (OAK_NAME,),
                ).fetchone()[0]

            _OAK_IDLE["last_count_id"] = newest_id
            _OAK_IDLE["since"] = None  # Begin after first real tyre count.
            _OAK_IDLE["sent"] = False
            _OAK_IDLE["retry_after"] = 0.0

        _OAK_IDLE["last_frame"] = now


def oak_no_count_alert_tick():
    if not oak_in_operating_hours():
        return

    now = time.monotonic()

    with _OAK_IDLE_LOCK:
        last_frame = _OAK_IDLE["last_frame"]
        since = _OAK_IDLE["since"]
        if (
            last_frame is None
            or since is None
            or now - last_frame > 30
        ):
            return

        with sqlite3.connect(DB, timeout=15) as con:
            newest_id = con.execute(
                "SELECT COALESCE(MAX(id),0) FROM counts "
                "WHERE source=?",
                (OAK_NAME,),
            ).fetchone()[0]

        if newest_id != _OAK_IDLE["last_count_id"]:
            _OAK_IDLE["last_count_id"] = newest_id
            _OAK_IDLE["since"] = None  # Begin after first real tyre count.
            _OAK_IDLE["sent"] = False
            _OAK_IDLE["retry_after"] = 0.0
            return

        if (
            _OAK_IDLE["sent"]
            or now - since < _OAK_IDLE_SECONDS
            or now < _OAK_IDLE["retry_after"]
        ):
            return

        _OAK_IDLE["retry_after"] = now + 300

    sent = send_message(
        "TYRETRACK Alert - No OAK counts for 10 minutes",
        "TYRECYCLE | TYRETRACK\n\n"
        "No OAK tyre counts have been recorded for at least "
        "10 minutes while OAK frames were being received.\n\n"
        f"Checked: {datetime.now():%Y-%m-%d %H:%M} "
        "(Brisbane time)\n\n"
        "This does not prove that a tyre passed undetected. "
        "Check the conveyor and the live camera view.\n",
    )

    if sent:
        with _OAK_IDLE_LOCK:
            if newest_id == _OAK_IDLE["last_count_id"]:
                _OAK_IDLE["sent"] = True


def process_scheduled_emails(now):
    """Send 05:00-17:00 hourly reports plus final 17:00-17:05 report.

    `now` is expected in Brisbane local time. Never send outside production
    reporting boundaries; failed sends remain queued for retry.
    """
    if not _EMAIL_LOCK.acquire(blocking=False):
        return
    try:
        if now.tzinfo is not None:
            now = now.astimezone(timezone(timedelta(hours=10))).replace(tzinfo=None)
        oak_no_count_alert_tick()
        today_open = now.replace(hour=5, minute=0, second=0, microsecond=0)
        today_close = today_open.replace(hour=17, minute=5)
        if now < today_open:
            return
        stored = get_state("hourly_next_start", "")
        try:
            cursor = datetime.fromisoformat(stored) if stored else today_open
        except ValueError:
            cursor = today_open
        if cursor < today_open or cursor > today_close:
            cursor = today_open
        for _ in range(13):
            if cursor >= today_close:
                break
            end = min(cursor + timedelta(hours=1), today_close)
            if now < end:
                break
            if not send_hourly_email(cursor, end):
                return
            cursor = end
            set_state("hourly_next_start", cursor.isoformat())
    except Exception:
        log("Hourly scheduler error:\n" + traceback.format_exc())
    finally:
        _EMAIL_LOCK.release()


def set_oak_recording(enabled):
    global RECORD_OAK_VIDEO
    RECORD_OAK_VIDEO = bool(enabled)
    log(
        "OAK recording "
        + ("enabled" if enabled else "disabled")
    )


def acceptable_box(box, width, height):
    x1, y1, x2, y2 = box
    box_width = max(1.0, float(x2 - x1))
    box_height = max(1.0, float(y2 - y1))
    aspect = max(
        box_width / box_height,
        box_height / box_width,
    )

    return not (
        box_width / width > MAX_BOX_WIDTH_RATIO
        or box_height / height > MAX_BOX_HEIGHT_RATIO
        or (box_width * box_height) / (width * height)
        > MAX_BOX_AREA_RATIO
        or aspect > MAX_BOX_ASPECT_RATIO
    )


def draw_detections(frame, detections):
    height, width = frame.shape[:2]

    for x1, y1, x2, y2, track_id, confidence in detections:
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width - 1, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height - 1, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        centre_x = (x1 + x2) // 2
        centre_y = (y1 + y2) // 2

        inset_x = round((x2 - x1) * VISUAL_BOX_INSET)
        inset_y = round((y2 - y1) * VISUAL_BOX_INSET)

        bx1, by1 = x1 + inset_x, y1 + inset_y
        bx2, by2 = x2 - inset_x, y2 - inset_y

        cv2.rectangle(
            frame,
            (bx1, by1),
            (bx2, by2),
            (20, 255, 57),
            1,
        )
        cv2.circle(
            frame,
            (centre_x, centre_y),
            5,
            (0, 0, 255),
            -1,
        )
        cv2.putText(
            frame,
            f"TYRE ID {track_id}  D:{confidence:.2f}",
            (bx1, max(24, by1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (20, 255, 57),
            2,
        )


def draw_line_and_total(
    frame, line_ratio, line_angle, line_length, total, end_ratio=None, line_totals=None
):
    height, width = frame.shape[:2]
    left, right, line_y, slope = segment_bounds(
        width,
        height,
        line_ratio,
        line_angle,
        line_length,
    )
    centre_x = (width - 1) / 2
    y_left = round(line_y + slope * (left - centre_x))
    y_right = round(line_y + slope * (right - centre_x))

    # Yellow in OpenCV's BGR colour order.
    yellow = (0, 255, 255)

    cv2.line(
        frame,
        (round(left), y_left),
        (round(right), y_right),
        yellow,
        4,
    )
    cv2.putText(
        frame,
        "FRONT LINE" if end_ratio is not None else "COUNT LINE",
        (18, max(32, round(line_y) - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        yellow,
        2,
    )
    if end_ratio is not None:
        el,er,ey,es=segment_bounds(width,height,end_ratio,line_angle,line_length)
        cv2.line(frame,(round(el),round(ey+es*(el-centre_x))),
                 (round(er),round(ey+es*(er-centre_x))),(255,200,0),3)
        cv2.putText(frame,"END LINE",(18,max(32,round(ey)-12)),
                    cv2.FONT_HERSHEY_SIMPLEX,.65,(255,200,0),2)
    # Small, content-sized video overlay; counting logic is unchanged.
    total_text = f"TOTAL {total}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    total_width = cv2.getTextSize(total_text, font, 0.55, 1)[0][0]
    box_right = min(width - 8, 18 + total_width)
    box_bottom = 35

    cv2.rectangle(
        frame, (8, 8), (box_right, box_bottom), (0, 0, 0), -1
    )
    cv2.putText(
        frame, total_text, (13, 27), font, 0.55,
        (255, 255, 255), 1, cv2.LINE_AA
    )



class OakSource:
    """
    Capture the newest OAK frame independently of inference.

    If recording is enabled, write received frames as unannotated
    640x360 MP4 segments, each at most 15 minutes long.
    """

    def __init__(self, ip=""):
        if not DEPTHAI_OK:
            raise RuntimeError(
                "DepthAI is not installed. Run INSTALL_DEPTHAI.bat."
            )

        self.device = None
        self.queue = None
        self.reader = None
        self.recorder = None
        self.record_queue = queue.Queue(maxsize=60)
        self.record_drops = 0
        self.last_read_frame_at = None

        self.stop_event = threading.Event()
        self.condition = threading.Condition()
        self.latest_frame = None
        self.latest_frame_at = None
        self.sequence = 0
        self.last_read_sequence = 0
        self.reader_error = None

        self.writer = None
        self.segment_started = None
        self.segment_path = None
        self.segment_frames = 0
        self.capture_fps = 20.0
        self.previous_camera_sequence = None

        pipeline = dai.Pipeline()
        camera = pipeline.create(dai.node.ColorCamera)
        camera.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        camera.setResolution(
            dai.ColorCameraProperties.SensorResolution.THE_1080_P
        )
        camera.setPreviewSize(480, 270)
        camera.setInterleaved(False)
        camera.setFps(20)

        output = pipeline.create(dai.node.XLinkOut)
        output.setStreamName("rgb")
        camera.preview.link(output.input)

        try:
            if ip.strip():
                self.device = dai.Device(
                    pipeline, dai.DeviceInfo(ip.strip())
                )
            else:
                self.device = dai.Device(pipeline)

            self.queue = self.device.getOutputQueue(
                "rgb",
                maxSize=4,
                blocking=False,
            )

            self.reader = threading.Thread(
                target=self._capture_loop,
                name="OAK RGB capture and recorder",
                daemon=True,
            )
            self.recorder = threading.Thread(target=self._record_loop,
                                             name="OAK MP4 writer", daemon=True)
            self.recorder.start()
            self.reader.start()

        except Exception:
            self.release()
            raise

    def _close_segment(self):
        if self.writer is None:
            return

        self.writer.release()
        log(
            f"OAK recording closed: {self.segment_path}; "
            f"frames written: {self.segment_frames}"
        )
        self.writer = None
        self.segment_path = None
        self.segment_frames = 0

    def _open_segment(self, now):
        self._close_segment()

        folder = OAK_RECORDINGS / now.strftime("%Y-%m-%d")
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"OAK_RGB_{now:%Y%m%d_%H%M%S_%f}.mp4"

        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.capture_fps,
            (480, 270),
        )
        if not writer.isOpened():
            raise RuntimeError(
                f"Could not start OAK recording: {path}. "
                "Check disk space and codec support."
            )

        self.writer = writer
        self.segment_started = now
        self.segment_path = path
        self.segment_frames = 0
        log(f"OAK recording started: {path}")

    def _capture_loop(self):
        last_packet_at = time.monotonic()

        try:
            while not self.stop_event.is_set():
                packet = self.queue.tryGet()

                if packet is None:
                    if time.monotonic() - last_packet_at > 6:
                        raise RuntimeError(
                            "No OAK RGB frames received for 6 seconds"
                        )
                    time.sleep(0.005)
                    continue

                last_packet_at = time.monotonic()
                received_at = datetime.now()
                frame = packet.getCvFrame()

                oak_frame_received()

                try:
                    camera_sequence = int(packet.getSequenceNum())
                    previous = self.previous_camera_sequence

                    if (
                        previous is not None
                        and camera_sequence > previous + 1
                    ):
                        log(
                            "OAK recording WARNING: "
                            f"{camera_sequence - previous - 1} "
                            "camera frames were not received"
                        )

                    self.previous_camera_sequence = camera_sequence
                except Exception:
                    pass

                with self.condition:
                    self.latest_frame = frame
                    self.latest_frame_at = time.monotonic()
                    self.sequence += 1
                    self.condition.notify_all()
                if RECORD_OAK_VIDEO:
                    try:
                        self.record_queue.put_nowait((frame, received_at))
                    except queue.Full:
                        self.record_drops += 1
                        if self.record_drops == 1 or self.record_drops % 30 == 0:
                            log(f"OAK MP4 overload: {self.record_drops} recording frames dropped; counting continues")


        except Exception as exc:
            self.reader_error = exc
            log(
                "OAK capture/recording failed:\n"
                + traceback.format_exc()
            )
            with self.condition:
                self.condition.notify_all()

        finally:
            oak_stream_stopped()
            self.stop_event.set()

    def _record_loop(self):
        try:
            while not self.stop_event.is_set() or not self.record_queue.empty():
                try:
                    frame, received_at = self.record_queue.get(timeout=0.1)
                except queue.Empty:
                    if not RECORD_OAK_VIDEO:
                        self._close_segment()
                    continue
                if (self.writer is None or
                        (received_at-self.segment_started).total_seconds() >= 900):
                    self._open_segment(received_at)
                small_frame = cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA)
                self.writer.write(small_frame)
                self.segment_frames += 1
        except Exception:
            log("OAK MP4 writer failed; live counting continues:\n" + traceback.format_exc())
        finally:
            self._close_segment()

    def read(self):
        """For inference: return the newest frame and its sequence."""
        with self.condition:
            self.condition.wait_for(
                lambda: (
                    self.sequence != self.last_read_sequence
                    or self.reader_error is not None
                    or self.stop_event.is_set()
                ),
                timeout=1.0,
            )

            if self.reader_error is not None:
                raise RuntimeError(
                    "OAK capture/recording stopped: "
                    f"{self.reader_error}"
                ) from self.reader_error

            if self.sequence == self.last_read_sequence:
                return False, None, None

            self.last_read_sequence = self.sequence
            self.last_read_frame_at = self.latest_frame_at
            return (
                True,
                self.latest_frame.copy(),
                self.sequence,
            )

    def preview_snapshot(self):
        """For the GUI: one consistent newest-frame snapshot."""
        with self.condition:
            if self.latest_frame is None:
                return None, self.sequence, None

            return (
                self.latest_frame.copy(),
                self.sequence,
                self.latest_frame_at,
            )

    def release(self):
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()

        if self.reader is not None and self.reader.is_alive():
            self.reader.join(timeout=5)

        if self.recorder is not None and self.recorder.is_alive():
            self.recorder.join(timeout=5)
            if self.recorder.is_alive():
                log("WARNING: OAK MP4 writer still finishing queued frames")

        if self.device is not None:
            try:
                self.device.close()
            except Exception:
                pass

        if self.reader is not None and self.reader.is_alive():
            self.reader.join(timeout=2)

        if self.reader is not None and self.reader.is_alive():
            log(
                "WARNING: OAK capture thread did not stop cleanly"
            )



def v48_mask_bottom(mask, box, frame_width, frame_height):
    """
    Return the bottom-most segmentation-mask pixel.

    If the detector has no usable segmentation mask,
    fall back to the bounding-box bottom edge.

    The returned coordinate is in original-frame pixels.
    """
    import numpy as np

    x1, y1, x2, y2 = box

    if mask is None:
        return float(y2)

    try:
        mask = np.asarray(mask)

        if mask.ndim != 2:
            return float(y2)

        if mask.shape != (frame_height, frame_width):
            mask = cv2.resize(
                mask.astype("float32"),
                (frame_width, frame_height),
                interpolation=cv2.INTER_NEAREST
            )

        active_rows = np.where(mask > 0.5)[0]

        if active_rows.size == 0:
            return float(y2)

        return float(active_rows[-1])

    except Exception:
        return float(y2)




def v481_draw_masks(frame, result, opacity=0.38, id_offset=0, show_labels=False):
    """
    Draw genuine segmentation masks and tracking IDs.

    No bounding rectangles are drawn.
    The original frame is modified in place.
    """

    import numpy as np

    if result.masks is None:
        return frame

    height, width = frame.shape[:2]

    masks = result.masks.data.cpu().numpy()

    if result.boxes is None:
        return frame

    boxes = result.boxes.xyxy.cpu().numpy()

    if result.boxes.id is not None:
        ids = [int(v) + id_offset for v in result.boxes.id.int().cpu().tolist()]
    else:
        ids = [-1] * len(masks)

    confidences = result.boxes.conf.cpu().tolist()

    colours = [
        (255, 90, 40),
        (40, 210, 120),
        (220, 80, 220),
        (40, 180, 255),
    ]

    for index, mask in enumerate(masks):

        if index >= len(boxes):
            break

        colour = colours[index % len(colours)]

        if mask.shape != (height, width):
            mask = cv2.resize(
                mask.astype("float32"),
                (width, height),
                interpolation=cv2.INTER_NEAREST
            )

        binary = (mask > 0.5).astype("uint8")

        if not binary.any():
            continue

        # Transparent mask fill.
        frame[binary > 0] = (
            frame[binary > 0].astype("float32")
            * (1.0 - opacity)
            + np.array(colour, dtype="float32")
            * opacity
        ).astype("uint8")

        # Actual tyre outline.
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            frame,
            contours,
            -1,
            colour,
            2
        )

        x1, y1, x2, y2 = boxes[index].astype(int)

        tyre_id = ids[index]
        confidence = confidences[index]

        label = (
            f"TYRE {tyre_id} "
            f"{confidence:.2f}"
        )

        if show_labels:
            cv2.putText(
                frame,
                label,
                (max(0, x1), max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

    return frame


class CounterWorker(QThread):
    frame_ready = pyqtSignal(object)
    counted = pyqtSignal(int, float)
    line_progress = pyqtSignal(int, int, int)
    status = pyqtSignal(str)
    finished_clean = pyqtSignal()

    def __init__(
        self,
        source_kind,
        source,
        direction,
        confidence,
        iou,
        line_ratio=0.59,
        line_angle=0,
        line_length=1.0,
        end_ratio=0.30,
    ):
        super().__init__()

        self.source_kind = source_kind
        self.source = source
        self.direction = direction
        self.confidence = confidence
        self.iou = iou

        self.line_ratio = max(
            0.10, min(0.90, float(line_ratio))
        )
        self.line_angle = max(
            -30, min(30, int(line_angle))
        )
        self.line_length = max(
            0.10, min(1.0, float(line_length))
        )

        self.end_ratio = max(.10,min(.90,float(end_ratio)))
        self.line_ids = [set(), set()]
        self.running = True
        self.session_count = 0
        self.counted_ids = set()
        self.show_tyre_labels = False
        self.class_votes = defaultdict(
            lambda: defaultdict(float)
        )
        self.session_type_counts = {
            "passenger": 0,
            "4x4": 0,
            "unknown": 0
        }
        self.previous_side = {}
        self.previous_x = {}
        self.previous_observation_frame = {}
        self.confirmed_ids = set()
        self.last_seen_frame = {}
        self.processing_frame = 0
        self.video_fps = 20.0
        self.tracker_recoveries = 0
        self.oak_source = None
        self.overlay_mode = "None"

        # V4.8.1 recorded-video review export.
        self.review_writer = None
        self.review_path = None


        self.overlay_lock = threading.Lock()
        self.oak_overlay = None

    def line_totals(self):
        return (len(self.line_ids[0]),len(self.line_ids[1]),
                len(self.line_ids[0] & self.line_ids[1]))

    def stop(self):
        self.running = False

    def latest_oak_overlay(self):
        with self.overlay_lock:
            return self.oak_overlay

    def run(self):
        source = None
        perf_since = time.monotonic()
        perf_frames = 0
        perf_track_seconds = 0.0

        try:
            model_path = resource_path(MODEL_NAME)
            if not model_path.exists():
                raise FileNotFoundError(
                    f"{MODEL_NAME} not found beside the app."
                )

            self.status.emit("Loading tyre detector...")
            model = YOLO(str(model_path))

            if self.source_kind == "oak":
                source = OakSource(str(self.source or ""))
                self.oak_source = source
                source_name = OAK_NAME
            else:
                source = cv2.VideoCapture(self.source)
                if not source.isOpened():
                    raise RuntimeError(
                        "Could not open video source."
                    )
                source_name = (
                    Path(self.source).name
                    if self.source_kind == "video"
                    else f"Camera {self.source}"
                )

            self.status.emit("COUNTING")

            if self.source_kind == "video":
                fps = source.get(cv2.CAP_PROP_FPS)

                if not fps or fps <= 0:
                    fps = 20.0
                self.video_fps = float(fps)

                review_folder = REPORTS / "video_reviews"
                review_folder.mkdir(
                    parents=True,
                    exist_ok=True
                )

                timestamp = datetime.now().strftime(
                    "%Y%m%d_%H%M%S"
                )

                self.review_path = (
                    review_folder
                    / f"TyreTrack_Review_{timestamp}.mp4"
                )

                frame_width = int(
                    source.get(cv2.CAP_PROP_FRAME_WIDTH)
                )

                frame_height = int(
                    source.get(cv2.CAP_PROP_FRAME_HEIGHT)
                )

                self.review_writer = cv2.VideoWriter(
                    str(self.review_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (frame_width, frame_height)
                )

                if not self.review_writer.isOpened():
                    raise RuntimeError(
                        "Could not create review MP4. "
                        "Check codec support and disk space."
                    )

            while self.running:
                if self.source_kind == "oak":
                    ok, frame, frame_sequence = source.read()
                else:
                    ok, frame = source.read()
                    frame_sequence = None

                if not ok:
                    if self.source_kind == "oak":
                        continue
                    break

                self.processing_frame += 1
                height, width = frame.shape[:2]
                left, right, line_y, slope = segment_bounds(
                    width,
                    height,
                    self.line_ratio,
                    self.line_angle,
                    self.line_length,
                )
                centre_x = (width - 1) / 2

                started = time.monotonic()
                try:
                    result = model.track(
                        frame,
                        persist=True,
                        tracker="bytetrack.yaml",
                        # ByteTrack can associate weak detections with established tracks.
                        # Only tracks observed at the user's threshold are allowed to count.
                        conf=min(self.confidence, 0.10),
                        iou=self.iou,
                        imgsz=INFERENCE_SIZE,
                        agnostic_nms=V48_AGNOSTIC_NMS,
                        verbose=False,
                    )[0]
                except np.linalg.LinAlgError as exc:
                    # A corrupt ByteTrack Kalman covariance can poison subsequent frames.
                    # Reset the tracker; never reset completed counts or session totals.
                    self.tracker_recoveries += 1
                    log(f"Tracker numerical failure at frame {self.processing_frame}: {exc}; "
                        f"recovering (attempt {self.tracker_recoveries})")
                    if self.tracker_recoveries > 20:
                        raise RuntimeError("Too many tracker numerical failures; stopping safely") from exc
                    predictor = getattr(model, "predictor", None)
                    trackers = getattr(predictor, "trackers", None)
                    if trackers:
                        for tracker in trackers:
                            tracker.reset()
                    # Track IDs can restart at 1. Namespace them by recovery generation.
                    # Discard old positions/votes to avoid false line crossings.
                    self.previous_side.clear()
                    self.previous_x.clear()
                    self.previous_observation_frame.clear()
                    self.confirmed_ids.clear()
                    self.last_seen_frame.clear()
                    self.class_votes.clear()
                    # Completed line counts remain; new track IDs use a new namespace.
                    # Continue reading frames, including recording this one.
                    if self.review_writer is not None:
                        self.review_writer.write(frame)
                    continue
                perf_track_seconds += (
                    time.monotonic() - started
                )
                perf_frames += 1

                detections = []

                if (
                    result.boxes is not None
                    and result.boxes.id is not None
                ):
                    boxes = result.boxes.xyxy.cpu().numpy()
                    ids = [int(value) + self.tracker_recoveries * 1000000
                           for value in result.boxes.id.int().cpu().tolist()]
                    confidences = (
                        result.boxes.conf.cpu().tolist()
                    )
                    class_ids = (
                        result.boxes.cls.int().cpu().tolist()
                    )

                    # Masks and boxes are aligned by detection index.
                    masks = None
                    if result.masks is not None:
                        masks = result.masks.data.cpu().numpy()

                    for detection_index, (
                        box, track_id, confidence, class_id
                    ) in enumerate(
                        zip(boxes, ids, confidences, class_ids)
                    ):
                        if not acceptable_box(
                            box, width, height
                        ):
                            continue

                        class_name = tyre_class_name(
                            model.names.get(
                                int(class_id), "unknown"
                            )
                        )

                        if (
                            class_name != "unknown"
                            and track_id not in self.counted_ids
                            and confidence >= self.confidence
                        ):
                            self.class_votes[track_id][class_name] += (
                                float(confidence)
                            )

                        x1, y1, x2, y2 = (
                            int(value) for value in box
                        )
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2

                        tyre_mask = None
                        if (
                            masks is not None
                            and detection_index < len(masks)
                        ):
                            tyre_mask = masks[detection_index]

                        counting_y = v48_mask_bottom(
                            tyre_mask,
                            box,
                            width,
                            height
                        )

                        self.last_seen_frame[track_id] = self.processing_frame
                        if confidence >= self.confidence:
                            self.confirmed_ids.add(track_id)
                        detections.append(
                            (
                                x1, y1, x2, y2,
                                int(track_id),
                                float(confidence),
                            )
                        )

                        # Classification label for this tracked tyre.
                        tyre_type, class_score = tyre_class_vote(
                            self.class_votes.get(track_id, {})
                        )

                        if tyre_type == "passenger":
                            display_type = "PASSENGER"
                            display_colour = (255, 180, 40)
                        elif tyre_type == "4x4":
                            display_type = "SUV / 4X4"
                            display_colour = (80, 220, 100)
                        else:
                            display_type = "UNCLASSIFIED"
                            display_colour = (180, 180, 180)

                        label = (
                            f"{display_type} "
                            f"ID:{int(track_id)} "
                            f"vote:{class_score:.0%}"
                        )

                        if self.show_tyre_labels:
                            cv2.putText(
                                frame,
                                label,
                                (max(0, x1), max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.52,
                                display_colour,
                                2,
                                cv2.LINE_AA
                            )



                        changed = False
                        for line_index, ratio in enumerate((self.line_ratio,self.end_ratio)):
                            ll,rr,ly,ls=segment_bounds(width,height,ratio,self.line_angle,self.line_length)
                            side=counting_y-(ly+ls*(cx-centre_x))
                            key=(line_index,track_id)
                            previous=self.previous_side.get(key)
                            margin=V48_HYSTERESIS_PX*math.sqrt(1+ls*ls)
                            stable=abs(side)>=margin
                            crossed=False
                            # Compare only confirmed observations outside the deadband.
                            # Require the same ID to have been seen recently; stale
                            # re-associations must not create phantom crossings.
                            gap=self.processing_frame-self.previous_observation_frame.get(key,-10000)
                            if (stable and previous is not None and gap<=V48_MAX_TRACK_GAP
                                    and track_id in self.confirmed_ids
                                    and track_id not in self.line_ids[line_index]):
                                if self.direction == "Bottom to Top":
                                    crossed=previous>=margin and side<=-margin
                                elif self.direction == "Top to Bottom":
                                    crossed=previous<=-margin and side>=margin
                                else:
                                    crossed=(previous<=-margin and side>=margin) or (previous>=margin and side<=-margin)
                            if crossed:
                                crossed=crosses_segment(previous,side,self.previous_x.get(key,cx),cx,ll,rr)
                            if crossed:
                                if track_id not in self.counted_ids:
                                    # Persist the union count once, before updating any totals.
                                    tyre_type, class_conf = tyre_class_vote(
                                        self.class_votes.get(track_id, {})
                                    )

                                    log_count(
                                        source_name,
                                        track_id,
                                        confidence,
                                        tyre_type,
                                        class_conf
                                    )

                                    self.session_type_counts[tyre_type] += 1
                                    self.counted_ids.add(track_id)
                                    self.session_count += 1
                                    self.counted.emit(track_id,float(confidence))
                                self.line_ids[line_index].add(track_id)
                                changed=True
                            if stable:
                                self.previous_side[key]=side
                                self.previous_x[key]=cx
                                self.previous_observation_frame[key]=self.processing_frame
                        if changed:
                            self.line_progress.emit(*self.line_totals())

                # Retire stale positions after 90 processed frames, not wall time.
                # Keep counted IDs for the entire session to prevent repeat counts.
                for track_id, seen in list(self.last_seen_frame.items()):
                    if self.processing_frame-seen > 90:
                        for index in (0,1):
                            self.previous_side.pop((index,track_id), None)
                            self.previous_x.pop((index,track_id), None)
                            self.previous_observation_frame.pop((index,track_id), None)
                        self.confirmed_ids.discard(track_id)
                        self.last_seen_frame.pop(track_id, None)

                if self.source_kind == "oak":
                    # One immutable snapshot, including the frame
                    # sequence used for inference. An empty result
                    # clears boxes from earlier detections.
                    with self.overlay_lock:
                        self.oak_overlay = (
                            source.last_read_frame_at,
                            frame_sequence,
                            (width, height),
                            tuple(detections),
                            frame.copy(),
                            result,
                        )
                else:
                    if self.overlay_mode == "Segmentation Mask":
                        v481_draw_masks(frame, result)
                    elif self.overlay_mode == "Bounding Boxes":
                        draw_detections(frame, detections)
                    draw_line_and_total(
                        frame,
                        self.line_ratio,
                        self.line_angle,
                        self.line_length,
                        self.session_count,
                        self.end_ratio, self.line_totals(),
                    )
                    if self.review_writer is not None:
                        self.review_writer.write(frame)

                    self.frame_ready.emit(
                        (frame.copy(), time.monotonic())
                    )

                if self.source_kind == "oak":
                    elapsed = time.monotonic() - perf_since
                    if elapsed >= 10:
                        log(
                            "PERF OAK processing: "
                            f"fps={perf_frames / elapsed:.1f}, "
                            "track_avg_ms="
                            f"{1000 * perf_track_seconds / max(1, perf_frames):.0f}"
                        )
                        perf_since = time.monotonic()
                        perf_frames = 0
                        perf_track_seconds = 0.0

            self.status.emit("STOPPED")

        except Exception as exc:
            log(
                "Counter worker failed:\n"
                + traceback.format_exc()
            )
            self.status.emit(f"ERROR: {exc}")

        finally:
            if self.review_writer is not None:
                try:
                    self.review_writer.release()

                    if self.review_path is not None:
                        log(
                            "Review video saved: "
                            + str(self.review_path)
                        )

                except Exception:
                    log(
                        "Review video close failed:\\n"
                        + traceback.format_exc()
                    )

                self.review_writer = None

            if source is not None:
                try:
                    source.release()
                except Exception:
                    log(
                        "Source release failed:\n"
                        + traceback.format_exc()
                    )
            self.finished_clean.emit()


class VideoLabel(QLabel):
    line_dragged = pyqtSignal(float, int, float)
    fullscreen_requested = pyqtSignal()

    def __init__(self):
        super().__init__("NO VIDEO SOURCE")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setStyleSheet(
            "background:#05080d;color:#718096;"
            "border:1px solid #263241;"
            "border-radius:12px;font-size:18px;"
        )

        self._pix = None
        self.line_ratio = 0.59
        self.line_angle = 0
        self.line_length = 1.0
        self.end_line_ratio = .30
        self.editable = True
        self.drag_mode = None
        self.setCursor(Qt.CrossCursor)

    def set_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape

        image = QImage(
            rgb.data,
            width,
            height,
            channels * width,
            QImage.Format_RGB888,
        ).copy()
        self._pix = QPixmap.fromImage(image)
        self._redraw()

    def resizeEvent(self, event):
        self._redraw()
        super().resizeEvent(event)

    def _redraw(self):
        if self._pix is not None:
            self.setPixmap(
                self._pix.scaled(
                    self.size(),
                    Qt.KeepAspectRatio,
                    Qt.FastTransformation,
                )
            )

    def image_geometry(self):
        if self._pix is None:
            return None

        scale = min(
            self.width() / self._pix.width(),
            self.height() / self._pix.height(),
        )
        width = self._pix.width() * scale
        height = self._pix.height() * scale

        return (
            (self.width() - width) / 2,
            (self.height() - height) / 2,
            width,
            height,
            scale,
        )

    def line_points(self):
        geometry = self.image_geometry()
        if geometry is None:
            return None

        ox, oy, _, _, scale = geometry
        left, right, line_y, slope = segment_bounds(
            self._pix.width(),
            self._pix.height(),
            self.line_ratio,
            self.line_angle,
            self.line_length,
        )
        centre_x = (self._pix.width() - 1) / 2

        return (
            ox + left * scale,
            oy + (
                line_y + slope * (left - centre_x)
            ) * scale,
            ox + right * scale,
            oy + (
                line_y + slope * (right - centre_x)
            ) * scale,
        )

    def paintEvent(self, event):
        super().paintEvent(event)

        if not self.editable:
            return

        points = self.line_points()
        if points is None:
            return

        x1, y1, x2, y2 = points
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor("#ffd400"), 3))
        painter.drawLine(
            round(x1),
            round(y1),
            round(x2),
            round(y2),
        )

        for x, y in (
            (x1, y1),
            (x2, y2),
            ((x1 + x2) / 2, (y1 + y2) / 2),
        ):
            painter.drawEllipse(
                round(x) - 7,
                round(y) - 7,
                14,
                14,
            )
        geometry=self.image_geometry()
        if geometry is not None:
            ox,oy,_,_,scale=geometry
            w,h=self._pix.width(),self._pix.height()
            ll,rr,ly,ls=segment_bounds(w,h,self.end_line_ratio,self.line_angle,self.line_length)
            centre=(w-1)/2
            painter.setPen(QPen(QColor("#00c8ff"),3))
            painter.drawLine(round(ox+ll*scale),round(oy+(ly+ls*(ll-centre))*scale),
                             round(ox+rr*scale),round(oy+(ly+ls*(rr-centre))*scale))
        painter.end()

    def mousePressEvent(self, event):
        if (
            not self.editable
            or event.button() != Qt.LeftButton
        ):
            return

        points = self.line_points()
        if points is None:
            return

        x1, y1, x2, y2 = points
        x, y = event.x(), event.y()

        if not x1 <= x <= x2:
            return

        expected = (
            y1 + (y2 - y1)
            * (x - x1)
            / max(1, x2 - x1)
        )
        if abs(y - expected) > 18:
            return

        self.drag_mode = (
            "resize"
            if abs(x - x1) < 22
            or abs(x - x2) < 22
            else "move"
        )

    def mouseMoveEvent(self, event):
        if not self.drag_mode or not self.editable:
            return

        geometry = self.image_geometry()
        if geometry is None:
            return

        ox, oy, width, height, _ = geometry

        if self.drag_mode == "move":
            ratio = (
                event.y()
                - oy
                - math.tan(
                    math.radians(self.line_angle)
                ) * (event.x() - ox - width / 2)
            ) / height
            self.line_ratio = max(
                0.10, min(0.90, ratio)
            )
        else:
            dx = event.x() - ox - width / 2
            if abs(dx) < 15:
                return

            angle = math.degrees(
                math.atan(
                    (
                        event.y()
                        - oy
                        - height * self.line_ratio
                    ) / dx
                )
            )
            self.line_angle = max(
                -30, min(30, round(angle))
            )
            self.line_length = max(
                0.10,
                min(1.0, 2 * abs(dx) / width),
            )

        self.line_dragged.emit(
            self.line_ratio,
            self.line_angle,
            self.line_length,
        )
        self.update()

    def mouseDoubleClickEvent(self, event):
        self.drag_mode = None
        if event.button() == Qt.LeftButton:
            self.fullscreen_requested.emit()

    def mouseReleaseEvent(self, event):
        self.drag_mode = None


class Main(QMainWindow):
    def __init__(self):
        super().__init__()

        init_db()
        ensure_email_config()

        if not get_state("hourly_next_start"):
            boundary = datetime.now().replace(
                minute=0, second=0, microsecond=0
            )
            set_state(
                "hourly_next_start",
                boundary.isoformat(),
            )

        self.email_thread = None
        self.email_retry_at = 0.0
        self.worker = None
        self.video_path = ""
        self.fullscreen_dialog = None
        self.fullscreen_video = None
        self.session = 0
        self.test_events = []  # Video playback seconds at confirmed crossings.

        self.setWindowTitle(
            APP_NAME + " - TOTAL TYRE COUNT [Controls 2026-10-12]"
        )
        self.resize(1450, 850)
        self.build_ui()
        self.refresh_counts()

        self.oak_preview_timer = QTimer(self)
        self.oak_preview_timer.setInterval(
            OAK_PREVIEW_INTERVAL_MS
        )
        self.oak_preview_timer.timeout.connect(
            self.display_oak_preview
        )
        self._oak_preview_sequence = 0
        self._oak_preview_since = time.monotonic()
        self._oak_preview_frames = 0

        self.clock = QTimer(self)
        self.clock.timeout.connect(self.tick)
        self.clock.start(1000)
        self.tick()

    def card(self, heading):
        frame = QFrame()
        frame.setObjectName("card")

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(10)

        title = QLabel(heading)
        title.setObjectName("cardTitle")
        layout.addWidget(title)

        return frame, layout

    def build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)

        outer = QVBoxLayout(root)
        outer.setContentsMargins(12, 8, 12, 10)
        outer.setSpacing(7)

        header = QHBoxLayout()
        title_column = QVBoxLayout()
        title_column.setSpacing(1)

        brand = QLabel("TYRECYCLE  |  TYRETRACK")
        brand.setObjectName("brand")
        subtitle = QLabel("AI CONVEYOR MONITORING  /  PRODUCTION CONTROL")
        subtitle.setObjectName("subtitle")

        title_column.addWidget(brand)
        title_column.addWidget(subtitle)

        self.health = QLabel("● READY")
        self.health.setObjectName("health")

        # Optional official logo: place tyrecycle_logo.png beside this Python file.
        # This also works in a packaged executable via resource_path().
        logo_path = resource_path("tyrecycle_logo.png")
        if logo_path.exists():
            logo_pixmap = QPixmap(str(logo_path))
            if not logo_pixmap.isNull():
                logo_label = QLabel()
                logo_label.setObjectName("tyrecycleLogo")
                logo_label.setPixmap(logo_pixmap.scaled(
                    136, 52, Qt.KeepAspectRatio, Qt.SmoothTransformation
                ))
                logo_label.setFixedSize(140, 54)
                logo_label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
                header.addWidget(logo_label)
        header.addLayout(title_column)
        header.addStretch()

        self.counters_btn = QPushButton("Hide counters")
        self.counters_btn.setToolTip("Show or hide production counters to enlarge the camera")
        self.counters_btn.clicked.connect(self.toggle_counters)

        self.controls_btn = QPushButton("Hide controls")
        self.controls_btn.clicked.connect(
            self.toggle_controls
        )
        fullscreen_btn = QPushButton("Fullscreen video")
        fullscreen_btn.clicked.connect(
            self.open_fullscreen
        )

        header.addWidget(self.counters_btn)
        header.addWidget(self.controls_btn)
        header.addWidget(fullscreen_btn)
        header.addWidget(self.health)
        outer.addLayout(header)

        # Compact totals: existing live classification QLabel objects are reused.
        self.passenger_lbl = QLabel("0")
        self.suv_lbl = QLabel("0")
        self.passenger_lbl.setToolTip("Passenger tyres in the current session; resets with SESSION.")
        self.suv_lbl.setToolTip("SUV / 4x4 tyres in the current session; resets with SESSION.")
        self.truck_lbl = QLabel("0")  # Reserved for future truck classification.
        self.session_lbl = QLabel("0")
        self.no_count_lbl = QLabel("0")
        self.flow_lbl = QLabel("0.0/min")
        self.gap_lbl = QLabel("0 min")
        self.production_status_lbl = QLabel("WAITING")
        self.hour_lbl = QLabel("0")
        self.day_lbl = QLabel("0")
        self.month_lbl = QLabel("0")
        self.total_lbl = QLabel("0")

        def metric_card(name, label, accent=None):
            frame = QFrame()
            frame.setObjectName("metric")
            layout = QVBoxLayout(frame)
            layout.setContentsMargins(10, 3, 10, 3)
            layout.setSpacing(0)
            title = QLabel(name)
            title.setObjectName("metricName")
            label.setObjectName("metricValue")
            label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            if accent:
                label.setStyleSheet("color:" + accent + ";font-size:22px;font-weight:900;")
            layout.addWidget(title)
            layout.addWidget(label)
            return frame

        self.counters_panel = QWidget()
        counters_layout = QVBoxLayout(self.counters_panel)
        counters_layout.setContentsMargins(0, 0, 0, 0)
        counters_layout.setSpacing(6)

        summary = QHBoxLayout()
        summary.setSpacing(8)
        for name, label in (
            ("THIS HOUR", self.hour_lbl),
            ("DAILY TOTAL", self.day_lbl),
            ("ALL-TIME TOTAL", self.total_lbl),
            ("BEST HOUR", self.session_lbl),
        ):
            summary.addWidget(metric_card(name, label), 1)
        counters_layout.addLayout(summary)

        monitor = QHBoxLayout()
        monitor.setSpacing(8)
        for title, label in (("FLOW RATE", self.flow_lbl),
                             ("NO COUNT ALERTS", self.no_count_lbl),
                             ("NO-FLOW TIME", self.gap_lbl),
                             ("PRODUCTION STATUS", self.production_status_lbl)):
            monitor.addWidget(metric_card(title, label, "#FF5C68" if title in ("NO COUNT ALERTS", "NO-FLOW TIME") else None), 1)
        counters_layout.addLayout(monitor)

        classes = QHBoxLayout()
        classes.setSpacing(8)
        classes.addWidget(metric_card("PASSENGER", self.passenger_lbl, "#42B7FF"), 1)
        classes.addWidget(metric_card("SUV / 4X4", self.suv_lbl, "#38D996"), 1)

        counters_layout.addLayout(classes)

        # Permanent production metrics live outside the hideable controls panel.
        production = QHBoxLayout()
        production.setSpacing(8)
        self.production_best = QLabel("0")
        self.production_lowest = QLabel("0")
        for heading, label, colour in (
            ("BEST COMPLETED HOUR", self.production_best, "#38D996"),
            ("LOWEST COMPLETED HOUR", self.production_lowest, "#FF5C68"),
        ):
            tile = metric_card(heading, label, colour)
            label.setStyleSheet("color:"+colour+";font-size:26px;font-weight:900;")
            production.addWidget(tile, 1)
        history_frame = QFrame()
        history_frame.setObjectName("metric")
        history_layout = QVBoxLayout(history_frame)
        history_layout.setContentsMargins(10, 3, 10, 3)
        self.history_heading = QLabel("HOURLY HISTORY  ·  starts 05:00 Brisbane")
        self.history_heading.setObjectName("metricName")
        history_layout.addWidget(self.history_heading)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedHeight(42)
        content = QWidget()
        row = QHBoxLayout(content)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(7)
        self.production_slots = []
        for index in range(24):
            label = QLabel(f"H{index+1}  0")
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumWidth(76)
            row.addWidget(label)
            self.production_slots.append(label)
        scroll.setWidget(content)
        self.production_scroll = scroll
        history_layout.addWidget(scroll)
        production.addWidget(history_frame, 3)
        counters_layout.addLayout(production)
        outer.addWidget(self.counters_panel)

        body = QHBoxLayout()
        outer.addLayout(body, 1)

        left = QVBoxLayout()
        left.setSpacing(6)
        body.addLayout(left, 3)

        self.video = VideoLabel()
        self.video.fullscreen_requested.connect(
            self.open_fullscreen
        )
        self.video.line_dragged.connect(
            self.mouse_line_changed
        )
        left.addWidget(self.video, 1)

        run_row = QHBoxLayout()

        self.start_btn = QPushButton(
            "▶  START COUNTING"
        )
        self.start_btn.setObjectName("start")
        self.start_btn.clicked.connect(self.start)

        self.stop_btn = QPushButton(
            "■  STOP COUNTING"
        )
        self.stop_btn.setObjectName("stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop)

        run_row.addWidget(self.start_btn)
        run_row.addWidget(self.stop_btn)
        left.addLayout(run_row)

        self.status_lbl = QLabel(
            "Ready  •  Choose a source, then start counting"
        )
        self.status_lbl.setObjectName("status")
        left.addWidget(self.status_lbl)

        footer = QHBoxLayout()

        self.all_time_lbl = QLabel("ALL TIME  0")
        self.all_time_lbl.setObjectName("footer")

        self.source_status = QLabel(
            "Source: none selected"
        )
        self.source_status.setObjectName("footer")

        footer.addWidget(self.all_time_lbl)
        footer.addStretch()
        footer.addWidget(self.source_status)
        left.addLayout(footer)

        tabs = QTabWidget()
        self.controls_tabs = tabs
        tabs.setMinimumWidth(315)
        tabs.setMaximumWidth(365)
        body.addWidget(tabs, 1)

        def tab_page(name):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(
                Qt.ScrollBarAlwaysOff
            )

            host = QWidget()
            layout = QVBoxLayout(host)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setSpacing(10)

            scroll.setWidget(host)
            tabs.addTab(scroll, name)
            return layout

        source_tab = tab_page("CAMERA")
        line_tab = tab_page("COUNTING")
        reports_tab = tab_page("REPORTS")
        hours_tab = tab_page("PERFORMANCE")

        source_card, source_layout = self.card(
            "01  CAMERA & VIDEO SOURCE"
        )
        self.source_combo = QComboBox()
        self.source_combo.addItems(
            [
                "Video File",
                "Camera 0",
                "Camera 1",
                OAK_NAME,
            ]
        )
        self.source_combo.currentTextChanged.connect(
            self.update_source_status
        )
        source_layout.addWidget(self.source_combo)

        self.record_oak_checkbox = QCheckBox(
            "Record OAK video (MP4 — uses disk space)"
        )
        self.record_oak_checkbox.setChecked(False)
        self.record_oak_checkbox.toggled.connect(
            set_oak_recording
        )
        source_layout.addWidget(
            self.record_oak_checkbox
        )
        self.overlay_mode_combo = QComboBox()
        self.overlay_mode_combo.addItems(["None", "Bounding Boxes", "Segmentation Mask"])
        self.overlay_mode_combo.setCurrentText("None")
        self.overlay_mode_combo.currentTextChanged.connect(
            self.update_overlay_mode)
        source_layout.addWidget(QLabel("AI detection display (counting stays active)"))
        source_layout.addWidget(self.overlay_mode_combo)

        self.show_tyre_labels = QCheckBox("Show Tyre Labels")
        self.show_tyre_labels.setChecked(False)
        self.show_tyre_labels.setToolTip(
            "Display tyre classification labels and IDs."
        )
        source_layout.addWidget(self.show_tyre_labels)

        self.show_tyre_labels.toggled.connect(
            self.update_tyre_label_visibility
        )

        self.browse_btn = QPushButton(
            "Choose video file…"
        )
        self.browse_btn.clicked.connect(self.browse)
        source_layout.addWidget(self.browse_btn)

        self.file_lbl = QLabel(
            "No video selected"
        )
        self.file_lbl.setObjectName("helper")
        self.file_lbl.setWordWrap(True)
        source_layout.addWidget(self.file_lbl)

        source_tab.addWidget(source_card)
        source_tab.addStretch()

        line_card, line_layout = self.card(
            "02  COUNTING CONFIGURATION"
        )

        self.dir_combo = QComboBox()
        self.dir_combo.addItems(
            [
                "Either Direction",
                "Bottom to Top",
                "Top to Bottom",
            ]
        )
        self.dir_combo.setCurrentText(
            "Bottom to Top"
        )
        line_layout.addWidget(
            QLabel("Crossing direction")
        )
        line_layout.addWidget(self.dir_combo)

        conf_row = QHBoxLayout()
        conf_row.addWidget(
            QLabel("Detection sensitivity")
        )
        conf_row.addStretch()

        self.conf_lbl = QLabel("35%")
        self.conf_lbl.setObjectName("settingValue")
        conf_row.addWidget(self.conf_lbl)
        line_layout.addLayout(conf_row)

        self.conf = QSlider(Qt.Horizontal)
        self.conf.setRange(20, 80)
        self.conf.setValue(35)
        self.conf.valueChanged.connect(
            lambda value: self.conf_lbl.setText(
                f"{value}%"
            )
        )
        line_layout.addWidget(self.conf)

        try:
            saved_position = int(
                float(get_state("line_ratio", "0.59"))
                * 100
            )
        except (ValueError, TypeError):
            saved_position = 59

        try:
            saved_angle = int(
                get_state("line_angle", "0")
            )
        except (ValueError, TypeError):
            saved_angle = 0

        try:
            saved_length = round(
                float(
                    get_state("line_length", "1.0")
                ) * 100
            )
        except (ValueError, TypeError):
            saved_length = 100

        self.line_pos = QSlider(Qt.Horizontal)
        self.line_pos.setRange(10, 90)
        self.line_pos.setValue(
            max(10, min(90, saved_position))
        )

        self.line_angle = QSlider(Qt.Horizontal)
        self.line_angle.setRange(-30, 30)
        self.line_angle.setValue(
            max(-30, min(30, saved_angle))
        )

        self.line_length = QSlider(Qt.Horizontal)
        self.line_length.setRange(10, 100)
        self.line_length.setValue(
            max(10, min(100, saved_length))
        )

        self.line_pos_lbl = QLabel()
        self.line_angle_lbl = QLabel()
        self.line_length_lbl = QLabel()

        for heading, slider, value_label in (
            (
                "Count line — UP / DOWN",
                self.line_pos,
                self.line_pos_lbl,
            ),
            (
                "Count line — ANGLE",
                self.line_angle,
                self.line_angle_lbl,
            ),
            (
                "Count line — LENGTH",
                self.line_length,
                self.line_length_lbl,
            ),
        ):
            line_layout.addWidget(
                QLabel(heading)
            )
            line_layout.addWidget(slider)
            line_layout.addWidget(value_label)
            slider.valueChanged.connect(
                self.line_changed
            )

        line_buttons = QHBoxLayout()
        up = QPushButton("▲ UP")
        down = QPushButton("▼ DOWN")
        reset = QPushButton("RESET LINE")

        up.clicked.connect(
            lambda: self.line_pos.setValue(
                max(10, self.line_pos.value() - 1)
            )
        )
        down.clicked.connect(
            lambda: self.line_pos.setValue(
                min(90, self.line_pos.value() + 1)
            )
        )
        reset.clicked.connect(self.reset_line)

        self.line_buttons = [up, down, reset]
        for button in self.line_buttons:
            line_buttons.addWidget(button)

        self.end_line_pos = QSlider(Qt.Horizontal)
        self.end_line_pos.setRange(10,90)
        try:
            end_position=round(float(get_state("end_line_ratio","0.30"))*100)
        except (ValueError,TypeError):
            end_position=30
        self.end_line_pos.setValue(end_position)
        self.end_line_lbl = QLabel()
        line_layout.addWidget(QLabel("End line position (shares front-line angle and length)"))
        line_layout.addWidget(self.end_line_pos)
        line_layout.addWidget(self.end_line_lbl)
        self.end_line_pos.valueChanged.connect(self.line_changed)
        self.line_totals_lbl = QLabel("Front 0 | End 0 | Both 0 | Final 0")
        self.line_totals_lbl.setWordWrap(True)
        line_layout.addWidget(self.line_totals_lbl)
        line_layout.addLayout(line_buttons)
        line_tab.addWidget(line_card)

        hint = QLabel(
            "Stop counting to adjust the line. "
            "Drag the yellow front line to move it; drag either end "
            "to shorten or tilt it. Set the blue end line with its position slider. "
            "Final counts each ID once at either line; Both shows confirmation at both."
        )
        hint.setWordWrap(True)
        line_tab.addWidget(hint)
        line_tab.addStretch()

        reports_card, reports_layout = self.card(
            "3  RUN & REPORT"
        )

        export_btn = QPushButton(
            "Export today's results"
        )
        export_btn.clicked.connect(
            self.export_results
        )

        self.reset_btn = QPushButton(
            "Reset session totals"
        )
        self.reset_btn.clicked.connect(
            self.reset_session
        )

        recordings_btn = QPushButton(
            "Open OAK recordings folder"
        )
        recordings_btn.clicked.connect(
            self.open_oak_recordings
        )

        reports_layout.addWidget(export_btn)
        reports_layout.addWidget(
            self.reset_btn
        )
        reports_layout.addWidget(
            recordings_btn
        )

        reports_tab.addWidget(reports_card)
        reports_tab.addStretch()

        best_card, best_layout = self.card(
            "BEST COMPLETED PRODUCTION HOUR"
        )
        self.best_hour_period = QLabel()
        self.best_hour_total = QLabel()
        self.best_hour_types = QLabel(
            "OAK tyre counts only"
        )

        for label in (
            self.best_hour_period,
            self.best_hour_total,
            self.best_hour_types,
        ):
            label.setWordWrap(True)
            best_layout.addWidget(label)

        hours_tab.addWidget(best_card)

        lowest_card, lowest_layout = self.card(
            "LOWEST COMPLETED PRODUCTION HOUR"
        )
        self.lowest_hour_period = QLabel()
        self.lowest_hour_total = QLabel()
        self.lowest_hour_note = QLabel(
            "OAK tyre counts only; "
            "zero-count hours excluded"
        )

        for label in (
            self.lowest_hour_period,
            self.lowest_hour_total,
            self.lowest_hour_note,
        ):
            label.setWordWrap(True)
            lowest_layout.addWidget(label)

        hours_tab.addWidget(lowest_card)
        hours_tab.addStretch()

        self.line_changed()
        self.update_source_status()

        self.setStyleSheet(
            """
            QMainWindow,QWidget{
                background:#101b20;color:#f2f6f4;
                font-family:Segoe UI;font-size:14px
            }
            #brand{
                font-size:21px;font-weight:800;
                color:#ffffff;letter-spacing:1px
            }
            #subtitle,#helper,#footer{
                color:#a8bdb7;font-size:12px
            }
            #health{
                color:#8de2a4;font-weight:800;
                padding:9px 15px;background:#183c32;
                border:1px solid #36795d;
                border-radius:12px
            }
            #card{
                background:#1b2d32;
                border:1px solid #2d4050;
                border-radius:14px
            }
            #cardTitle{
                font-size:13px;font-weight:800;
                color:#a9bfd0
            }
            #metric{
                background:#172633;
                border:1px solid #38535a;
                border-radius:12px
            }
            #metricName{
                color:#c2d3ce;font-size:10px;
                font-weight:800
            }
            #metricValue{
                font-size:22px;font-weight:900;
                color:#f7fbff
            }
            #settingValue{
                color:#9ce6aa;font-weight:800
            }
            QPushButton{
                background:#284047;
                border:1px solid #45666a;
                border-radius:9px;padding:10px;
                font-weight:700;text-align:center
            }
            QPushButton:hover{background:#315066}
            QPushButton:disabled{
                color:#819793;background:#202f34;
                border-color:#34464b
            }
            #start{
                background:#287a49;
                border-color:#78c67e;
                color:white;font-size:17px;
                padding:8px
            }
            #start:hover{background:#35945a}
            #stop{
                background:#772f37;
                border-color:#a8535c;
                color:white;font-size:15px;
                padding:8px
            }
            QTabWidget::pane{
                border:1px solid #38535a;
                border-radius:8px
            }
            QTabBar::tab{
                background:#1b2d32;color:#c2d3ce;
                padding:4px 5px;
                font-size:10px;
                min-height:18px
            }
            QTabBar::tab:selected{
                background:#284047;
                color:#9ce6aa
            }
            QComboBox{
                background:#16282c;
                border:1px solid #45666a;
                border-radius:7px;padding:8px
            }
            QSlider::groove:horizontal{
                height:6px;background:#395458;
                border-radius:3px
            }
            QSlider::handle:horizontal{
                background:#9add8b;width:18px;
                margin:-6px 0;border-radius:9px
            }
            #status{
                background:#1b2d32;
                border:1px solid #38535a;
                border-radius:9px;padding:8px;
                color:#edf4f0;font-weight:600
            }
            """
        )

    def update_overlay_mode(self, mode):
        worker = getattr(self, "worker", None)
        if worker is not None:
            worker.overlay_mode = mode
        self._oak_preview_sequence = -1

    def update_tyre_label_visibility(self, checked):
        worker = getattr(self, "worker", None)

        if worker is not None:
            worker.show_tyre_labels = bool(checked)
        self._oak_preview_sequence = -1

    def toggle_counters(self):
        visible = not self.counters_panel.isVisible()
        self.counters_panel.setVisible(visible)
        self.counters_btn.setText("Hide counters" if visible else "Show counters")

    def toggle_controls(self):
        visible = not self.controls_tabs.isVisible()
        self.controls_tabs.setVisible(visible)
        self.controls_btn.setText(
            "Hide controls"
            if visible else "Show controls"
        )

    def open_fullscreen(self):
        if self.fullscreen_dialog is not None:
            self.fullscreen_dialog.raise_()
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(
            "TyreTrack — Fullscreen video"
        )

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(0, 0, 0, 0)

        video = VideoLabel()
        video.editable = False
        video.fullscreen_requested.connect(
            dialog.reject
        )
        layout.addWidget(video, 1)

        close_btn = QPushButton(
            "Return to dashboard  •  "
            "Esc or double-click video"
        )
        close_btn.clicked.connect(
            dialog.reject
        )
        layout.addWidget(close_btn)

        if self.video._pix is not None:
            video._pix = QPixmap(
                self.video._pix
            )
            video._redraw()

        self.fullscreen_dialog = dialog
        self.fullscreen_video = video
        dialog.finished.connect(
            self.fullscreen_closed
        )
        dialog.showFullScreen()

    def fullscreen_closed(self, _result):
        dialog = self.fullscreen_dialog
        self.fullscreen_dialog = None
        self.fullscreen_video = None

        if dialog is not None:
            dialog.deleteLater()

    def display_frame(self, value):
        emitted_at = None
        frame = value

        if isinstance(value, tuple):
            frame, emitted_at = value

        self.video.set_frame(frame)

        if self.fullscreen_video is not None:
            self.fullscreen_video.set_frame(frame)

        if emitted_at is not None:
            if not hasattr(self, "_perf_gui_since"):
                self._perf_gui_since = time.monotonic()
                self._perf_gui_frames = 0
                self._perf_gui_delay_sum = 0.0
                self._perf_gui_delay_max = 0.0

            delay = max(
                0.0,
                time.monotonic() - emitted_at,
            )
            self._perf_gui_frames += 1
            self._perf_gui_delay_sum += delay
            self._perf_gui_delay_max = max(
                self._perf_gui_delay_max,
                delay,
            )

            elapsed = (
                time.monotonic()
                - self._perf_gui_since
            )
            if elapsed >= 10:
                count = max(
                    1, self._perf_gui_frames
                )
                log(
                    "PERF GUI display: "
                    f"fps={self._perf_gui_frames / elapsed:.1f}, "
                    "queue_avg_ms="
                    f"{1000 * self._perf_gui_delay_sum / count:.0f}, "
                    "queue_max_ms="
                    f"{1000 * self._perf_gui_delay_max:.0f}"
                )

                self._perf_gui_since = (
                    time.monotonic()
                )
                self._perf_gui_frames = 0
                self._perf_gui_delay_sum = 0.0
                self._perf_gui_delay_max = 0.0

    def display_oak_preview(self):
        worker = self.worker

        if (
            worker is None
            or not worker.isRunning()
            or worker.source_kind != "oak"
            or worker.oak_source is None
        ):
            return

        detections = ()
        mode = self.overlay_mode_combo.currentText()
        # Labels require the matched inference frame even without boxes or masks.
        if mode != "None" or self.show_tyre_labels.isChecked():
            overlay = worker.latest_oak_overlay()
            if overlay is None:
                return
            captured_at, sequence, dimensions, detections, processed_frame, result = overlay
            frame = processed_frame.copy()
        else:
            frame, sequence, captured_at = worker.oak_source.preview_snapshot()
        if frame is None or sequence == self._oak_preview_sequence:
            return
        self._oak_preview_sequence = sequence
        if mode == "Bounding Boxes":
            draw_detections(frame, detections)
        elif mode == "Segmentation Mask":
            v481_draw_masks(frame, result)

        draw_line_and_total(
            frame,
            worker.line_ratio,
            worker.line_angle,
            worker.line_length,
            worker.session_count,
            worker.end_ratio, worker.line_totals(),
        )

        self.display_frame(frame)

        self._oak_preview_frames += 1
        elapsed = (
            time.monotonic()
            - self._oak_preview_since
        )

        if elapsed >= 10:
            age_ms = (
                1000 * max(
                    0.0,
                    time.monotonic() - captured_at,
                )
                if captured_at is not None
                else -1
            )
            log(
                "PERF OAK live preview: "
                f"fps={self._oak_preview_frames / elapsed:.1f}, "
                f"latest_frame_age_ms={age_ms:.0f}"
            )
            self._oak_preview_since = (
                time.monotonic()
            )
            self._oak_preview_frames = 0

    def update_source_status(self):
        source = self.source_combo.currentText()
        is_file = source == "Video File"

        self.browse_btn.setVisible(is_file)
        self.file_lbl.setVisible(is_file)

        if is_file:
            name = (
                Path(self.video_path).name
                if self.video_path
                else "select a video"
            )
            self.source_status.setText(
                f"Source: {name}"
            )
        else:
            self.source_status.setText(
                f"Source: {source}"
            )

    def browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select conveyor video",
            "",
            "Video (*.mp4 *.avi *.mov *.mkv)",
        )
        if not path:
            return

        self.video_path = path
        self.file_lbl.setText(
            Path(path).name
        )
        self.source_combo.setCurrentText(
            "Video File"
        )
        self.update_source_status()

        cap = cv2.VideoCapture(path)
        try:
            ok, frame = cap.read()
            if ok:
                self.display_frame(frame)
        finally:
            cap.release()

    def selected_source(self):
        source = self.source_combo.currentText()

        if source == "Video File":
            if not self.video_path:
                self.browse()

            if not self.video_path:
                return None

            return "video", self.video_path

        if source == "Camera 0":
            return "camera", 0

        if source == "Camera 1":
            return "camera", 1

        return "oak", ""

    def set_running_controls(self, running):
        self.video.editable = not running
        self.video.update()

        self.start_btn.setEnabled(
            not running
        )
        self.stop_btn.setEnabled(running)
        self.reset_btn.setEnabled(
            not running
        )
        self.source_combo.setEnabled(
            not running
        )
        self.browse_btn.setEnabled(
            not running
        )
        self.dir_combo.setEnabled(
            not running
        )
        self.conf.setEnabled(
            not running
        )

        for slider in (
            self.line_pos,
            self.line_angle,
            self.line_length,
            self.end_line_pos,
        ):
            slider.setEnabled(
                not running
            )

        for button in self.line_buttons:
            button.setEnabled(
                not running
            )

    def start(self):
        if (
            self.worker is not None
            and self.worker.isRunning()
        ):
            return

        selected = self.selected_source()
        if selected is None:
            return

        try:
            if abs(self.line_pos.value()-self.end_line_pos.value()) < 3:
                raise ValueError("Place the front and end lines at least 3% apart.")
            self.session = 0
            self.test_events = []
            self.line_totals_lbl.setText("Front 0 | End 0 | Both 0 | Final 0")
            self.refresh_counts()

            worker = CounterWorker(
                selected[0],
                selected[1],
                self.dir_combo.currentText(),
                self.conf.value() / 100,
                IOU_DEFAULT,
                self.line_pos.value() / 100.0,
                self.line_angle.value(),
                self.line_length.value() / 100.0,
                self.end_line_pos.value() / 100.0,
            )
            worker.overlay_mode = self.overlay_mode_combo.currentText()
            worker.show_tyre_labels = (
                self.show_tyre_labels.isChecked()
            )
            self.worker = worker
            self.refresh_counts()
            worker.line_progress.connect(self.on_line_progress)

            worker.frame_ready.connect(
                self.display_frame
            )
            worker.counted.connect(
                self.on_count
            )
            worker.status.connect(
                self.on_status
            )
            worker.finished_clean.connect(
                self.worker_done
            )

            self.set_running_controls(True)
            worker.start()

            if selected[0] == "oak":
                self._oak_preview_sequence = 0
                self._oak_preview_since = (
                    time.monotonic()
                )
                self._oak_preview_frames = 0
                self.oak_preview_timer.start()

        except Exception as exc:
            log(
                "START failed:\n"
                + traceback.format_exc()
            )
            self.set_running_controls(False)
            QMessageBox.critical(
                self,
                "Start Failed",
                str(exc),
            )

    def stop(self):
        if self.worker is not None:
            self.worker.stop()

        self.stop_btn.setEnabled(False)
        self.status_lbl.setText(
            "Stopping… finishing the current frame"
        )

    def worker_done(self):
        self.oak_preview_timer.stop()
        self.set_running_controls(False)

    def on_status(self, status):
        messages = {
            "COUNTING":
                "Counting in progress  •  Tyres count "
                "when they cross the yellow line",
            "STOPPED":
                "Stopped  •  Session totals are "
                "ready to export",
        }
        self.status_lbl.setText(
            messages.get(status, status)
        )

        if status.startswith("ERROR:"):
            self.health.setText(
                "● CHECK SYSTEM"
            )
            self.health.setStyleSheet(
                "color:#ff9b9b"
            )
            QMessageBox.critical(
                self,
                "TYRETRACK - System Error",
                status[6:].strip()
                + "\n\nFull details: "
                + str(LOG),
            )
        elif status == "COUNTING":
            self.health.setText(
                "● COUNTING"
            )
            self.health.setStyleSheet(
                "color:#7de7bc"
            )
        elif status == "STOPPED":
            self.health.setText(
                "● READY"
            )
            self.health.setStyleSheet("")

    def on_line_progress(self,front,end,both):
        self.line_totals_lbl.setText(
            f"Front {front} | End {end} | Both {both} | Final {front+end-both}")

    def on_count(self, _track_id, _confidence):
        # Already committed to SQLite in CounterWorker.
        self.session += 1
        if self.worker is not None and self.worker.source_kind == "video":
            self.test_events.append(self.worker.processing_frame / max(1.0, self.worker.video_fps))

        if self.worker is not None and self.worker.source_kind == "oak":
            try:
                export_daily_csv()
                export_classification_csv()
            except Exception as exc:
                log(f"Daily CSV update failed: {exc}")

        self.refresh_counts()

    def refresh_production_hours(self, test_mode=False, test_seconds=0.0):
        if test_mode:
            # Recorded footage uses a virtual production clock beginning at 05:00.
            # These values never affect the real OAK production database.
            slots = []
            for index in range(13):
                begin = index * 3600.0
                finish = min((index + 1) * 3600.0, 12 * 3600.0 + 300.0)
                slots.append(dict(number=index+1, start=begin, end=finish,
                                  count=sum(begin <= t < finish for t in self.test_events),
                                  completed=test_seconds >= finish))
            completed = [s for s in slots if s["completed"]]
            best = max(completed, key=lambda s:s["count"]) if completed else None
            lowest = min(completed, key=lambda s:s["count"]) if completed else None
        else:
            with sqlite3.connect(DB, timeout=15) as con:
                slots, best, lowest = hourly_snapshot(con, source=OAK_NAME)

        def display(slot):
            return f"H{slot['number']}  ·  {slot['count']:,}" if slot else "0  ·  waiting"

        self.production_best.setText(display(best))
        self.production_lowest.setText(display(lowest))
        for prefix, slot in (("best", best), ("lowest", lowest)):
            getattr(self, prefix + "_hour_total").setText(display(slot))
            if slot:
                if test_mode:
                    a = 5 * 3600 + slot['start']
                    b = 5 * 3600 + slot['end']
                    period = f"{int(a//3600):02d}:{int(a%3600//60):02d}–{int(b//3600):02d}:{int(b%3600//60):02d} VIDEO TEST"
                else:
                    period = f"{slot['start']:%H:%M:%S}–{slot['end']:%H:%M:%S} Brisbane"
            else:
                period = "No completed hour yet"
            getattr(self, prefix + "_hour_period").setText(period)
        self.best_hour_types.setText("VIDEO TEST · virtual start 05:00" if test_mode else
                                     "Production day · closes 17:05 Brisbane UTC+10")
        self.lowest_hour_note.setText("Completed empty hours included; current/future excluded")
        self.history_heading.setText("HOURLY HISTORY · VIDEO TEST (05:00 virtual)" if test_mode else
                                     "HOURLY HISTORY · starts 05:00 · cutoff 17:05 Brisbane UTC+10")
        now = brisbane_now()
        active = 0
        for index, slot in enumerate(slots):
            completed = slot['completed']
            current = (slot['start'] <= test_seconds < slot['end']) if test_mode else (slot['start'] <= now < slot['end'])
            colour = "#38D996" if completed else "#42B7FF" if current else "#84929E"
            tag = "✓" if completed else "live" if current else "–"
            self.production_slots[index].setText(f"H{index+1} {tag}\n{slot['count']:,}")
            self.production_slots[index].setStyleSheet(
                f"color:{colour};font-size:13px;font-weight:700;padding:2px;")
            if test_mode:
                self.production_slots[index].setToolTip("Video playback time, starting at virtual 05:00")
            else:
                self.production_slots[index].setToolTip(
                    f"{slot['start']:%H:%M:%S}–{slot['end']:%H:%M:%S}")
            if completed or current:
                active = index
        self.production_scroll.ensureWidgetVisible(self.production_slots[active])

    def refresh_counts(self):
        worker = self.worker
        test_mode = bool(worker is not None and worker.source_kind == "video")
        if test_mode:
            elapsed = worker.processing_frame / max(1.0, worker.video_fps)
            events = sorted(self.test_events)
            hour_index = min(12, int(elapsed // 3600))
            this_hour = sum(hour_index*3600 <= t < min((hour_index+1)*3600, 43500) for t in events)
            last_five = sum(max(0, elapsed-300) <= t <= elapsed for t in events)
            gaps = [b-a for a,b in zip(events,events[1:]) if b-a >= 600]
            trailing = max(0, elapsed-events[-1]) if events else 0
            if trailing >= 600:
                gaps.append(trailing)
            types = worker.session_type_counts
            # Worker commits before emitting the counted signal, so session totals agree.
            self.hour_lbl.setText(str(this_hour))
            self.day_lbl.setText(str(len(events)))
            self.total_lbl.setText(str(len(events)))
            completed = [sum(i*3600 <= t < min((i+1)*3600,43500) for t in events)
                         for i in range(13) if elapsed >= min((i+1)*3600,43500)]
            self.session_lbl.setText(str(max(completed,default=0)))
            self.no_count_lbl.setText(str(len(gaps)))
            self.flow_lbl.setText(f"{last_five / max(1/60, min(300, max(elapsed,1))) * 60:.1f}/min")
            self.gap_lbl.setText(f"{sum(gaps)/60:.0f} min")
            self.production_status_lbl.setText("TEST · NO COUNT" if trailing >= 600 and events else "TEST MODE")
            self.all_time_lbl.setText("TEST SESSION  " + str(len(events)))
            self.passenger_lbl.setText(str(types.get("passenger",0)))
            self.suv_lbl.setText(str(types.get("4x4",0)))
            self.refresh_production_hours(test_mode=True,test_seconds=elapsed)
        else:
            metrics = production_snapshot()
            self.hour_lbl.setText(str(metrics["current_hour"]))
            self.day_lbl.setText(str(metrics["daily"]))
            self.total_lbl.setText(str(metrics["all_time"]))
            self.session_lbl.setText(str(metrics["best_count"]))
            self.no_count_lbl.setText(str(metrics["no_count_incidents"]))
            self.flow_lbl.setText(f"{metrics['flow_per_minute']:.1f}/min")
            self.gap_lbl.setText(f"{metrics['no_flow_minutes']:.0f} min")
            self.production_status_lbl.setText(metrics["no_count_status"])
            self.all_time_lbl.setText("ALL TIME  " + str(metrics["all_time"]))
            with sqlite3.connect(DB,timeout=15) as con:
                types = con.execute("SELECT tyre_type,COUNT(*) FROM counts WHERE source=? AND metric_utc>=? AND metric_utc<? GROUP BY tyre_type",
                                    (OAK_NAME,metrics['day_start'].timestamp(),min(brisbane_now(), metrics['day_end']).timestamp())).fetchall()
            class_totals = {"passenger":0,"4x4":0}
            for name,n in types:
                key=tyre_class_name(name)
                if key in class_totals:
                    class_totals[key]+=n
            self.passenger_lbl.setText(str(class_totals['passenger']))
            self.suv_lbl.setText(str(class_totals['4x4']))
            self.refresh_production_hours()
        self.gap_lbl.setStyleSheet("color:#FF5C68;font-size:22px;font-weight:900;")
        self.no_count_lbl.setStyleSheet("color:#FF5C68;font-size:22px;font-weight:900;")
        alert = "NO COUNT" in self.production_status_lbl.text()
        self.production_status_lbl.setStyleSheet(
            "color:#FF5C68;font-size:22px;font-weight:900;" if alert else "")
        if getattr(self, "_last_dashboard_snapshot", 0) + 30 < time.monotonic():
            self._last_dashboard_snapshot = time.monotonic()
            self.grab().save(str(DASHBOARD_SNAPSHOT), "PNG")

    def reset_session(self):
        if (
            self.worker is not None
            and self.worker.isRunning()
        ):
            QMessageBox.information(
                self,
                "Stop counting first",
                "Stop the current run before "
                "resetting its total.",
            )
            return

        if self.session:
            answer = QMessageBox.question(
                self,
                "Reset session total",
                "Clear the session total on this "
                "screen? Saved records remain.",
            )
            if answer != QMessageBox.Yes:
                return

        self.session = 0
        self.test_events = []
        if self.worker is not None:
            self.worker.session_count = 0
            self.worker.session_type_counts = {"passenger": 0, "4x4": 0, "unknown": 0}
            self.worker.line_ids = [set(), set()]
            self.worker.counted_ids.clear()
        self.line_totals_lbl.setText("Front 0 | End 0 | Both 0 | Final 0")
        self.refresh_counts()

    def export_results(self):
        try:
            suggested = (
                f"TyreTrack_"
                f"{datetime.now().date().isoformat()}.csv"
            )
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Export TyreTrack Results",
                suggested,
                "CSV Files (*.csv)",
            )
            if not path:
                return

            if not path.lower().endswith(".csv"):
                path += ".csv"

            report = export_daily_csv()
            if Path(report).resolve() != Path(path).resolve():
                shutil.copy2(str(report), path)
            QMessageBox.information(
                self,
                "Export Complete",
                f"Results exported to:\n{path}",
            )
        except Exception as exc:
            log(
                "Manual export failed:\n"
                + traceback.format_exc()
            )
            QMessageBox.critical(
                self,
                "Export Failed",
                str(exc),
            )

    def open_oak_recordings(self):
        try:
            os.startfile(
                str(OAK_RECORDINGS)
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "OAK recordings",
                f"Cannot open folder: {exc}\n\n"
                f"{OAK_RECORDINGS}",
            )

    def mouse_line_changed(
        self, ratio, angle, length
    ):
        sliders = (
            self.line_pos,
            self.line_angle,
            self.line_length,
        )

        for slider in sliders:
            slider.blockSignals(True)

        self.line_pos.setValue(
            round(ratio * 100)
        )
        self.line_angle.setValue(angle)
        self.line_length.setValue(
            round(length * 100)
        )

        for slider in sliders:
            slider.blockSignals(False)

        self.line_changed()

    def line_changed(self):
        if (
            self.worker is not None
            and self.worker.isRunning()
        ):
            return

        ratio = self.line_pos.value() / 100.0
        angle = self.line_angle.value()
        length = self.line_length.value() / 100.0

        self.line_pos_lbl.setText(
            f"Position: {self.line_pos.value()}%"
        )
        self.line_angle_lbl.setText(
            f"Angle: {angle:+d}°"
        )
        self.line_length_lbl.setText(
            f"Length: {self.line_length.value()}%"
        )

        if hasattr(self,"end_line_pos"):
            end_ratio=self.end_line_pos.value()/100.0
            self.end_line_lbl.setText(f"End position: {self.end_line_pos.value()}%")
            self.video.end_line_ratio=end_ratio
            set_state("end_line_ratio",end_ratio)
        self.video.line_ratio = ratio
        self.video.line_angle = angle
        self.video.line_length = length
        self.video.update()

        set_state("line_ratio", ratio)
        set_state("line_angle", angle)
        set_state("line_length", length)

    def reset_line(self):
        if (
            self.worker is not None
            and self.worker.isRunning()
        ):
            return

        self.line_pos.setValue(59)
        self.line_angle.setValue(0)
        self.line_length.setValue(100)
        self.line_changed()

    def tick(self):
        try:
            self.refresh_counts()
        except Exception:
            log(
                "Count refresh failed:\n"
                + traceback.format_exc()
            )

        # Never send live production reports or no-count alerts during video tests.
        if self.worker is None or self.worker.source_kind != "oak" or not self.worker.isRunning():
            return

        email_thread_idle = (
            self.email_thread is None
            or not self.email_thread.is_alive()
        )
        if (
            email_thread_idle
            and time.monotonic()
            >= self.email_retry_at
        ):
            self.email_retry_at = (
                time.monotonic() + 60
            )
            self.email_thread = threading.Thread(
                target=process_scheduled_emails,
                args=(datetime.now(),),
                daemon=True,
            )
            self.email_thread.start()

    def closeEvent(self, event):
        self.oak_preview_timer.stop()

        if (
            self.worker is not None
            and self.worker.isRunning()
        ):
            self.worker.stop()

            if not self.worker.wait(15000):
                QMessageBox.warning(
                    self,
                    "Still stopping",
                    "The detector has not finished "
                    "its current frame. Wait, then "
                    "close the app again.",
                )
                if self.worker.source_kind == "oak":
                    self.oak_preview_timer.start()
                event.ignore()
                return

        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = Main()
    window.show()

    sys.exit(app.exec_())
