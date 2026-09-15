"""
alert_engine.py
---------------
Persistent Alert & Notification Engine for Upstox Scanner.
Supports:
1. Static Price Cross Alerts (>=, <=)
2. Trendline & Support/Resistance Level Alerts
3. Dynamic Indicator Cross Alerts (EMA crossover, RSI threshold, SuperTrend flip, Hilega Milega)
4. Chartink Waterfall Scan #364 (Stage 4 Full Alignment Trigger)
5. Multi-channel delivery to Telegram, WhatsApp (CallMeBot/Webhook), Email (SMTP), and Webhooks.
"""

import os
import mimetypes
import uuid
import sqlite3
import json
import logging
import smtplib
import urllib.request
import urllib.parse
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime, time
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

IST = ZoneInfo("Asia/Kolkata")


def is_market_hours_ist(now: Optional[datetime] = None) -> bool:
    """
    Returns True if current time is within Indian Market Alert Window:
    Monday through Friday between 09:00:00 AM IST and 04:00:00 PM IST (09:00 - 16:00).
    Outside these hours, automated alerts are suppressed.
    """
    if now is None:
        now = datetime.now(IST)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=IST)
    else:
        now = now.astimezone(IST)

    # Mon=0, Fri=4, Sat=5, Sun=6
    if now.weekday() >= 5:
        return False

    t = now.time()
    return (time(9, 0, 0) <= t <= time(16, 0, 0))


import config
import database
import scanner
import screener_engine

logger = logging.getLogger("alert_engine")

ALERTS_DB_PATH = config.DATA_DIR / "alerts.db"


def get_alerts_db() -> sqlite3.Connection:
    """Returns a thread-safe connection to the alerts SQLite database."""
    ALERTS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(ALERTS_DB_PATH), timeout=30.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.row_factory = sqlite3.Row
    return conn


def init_alerts_db():
    """Initializes the database schema for alerts, channels, and audit logs."""
    with get_alerts_db() as conn:
        cursor = conn.cursor()
        # Alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                condition_json TEXT NOT NULL,
                channels_json TEXT NOT NULL,
                trigger_mode TEXT DEFAULT 'ONCE',
                is_active INTEGER DEFAULT 1,
                trigger_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_triggered_at TIMESTAMP,
                note TEXT
            );
        """)

        # Alert logs / audit table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alert_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id INTEGER,
                symbol TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                trigger_price REAL,
                message TEXT NOT NULL,
                delivery_status_json TEXT NOT NULL,
                triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Channel configuration table (single row or key-value)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS channel_config (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()


# Initialize schema immediately on import
init_alerts_db()


# =========================================================================
# NOTIFICATION DISPATCHERS
# =========================================================================

def get_channel_config() -> Dict[str, Any]:
    """Retrieves saved credentials and configurations for all notification channels."""
    defaults = {
        "telegram": {"enabled": False, "bot_token": "", "chat_id": ""},
        "whatsapp": {"enabled": False, "phone": "", "api_key": "", "mode": "callmebot", "webhook_url": ""},
        "email": {"enabled": False, "smtp_host": "smtp.gmail.com", "smtp_port": 587, "username": "", "password": "", "to_address": ""},
        "webhook": {"enabled": False, "url": ""},
        "audio": {"enabled": True}
    }
    try:
        with get_alerts_db() as conn:
            row = conn.execute("SELECT value_json FROM channel_config WHERE key = 'notification_settings'").fetchone()
            if row and row["value_json"]:
                saved = json.loads(row["value_json"])
                for k, v in saved.items():
                    if k in defaults and isinstance(v, dict):
                        defaults[k].update(v)
                    else:
                        defaults[k] = v
    except Exception as e:
        logger.warning(f"Failed to read channel_config: {e}")
    return defaults


def save_channel_config(cfg: Dict[str, Any]):
    """Persists notification credentials to SQLite."""
    with get_alerts_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO channel_config (key, value_json, updated_at) VALUES ('notification_settings', ?, CURRENT_TIMESTAMP)",
            (json.dumps(cfg),)
        )
        conn.commit()


def send_telegram(bot_token: str, chat_id: str, message: str) -> Tuple[bool, str]:
    """Sends a Telegram notification via official Bot API."""
    if not bot_token or not chat_id:
        return False, "Bot token or Chat ID missing"
    try:
        url = f"https://api.telegram.org/bot{bot_token.strip()}/sendMessage"
        payload = {
            "chat_id": chat_id.strip(),
            "text": message,
            "parse_mode": "HTML"
        }
        data = urllib.parse.urlencode(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"User-Agent": "UpstoxAlertBot/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                return True, "Delivered"
            return False, f"HTTP {resp.status}"
    except Exception as e:
        logger.error(f"Telegram send error: {e}")
        return False, str(e)


def send_telegram_photo(bot_token: str, chat_id: str, photo_path: str, caption: str = "") -> Tuple[bool, str]:
    """Sends a photo image directly to Telegram Chat via sendPhoto API."""
    if not bot_token or not chat_id or not photo_path or not os.path.exists(photo_path):
        return False, "Bot token, Chat ID, or Photo file missing"

    try:
        boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
        with open(photo_path, "rb") as f:
            file_bytes = f.read()

        filename = os.path.basename(photo_path)
        content_type = mimetypes.guess_type(photo_path)[0] or "image/png"

        parts = []
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"chat_id\"\r\n\r\n{chat_id.strip()}\r\n".encode("utf-8"))
        if caption:
            parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"caption\"\r\n\r\n{caption}\r\n".encode("utf-8"))
            parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"parse_mode\"\r\n\r\nHTML\r\n".encode("utf-8"))
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"photo\"; filename=\"{filename}\"\r\nContent-Type: {content_type}\r\n\r\n".encode("utf-8"))
        parts.append(file_bytes)
        parts.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))

        payload = b"".join(parts)
        url = f"https://api.telegram.org/bot{bot_token.strip()}/sendPhoto"
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "User-Agent": "UpstoxAlertBot/1.0"
            }
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            if resp.status == 200:
                return True, "Delivered photo"
            return False, f"HTTP {resp.status}"
    except Exception as e:
        logger.error(f"Telegram sendPhoto error: {e}")
        return False, str(e)


def send_whatsapp(phone: str, api_key: str, message: str, mode: str = "callmebot", webhook_url: str = "") -> Tuple[bool, str]:
    """
    Sends a WhatsApp message via CallMeBot API or custom Webhook.
    CallMeBot API format: https://api.callmebot.com/whatsapp.php?phone=[phone]&text=[message]&apikey=[apikey]
    """
    if mode == "webhook" and webhook_url:
        return send_webhook(webhook_url, {"phone": phone, "message": message, "channel": "whatsapp"})

    if not phone or not api_key:
        return False, "Phone number or CallMeBot API key missing"

    try:
        clean_phone = phone.strip().replace("+", "").replace(" ", "").replace("-", "")
        # Remove HTML tags for WhatsApp plain text
        clean_msg = message.replace("<b>", "*").replace("</b>", "*").replace("<code>", "`").replace("</code>", "`")
        encoded_msg = urllib.parse.quote_plus(clean_msg)
        url = f"https://api.callmebot.com/whatsapp.php?phone={clean_phone}&text={encoded_msg}&apikey={api_key.strip()}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
            if "success" in body.lower() or "queued" in body.lower() or resp.status == 200:
                return True, "Delivered"
            return False, body[:100]
    except Exception as e:
        logger.error(f"WhatsApp send error: {e}")
        return False, str(e)


def send_email(smtp_host: str, smtp_port: int, username: str, password: str, to_address: str, subject: str, message: str) -> Tuple[bool, str]:
    """Sends an email alert using standard SMTP TLS/SSL."""
    if not smtp_host or not username or not password or not to_address:
        return False, "SMTP settings or recipient email missing"

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"Upstox Alert <{username}>"
        msg["To"] = to_address

        # Text and HTML parts
        text_part = MIMEText(message.replace("<b>", "").replace("</b>", "").replace("<code>", "").replace("</code>", ""), "plain")
        html_part = MIMEText(f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; padding: 20px; border: 1px solid #E2E8F0; border-radius: 8px;">
            <h3 style="color: #0284C7; margin-top: 0;">{subject}</h3>
            <div style="font-size: 14px; line-height: 1.6; color: #334155;">
                {message.replace(chr(10), '<br/>')}
            </div>
            <hr style="border: none; border-top: 1px solid #E2E8F0; margin: 20px 0;"/>
            <small style="color: #94A3B8;">Upstox Pro Multi-Timeframe Scanner & Alert Engine</small>
        </div>
        """, "html")

        msg.attach(text_part)
        msg.attach(html_part)

        port = int(smtp_port)
        if port == 465:
            server = smtplib.SMTP_SSL(smtp_host, port, timeout=10)
        else:
            server = smtplib.SMTP(smtp_host, port, timeout=10)
            server.starttls()

        server.login(username, password)
        server.sendmail(username, [to_address], msg.as_string())
        server.quit()
        return True, "Delivered"
    except Exception as e:
        logger.error(f"Email send error: {e}")
        return False, str(e)


def send_email_with_image(smtp_host: str, smtp_port: int, username: str, password: str, to_address: str, subject: str, message: str, image_path: str) -> Tuple[bool, str]:
    """Sends an email with an embedded and attached Quad-Chart screenshot."""
    if not smtp_host or not username or not password or not to_address:
        return False, "SMTP settings or recipient email missing"

    try:
        msg = MIMEMultipart("related")
        msg["Subject"] = subject
        msg["From"] = f"Upstox Alert <{username}>"
        msg["To"] = to_address

        html = f"""
        <div style="font-family: Arial, sans-serif; max-width: 800px; padding: 20px; border: 1px solid #E2E8F0; border-radius: 8px;">
            <h3 style="color: #0284C7; margin-top: 0;">{subject}</h3>
            <div style="font-size: 14px; line-height: 1.6; color: #334155; margin-bottom: 16px;">
                {message.replace(chr(10), '<br/>')}
            </div>
            <div>
                <img src="cid:quad_chart_img" style="max-width: 100%; height: auto; border-radius: 6px; border: 1px solid #334155;" />
            </div>
            <hr style="border: none; border-top: 1px solid #E2E8F0; margin: 20px 0;"/>
            <small style="color: #94A3B8;">Upstox Pro Multi-Timeframe Scanner & Alert Engine</small>
        </div>
        """
        msg.attach(MIMEText(html, "html"))

        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                img_part = MIMEImage(f.read())
                img_part.add_header("Content-ID", "<quad_chart_img>")
                img_part.add_header("Content-Disposition", "inline", filename=os.path.basename(image_path))
                msg.attach(img_part)

        port = int(smtp_port)
        if port == 465:
            server = smtplib.SMTP_SSL(smtp_host, port, timeout=15)
        else:
            server = smtplib.SMTP(smtp_host, port, timeout=15)
            server.starttls()

        server.login(username, password)
        server.sendmail(username, [to_address], msg.as_string())
        server.quit()
        return True, "Delivered email with screenshot"
    except Exception as e:
        logger.error(f"Email image send error: {e}")
        return False, str(e)


def send_webhook(webhook_url: str, payload: Dict[str, Any]) -> Tuple[bool, str]:
    """Dispatches a JSON payload to a Webhook URL (Discord / Slack / Zapier / n8n)."""
    if not webhook_url:
        return False, "Webhook URL missing"
    try:
        json_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            webhook_url.strip(),
            data=json_data,
            headers={"Content-Type": "application/json", "User-Agent": "UpstoxAlert/1.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status in (200, 201, 204):
                return True, "Delivered"
            return False, f"HTTP {resp.status}"
    except Exception as e:
        logger.error(f"Webhook send error: {e}")
        return False, str(e)


def dispatch_alert(
    symbol: str,
    alert_type: str,
    trigger_price: float,
    headline: str,
    details: str,
    selected_channels: List[str],
    screenshot_path: Optional[str] = None,
    ignore_market_hours: bool = False
) -> Dict[str, Tuple[bool, str]]:
    """
    Dispatches notifications across all selected active channels and records to audit logs.
    If screenshot_path is provided, sends photos via Telegram and embedded images via Email.
    Enforces market hours constraint (09:00 AM - 04:00 PM IST Mon-Fri) unless ignore_market_hours=True.
    """
    if not ignore_market_hours and not is_market_hours_ist():
        now_str = datetime.now(IST).strftime("%Y-%m-%d %I:%M:%S %p IST")
        logger.info(f"⏸️ Alert suppressed for {symbol}: Outside Market Timing (09:00 AM - 04:00 PM IST Mon-Fri). Current IST: {now_str}")
        return {
            "market_hours": (False, f"Suppressed: Outside Market Timing (09:00 AM - 04:00 PM IST Mon-Fri). Current IST: {now_str}")
        }

    cfg = get_channel_config()
    results = {}

    timestamp_str = datetime.now(IST).strftime("%d %b %Y %I:%M:%S %p IST")
    subject = f"🚨 ALERT: {symbol} triggered {alert_type} @ ₹{trigger_price:,.2f}"

    # Telegram format
    tg_msg = (
        f"🚨 <b>UPSTOX TRADING ALERT</b> 🚨\n\n"
        f"📌 <b>Stock:</b> <code>{symbol}</code>\n"
        f"💰 <b>LTP:</b> ₹{trigger_price:,.2f}\n"
        f"⚡ <b>Condition:</b> {headline}\n"
        f"📝 <b>Details:</b> {details}\n"
        f"⏰ <b>Time:</b> {timestamp_str}\n"
    )

    if "telegram" in selected_channels and cfg["telegram"].get("enabled"):
        token = cfg["telegram"]["bot_token"]
        chat = cfg["telegram"]["chat_id"]
        if screenshot_path and os.path.exists(screenshot_path):
            t_ok, t_msg = send_telegram_photo(token, chat, screenshot_path, caption=tg_msg)
        else:
            t_ok, t_msg = send_telegram(token, chat, tg_msg)
        results["telegram"] = (t_ok, t_msg)

    if "whatsapp" in selected_channels and cfg["whatsapp"].get("enabled"):
        w_ok, w_msg = send_whatsapp(
            cfg["whatsapp"]["phone"],
            cfg["whatsapp"]["api_key"],
            tg_msg,
            mode=cfg["whatsapp"].get("mode", "callmebot"),
            webhook_url=cfg["whatsapp"].get("webhook_url", "")
        )
        results["whatsapp"] = (w_ok, w_msg)

    if "email" in selected_channels and cfg["email"].get("enabled"):
        if screenshot_path and os.path.exists(screenshot_path):
            e_ok, e_msg = send_email_with_image(
                cfg["email"]["smtp_host"],
                cfg["email"]["smtp_port"],
                cfg["email"]["username"],
                cfg["email"]["password"],
                cfg["email"]["to_address"],
                subject,
                tg_msg,
                screenshot_path
            )
        else:
            e_ok, e_msg = send_email(
                cfg["email"]["smtp_host"],
                cfg["email"]["smtp_port"],
                cfg["email"]["username"],
                cfg["email"]["password"],
                cfg["email"]["to_address"],
                subject,
                tg_msg
            )
        results["email"] = (e_ok, e_msg)

    if "webhook" in selected_channels and cfg["webhook"].get("enabled"):
        payload = {
            "event": "alert_triggered",
            "symbol": symbol,
            "alert_type": alert_type,
            "price": trigger_price,
            "headline": headline,
            "details": details,
            "screenshot_path": screenshot_path,
            "timestamp": timestamp_str
        }
        wh_ok, wh_msg = send_webhook(cfg["webhook"]["url"], payload)
        results["webhook"] = (wh_ok, wh_msg)

    # In-app always recorded
    results["in_app"] = (True, "Logged")

    return results


# =========================================================================
# ALERT CRUD OPERATIONS
# =========================================================================

def create_alert(
    symbol: str,
    alert_type: str,
    condition: Dict[str, Any],
    channels: List[str],
    trigger_mode: str = "ONCE",
    note: str = ""
) -> int:
    """Creates and persists a new alert in SQLite."""
    with get_alerts_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO alerts (symbol, alert_type, condition_json, channels_json, trigger_mode, is_active, trigger_count, note)
            VALUES (?, ?, ?, ?, ?, 1, 0, ?);
        """, (symbol.upper(), alert_type, json.dumps(condition), json.dumps(channels), trigger_mode, note))
        conn.commit()
        return cursor.lastrowid


def get_all_alerts() -> List[Dict[str, Any]]:
    """Fetches all alerts from database."""
    with get_alerts_db() as conn:
        rows = conn.execute("SELECT * FROM alerts ORDER BY alert_id DESC").fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["condition"] = json.loads(d["condition_json"])
            d["channels"] = json.loads(d["channels_json"])
            result.append(d)
        return result


def get_active_alerts() -> List[Dict[str, Any]]:
    """Fetches only active alerts."""
    all_alerts = get_all_alerts()
    return [a for a in all_alerts if a["is_active"] == 1]


def update_alert_status(alert_id: int, is_active: bool):
    """Activates or pauses an alert."""
    with get_alerts_db() as conn:
        conn.execute("UPDATE alerts SET is_active = ? WHERE alert_id = ?", (1 if is_active else 0, alert_id))
        conn.commit()


def delete_alert(alert_id: int):
    """Deletes an alert permanently."""
    with get_alerts_db() as conn:
        conn.execute("DELETE FROM alerts WHERE alert_id = ?", (alert_id,))
        conn.commit()


def log_alert_trigger(alert_id: int, symbol: str, alert_type: str, trigger_price: float, message: str, delivery_results: Dict[str, Tuple[bool, str]]):
    """Records an alert execution in the audit log and updates trigger counts with IST timestamp."""
    deliv_str = json.dumps({k: {"success": v[0], "status": v[1]} for k, v in delivery_results.items()})
    now_ist_str = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    with get_alerts_db() as conn:
        conn.execute("""
            INSERT INTO alert_logs (alert_id, symbol, alert_type, trigger_price, message, delivery_status_json, triggered_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (alert_id, symbol, alert_type, trigger_price, message, deliv_str, now_ist_str))

        # Check if alert should be deactivated (if ONCE mode)
        alert_row = conn.execute("SELECT trigger_mode, trigger_count FROM alerts WHERE alert_id = ?", (alert_id,)).fetchone()
        if alert_row:
            new_count = alert_row["trigger_count"] + 1
            if alert_row["trigger_mode"] == "ONCE":
                conn.execute("UPDATE alerts SET trigger_count = ?, is_active = 0, last_triggered_at = ? WHERE alert_id = ?", (new_count, now_ist_str, alert_id))
            else:
                conn.execute("UPDATE alerts SET trigger_count = ?, last_triggered_at = ? WHERE alert_id = ?", (new_count, now_ist_str, alert_id))
        conn.commit()


def get_alert_logs(limit: int = 50) -> List[Dict[str, Any]]:
    """Fetches historical alert trigger logs with IST time annotation."""
    with get_alerts_db() as conn:
        rows = conn.execute("SELECT * FROM alert_logs ORDER BY log_id DESC LIMIT ?", (limit,)).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["delivery_status"] = json.loads(d["delivery_status_json"])
            t_raw = str(d.get("triggered_at", ""))
            if t_raw and not t_raw.endswith("IST"):
                d["triggered_at_ist"] = f"{t_raw[:19]} IST"
            else:
                d["triggered_at_ist"] = t_raw
            result.append(d)
        return result


# =========================================================================
# ALERT EVALUATION LOGIC
# =========================================================================

def evaluate_single_alert(
    alert: Dict[str, Any],
    daily_df: pd.DataFrame,
    intra_75_df: Optional[pd.DataFrame] = None
) -> Optional[Tuple[float, str, str]]:
    """
    Evaluates an alert against fresh OHLCV candle data.
    Supports complete TradingView alert types:
    - Static Price Crosses & Price Channels (Inside/Outside Range)
    - Trendlines & Horizontal Support/Resistance (Breakout, Breakdown, Bounce, Rejection)
    - Moving Average Crosses (Price vs EMA, EMA Golden/Death Cross)
    - RSI Oscillator (Level Crosses, Oversold/Overbought entries & exits)
    - MACD (Signal Line Crossover, Zero Line Cross, Histogram)
    - SuperTrend (Bullish & Bearish Trend Flips)
    - Bollinger Bands (Upper Breakout, Lower Breakdown, Band Squeeze)
    - Weekly CPR (R1 Breakout, S1 Breakdown, Pivot Cross, 0.5 Support)
    - Volume Surge (2x, 3x, 5x 20-day Average Volume)
    - Authentic Hilega-Milega Setup
    - Chartink Waterfall Positional Scan #364 (Stage 4 Full Alignment)
    Returns: (trigger_price, headline, details) if triggered, else None.
    """
    if daily_df is None or daily_df.empty or len(daily_df) < 5:
        return None

    close = daily_df["close"].values
    ltp = float(close[-1])
    prev_close = float(close[-2]) if len(close) >= 2 else ltp

    atype = alert["alert_type"]
    cond = alert["condition"]

    # 1. STATIC PRICE & CHANNEL ALERTS
    if atype == "STATIC_PRICE":
        target = float(cond.get("target_price", 0.0))
        op = cond.get("operator", ">=")

        if op in (">=", "crosses_above"):
            if ltp >= target and prev_close < target:
                return ltp, f"Price crossed above ₹{target:,.2f}", f"LTP: ₹{ltp:,.2f} (Target: ₹{target:,.2f})"
        elif op in ("<=", "crosses_below"):
            if ltp <= target and prev_close > target:
                return ltp, f"Price crossed below ₹{target:,.2f}", f"LTP: ₹{ltp:,.2f} (Target: ₹{target:,.2f})"
        elif op == "touches":
            if abs(ltp - target) / max(1.0, target) <= 0.005:  # within 0.5%
                return ltp, f"Price touched ₹{target:,.2f}", f"LTP: ₹{ltp:,.2f} is within 0.5% of Target ₹{target:,.2f}"
        elif op in ("between", "channel_enter"):
            p_low = float(cond.get("price_low", target * 0.98))
            p_high = float(cond.get("price_high", target * 1.02))
            if (p_low <= ltp <= p_high) and not (p_low <= prev_close <= p_high):
                return ltp, f"Entered Channel [₹{p_low:,.2f} - ₹{p_high:,.2f}]", f"LTP: ₹{ltp:,.2f} entered price range ₹{p_low:,.2f} to ₹{p_high:,.2f}"
        elif op in ("outside", "channel_exit"):
            p_low = float(cond.get("price_low", target * 0.98))
            p_high = float(cond.get("price_high", target * 1.02))
            if not (p_low <= ltp <= p_high) and (p_low <= prev_close <= p_high):
                return ltp, f"Exited Channel [₹{p_low:,.2f} - ₹{p_high:,.2f}]", f"LTP: ₹{ltp:,.2f} broke out of price range ₹{p_low:,.2f} to ₹{p_high:,.2f}"

    # 2. STATIC TRENDLINE / SUPPORT & RESISTANCE ALERT
    elif atype == "TRENDLINE_SR":
        sr_type = cond.get("sr_type", "Horizontal")  # "Horizontal" or "Sloping_Trendline"
        if sr_type == "Horizontal":
            level = float(cond.get("level", 0.0))
            kind = cond.get("kind", "Resistance")  # "Resistance", "Support", "Support_Bounce", "Resistance_Rejection"
            if kind == "Resistance" and ltp >= level and prev_close < level:
                return ltp, f"Breakout above Resistance ₹{level:,.2f}", f"LTP: ₹{ltp:,.2f} broke above horizontal resistance ₹{level:,.2f}"
            elif kind == "Support" and ltp <= level and prev_close > level:
                return ltp, f"Breakdown below Support ₹{level:,.2f}", f"LTP: ₹{ltp:,.2f} broke below horizontal support ₹{level:,.2f}"
            elif kind == "Support_Bounce":
                cur_low = float(daily_df["low"].iloc[-1])
                if cur_low <= level and ltp > level * 1.005:
                    return ltp, f"Support Bounce @ ₹{level:,.2f}", f"Low dipped to ₹{cur_low:,.2f} and bounced to ₹{ltp:,.2f}"
            elif kind == "Resistance_Rejection":
                cur_high = float(daily_df["high"].iloc[-1])
                if cur_high >= level and ltp < level * 0.995:
                    return ltp, f"Resistance Rejection @ ₹{level:,.2f}", f"High touched ₹{cur_high:,.2f} and rejected down to ₹{ltp:,.2f}"
        else:
            p1 = float(cond.get("price1", ltp))
            p2 = float(cond.get("price2", ltp))
            bars = max(int(cond.get("bars_span", 20)), 1)
            slope = (p2 - p1) / bars
            current_tl_val = p2 + slope
            tl_dir = cond.get("direction", "Breakout_Above")
            if tl_dir == "Breakout_Above" and ltp >= current_tl_val and prev_close < current_tl_val:
                return ltp, f"Upper Trendline Breakout @ ₹{current_tl_val:,.2f}", f"LTP: ₹{ltp:,.2f} crossed above sloping trendline (₹{current_tl_val:,.2f})"
            elif tl_dir == "Breakdown_Below" and ltp <= current_tl_val and prev_close > current_tl_val:
                return ltp, f"Lower Trendline Breakdown @ ₹{current_tl_val:,.2f}", f"LTP: ₹{ltp:,.2f} broke below sloping trendline (₹{current_tl_val:,.2f})"

    # 3. DYNAMIC TECHNICAL INDICATOR ALERT (TRADINGVIEW CORE SET)
    elif atype == "INDICATOR":
        ind_rule = cond.get("rule", "EMA_Cross")
        tf = cond.get("timeframe", "Daily")

        df_eval = daily_df
        if tf == "Weekly":
            df_eval = scanner.resample_ohlcv(daily_df, "weekly")
        elif tf == "Monthly":
            df_eval = scanner.resample_ohlcv(daily_df, "monthly")
        elif tf == "75-Min" and intra_75_df is not None and not intra_75_df.empty:
            df_eval = intra_75_df

        c_series = df_eval["close"]

        # 3.1 Price vs Moving Average (e.g. Price crosses 20 EMA, 50 EMA, 200 EMA)
        if ind_rule in ("Price_EMA", "Price_Cross_EMA"):
            span = int(cond.get("span", 20))
            direction = cond.get("direction", "crosses_above")
            ema_series = scanner.calculate_ema(c_series, span=span).values
            if len(ema_series) >= 2:
                cur_ema = ema_series[-1]
                prev_ema = ema_series[-2]
                if direction in ("crosses_above", ">") and (ltp >= cur_ema) and (prev_close < prev_ema):
                    return ltp, f"{tf} Price Crossed Above {span} EMA", f"LTP ₹{ltp:,.2f} broke above {span} EMA (₹{cur_ema:,.2f})"
                elif direction in ("crosses_below", "<") and (ltp <= cur_ema) and (prev_close > prev_ema):
                    return ltp, f"{tf} Price Crossed Below {span} EMA", f"LTP ₹{ltp:,.2f} broke below {span} EMA (₹{cur_ema:,.2f})"

        # 3.2 EMA Crossover & Crossunder (Fast x Slow)
        elif ind_rule == "EMA_Cross":
            fast = int(cond.get("fast_ema", 9))
            slow = int(cond.get("slow_ema", 20))
            direction = cond.get("direction", "bullish_cross")
            ema_f = scanner.calculate_ema(c_series, span=fast).values
            ema_s = scanner.calculate_ema(c_series, span=slow).values
            if len(ema_f) >= 2:
                if direction == "bullish_cross" and (ema_f[-1] > ema_s[-1]) and (ema_f[-2] <= ema_s[-2]):
                    return ltp, f"{tf} Bullish EMA Crossover ({fast} > {slow})", f"Fast EMA {fast} (₹{ema_f[-1]:,.2f}) crossed above Slow EMA {slow} (₹{ema_s[-1]:,.2f})"
                elif direction == "bearish_cross" and (ema_f[-1] < ema_s[-1]) and (ema_f[-2] >= ema_s[-2]):
                    return ltp, f"{tf} Bearish EMA Crossunder ({fast} < {slow})", f"Fast EMA {fast} (₹{ema_f[-1]:,.2f}) crossed below Slow EMA {slow} (₹{ema_s[-1]:,.2f})"

        # 3.3 RSI Level & Zone Alerts
        elif ind_rule == "RSI_Level":
            threshold = float(cond.get("threshold", 60.0))
            operator = cond.get("operator", "crosses_above")
            span = int(cond.get("span", 14))
            rsi_vals = scanner.calculate_rsi(c_series, span=span).values
            if len(rsi_vals) >= 2:
                cur_rsi = rsi_vals[-1]
                prev_rsi = rsi_vals[-2]
                if operator in (">", "crosses_above") and (cur_rsi > threshold) and (prev_rsi <= threshold):
                    return ltp, f"{tf} RSI({span}) Crossed Above {threshold}", f"Current RSI: {cur_rsi:.1f} (Previous: {prev_rsi:.1f})"
                elif operator in ("<", "crosses_below") and (cur_rsi < threshold) and (prev_rsi >= threshold):
                    return ltp, f"{tf} RSI({span}) Crossed Below {threshold}", f"Current RSI: {cur_rsi:.1f} (Previous: {prev_rsi:.1f})"
                elif operator == "enters_oversold" and (cur_rsi < 30.0) and (prev_rsi >= 30.0):
                    return ltp, f"{tf} RSI Entered Oversold (<30)", f"RSI({span}) dropped to {cur_rsi:.1f} (Oversold zone)"
                elif operator == "exits_oversold" and (cur_rsi > 30.0) and (prev_rsi <= 30.0):
                    return ltp, f"{tf} RSI Exited Oversold (>30)", f"RSI({span}) recovered to {cur_rsi:.1f} out of oversold zone"
                elif operator == "enters_overbought" and (cur_rsi > 70.0) and (prev_rsi <= 70.0):
                    return ltp, f"{tf} RSI Entered Overbought (>70)", f"RSI({span}) surged to {cur_rsi:.1f} (Overbought zone)"
                elif operator == "exits_overbought" and (cur_rsi < 70.0) and (prev_rsi >= 70.0):
                    return ltp, f"{tf} RSI Exited Overbought (<70)", f"RSI({span}) pulled back to {cur_rsi:.1f} from overbought zone"

        # 3.4 MACD Alerts
        elif ind_rule == "MACD":
            fast = int(cond.get("fast", 12))
            slow = int(cond.get("slow", 26))
            sig = int(cond.get("signal", 9))
            m_type = cond.get("macd_type", "bullish_cross")
            macd_df = scanner.calculate_macd(c_series, fast=fast, slow=slow, signal=sig)
            if len(macd_df) >= 2:
                m_line = macd_df["MACD_Line"].values
                s_line = macd_df["MACD_Signal"].values
                if m_type == "bullish_cross" and (m_line[-1] > s_line[-1]) and (m_line[-2] <= s_line[-2]):
                    return ltp, f"{tf} MACD Bullish Crossover", f"MACD Line ({m_line[-1]:.2f}) crossed above Signal Line ({s_line[-1]:.2f})"
                elif m_type == "bearish_cross" and (m_line[-1] < s_line[-1]) and (m_line[-2] >= s_line[-2]):
                    return ltp, f"{tf} MACD Bearish Crossunder", f"MACD Line ({m_line[-1]:.2f}) crossed below Signal Line ({s_line[-1]:.2f})"
                elif m_type == "zero_cross_above" and (m_line[-1] > 0) and (m_line[-2] <= 0):
                    return ltp, f"{tf} MACD Crossed Above Zero", f"MACD Line crossed above baseline into bullish territory ({m_line[-1]:.2f})"
                elif m_type == "zero_cross_below" and (m_line[-1] < 0) and (m_line[-2] >= 0):
                    return ltp, f"{tf} MACD Crossed Below Zero", f"MACD Line crossed below baseline into bearish territory ({m_line[-1]:.2f})"

        # 3.5 SuperTrend Alerts (Bullish & Bearish Flips)
        elif ind_rule == "SuperTrend":
            period = int(cond.get("period", 10))
            multiplier = float(cond.get("multiplier", 3.0))
            st_dir_choice = cond.get("direction", "bullish_flip")
            st_df = scanner.calculate_supertrend(df_eval, period=period, multiplier=multiplier)
            st_dir = st_df.get("Trend_Direction", st_df.get("ST_Direction", pd.Series(1, index=df_eval.index))).values
            if len(st_dir) >= 2:
                if st_dir_choice == "bullish_flip" and (st_dir[-1] == 1) and (st_dir[-2] == -1):
                    return ltp, f"{tf} SuperTrend Bullish Flip (+1 🟢)", f"SuperTrend({period}, {multiplier}) turned BULLISH at ₹{ltp:,.2f}"
                elif st_dir_choice == "bearish_flip" and (st_dir[-1] == -1) and (st_dir[-2] == 1):
                    return ltp, f"{tf} SuperTrend Bearish Flip (-1 🔴)", f"SuperTrend({period}, {multiplier}) turned BEARISH at ₹{ltp:,.2f}"

        # 3.6 Bollinger Bands Alerts
        elif ind_rule == "Bollinger_Bands":
            period = int(cond.get("period", 20))
            std = float(cond.get("std", 2.0))
            bb_type = cond.get("bb_type", "upper_cross")
            bb_df = scanner.calculate_bollinger_bands(c_series, period=period, num_std=std)
            if len(bb_df) >= 2:
                up = bb_df["Upper"].values
                mid = bb_df["Middle"].values
                low = bb_df["Lower"].values
                if bb_type == "upper_cross" and (ltp >= up[-1]) and (prev_close < up[-2]):
                    return ltp, f"{tf} Bollinger Upper Band Breakout", f"LTP ₹{ltp:,.2f} crossed above Upper Band ₹{up[-1]:,.2f}"
                elif bb_type == "lower_cross" and (ltp <= low[-1]) and (prev_close > low[-2]):
                    return ltp, f"{tf} Bollinger Lower Band Breakdown", f"LTP ₹{ltp:,.2f} broke below Lower Band ₹{low[-1]:,.2f}"
                elif bb_type == "squeeze":
                    bw = (up[-1] - low[-1]) / max(1.0, mid[-1])
                    thresh_bw = float(cond.get("bandwidth_threshold", 0.05))
                    if bw < thresh_bw:
                        return ltp, f"{tf} Bollinger Band Squeeze ({bw*100:.1f}%)", f"Volatility contraction: Bandwidth {bw*100:.1f}% is below {thresh_bw*100:.1f}%"

        # 3.7 Weekly CPR Alerts (Pivot, R1, S1, 0.5 Support)
        elif ind_rule == "Weekly_CPR":
            cpr_level = cond.get("cpr_level", "R1")
            wcpr_df = scanner.calculate_weekly_cpr(daily_df)
            if not wcpr_df.empty:
                last_cpr = wcpr_df.iloc[-1]
                target_lvl = float(last_cpr.get(cpr_level, 0.0))
                if cpr_level == "R1" and ltp >= target_lvl and prev_close < target_lvl:
                    return ltp, f"Weekly CPR R1 Resistance Breakout @ ₹{target_lvl:,.2f}", f"LTP ₹{ltp:,.2f} crossed above Weekly R1 level ₹{target_lvl:,.2f}"
                elif cpr_level == "S1" and ltp <= target_lvl and prev_close > target_lvl:
                    return ltp, f"Weekly CPR S1 Support Breakdown @ ₹{target_lvl:,.2f}", f"LTP ₹{ltp:,.2f} crossed below Weekly S1 level ₹{target_lvl:,.2f}"
                elif cpr_level == "P" and ltp >= target_lvl and prev_close < target_lvl:
                    return ltp, f"Weekly CPR Pivot (P) Cross Above @ ₹{target_lvl:,.2f}", f"LTP ₹{ltp:,.2f} crossed above Central Weekly Pivot ₹{target_lvl:,.2f}"
                elif cpr_level == "S_05" and abs(ltp - target_lvl) / max(1.0, target_lvl) <= 0.005:
                    return ltp, f"Weekly CPR 0.5 Support Touched @ ₹{target_lvl:,.2f}", f"LTP ₹{ltp:,.2f} reached Weekly CPR 0.5 Support ₹{target_lvl:,.2f}"

        # 3.8 Volume Surge Alerts
        elif ind_rule == "Volume_Surge":
            multiplier = float(cond.get("volume_multiplier", 2.0))
            if "volume" in df_eval.columns and len(df_eval) >= 21:
                v_series = df_eval["volume"].values
                avg_vol = float(np.mean(v_series[-21:-1]))
                cur_vol = float(v_series[-1])
                if avg_vol > 0 and cur_vol >= (avg_vol * multiplier):
                    return ltp, f"{tf} Volume Surge ({cur_vol/avg_vol:.1f}x 20-day avg)", f"Volume: {int(cur_vol):,} shares (20-day avg: {int(avg_vol):,})"

        # 3.9 Hilega Milega
        elif ind_rule == "Hilega_Milega":
            rsi9 = scanner.calculate_rsi(c_series, span=9)
            re3 = scanner.calculate_ema(rsi9, span=3).values
            rw21 = scanner.calculate_wma(rsi9, period=21).values
            r_vals = rsi9.values
            if len(re3) >= 2 and (re3[-1] > rw21[-1]) and (re3[-2] <= rw21[-2]) and (r_vals[-1] >= 50.0):
                return ltp, f"{tf} Hilega Milega Bullish Crossover", f"RSI EMA(3): {re3[-1]:.1f} crossed above WMA(21): {rw21[-1]:.1f} with RSI: {r_vals[-1]:.1f}"

    # 4. CHARTINK WATERFALL POSITIONAL SCAN (Stage 4 Trigger)
    elif atype == "WATERFALL_STAGE4":
        wf_res = screener_engine.evaluate_stock_waterfall(alert["symbol"], daily_df, intra_75_df)
        if wf_res is not None and wf_res.get("Stage") == 4:
            return ltp, "🏆 Positional Scan #364 Full Alignment (Stage 4)", "Stock has qualified on Monthly + Weekly + Daily + 75m triggers simultaneously!"

    # 5. CHARTINK INTRADAY SCAN #19122704 (Stage 4/5 Breakdown Trigger)
    elif atype == "INTRADAY_SCAN_19122704":
        wf_res = screener_engine.evaluate_intraday_scan_19122704(alert["symbol"], daily_df, intra_75_df)
        if wf_res is not None:
            stage = wf_res.get("Stage", 0)
            if stage == 5:
                return ltp, "🚨 Chartink Intraday #19122704 — FULL SIGNAL (Stage 5)", \
                    "Stock passed ALL 5 tiers: Monthly + Weekly + Daily + 75m + 15m Precision Entry (gap-down, bearish bar, MA squeeze)!"
            elif stage == 4:
                return ltp, "⚡ Chartink Intraday Scan #19122704 Full Alignment (Stage 4)", \
                    "Stock has qualified on Monthly + Weekly + Daily + 75m Breakdown triggers simultaneously!"

    return None


def check_all_active_alerts(data_provider_fn=None, ignore_market_hours: bool = False) -> List[Dict[str, Any]]:
    """
    Scans and evaluates all active alerts. Dispatches notifications when conditions are met.
    Restricted strictly to 09:00 AM - 04:00 PM IST (Mon-Fri) unless ignore_market_hours=True.
    Returns list of triggered events.
    """
    if not ignore_market_hours and not is_market_hours_ist():
        logger.info("⏸️ Skipping active alerts scan: Outside Market Hours (09:00 AM - 04:00 PM IST Mon-Fri).")
        return []

    active_alerts = get_active_alerts()
    if not active_alerts:
        return []

    triggered_events = []

    for alert in active_alerts:
        sym = alert["symbol"]
        daily_df = None
        intra_df = None

        if data_provider_fn:
            daily_df = data_provider_fn(sym, "Daily")
            intra_df = data_provider_fn(sym, "75-Min")
        else:
            daily_df = database.get_candles_df(sym)
            intra_df = parquet_loader.ensure_symbol_75m_candles(sym, min_bars=20)

        if daily_df is None or daily_df.empty:
            continue

        eval_res = evaluate_single_alert(alert, daily_df, intra_df)
        if eval_res is not None:
            trig_price, headline, details = eval_res
            channels = alert.get("channels", ["telegram", "in_app"])

            # Dispatch notification
            deliv_results = dispatch_alert(
                symbol=sym,
                alert_type=alert["alert_type"],
                trigger_price=trig_price,
                headline=headline,
                details=details,
                selected_channels=channels
            )

            # Log to database
            log_alert_trigger(
                alert_id=alert["alert_id"],
                symbol=sym,
                alert_type=alert["alert_type"],
                trigger_price=trig_price,
                message=f"{headline} - {details}",
                delivery_results=deliv_results
            )

            triggered_events.append({
                "alert_id": alert["alert_id"],
                "symbol": sym,
                "type": alert["alert_type"],
                "price": trig_price,
                "headline": headline,
                "details": details,
                "delivery": deliv_results
            })

    return triggered_events
