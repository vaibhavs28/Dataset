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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

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
    screenshot_path: Optional[str] = None
) -> Dict[str, Tuple[bool, str]]:
    """
    Dispatches notifications across all selected active channels and records to audit logs.
    If screenshot_path is provided, sends photos via Telegram and embedded images via Email.
    """
    cfg = get_channel_config()
    results = {}

    timestamp_str = datetime.now().strftime("%d %b %Y %H:%M:%S")
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
    """Records an alert execution in the audit log and updates trigger counts."""
    deliv_str = json.dumps({k: {"success": v[0], "status": v[1]} for k, v in delivery_results.items()})
    with get_alerts_db() as conn:
        conn.execute("""
            INSERT INTO alert_logs (alert_id, symbol, alert_type, trigger_price, message, delivery_status_json)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (alert_id, symbol, alert_type, trigger_price, message, deliv_str))

        # Check if alert should be deactivated (if ONCE mode)
        alert_row = conn.execute("SELECT trigger_mode, trigger_count FROM alerts WHERE alert_id = ?", (alert_id,)).fetchone()
        if alert_row:
            new_count = alert_row["trigger_count"] + 1
            if alert_row["trigger_mode"] == "ONCE":
                conn.execute("UPDATE alerts SET trigger_count = ?, is_active = 0, last_triggered_at = CURRENT_TIMESTAMP WHERE alert_id = ?", (new_count, alert_id))
            else:
                conn.execute("UPDATE alerts SET trigger_count = ?, last_triggered_at = CURRENT_TIMESTAMP WHERE alert_id = ?", (new_count, alert_id))
        conn.commit()


def get_alert_logs(limit: int = 50) -> List[Dict[str, Any]]:
    """Fetches historical alert trigger logs."""
    with get_alerts_db() as conn:
        rows = conn.execute("SELECT * FROM alert_logs ORDER BY log_id DESC LIMIT ?", (limit,)).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["delivery_status"] = json.loads(d["delivery_status_json"])
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
    Returns: (trigger_price, headline, details) if triggered, else None.
    """
    if daily_df is None or daily_df.empty or len(daily_df) < 5:
        return None

    close = daily_df["close"].values
    ltp = float(close[-1])
    prev_close = float(close[-2]) if len(close) >= 2 else ltp

    atype = alert["alert_type"]
    cond = alert["condition"]

    # 1. STATIC PRICE ALERT
    if atype == "STATIC_PRICE":
        target = float(cond.get("target_price", 0.0))
        op = cond.get("operator", ">=")

        if op == ">=":
            if ltp >= target and prev_close < target:
                return ltp, f"Price crossed above ₹{target:,.2f}", f"LTP: ₹{ltp:,.2f} (Target: ₹{target:,.2f})"
        elif op == "<=":
            if ltp <= target and prev_close > target:
                return ltp, f"Price crossed below ₹{target:,.2f}", f"LTP: ₹{ltp:,.2f} (Target: ₹{target:,.2f})"
        elif op == "touches":
            if abs(ltp - target) / target <= 0.005:  # within 0.5%
                return ltp, f"Price touched ₹{target:,.2f}", f"LTP: ₹{ltp:,.2f} is within 0.5% of Target ₹{target:,.2f}"

    # 2. STATIC TRENDLINE / SUPPORT & RESISTANCE ALERT
    elif atype == "TRENDLINE_SR":
        sr_type = cond.get("sr_type", "Horizontal")  # "Horizontal" or "Sloping_Trendline"
        if sr_type == "Horizontal":
            level = float(cond.get("level", 0.0))
            kind = cond.get("kind", "Resistance")  # "Resistance" or "Support"
            if kind == "Resistance" and ltp >= level and prev_close < level:
                return ltp, f"Breakout above Resistance ₹{level:,.2f}", f"LTP: ₹{ltp:,.2f} broke above horizontal level ₹{level:,.2f}"
            elif kind == "Support" and ltp <= level and prev_close > level:
                return ltp, f"Breakdown below Support ₹{level:,.2f}", f"LTP: ₹{ltp:,.2f} broke below support level ₹{level:,.2f}"
        else:
            # Sloping Trendline: y = slope * bar_index + intercept
            # Or price1 at t1, price2 at t2
            p1 = float(cond.get("price1", ltp))
            p2 = float(cond.get("price2", ltp))
            bars = max(int(cond.get("bars_span", 20)), 1)
            # Extrapolate current trendline price
            slope = (p2 - p1) / bars
            current_tl_val = p2 + slope
            if ltp >= current_tl_val and prev_close < current_tl_val:
                return ltp, f"Trendline Breakout @ ₹{current_tl_val:,.2f}", f"LTP: ₹{ltp:,.2f} crossed above sloping trendline (estimated ₹{current_tl_val:,.2f})"

    # 3. DYNAMIC TECHNICAL INDICATOR ALERT
    elif atype == "INDICATOR":
        ind_rule = cond.get("rule", "EMA_Cross")
        tf = cond.get("timeframe", "Daily")

        df_eval = daily_df
        if tf == "Weekly":
            df_eval = scanner.resample_ohlcv(daily_df, "weekly")
        elif tf == "75-Min" and intra_75_df is not None and not intra_75_df.empty:
            df_eval = intra_75_df

        c_series = df_eval["close"]

        if ind_rule == "EMA_Cross":
            fast = int(cond.get("fast_ema", 9))
            slow = int(cond.get("slow_ema", 20))
            ema_f = scanner.calculate_ema(c_series, span=fast).values
            ema_s = scanner.calculate_ema(c_series, span=slow).values
            if len(ema_f) >= 2:
                if (ema_f[-1] > ema_s[-1]) and (ema_f[-2] <= ema_s[-2]):
                    return ltp, f"{tf} Bullish EMA Cross ({fast} > {slow})", f"EMA {fast} (₹{ema_f[-1]:,.2f}) crossed above EMA {slow} (₹{ema_s[-1]:,.2f})"

        elif ind_rule == "RSI_Level":
            threshold = float(cond.get("threshold", 60.0))
            operator = cond.get("operator", ">")
            span = int(cond.get("span", 14))
            rsi_vals = scanner.calculate_rsi(c_series, span=span).values
            if len(rsi_vals) >= 2:
                if operator in (">", "crosses_above") and (rsi_vals[-1] > threshold) and (rsi_vals[-2] <= threshold):
                    return ltp, f"{tf} RSI({span}) Crossed Above {threshold}", f"Current RSI: {rsi_vals[-1]:.1f} (Previous: {rsi_vals[-2]:.1f})"
                elif operator in ("<", "crosses_below") and (rsi_vals[-1] < threshold) and (rsi_vals[-2] >= threshold):
                    return ltp, f"{tf} RSI({span}) Crossed Below {threshold}", f"Current RSI: {rsi_vals[-1]:.1f} (Previous: {rsi_vals[-2]:.1f})"

        elif ind_rule == "Hilega_Milega":
            rsi9 = scanner.calculate_rsi(c_series, span=9)
            re3 = scanner.calculate_ema(rsi9, span=3).values
            rw21 = scanner.calculate_wma(rsi9, period=21).values
            r_vals = rsi9.values
            if len(re3) >= 2 and (re3[-1] > rw21[-1]) and (re3[-2] <= rw21[-2]) and (r_vals[-1] >= 50.0):
                return ltp, f"{tf} Hilega Milega Bullish Crossover", f"RSI EMA(3): {re3[-1]:.1f} crossed above WMA(21): {rw21[-1]:.1f} with RSI: {r_vals[-1]:.1f}"

        elif ind_rule == "SuperTrend":
            st_df = scanner.calculate_supertrend(df_eval, period=10, multiplier=3.0)
            st_dir = st_df.get("Trend_Direction", st_df.get("ST_Direction", pd.Series(1, index=df_eval.index))).values
            if len(st_dir) >= 2 and (st_dir[-1] == 1) and (st_dir[-2] == -1):
                return ltp, f"{tf} SuperTrend Bullish Flip (+1)", f"SuperTrend reversed from Bearish to Bullish at ₹{ltp:,.2f}"

    # 4. CHARTINK WATERFALL POSITIONAL SCAN (Stage 4 Trigger)
    elif atype == "WATERFALL_STAGE4":
        wf_res = screener_engine.evaluate_stock_waterfall(alert["symbol"], daily_df, intra_75_df)
        if wf_res is not None and wf_res.get("Stage") == 4:
            return ltp, "🏆 Positional Scan #364 Full Alignment (Stage 4)", "Stock has qualified on Monthly + Weekly + Daily + 75m triggers simultaneously!"

    return None


def check_all_active_alerts(data_provider_fn=None) -> List[Dict[str, Any]]:
    """
    Scans and evaluates all active alerts. Dispatches notifications when conditions are met.
    Returns list of triggered events.
    """
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
