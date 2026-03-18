"""
Netra AI — alert channel backends.
Ported from edgeguard/src/alerts/ and hardened.
"""
from __future__ import annotations

import smtplib
from abc import ABC, abstractmethod
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from netra.types import IncidentEvent, AlertSeverity
from backend.core.settings import get_settings
from backend.core.logging import get_logger

log      = get_logger("alerting.channels")
settings = get_settings()


def _narrative(incident: IncidentEvent) -> str:
    lines = [
        f"🚨 NETRA ALERT — {incident.severity.value}",
        f"Camera: {incident.camera_id}  |  Track: {incident.track_id}",
        f"Stage: {incident.theft_stage.value}",
        f"Concealment: {incident.concealment_type.value}",
        f"Risk Score: {incident.risk_score:.2%}",
        f"FSM: {incident.fsm_score:.1f}  ShopFormer: {incident.shopformer_score:.2%}",
        f"Incident ID: {incident.incident_id}",
    ]
    return "\n".join(lines)


class AlertChannel(ABC):
    @abstractmethod
    async def send(self, incident: IncidentEvent) -> bool:
        ...


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

class TelegramChannel(AlertChannel):
    def __init__(self) -> None:
        import httpx
        self._client = httpx.AsyncClient()
        self._token  = settings.telegram_bot_token
        self._chat   = settings.telegram_chat_id

    async def send(self, incident: IncidentEvent) -> bool:
        if not self._token or not self._chat:
            return False
        text = _narrative(incident)
        url  = f"https://api.telegram.org/bot{self._token}/sendMessage"
        try:
            r = await self._client.post(url, json={"chat_id": self._chat, "text": text})
            if r.status_code != 200:
                log.warning(f"Telegram send failed: {r.status_code} {r.text}")
                return False
            return True
        except Exception as e:
            log.error(f"Telegram error: {e}")
            return False


# ---------------------------------------------------------------------------
# Email (SMTP)
# ---------------------------------------------------------------------------

class EmailChannel(AlertChannel):
    async def send(self, incident: IncidentEvent) -> bool:
        if not settings.email_smtp_host or not settings.email_to:
            return False
        try:
            msg            = MIMEMultipart("alternative")
            msg["Subject"] = f"[{incident.severity.value}] Netra Alert — {incident.camera_id}"
            msg["From"]    = settings.email_from
            msg["To"]      = ", ".join(settings.email_to)
            body           = _narrative(incident)
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(settings.email_smtp_host, settings.email_smtp_port, timeout=10) as srv:
                srv.starttls()
                srv.login(settings.email_smtp_user, settings.email_smtp_password)
                srv.sendmail(settings.email_from, settings.email_to, msg.as_string())
            return True
        except Exception as e:
            log.error(f"Email error: {e}")
            return False


# ---------------------------------------------------------------------------
# SMS (Twilio)
# ---------------------------------------------------------------------------

class SMSChannel(AlertChannel):
    async def send(self, incident: IncidentEvent) -> bool:
        if not settings.twilio_account_sid or not settings.sms_to_numbers:
            return False
        try:
            import httpx
            auth = (settings.twilio_account_sid, settings.twilio_auth_token)
            body = f"NETRA [{incident.severity.value}] Camera {incident.camera_id} | Risk {incident.risk_score:.0%} | ID {incident.incident_id[:8]}"
            url  = f"https://api.twilio.com/2010-04-01/Accounts/{settings.twilio_account_sid}/Messages.json"
            async with httpx.AsyncClient(auth=auth) as client:
                for number in settings.sms_to_numbers:
                    await client.post(url, data={"From": settings.twilio_from_number, "To": number, "Body": body})
            return True
        except Exception as e:
            log.error(f"SMS error: {e}")
            return False


# ---------------------------------------------------------------------------
# Webhook (generic HTTP POST)
# ---------------------------------------------------------------------------

class WebhookChannel(AlertChannel):
    async def send(self, incident: IncidentEvent) -> bool:
        if not settings.webhook_url:
            return False
        try:
            import httpx, dataclasses
            payload = {
                "incident_id":      incident.incident_id,
                "camera_id":        incident.camera_id,
                "track_id":         incident.track_id,
                "risk_score":       incident.risk_score,
                "severity":         incident.severity.value,
                "theft_stage":      incident.theft_stage.value,
                "concealment_type": incident.concealment_type.value,
                "timestamp":        incident.timestamp,
            }
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.post(settings.webhook_url, json=payload)
                return r.status_code < 400
        except Exception as e:
            log.error(f"Webhook error: {e}")
            return False


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

class AlertDispatcher:
    def __init__(self) -> None:
        self._channels: list[AlertChannel] = []
        if settings.telegram_enabled:
            self._channels.append(TelegramChannel())
        if settings.email_enabled:
            self._channels.append(EmailChannel())
        if settings.sms_enabled:
            self._channels.append(SMSChannel())
        if settings.webhook_enabled:
            self._channels.append(WebhookChannel())
        log.info(f"AlertDispatcher ready with {len(self._channels)} channel(s)")

    async def dispatch(self, incident: IncidentEvent) -> None:
        for ch in self._channels:
            try:
                ok = await ch.send(incident)
                if not ok:
                    log.warning(f"{ch.__class__.__name__} returned failure for {incident.incident_id}")
            except Exception as e:
                log.error(f"{ch.__class__.__name__} dispatch exception: {e}")
