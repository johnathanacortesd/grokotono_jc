"""Registro local de uso por cliente y aviso opcional por correo."""

from __future__ import annotations

import csv
import json
import os
import smtplib
import ssl
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib import error, request

USAGE_DIR = Path("data")
USAGE_CSV = USAGE_DIR / "uso_clientes.csv"

# Destino por defecto (A.C.). Se puede anular con el secret USAGE_NOTIFY_EMAIL.
# Contraseñas SMTP / API keys NUNCA van en el código: solo secrets o env.
DEFAULT_USAGE_NOTIFY_EMAIL = "cortesalexander8@gmail.com"

USAGE_FIELDS = (
    "timestamp",
    "marca",
    "aliases",
    "n_rows",
    "positivo",
    "negativo",
    "neutro",
    "model",
    "elapsed_s",
    "cost_usd",
)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def aliases_summary(aliases: Sequence[str] | None, *, limit: int = 8) -> str:
    items = [a.strip() for a in (aliases or []) if a and str(a).strip()]
    if not items:
        return ""
    shown = items[:limit]
    extra = len(items) - len(shown)
    text = ", ".join(shown)
    if extra > 0:
        text = f"{text} (+{extra})"
    return text


def _secret(name: str, secrets: Mapping[str, Any] | None = None) -> str:
    if secrets:
        for key in (name, name.lower(), name.upper()):
            try:
                val = secrets[key]
            except Exception:
                val = None
            if val:
                return str(val).strip()
    return (os.environ.get(name) or os.environ.get(name.upper()) or "").strip()


def record_run(
    *,
    marca: str,
    aliases: Sequence[str] | None = None,
    n_rows: int,
    tono_counts: Mapping[str, int] | None = None,
    model: str = "",
    elapsed_s: float = 0.0,
    cost_usd: float | None = None,
    path: Path | None = None,
) -> dict[str, str]:
    """Append one successful classification run to the local CSV log."""
    counts = tono_counts or {}
    row = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "marca": _as_text(marca),
        "aliases": aliases_summary(aliases),
        "n_rows": str(int(n_rows)),
        "positivo": str(int(counts.get("Positivo", 0))),
        "negativo": str(int(counts.get("Negativo", 0))),
        "neutro": str(int(counts.get("Neutro", 0))),
        "model": _as_text(model),
        "elapsed_s": f"{float(elapsed_s):.3f}",
        "cost_usd": "" if cost_usd is None else f"{float(cost_usd):.6f}",
    }
    dest = path or USAGE_CSV
    dest.parent.mkdir(parents=True, exist_ok=True)
    new_file = not dest.exists() or dest.stat().st_size == 0
    with dest.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=USAGE_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    return row


def load_recent_runs(limit: int = 20, path: Path | None = None) -> list[dict[str, str]]:
    dest = path or USAGE_CSV
    if not dest.exists():
        return []
    with dest.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if limit and limit > 0:
        return rows[-limit:]
    return rows


def _compose_email(row: Mapping[str, str]) -> tuple[str, str]:
    marca = row.get("marca") or "(sin marca)"
    subject = f"grokotono · uso · {marca} · {row.get('n_rows') or 0} filas"
    body = (
        "Clasificación completada.\n\n"
        f"Fecha (UTC): {row.get('timestamp', '')}\n"
        f"Marca: {marca}\n"
        f"Alias: {row.get('aliases') or '(ninguno)'}\n"
        f"Filas: {row.get('n_rows', '0')}\n"
        f"Tono Positivo/Negativo/Neutro: "
        f"{row.get('positivo', '0')}/{row.get('negativo', '0')}/{row.get('neutro', '0')}\n"
        f"Modelo: {row.get('model') or '(n/d)'}\n"
        f"Tiempo (s): {row.get('elapsed_s', '')}\n"
        f"Costo USD: {row.get('cost_usd') or '(n/d)'}\n"
    )
    return subject, body


def _send_resend(
    *,
    api_key: str,
    to_addr: str,
    from_addr: str,
    subject: str,
    body: str,
) -> None:
    payload = json.dumps(
        {
            "from": from_addr,
            "to": [to_addr],
            "subject": subject,
            "text": body,
        }
    ).encode("utf-8")
    req = request.Request(
        "https://api.resend.com/emails",
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with request.urlopen(req, timeout=12) as resp:
            resp.read()
    except error.URLError as exc:
        raise RuntimeError(f"Resend no pudo enviar el aviso: {exc}") from exc


def _send_smtp(
    *,
    host: str,
    port: int,
    user: str,
    password: str,
    to_addr: str,
    from_addr: str,
    subject: str,
    body: str,
) -> None:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.set_content(body)
    context = ssl.create_default_context()
    if port == 465:
        with smtplib.SMTP_SSL(host, port, timeout=12, context=context) as smtp:
            if user:
                smtp.login(user, password)
            smtp.send_message(msg)
        return
    with smtplib.SMTP(host, port, timeout=12) as smtp:
        smtp.ehlo()
        try:
            smtp.starttls(context=context)
            smtp.ehlo()
        except smtplib.SMTPException:
            pass
        if user:
            smtp.login(user, password)
        smtp.send_message(msg)


def notify_destination(secrets: Mapping[str, Any] | None = None) -> str:
    """To-address: secret USAGE_NOTIFY_EMAIL, else the confirmed default."""
    return _secret("USAGE_NOTIFY_EMAIL", secrets) or DEFAULT_USAGE_NOTIFY_EMAIL


def maybe_notify_email(
    row: Mapping[str, str],
    *,
    secrets: Mapping[str, Any] | None = None,
) -> str | None:
    """Send usage email after a successful run if SMTP or Resend secrets exist.

    Destino por defecto: DEFAULT_USAGE_NOTIFY_EMAIL. Anulable con
    USAGE_NOTIFY_EMAIL. Sin SMTP_HOST ni RESEND_API_KEY no se envía.
    SMTP_PASSWORD / RESEND_API_KEY se leen solo de secrets/env.
    """
    to_addr = notify_destination(secrets)
    if not to_addr:
        return None
    subject, body = _compose_email(row)
    from_addr = (
        _secret("SMTP_FROM", secrets)
        or _secret("USAGE_NOTIFY_FROM", secrets)
        or _secret("SMTP_USER", secrets)
        or "grokotono@localhost"
    )
    resend_key = _secret("RESEND_API_KEY", secrets)
    host = _secret("SMTP_HOST", secrets)
    if not resend_key and not host:
        return None
    if resend_key:
        _send_resend(
            api_key=resend_key,
            to_addr=to_addr,
            from_addr=from_addr,
            subject=subject,
            body=body,
        )
        return "resend"
    port_raw = _secret("SMTP_PORT", secrets) or "587"
    try:
        port = int(port_raw)
    except ValueError:
        port = 587
    _send_smtp(
        host=host,
        port=port,
        user=_secret("SMTP_USER", secrets),
        password=_secret("SMTP_PASSWORD", secrets),
        to_addr=to_addr,
        from_addr=from_addr,
        subject=subject,
        body=body,
    )
    return "smtp"
