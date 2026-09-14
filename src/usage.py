"""Registro local de uso por cliente y aviso por correo (sin UI)."""

from __future__ import annotations

import csv
import json
import logging
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
DEFAULT_USAGE_NOTIFY_FROM = "onboarding@resend.dev"

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
    "email_status",
)

_LOG = logging.getLogger("grokotono.usage")

# Formas anidadas de Streamlit secrets.toml (además de la clave plana).
_NESTED_SECRET_KEYS: dict[str, tuple[tuple[str, str], ...]] = {
    "RESEND_API_KEY": (
        ("resend", "api_key"),
        ("resend", "API_KEY"),
        ("Resend", "api_key"),
        ("RESEND", "API_KEY"),
    ),
    "USAGE_NOTIFY_EMAIL": (
        ("usage", "notify_email"),
        ("usage", "email"),
        ("resend", "to"),
    ),
    "USAGE_NOTIFY_FROM": (
        ("usage", "notify_from"),
        ("usage", "from"),
        ("resend", "from"),
        ("resend", "from_addr"),
    ),
    "SMTP_HOST": (("smtp", "host"), ("SMTP", "HOST")),
    "SMTP_PORT": (("smtp", "port"), ("SMTP", "PORT")),
    "SMTP_USER": (("smtp", "user"), ("smtp", "username"), ("SMTP", "USER")),
    "SMTP_PASSWORD": (("smtp", "password"), ("SMTP", "PASSWORD")),
    "SMTP_FROM": (("smtp", "from"), ("SMTP", "FROM")),
}


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


def _mapping_get(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    try:
        return obj[key]
    except Exception:
        pass
    getter = getattr(obj, "get", None)
    if callable(getter):
        try:
            return getter(key)
        except Exception:
            return None
    return None


def _is_nested_table(val: Any) -> bool:
    if val is None or isinstance(val, (str, bytes, int, float, bool)):
        return False
    return isinstance(val, Mapping) or (hasattr(val, "__getitem__") and hasattr(val, "keys"))


def _secret(name: str, secrets: Mapping[str, Any] | None = None) -> str:
    if secrets is not None:
        for key in (name, name.lower(), name.upper()):
            val = _mapping_get(secrets, key)
            if val is None or _is_nested_table(val):
                continue
            text = str(val).strip()
            if text:
                return text
        for section, inner in _NESTED_SECRET_KEYS.get(name, ()):
            nested = _mapping_get(secrets, section)
            if nested is None:
                continue
            val = _mapping_get(nested, inner)
            if val is None or _is_nested_table(val):
                continue
            text = str(val).strip()
            if text:
                return text
    return (os.environ.get(name) or os.environ.get(name.upper()) or "").strip()


def _build_row(
    *,
    marca: str,
    aliases: Sequence[str] | None = None,
    n_rows: int,
    tono_counts: Mapping[str, int] | None = None,
    model: str = "",
    elapsed_s: float = 0.0,
    cost_usd: float | None = None,
    email_status: str = "",
) -> dict[str, str]:
    counts = tono_counts or {}
    return {
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
        "email_status": _as_text(email_status),
    }


def _migrate_csv_header(dest: Path) -> None:
    if not dest.exists() or dest.stat().st_size == 0:
        return
    with dest.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        old_fields = list(reader.fieldnames or [])
        rows = list(reader)
    if old_fields == list(USAGE_FIELDS):
        return
    with dest.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=USAGE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in USAGE_FIELDS})


def record_run(
    *,
    marca: str,
    aliases: Sequence[str] | None = None,
    n_rows: int,
    tono_counts: Mapping[str, int] | None = None,
    model: str = "",
    elapsed_s: float = 0.0,
    cost_usd: float | None = None,
    email_status: str = "",
    path: Path | None = None,
) -> dict[str, str]:
    """Append one successful classification run to the local CSV log."""
    row = _build_row(
        marca=marca,
        aliases=aliases,
        n_rows=n_rows,
        tono_counts=tono_counts,
        model=model,
        elapsed_s=elapsed_s,
        cost_usd=cost_usd,
        email_status=email_status,
    )
    dest = path or USAGE_CSV
    dest.parent.mkdir(parents=True, exist_ok=True)
    _migrate_csv_header(dest)
    new_file = not dest.exists() or dest.stat().st_size == 0
    with dest.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=USAGE_FIELDS, extrasaction="ignore")
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
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:240]
        raise RuntimeError(f"Resend HTTP {exc.code}: {detail or exc}") from exc
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


def _from_address(secrets: Mapping[str, Any] | None, *, resend: bool) -> str:
    custom = _secret("USAGE_NOTIFY_FROM", secrets) or _secret("SMTP_FROM", secrets)
    if resend:
        return custom or DEFAULT_USAGE_NOTIFY_FROM
    return custom or _secret("SMTP_USER", secrets) or "grokotono@localhost"


def _log_status(status: str) -> None:
    line = f"[grokotono] email_status={status}"
    _LOG.info(line)
    print(line, flush=True)


def maybe_notify_email(
    row: Mapping[str, str],
    *,
    secrets: Mapping[str, Any] | None = None,
) -> str:
    """Siempre intenta avisar tras una corrida exitosa.

    Prefiere Resend si hay RESEND_API_KEY; si no, SMTP. Destino por defecto:
    DEFAULT_USAGE_NOTIFY_EMAIL. Anulable con USAGE_NOTIFY_EMAIL.
    Sin canal: 'skipped_no_secrets'. Un fallo de envío no se relanza:
    se devuelve 'error:…'. SMTP_PASSWORD / RESEND_API_KEY solo de secrets/env.
    """
    try:
        to_addr = notify_destination(secrets)
        if not to_addr:
            return "skipped_no_secrets"
        subject, body = _compose_email(row)
        resend_key = _secret("RESEND_API_KEY", secrets)
        host = _secret("SMTP_HOST", secrets)
        if not resend_key and not host:
            return "skipped_no_secrets"
        if resend_key:
            _send_resend(
                api_key=resend_key,
                to_addr=to_addr,
                from_addr=_from_address(secrets, resend=True),
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
            from_addr=_from_address(secrets, resend=False),
            subject=subject,
            body=body,
        )
        return "smtp"
    except Exception as exc:  # noqa: BLE001 — el correo no debe tumbar la corrida
        return f"error:{exc}"[:180]


def record_run_and_notify(
    *,
    marca: str,
    aliases: Sequence[str] | None = None,
    n_rows: int,
    tono_counts: Mapping[str, int] | None = None,
    model: str = "",
    elapsed_s: float = 0.0,
    cost_usd: float | None = None,
    secrets: Mapping[str, Any] | None = None,
    path: Path | None = None,
) -> dict[str, str]:
    """Append CSV + always attempt email. email_status queda en el log."""
    draft = _build_row(
        marca=marca,
        aliases=aliases,
        n_rows=n_rows,
        tono_counts=tono_counts,
        model=model,
        elapsed_s=elapsed_s,
        cost_usd=cost_usd,
    )
    status = maybe_notify_email(draft, secrets=secrets)
    _log_status(status)
    return record_run(
        marca=marca,
        aliases=aliases,
        n_rows=n_rows,
        tono_counts=tono_counts,
        model=model,
        elapsed_s=elapsed_s,
        cost_usd=cost_usd,
        email_status=status,
        path=path,
    )
