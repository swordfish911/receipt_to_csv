import base64
import csv
import io
import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from anthropic import Anthropic

MODEL = "claude-sonnet-5-5"
APP_DIR = Path(__file__).parent
KEY_FILE = APP_DIR / "api_key.txt"

PROMPT = """You are reading a single receipt photo. Return ONLY a JSON object, with no preamble and no Markdown fences, using this exact shape:
{
  "date": "YYYY-MM-DD or empty string if unreadable",
  "merchant": "",
  "receipt_id": "",
  "currency": "ISO code, e.g. MYR",
  "total": number (the balance due or final total printed),
  "tax": number (0 if none shown),
  "adjustments": [ { "label": "name as printed, e.g. Service Charge, GST, SST, Delivery Fee, Rounding, Discount", "amount": number (positive for charges, negative for discounts) } ],
  "line_items": [ { "description": "", "quantity": "text with unit, e.g. 0.33 kg", "unit_price": number or null, "amount": number or null } ],
  "notes": "under 15 words, only naming text or numbers you could not read clearly. Do not comment on dates or arithmetic. Empty string if none."
}
Put every printed charge or discount that is not a line item into adjustments, including service charge and rounding. Tax goes in the tax field, not in adjustments. Formatting rules: inside any string value, never use double quotes or line breaks; use single quotes instead. Keep quantity as plain text such as "0.33 kg" with no unit repeated, and use null (not text) for unknown numbers. Rules: read the date as printed and convert to YYYY-MM-DD (Malaysian receipts are usually DD/MM/YYYY). Do not guess numbers you cannot read; leave amount null and flag it in notes."""


def load_api_key():
    """Read the key from api_key.txt next to this script, or from the environment."""
    if KEY_FILE.exists():
        key = KEY_FILE.read_text(encoding="utf-8").strip()
        if key:
            return key
    import os
    return os.environ.get("ANTHROPIC_API_KEY", "").strip()


ROUNDING_TOLERANCE = 0.05  # allows for cash rounding on the printed total


def parse_qty(text):
    """Pull the number out of text like '0.33 kg', '2.00' or '.00'. Returns None if none found."""
    if text is None:
        return None
    match = re.search(r"\d*\.?\d+", str(text))
    return float(match.group()) if match else None


def apply_checks(d):
    """Run arithmetic checks and add any problems to the notes field."""
    issues = []
    items = d.get("line_items") or []
    amounts = []

    for i, li in enumerate(items, start=1):
        qty = parse_qty(li.get("quantity"))
        price = li.get("unit_price")
        amount = li.get("amount")
        if amount is not None:
            amounts.append(float(amount))
        if qty is not None and price is not None and amount is not None:
            expected = qty * float(price)
            if abs(expected - float(amount)) > ROUNDING_TOLERANCE:
                issues.append(
                    f"Item {i} ({li.get('description', '')}): quantity x price = {expected:.2f}, "
                    f"but amount shows {float(amount):.2f}"
                )

    total = d.get("total")
    tax = float(d.get("tax") or 0)
    adjustments = d.get("adjustments") or []
    adj_sum = sum(float(a.get("amount") or 0) for a in adjustments)
    if amounts and total is not None:
        items_sum = sum(amounts)
        expected_total = items_sum + tax + adj_sum
        diff = expected_total - float(total)
        if abs(diff) > ROUNDING_TOLERANCE:
            issues.append(
                f"Items {items_sum:.2f} + tax {tax:.2f} + other charges {adj_sum:+.2f} = {expected_total:.2f}, "
                f"but printed total is {float(total):.2f} (difference {diff:+.2f})"
            )

    if issues:
        existing = (d.get("notes") or "").strip()
        check_text = "Arithmetic check failed: " + " | ".join(issues)
        d["notes"] = (existing + " " + check_text).strip()
    return d


def ask_claude(client, media_type, data, prompt):
    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def parse_json_reply(text):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON found in reply")
    return json.loads(text[start:end + 1], strict=False)


def read_receipt(client, uploaded_file):
    data = base64.standard_b64encode(uploaded_file.getvalue()).decode("utf-8")
    media_type = uploaded_file.type or "image/jpeg"

    text = ask_claude(client, media_type, data, PROMPT)
    try:
        return apply_checks(parse_json_reply(text))
    except (ValueError, json.JSONDecodeError):
        pass

    # Retry once with a reminder about the most common formatting mistake.
    retry_prompt = PROMPT + (
        "\n\nYour previous reply was not valid JSON. Output only the JSON object. "
        "Check that every quote, comma and bracket is correct and no string contains a double quote or line break."
    )
    text = ask_claude(client, media_type, data, retry_prompt)
    try:
        return apply_checks(parse_json_reply(text))
    except (ValueError, json.JSONDecodeError) as e:
        raise ValueError(f"{e}. Reply started with: {text[:200]}")


def num(value):
    if value is None or value == "":
        return ""
    return f"{float(value):.2f}"


def other_charges_text(d):
    parts = []
    for a in d.get("adjustments") or []:
        label = a.get("label") or "Other"
        parts.append(f"{label} {num(a.get('amount'))}")
    return "; ".join(parts)


def build_receipt_csv(results):
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["date", "merchant", "description", "currency", "amount", "tax", "other_charges", "category", "receipt_id", "notes"])
    for r in results:
        if r.get("error"):
            continue
        d = r["data"]
        desc = "; ".join(li.get("description", "") for li in d.get("line_items", []))
        writer.writerow([d.get("date", ""), d.get("merchant", ""), desc, d.get("currency", ""),
                         num(d.get("total")), num(d.get("tax")), other_charges_text(d), "", d.get("receipt_id", ""), d.get("notes", "")])
    return out.getvalue()


def build_item_csv(results):
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["date", "merchant", "description", "quantity", "unit_price", "amount", "currency", "receipt_id", "notes"])
    for r in results:
        if r.get("error"):
            continue
        d = r["data"]
        for idx, li in enumerate(d.get("line_items", [])):
            note = d.get("notes", "") if idx == 0 else ""
            writer.writerow([d.get("date", ""), d.get("merchant", ""), li.get("description", ""),
                             li.get("quantity", ""), num(li.get("unit_price")), num(li.get("amount")),
                             d.get("currency", ""), d.get("receipt_id", ""), note])
    return out.getvalue()


def build_table(results):
    rows = []
    for r in results:
        if r.get("error"):
            rows.append({"Date": "", "Merchant": "", "Receipt ID": "", "Item": r["error"],
                         "Qty": "", "Amount": "", "Flag": "Error", "File": r["file"]})
            continue
        d = r["data"]
        items = d.get("line_items") or [{"description": "(no items read)", "quantity": "", "amount": d.get("total")}]
        for li in items:
            rows.append({
                "Date": d.get("date", ""),
                "Merchant": d.get("merchant", ""),
                "Receipt ID": d.get("receipt_id", ""),
                "Item": li.get("description", ""),
                "Qty": li.get("quantity", ""),
                "Amount": num(li.get("amount")),
                "Flag": "Check" if d.get("notes") else "",
                "File": r["file"],
            })
    return pd.DataFrame(rows)


def highlight_flags(row):
    # Set the text colour explicitly, so it stays readable on the coloured background
    # whether the phone or browser is in dark or light mode.
    if row["Flag"] == "Check":
        return ["background-color: #fff4d6; color: #111111"] * len(row)
    if row["Flag"] == "Error":
        return ["background-color: #fde2e2; color: #111111"] * len(row)
    return [""] * len(row)


st.set_page_config(page_title="Receipt to CSV", layout="wide")
st.title("Alvin's Receipt to CSV Prototype 0.01")
st.caption("Upload receipt photos. Claude reads each one and produces CSV rows you can download.")

api_key = load_api_key()
if not api_key:
    st.info(
        "No API key yet. Create api_key.txt in this folder containing only your key, "
        "then restart the app. Until then, Read receipts stays disabled and nothing is sent to Claude."
    )

uploads = st.file_uploader(
    "Receipt photos (JPG or PNG)",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

if "results" not in st.session_state:
    st.session_state.results = []

if st.button("Read receipts", disabled=(not uploads or not api_key), type="primary"):
    client = Anthropic(api_key=api_key)
    results = []
    progress = st.progress(0.0, text="Starting...")
    for i, f in enumerate(uploads):
        progress.progress((i) / len(uploads), text=f"Reading {i + 1} of {len(uploads)}: {f.name}")
        try:
            results.append({"file": f.name, "data": read_receipt(client, f)})
        except Exception as e:
            results.append({"file": f.name, "error": f"Could not read: {e}"})
    progress.progress(1.0, text="Done")
    st.session_state.results = results

results = st.session_state.results
if results:
    ok = sum(1 for r in results if not r.get("error"))
    st.write(f"{ok} of {len(results)} receipt(s) read.")

    df = build_table(results)
    if not df.empty:
        st.dataframe(df.style.apply(highlight_flags, axis=1), use_container_width=True, hide_index=True)
        st.caption("Rows highlighted yellow need checking against the paper. Red rows failed to read.")

    col1, col2 = st.columns(2)
    col1.download_button("Download CSV (one row per receipt)", build_receipt_csv(results),
                         file_name="receipts.csv", mime="text/csv", disabled=ok == 0)
    col2.download_button("Download CSV (one row per item)", build_item_csv(results),
                         file_name="receipt_items.csv", mime="text/csv", disabled=ok == 0)
