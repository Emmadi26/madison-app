import io
import json
import time
from datetime import date

import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------
MODEL = "gemini-2.5-flash-lite"
MAX_FEEDS = 50          # keeps the page fast and protects the free AI quota
MAX_TRIES = 3           # AI retries
REQUIRED = ["feed_name", "rows_today", "baseline_rows", "blank_percent", "hours_late"]

SAMPLE_CSV = """feed_name,rows_today,baseline_rows,blank_percent,hours_late
Vendor Pricing Feed,50,50,0.5,0
Benchmark Rates Feed,29,29,0,0.2
Positions Feed,410,900,1.2,0
Index Constituents Feed,1500,1520,34,0
FX Rates Feed,180,185,2,6.5
Corporate Actions Feed,72,100,3,0.5
Holdings Feed,230,520,28,7
Reference Data Feed,800,805,0.8,0
"""

TEMPLATE_CSV = "feed_name,rows_today,baseline_rows,blank_percent,hours_late\nExample Feed,100,100,0,0\n"


# ---------------------------------------------------------------
# STEP 1: READ AND CHECK THE FILE (plain-English errors)
# ---------------------------------------------------------------
class InputProblem(Exception):
    """A problem with the user's file, written so a non-technical person can fix it."""


def load_feeds(raw_bytes):
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception:
        raise InputProblem(
            "I could not read that file. Please upload a CSV file "
            "(in Excel, choose File > Save As > CSV)."
        )
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    if df.empty:
        raise InputProblem("The file has no rows. Add one row per vendor feed and try again.")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise InputProblem(
            "Your file is missing these columns: " + ", ".join(missing)
            + ". Click 'Download the blank template' above to see the exact column names."
        )
    df = df[REQUIRED].copy()
    for col in REQUIRED[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    bad = df[df[REQUIRED[1:]].isna().any(axis=1)]
    if not bad.empty:
        names = ", ".join(str(n) for n in bad["feed_name"].head(3))
        raise InputProblem(
            f"Some numbers are missing or are not numbers (check these feeds: {names}). "
            "Every cell except the feed name must contain a number."
        )
    if (df[REQUIRED[1:]] < 0).any().any():
        raise InputProblem("Numbers cannot be negative. Please check rows_today, baseline_rows, blank_percent and hours_late.")
    if (df["baseline_rows"] <= 0).any():
        raise InputProblem("baseline_rows must be above zero for every feed (it is the normal number of rows).")
    if (df["blank_percent"] > 100).any():
        raise InputProblem("blank_percent must be between 0 and 100.")
    if len(df) > MAX_FEEDS:
        raise InputProblem(f"That file has {len(df)} feeds. This tool checks up to {MAX_FEEDS} at a time. Please split the file.")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------
# STEP 2: THE RULES (same as my Assignment 4 workflow)
# ---------------------------------------------------------------
RANK = {"OK": 0, "WARNING": 1, "CRITICAL": 2}


def check_feed(row):
    status = "OK"
    flags = []

    def bump(s):
        nonlocal status
        if RANK[s] > RANK[status]:
            status = s

    drop = round((row["baseline_rows"] - row["rows_today"]) / row["baseline_rows"] * 100)
    if drop > 20:
        flags.append(f"Row count is down {drop}% compared with normal")
        bump("WARNING")
    if drop > 50:
        bump("CRITICAL")
    if row["blank_percent"] > 5:
        flags.append(f"{row['blank_percent']:g}% of fields are blank (limit is 5%)")
        bump("WARNING")
    if row["blank_percent"] > 20:
        bump("CRITICAL")
    if row["hours_late"] > 1:
        flags.append(f"Arrived {row['hours_late']:g} hours later than usual")
        bump("WARNING")
    if row["hours_late"] > 4:
        bump("CRITICAL")
    return status, flags, drop


def run_rules(df):
    out = []
    for _, r in df.iterrows():
        status, flags, drop = check_feed(r)
        out.append({
            "feed": str(r["feed_name"]),
            "rows_today": int(r["rows_today"]),
            "baseline_rows": int(r["baseline_rows"]),
            "row_change_pct": -drop,
            "blank_percent": float(r["blank_percent"]),
            "hours_late": float(r["hours_late"]),
            "status": status,
            "flags": flags,
        })
    return out


# ---------------------------------------------------------------
# STEP 3: THE AI EXPLANATION (one call for all problem feeds)
# ---------------------------------------------------------------
def _google_error(e):
    code = getattr(getattr(e, "response", None), "status_code", None)
    detail = ""
    try:
        detail = e.response.json()["error"]["message"][:200]
    except Exception:
        pass
    return code, detail


def _call(url, api_key, prompt):
    resp = requests.post(
        url,
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        json={
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"},
        },
        timeout=60,
    )
    resp.raise_for_status()
    text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    data = json.loads(text.replace("```json", "").replace("```", "").strip())
    result = {}
    for item in data:
        if (item.get("feed") and item.get("explanation") and item.get("action")
                and item.get("severity") in RANK):
            result[item["feed"]] = item
    if not result:
        raise ValueError("The AI reply had no usable items")
    return result


def discover_models(api_key):
    """Asks Google which models this key can use, newest flash models first."""
    try:
        r = requests.get("https://generativelanguage.googleapis.com/v1beta/models?pageSize=100",
                         headers={"x-goog-api-key": api_key}, timeout=30)
        r.raise_for_status()
        names = []
        for m in r.json().get("models", []):
            n = m.get("name", "").replace("models/", "")
            if ("generateContent" in m.get("supportedGenerationMethods", []) and "flash" in n
                    and not any(w in n for w in ["lite", "image", "tts", "live", "audio", "thinking", "preview", "exp"])):
                names.append(n)
        return sorted(names, reverse=True)[:3]
    except Exception:
        return []


def ask_gemini(problems, api_key):
    lines = []
    for p in problems:
        lines.append(
            f"- {p['feed']}: rows {p['rows_today']} (normal {p['baseline_rows']}, change {p['row_change_pct']}%), "
            f"blanks {p['blank_percent']}%, hours late {p['hours_late']}, rule status {p['status']}. "
            f"Flags: {'; '.join(p['flags'])}"
        )
    prompt = (
        "You are Madison, a data quality assistant for an investment data operations analyst. "
        "These overnight vendor feeds failed checks against their own normal pattern.\n"
        + "\n".join(lines)
        + "\nFor EACH feed reply with JSON only: a list of objects with keys "
        '"feed" (exactly as given), "explanation" (2 plain sentences using the real numbers), '
        '"action" (one concrete step to take before the trading desk opens), '
        '"severity" (OK, WARNING or CRITICAL), "reason" (why you kept or changed the rule status).'
    )
    api_key = str(api_key).strip().strip('"').strip("'")
    vertex = "https://aiplatform.googleapis.com/v1/publishers/google/models/{m}:generateContent"
    studio = "https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"
    order = [vertex, studio] if api_key.startswith("AQ.") else [studio, vertex]
    last = "no answer"

    def try_urls(models):
        nonlocal last
        for m in models:
            for u in order:
                try:
                    return _call(u.format(m=m), api_key, prompt)
                except Exception as e:
                    code, detail = _google_error(e)
                    last = f"status {code}. Google said: {detail}" if code else type(e).__name__
                    if code == 429:
                        raise RuntimeError(last)
        return None

    for attempt in range(1, MAX_TRIES + 1):
        out = try_urls([MODEL])
        if out:
            return out
        if "status 404" in last or "status 400" in last or "status 403" in last:
            out = try_urls(discover_models(api_key))   # ask Google which models this key can use
            if out:
                return out
        if attempt < MAX_TRIES:
            time.sleep(2 * 2 ** (attempt - 1))
    raise RuntimeError(last)


def friendly_reason(err, has_key):
    if not has_key:
        return "No AI key was found in the app settings (Secrets)."
    e = str(err)
    if "status 429" in e:
        return "The free AI limit was reached. Wait a minute and try again."
    return "The AI service said no. " + e


def add_explanations(results, api_key):
    """Fills in explanation, action and final status. Falls back to plain rule text."""
    problems = [r for r in results if r["status"] != "OK"]
    ai = {}
    ai_ok = True
    reason = ""
    if problems and api_key:
        try:
            ai = ask_gemini(problems, api_key)
        except Exception as e:
            ai_ok = False
            reason = friendly_reason(e, True)
    elif problems:
        ai_ok = False
        reason = friendly_reason("", False)
    for r in results:
        if r["status"] == "OK":
            r.update(final_status="OK", explanation="No issues found.", action="Nothing to do.", ai_used=False)
            continue
        hit = ai.get(r["feed"])
        if hit:
            r.update(final_status=hit["severity"], explanation=hit["explanation"],
                     action=hit["action"], ai_used=True)
        else:
            r.update(final_status=r["status"],
                     explanation="Automatic explanation unavailable. Problems found: " + "; ".join(r["flags"]) + ".",
                     action="Check this feed by hand before the trading desk opens.", ai_used=False)
    return results, ai_ok, reason


# ---------------------------------------------------------------
# STEP 4: THE REPORT FILE (HTML the user can download and email)
# ---------------------------------------------------------------
def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_html(results):
    colors = {"CRITICAL": "#fde2e2", "WARNING": "#fff3cd", "OK": "#e3f6e8"}
    rows = ""
    for r in results:
        rows += (
            f"<tr style='background:{colors[r['final_status']]}'><td><b>{esc(r['feed'])}</b></td>"
            f"<td><b>{r['final_status']}</b></td><td>{r['rows_today']} / {r['baseline_rows']} ({r['row_change_pct']}%)</td>"
            f"<td>{r['blank_percent']:g}%</td><td>{r['hours_late']:g}h</td>"
            f"<td>{esc(r['explanation'])}</td><td>{esc(r['action'])}</td></tr>"
        )
    return (
        "<html><body style='font-family:Arial,sans-serif;max-width:1000px;margin:auto;padding:20px'>"
        f"<h1>Madison Morning Feed Check: {date.today()}</h1>"
        "<table border='1' cellpadding='8' style='border-collapse:collapse;width:100%;font-size:13px'>"
        "<tr style='background:#222;color:#fff'><th>Feed</th><th>Status</th><th>Rows vs normal</th>"
        "<th>Blanks</th><th>Late</th><th>What happened</th><th>What to do</th></tr>"
        f"{rows}</table></body></html>"
    )


# ---------------------------------------------------------------
# THE PAGE
# ---------------------------------------------------------------
def main():
    st.set_page_config(page_title="Madison: Morning Feed Check", page_icon="📋", layout="wide")
    st.title("Madison: Morning Vendor Feed Check")
    st.write(
        "Upload last night's vendor feed numbers. Madison checks each feed against its normal pattern "
        "and tells you in plain English which ones to fix before the trading desk opens."
    )

    with st.expander("What goes in the file? (click to open)"):
        st.write(
            "One row per vendor feed, with 5 columns:\n"
            "- **feed_name**: the name of the feed\n"
            "- **rows_today**: how many rows arrived last night\n"
            "- **baseline_rows**: how many rows it normally has\n"
            "- **blank_percent**: percent of fields that are empty (0 to 100)\n"
            "- **hours_late**: hours later than usual it arrived (0 if on time)"
        )
        st.download_button("Download the blank template", TEMPLATE_CSV, "madison_template.csv", "text/csv")

    uploaded = st.file_uploader("Step 1: Upload your feed file (CSV)", type=["csv"])
    use_sample = st.button("Or try it with sample data")

    raw = None
    if uploaded is not None:
        raw = uploaded.getvalue()
    elif use_sample or st.session_state.get("sample_on"):
        st.session_state["sample_on"] = True
        raw = SAMPLE_CSV.encode()
        st.info("Showing sample data. Upload your own file above to check real feeds.")

    if raw is None:
        st.stop()

    try:
        df = load_feeds(raw)
    except InputProblem as e:
        st.error(str(e))
        st.stop()

    api_key = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""
    with st.spinner("Checking your feeds. This can take up to a minute..."):
        results = run_rules(df)
        results, ai_ok, reason = add_explanations(results, api_key)

    results.sort(key=lambda r: RANK[r["final_status"]], reverse=True)
    crit = sum(r["final_status"] == "CRITICAL" for r in results)
    warn = sum(r["final_status"] == "WARNING" for r in results)
    good = sum(r["final_status"] == "OK" for r in results)

    if crit:
        st.error(f"{crit} feed(s) need action before the desk opens.")
    elif warn:
        st.warning("No critical problems, but some feeds need a look.")
    else:
        st.success("All feeds look normal. The desk can open.")
    if not ai_ok:
        st.info("The AI explanation service was not available, so these results use simple rule-based messages. Reason: " + reason)

    c1, c2, c3 = st.columns(3)
    c1.metric("Critical", crit)
    c2.metric("Warning", warn)
    c3.metric("OK", good)

    st.subheader("Results, worst first")
    for r in results:
        title = f"{r['final_status']}: {r['feed']}"
        box = {"CRITICAL": st.error, "WARNING": st.warning, "OK": st.success}[r["final_status"]]
        box(title)
        if r["final_status"] != "OK":
            st.write(f"**What happened:** {r['explanation']}")
            st.write(f"**What to do:** {r['action']}")
            st.caption(
                f"Rows {r['rows_today']} vs normal {r['baseline_rows']} ({r['row_change_pct']}%), "
                f"blanks {r['blank_percent']:g}%, {r['hours_late']:g} hours late"
            )

    st.caption("App version 6")
    st.download_button("Download this report (HTML)", build_html(results),
                       f"madison_report_{date.today()}.html", "text/html")


if __name__ == "__main__":
    main()
