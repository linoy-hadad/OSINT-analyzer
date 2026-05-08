from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st


DATA_JSON_PATH = Path("data.json")
SCREENSHOTS_DIR = Path("screenshots")

COMPACT_ANALYSIS_KEYS = {
    "analyst_number",
    "base_id",
    "findings",
    "analysis",
    "things_to_continue_analyzing",
    "action",
    "question",
    "answer",
}


def load_data() -> dict[str, Any]:
    if not DATA_JSON_PATH.exists():
        return {}

    with DATA_JSON_PATH.open("r", encoding="utf-8") as data_file:
        loaded = json.load(data_file)

    if not isinstance(loaded, dict):
        return {}

    return loaded


def list_text(value: object) -> list[str]:
    if not isinstance(value, list):
        return []

    return [str(item).strip() for item in value if str(item).strip()]


def text_preview(value: object, max_length: int = 220) -> str:
    text = str(value or "").strip().replace("\n", " ")
    if len(text) <= max_length:
        return text
    return f"{text[: max_length - 1].rstrip()}..."


def as_record(base_id: str, value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        record = dict(value)
    else:
        record = {}

    record.setdefault("base_id", base_id)
    record.setdefault("country", "Unknown")
    record.setdefault("analyses", [])
    record.setdefault("commander", {})
    return record


def get_screenshot_folder(record: dict[str, Any]) -> Path:
    folder = str(record.get("screenshot_folder") or "").strip()
    if folder:
        return Path(folder)

    return SCREENSHOTS_DIR / f"base_{record.get('base_id', '')}"


def overview_image_path(record: dict[str, Any]) -> Path | None:
    base_id = str(record.get("base_id") or "").strip()
    folder = get_screenshot_folder(record)
    candidates = [
        folder / "analyst_1.jpg",
        SCREENSHOTS_DIR / f"base_id_{base_id}.jpg",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def analyst_image_path(record: dict[str, Any], analyst_number: int) -> Path | None:
    candidate = get_screenshot_folder(record) / f"analyst_{analyst_number}.jpg"
    if candidate.exists():
        return candidate
    return None


def get_analyses(record: dict[str, Any]) -> list[dict[str, Any]]:
    analyses = record.get("analyses")
    if not isinstance(analyses, list):
        return []

    cleaned: list[dict[str, Any]] = []
    for item in analyses:
        if isinstance(item, dict):
            cleaned.append(item)

    return sorted(cleaned, key=lambda item: int(item.get("analyst_number") or 0))


def total_findings(analyses: list[dict[str, Any]]) -> int:
    return sum(len(list_text(item.get("findings"))) for item in analyses)


def final_action(analyses: list[dict[str, Any]]) -> str:
    if not analyses:
        return "unknown"

    return str(analyses[-1].get("action") or "unknown")


def skipped_analysts(analyses: list[dict[str, Any]]) -> list[int]:
    numbers = {
        int(item.get("analyst_number"))
        for item in analyses
        if str(item.get("analyst_number") or "").isdigit()
    }
    if not numbers:
        return []

    return [number for number in range(min(numbers), max(numbers) + 1) if number not in numbers]


def commander(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("commander")
    if isinstance(value, dict):
        return value
    return {}


def commander_summary(record: dict[str, Any]) -> str:
    value = commander(record).get("analysis", "")
    return str(value or "").strip()


def render_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(59, 92, 138, 0.15), transparent 30rem),
                #0b1016;
        }
        [data-testid="stHeader"] {
            background: rgba(11, 16, 22, 0.76);
        }
        .block-container {
            max-width: 1240px;
            padding-top: 2.4rem;
            padding-bottom: 3rem;
        }
        .intel-kicker {
            color: #8fb3ff;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }
        .intel-title {
            color: #f3f7fb;
            font-size: 2.35rem;
            font-weight: 800;
            line-height: 1.08;
            margin: 0.2rem 0 0.4rem;
        }
        .intel-subtitle {
            color: #9aa9b8;
            font-size: 1rem;
            max-width: 780px;
        }
        .report-card,
        .panel,
        .analyst-card {
            border: 1px solid rgba(143, 179, 255, 0.18);
            background: rgba(19, 27, 36, 0.76);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 18px 50px rgba(0, 0, 0, 0.25);
        }
        .report-card {
            min-height: 230px;
        }
        .card-title {
            color: #f3f7fb;
            font-size: 1.25rem;
            font-weight: 760;
            margin-bottom: 0.15rem;
        }
        .muted {
            color: #9aa9b8;
        }
        .metric-row {
            display: flex;
            gap: 0.55rem;
            flex-wrap: wrap;
            margin: 0.8rem 0;
        }
        .pill {
            border: 1px solid rgba(143, 179, 255, 0.22);
            color: #c8d7e6;
            background: rgba(143, 179, 255, 0.08);
            border-radius: 999px;
            padding: 0.22rem 0.6rem;
            font-size: 0.78rem;
            font-weight: 650;
        }
        .warning-pill {
            border-color: rgba(255, 196, 87, 0.34);
            color: #ffd58c;
            background: rgba(255, 196, 87, 0.08);
        }
        .section-title {
            color: #f3f7fb;
            font-size: 1.35rem;
            font-weight: 780;
            margin: 1.4rem 0 0.65rem;
        }
        .analyst-heading {
            color: #f3f7fb;
            font-size: 1.1rem;
            font-weight: 760;
            margin-bottom: 0.25rem;
        }
        .action-tag {
            display: inline-block;
            border: 1px solid rgba(120, 210, 175, 0.3);
            color: #95e7c4;
            background: rgba(120, 210, 175, 0.08);
            border-radius: 6px;
            padding: 0.18rem 0.5rem;
            font-size: 0.78rem;
            font-weight: 700;
        }
        .question-box {
            border-left: 3px solid #8fb3ff;
            background: rgba(143, 179, 255, 0.08);
            padding: 0.75rem 0.85rem;
            border-radius: 0 6px 6px 0;
            margin-top: 0.7rem;
        }
        .answer-box {
            border-left: 3px solid #95e7c4;
            background: rgba(120, 210, 175, 0.07);
            padding: 0.75rem 0.85rem;
            border-radius: 0 6px 6px 0;
            margin-top: 0.7rem;
        }
        div[data-testid="stButton"] > button {
            width: 100%;
            border-radius: 6px;
            border: 1px solid rgba(143, 179, 255, 0.28);
            background: rgba(143, 179, 255, 0.12);
            color: #f3f7fb;
            font-weight: 700;
        }
        div[data-testid="stButton"] > button:hover {
            border-color: rgba(143, 179, 255, 0.75);
            background: rgba(143, 179, 255, 0.2);
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_missing_image(label: str) -> None:
    st.markdown(
        f"""
        <div class="panel">
          <div class="muted">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_list(title: str, items: list[str]) -> None:
    if not items:
        st.caption(f"No {title.lower()} recorded.")
        return

    with st.expander(title, expanded=False):
        for item in items:
            st.markdown(f"- {item}")


def render_gallery(records: dict[str, dict[str, Any]]) -> None:
    st.markdown('<div class="intel-kicker">OSINT ANALYSIS VIEWER</div>', unsafe_allow_html=True)
    st.markdown('<div class="intel-title">Intelligence Report Gallery</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="intel-subtitle">Browse analyzed bases, inspect analyst reasoning, and review commander conclusions from the compact data file.</div>',
        unsafe_allow_html=True,
    )

    query = st.text_input("Search by base ID or country", placeholder="Example: 147 or Egypt")
    filtered_items = []
    for base_id, record in records.items():
        haystack = f"{base_id} {record.get('country', '')}".lower()
        if query.strip().lower() in haystack:
            filtered_items.append((base_id, record))

    if not filtered_items:
        st.info("No matching bases found.")
        return

    cols = st.columns(3)
    for index, (base_id, record) in enumerate(filtered_items):
        analyses = get_analyses(record)
        image_path = overview_image_path(record)
        missing = []
        if image_path is None:
            missing.append("overview image missing")
        if not commander_summary(record):
            missing.append("commander output missing")
        skipped = skipped_analysts(analyses)
        if skipped:
            missing.append(f"skipped analysts: {', '.join(str(item) for item in skipped)}")

        with cols[index % 3]:
            st.markdown('<div class="report-card">', unsafe_allow_html=True)
            if image_path is not None:
                st.image(str(image_path), use_container_width=True)
            else:
                render_missing_image("No overview image available")

            st.markdown(
                f"""
                <div class="card-title">Base {base_id}</div>
                <div class="muted">{record.get("country", "Unknown")}</div>
                <div class="metric-row">
                  <span class="pill">{len(analyses)} analysts</span>
                  <span class="pill">{total_findings(analyses)} findings</span>
                  <span class="pill">final action: {final_action(analyses)}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            for warning in missing:
                st.markdown(
                    f'<span class="pill warning-pill">{warning}</span>',
                    unsafe_allow_html=True,
                )

            preview = text_preview(commander_summary(record), 260)
            if preview:
                st.caption(preview)

            if st.button("Open dossier", key=f"open_{base_id}"):
                st.session_state.selected_base_id = base_id
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)


def render_top_links(record: dict[str, Any]) -> None:
    link_cols = st.columns(2)
    initial_link = str(record.get("initial_google_earth_link") or "").strip()
    final_link = str(record.get("final_google_earth_link") or "").strip()

    with link_cols[0]:
        if initial_link:
            st.link_button("Initial Google Earth View", initial_link)
        else:
            st.caption("Initial Google Earth link missing.")

    with link_cols[1]:
        if final_link:
            st.link_button("Final Google Earth View", final_link)
        else:
            st.caption("Final Google Earth link missing.")


def render_commander_panel(record: dict[str, Any], compact: bool = False) -> None:
    value = commander(record)
    findings = list_text(value.get("findings"))
    analysis = str(value.get("analysis") or "").strip()
    final_steps = list_text(value.get("final_steps"))

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### Commander Assessment")
    if analysis:
        st.write(analysis if not compact else text_preview(analysis, 520))
    else:
        st.warning("Commander output is missing.")

    if not compact:
        render_list("Commander Findings", findings)
        render_list("Final Steps", final_steps)
        summary = {
            "findings": findings,
            "analysis": analysis,
            "final_steps": final_steps,
        }
        with st.expander("Copy commander summary", expanded=False):
            st.code(json.dumps(summary, indent=2, ensure_ascii=False), language="json")

    st.markdown("</div>", unsafe_allow_html=True)


def render_analysis(record: dict[str, Any], analysis: dict[str, Any]) -> None:
    analyst_number = int(analysis.get("analyst_number") or 0)
    image_path = analyst_image_path(record, analyst_number)

    st.markdown('<div class="analyst-card">', unsafe_allow_html=True)
    left, right = st.columns([0.95, 1.25], gap="large")

    with left:
        if image_path is not None:
            st.image(str(image_path), use_container_width=True)
        else:
            render_missing_image(f"Missing image for analyst {analyst_number}")

    with right:
        action = str(analysis.get("action") or "unknown")
        st.markdown(
            f"""
            <div class="analyst-heading">Analyst {analyst_number}</div>
            <span class="action-tag">{action}</span>
            """,
            unsafe_allow_html=True,
        )

        body = str(analysis.get("analysis") or "").strip()
        if body:
            st.write(body)
        else:
            st.caption("No analysis text recorded.")

        render_list("Findings", list_text(analysis.get("findings")))
        render_list(
            "Things To Continue Analyzing",
            list_text(analysis.get("things_to_continue_analyzing")),
        )

        question = str(analysis.get("question") or "").strip()
        answer = str(analysis.get("answer") or "").strip()
        if question:
            st.markdown(
                f"""
                <div class="question-box">
                  <strong>Consultant Question</strong><br>{question}
                </div>
                """,
                unsafe_allow_html=True,
            )
        if answer:
            st.markdown(
                f"""
                <div class="answer-box">
                  <strong>Moondream Answer</strong><br>{answer}
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif question:
            st.caption("No Moondream answer recorded for this question.")

    st.markdown("</div>", unsafe_allow_html=True)


def render_dossier(record: dict[str, Any]) -> None:
    base_id = str(record.get("base_id") or "Unknown")
    if st.button("Back to gallery", key="back_to_gallery"):
        st.session_state.selected_base_id = None
        st.rerun()

    st.markdown('<div class="intel-kicker">INTELLIGENCE DOSSIER</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="intel-title">Base {base_id}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="intel-subtitle">{record.get("country", "Unknown")} | Analyst review and commander conclusion</div>',
        unsafe_allow_html=True,
    )

    render_top_links(record)

    analyses = get_analyses(record)
    skipped = skipped_analysts(analyses)
    if skipped:
        st.warning(f"Skipped analyst numbers detected: {', '.join(str(item) for item in skipped)}")

    hero_left, hero_right = st.columns([1.15, 0.85], gap="large")
    with hero_left:
        image_path = overview_image_path(record)
        if image_path is not None:
            st.image(str(image_path), caption="Initial analyst image", use_container_width=True)
        else:
            render_missing_image("No first image available for this base")
    with hero_right:
        render_commander_panel(record, compact=True)

    st.markdown('<div class="section-title">Analyst Brainstorming</div>', unsafe_allow_html=True)
    if not analyses:
        st.info("No analyst records are available for this base.")
    for analysis in analyses:
        render_analysis(record, analysis)

    st.markdown('<div class="section-title">Final Commander Conclusion</div>', unsafe_allow_html=True)
    render_commander_panel(record, compact=False)


def main() -> None:
    st.set_page_config(
        page_title="OSINT Intelligence Reports",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    render_css()

    raw_data = load_data()
    records = {base_id: as_record(base_id, value) for base_id, value in raw_data.items()}

    if "selected_base_id" not in st.session_state:
        st.session_state.selected_base_id = None

    selected_base_id = st.session_state.selected_base_id
    if selected_base_id and selected_base_id in records:
        render_dossier(records[selected_base_id])
    else:
        st.session_state.selected_base_id = None
        render_gallery(records)


if __name__ == "__main__":
    main()
