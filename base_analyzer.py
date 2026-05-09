from __future__ import annotations

import csv
import json
import os
import shutil
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import moondream as md
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


ROWS_TO_PROCESS = 1
ANALYST_COUNT = 8

CSV_PATH = Path("military_bases.csv")
OUTPUT_DIR = Path("screenshots")
DATA_JSON_PATH = Path("data.json")
DEBUG_OUTPUT_DIR = Path("analyses")

SCREENSHOT_WIDTH = 1024
PAGE_LOAD_WAIT_SECONDS = 6
VISUAL_READY_TIMEOUT_SECONDS = 45
VISUAL_READY_POLL_SECONDS = 2
WINDOW_SIZE = (1600, 900)

DEFAULT_ALTITUDE = 10.04969521
DEFAULT_RANGE = 1825.78590766
DEFAULT_YAW = 30.00000016
DEFAULT_HEADING = -0.0
DEFAULT_TILT = 0.0
DEFAULT_ROLL = 0.0

ZOOM_IN_FACTOR = 0.85
ZOOM_OUT_FACTOR = 1.20
LON_STEP_CITY = 0.001

ANALYST_MODEL = "gemini-2.5-flash-lite"
COMMANDER_MODEL = "gemini-2.5-flash"
MOONDREAM_MODEL = "moondream-3-preview"
GOOGLE_API_KEY_ENV_VAR = "GOOGLE_API_KEY"
MOONDREAM_API_KEY_ENV_VAR = "MOONDREAM_API_KEY"
GEMINI_MAX_RETRIES = 3
GEMINI_RETRY_WAIT_SECONDS = 5
GEMINI_RATE_LIMIT_DELAY_SECONDS = 2  # Small delay between consecutive calls to avoid rate limits

ALLOWED_ACTIONS = {"zoom-in", "zoom-out", "move-left", "move-right", "finish"}
REQUIRED_ANALYST_KEYS = {
    "findings",
    "analysis",
    "things_to_continue_analyzing",
    "action",
    "question",
}
REQUIRED_COMMANDER_KEYS = {
    "findings",
    "analysis",
    "final_steps",
}


class GeminiAnalysisError(Exception):
    def __init__(self, message: str, raw_response_text: str | None = None) -> None:
        super().__init__(message)
        self.raw_response_text = raw_response_text


def build_consultant_question_instructions() -> str:
    return """
You also have access to an external visual consultant. She specializes in object-level image analysis: counting objects, identifying visual patterns, estimating object locations, and describing coordinates or areas within the image.

In the "question" field, ask her one specific question that would help the next analyst verify or deepen your assessment. Prefer concrete visual questions about counts, object types, locations, coordinates, suspicious structures, vehicles, aircraft, launchers, bunkers, radars, storage areas, roads, or infrastructure.

Ask only one question. If you are uncertain what to ask, ask a broad visual question about suspicious military-related objects or infrastructure in the image.
""".strip()


def format_camera_value(value: float) -> str:
    return format(value, ".14f").rstrip("0").rstrip(".")


def build_google_earth_url(
    latitude: float,
    longitude: float,
    altitude: float = DEFAULT_ALTITUDE,
    range_value: float = DEFAULT_RANGE,
    yaw: float = DEFAULT_YAW,
    heading: float = DEFAULT_HEADING,
    tilt: float = DEFAULT_TILT,
    roll: float = DEFAULT_ROLL,
) -> str:
    return (
        "https://earth.google.com/web/@"
        f"{format_camera_value(latitude)},"
        f"{format_camera_value(longitude)},"
        f"{format_camera_value(altitude)}a,"
        f"{format_camera_value(range_value)}d,"
        f"{format_camera_value(yaw)}y,"
        f"{format_camera_value(heading)}h,"
        f"{format_camera_value(tilt)}t,"
        f"{format_camera_value(roll)}r"
    )


def parse_camera_component(component: str, suffix: str, default: float) -> float:
    cleaned = component.strip()
    if cleaned.endswith(suffix):
        cleaned = cleaned[:-1]

    try:
        return float(cleaned)
    except ValueError:
        return default


def parse_google_earth_url(url: str) -> dict[str, float]:
    if "/@" not in url:
        raise ValueError(f"Unsupported Google Earth URL format: {url}")

    coordinate_part = url.split("/@", 1)[1].split("?", 1)[0]
    parts = coordinate_part.split(",")
    if len(parts) < 2:
        raise ValueError(f"Google Earth URL is missing latitude/longitude: {url}")

    latitude = float(parts[0])
    longitude = float(parts[1])

    return {
        "latitude": latitude,
        "longitude": longitude,
        "altitude": parse_camera_component(parts[2], "a", DEFAULT_ALTITUDE)
        if len(parts) > 2
        else DEFAULT_ALTITUDE,
        "range_value": parse_camera_component(parts[3], "d", DEFAULT_RANGE)
        if len(parts) > 3
        else DEFAULT_RANGE,
        "yaw": parse_camera_component(parts[4], "y", DEFAULT_YAW)
        if len(parts) > 4
        else DEFAULT_YAW,
        "heading": parse_camera_component(parts[5], "h", DEFAULT_HEADING)
        if len(parts) > 5
        else DEFAULT_HEADING,
        "tilt": parse_camera_component(parts[6], "t", DEFAULT_TILT)
        if len(parts) > 6
        else DEFAULT_TILT,
        "roll": parse_camera_component(parts[7], "r", DEFAULT_ROLL)
        if len(parts) > 7
        else DEFAULT_ROLL,
    }


def build_google_earth_url_from_state(camera_state: dict[str, float]) -> str:
    return build_google_earth_url(
        latitude=camera_state["latitude"],
        longitude=camera_state["longitude"],
        altitude=camera_state["altitude"],
        range_value=camera_state["range_value"],
        yaw=camera_state["yaw"],
        heading=camera_state["heading"],
        tilt=camera_state["tilt"],
        roll=camera_state["roll"],
    )


def apply_action_to_camera_state(camera_state: dict[str, float], action: str) -> dict[str, float]:
    next_state = dict(camera_state)

    if action == "zoom-in":
        next_state["range_value"] *= ZOOM_IN_FACTOR
    elif action == "zoom-out":
        next_state["range_value"] *= ZOOM_OUT_FACTOR
    elif action == "move-left":
        next_state["longitude"] -= LON_STEP_CITY
    elif action == "move-right":
        next_state["longitude"] += LON_STEP_CITY

    return next_state


def build_base_prompt(country: str) -> str:
    consultant_instructions = build_consultant_question_instructions()
    return f"""
You are the world first expert in understanding satellite imagery and you work for the US army. We got intel that this area is a base/facility of the millitary of {country}. As an expert, analyze this image, find millitary related things - structures and anything suspicous.

Respond ONLY with a valid JSON object with exactly these keys:
{{
  "findings": [
    "A list of findings that you think are important for the US army to know, including all man-made structures, military equipment, and infrastructure. We are trying to find which systems, weapons, or equipment are present and used so focus on that."
  ],
  "analysis": "A detailed analysis of your findings. For example if you saw an F-35 airplane, explain what the capabilities of this airplane are and what it is mostly used for.",
  "things_to_continue_analyzing": [
    "A list of things that you think are important to continue analyzing in further images, like areas to focus on, zoom into, structures to investigate, and so on."
  ],
  "action": "One of: zoom-in, zoom-out, move-left, move-right, finish",
  "question": "One specific question for the external visual consultant."
}}

{consultant_instructions}

Action rules:
- Choose "zoom-in" if you need to zoom in the image.
- Choose "zoom-out" if you need more context of the surrounding area or if you are zoomed in too much.
- Choose "move-left" or "move-right" if you suspect there are important features just outside the current view.
- Choose "finish" if you have a complete understanding of the location.

Do not include markdown fences.
Do not include any text before or after the JSON.
If uncertain, say so clearly inside the JSON fields instead of inventing facts.
""".strip()


def build_analysis_prompt(
    country: str,
    history_of_analysts: dict[str, str] | None = None,
    consultant_qas: list[dict[str, Any]] | None = None,
) -> str:
    if not history_of_analysts:
        return build_base_prompt(country)

    serialized_history = json.dumps(history_of_analysts, indent=2, ensure_ascii=False)
    serialized_consultant_qas = json.dumps(consultant_qas or [], indent=2, ensure_ascii=False)
    consultant_instructions = build_consultant_question_instructions()
    return f"""
You are the world first expert in understanding satellite imagery and you work for the US army. We got intel that this area is a base/facility of the millitary of {country}. Here is the analysis of previous analysts about this area and their recommendations. You can use this data but don't use it as fact, think for yourself: {serialized_history}

Previous analysts asked the external consultant these questions, and here are her answers:
{serialized_consultant_qas}

Use this as helpful context, but do not treat it as guaranteed fact. Think independently from the current image.

Respond ONLY with a valid JSON object with exactly these keys:
{{
  "findings": [
    "A list of findings that you think are important for the US army to know, including all man-made structures, military equipment, and infrastructure. We are trying to find which systems, weapons, or equipment are present and used so focus on that."
  ],
  "analysis": "A detailed analysis of your findings. For example if you saw an F-35 airplane, explain what the capabilities of this airplane are and what it is mostly used for.",
  "things_to_continue_analyzing": [
    "A list of things that you think are important to continue analyzing in further images, like areas to focus on, zoom into, structures to investigate, and so on."
  ],
  "action": "One of: zoom-in, zoom-out, move-left, move-right, finish",
  "question": "One specific question for the external visual consultant."
}}

{consultant_instructions}

Action rules:
- Choose "zoom-in" if you need to zoom in the image.
- Choose "zoom-out" if you need more context of the surrounding area or if you are zoomed in too much.
- Choose "move-left" or "move-right" if you suspect there are important features just outside the current view.
- Choose "finish" if you have a complete understanding of the location.

Do not include markdown fences.
Do not include any text before or after the JSON.
If uncertain, say so clearly inside the JSON fields instead of inventing facts.
""".strip()


def build_commander_prompt(
    history_of_analysts: dict[str, str],
    consultant_qas: list[dict[str, Any]] | None = None,
) -> str:
    serialized_history = json.dumps(history_of_analysts, indent=2, ensure_ascii=False)
    serialized_consultant_qas = json.dumps(consultant_qas or [], indent=2, ensure_ascii=False)
    return f"""
You are a commander of intelligence analysts in the us military. Your analysts got this image and an intel that this area is probably an enemy base/area. Here is the history of what the analysts said (each one was written by a different analyst) - {serialized_history}.

The analysts also asked an external visual consultant object-level questions. The consultant specializes in counting objects, locating visible objects, and answering concrete visual questions. Here are the consultant questions and answers: {serialized_consultant_qas}. Use these answers as supporting evidence, but do not treat them as guaranteed fact.

As their commander, you should read their estimates and give a final conclustion that includes:
1. findings: summary of all the findings your analysts found plus things you noticed yourself. This part should include only findings, structures, and everything that can be seen in the image.
2. analysis: your final analysis based on what your analysts think and what you additionally add. Tell us what this place is, the chances that the intel is correct, and the meaning of the findings.
3. final_steps: tell us what the us army should do with that, for example follow this base regularly, ignore it, get more images, or prepare an intelligence document.

Respond ONLY with a valid JSON object with exactly these keys:
{{
  "findings": [
    "A list summarizing the visible findings from the image and the analyst history."
  ],
  "analysis": "Your final commander analysis.",
  "final_steps": [
    "A list of recommended next steps."
  ]
}}

Do not include markdown fences.
Do not include any text before or after the JSON.
If uncertain, say so clearly inside the JSON fields instead of inventing facts.
""".strip()


def ensure_google_earth_links(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError("CSV file is missing a header row.")

        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    if "google_earth_link" not in fieldnames:
        fieldnames.append("google_earth_link")

    updated = False
    for row in rows:
        latitude_text = (row.get("latitude") or "").strip()
        longitude_text = (row.get("longitude") or "").strip()
        existing_link = (row.get("google_earth_link") or "").strip()

        if not latitude_text or not longitude_text:
            continue

        latitude = float(latitude_text)
        longitude = float(longitude_text)
        expected_link = build_google_earth_url(latitude, longitude)

        if existing_link != expected_link:
            row["google_earth_link"] = expected_link
            updated = True

    if updated or "google_earth_link" not in reader.fieldnames:
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return rows


def load_data_json() -> dict[str, Any]:
    if not DATA_JSON_PATH.exists():
        return {}

    with DATA_JSON_PATH.open("r", encoding="utf-8") as data_file:
        loaded = json.load(data_file)

    if not isinstance(loaded, dict):
        raise ValueError("data.json must contain a top-level JSON object.")

    return loaded


def save_data_json(data: dict[str, Any]) -> None:
    with DATA_JSON_PATH.open("w", encoding="utf-8") as data_file:
        json.dump(data, data_file, indent=2, ensure_ascii=False)


def save_debug_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as debug_file:
        json.dump(data, debug_file, indent=2, ensure_ascii=False)


def create_gemini_client() -> genai.Client:
    api_key = os.getenv(GOOGLE_API_KEY_ENV_VAR, "").strip()
    if not api_key:
        raise ValueError(
            f"Missing {GOOGLE_API_KEY_ENV_VAR} environment variable. "
            "Set it before running Gemini analysis."
        )

    return genai.Client(api_key=api_key)


def create_moondream_client() -> Any:
    api_key = os.getenv(MOONDREAM_API_KEY_ENV_VAR, "").strip()
    if not api_key:
        raise ValueError(
            f"Missing {MOONDREAM_API_KEY_ENV_VAR} environment variable. "
            "Set it before running Moondream consultant analysis."
        )

    return md.vl(api_key=api_key)


def create_driver() -> webdriver.Chrome:
    chrome_options = Options()
    chrome_options.add_argument(f"--window-size={WINDOW_SIZE[0]},{WINDOW_SIZE[1]}")
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--start-fullscreen")
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument("--hide-scrollbars")
    chrome_options.add_argument("--enable-webgl")
    chrome_options.add_argument("--ignore-gpu-blocklist")
    chrome_options.add_argument("--enable-accelerated-2d-canvas")
    chrome_options.add_argument("--use-angle=d3d11")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)
    return webdriver.Chrome(options=chrome_options)


def resize_and_save_as_jpeg(image_bytes: bytes, destination: Path) -> None:
    with Image.open(BytesIO(image_bytes)) as image:
        rgb_image = image.convert("RGB")
        width, height = rgb_image.size

        if width == 0:
            raise ValueError("Captured screenshot has invalid width 0.")

        new_height = int(height * (SCREENSHOT_WIDTH / width))
        resized = rgb_image.resize((SCREENSHOT_WIDTH, new_height), Image.Resampling.LANCZOS)
        resized.save(destination, format="JPEG", quality=90, optimize=True)


def is_mostly_black(image_bytes: bytes, threshold: int = 12, min_dark_ratio: float = 0.92) -> bool:
    with Image.open(BytesIO(image_bytes)) as image:
        grayscale = image.convert("L")
        pixels = list(grayscale.getdata())

    if not pixels:
        return True

    dark_pixels = sum(1 for pixel in pixels if pixel <= threshold)
    return (dark_pixels / len(pixels)) >= min_dark_ratio


def normalize_string_list(value: object, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"'{field_name}' must be a list.")

    normalized_items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"All items in '{field_name}' must be strings.")

        cleaned = item.strip()
        if cleaned:
            normalized_items.append(cleaned)

    return normalized_items


def validate_analyst_response(data: object) -> dict[str, object]:
    if not isinstance(data, dict):
        raise ValueError("Gemini response must be a JSON object.")

    missing_keys = REQUIRED_ANALYST_KEYS - data.keys()
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"Gemini response missing required keys: {missing}")

    findings = normalize_string_list(data["findings"], "findings")
    continue_items = normalize_string_list(
        data["things_to_continue_analyzing"],
        "things_to_continue_analyzing",
    )

    analysis = data["analysis"]
    if not isinstance(analysis, str):
        raise ValueError("'analysis' must be a string.")
    analysis = analysis.strip()

    action = data["action"]
    if not isinstance(action, str):
        raise ValueError("'action' must be a string.")
    action = action.strip()
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"'action' must be one of: {', '.join(sorted(ALLOWED_ACTIONS))}.")

    question = data["question"]
    if not isinstance(question, str):
        raise ValueError("'question' must be a string.")
    question = question.strip()
    if not question:
        raise ValueError("'question' must not be empty.")

    return {
        "findings": findings,
        "analysis": analysis,
        "things_to_continue_analyzing": continue_items,
        "action": action,
        "question": question,
    }


def validate_commander_response(data: object) -> dict[str, object]:
    if not isinstance(data, dict):
        raise ValueError("Commander response must be a JSON object.")

    missing_keys = REQUIRED_COMMANDER_KEYS - data.keys()
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"Commander response missing required keys: {missing}")

    findings = normalize_string_list(data["findings"], "findings")
    final_steps = normalize_string_list(data["final_steps"], "final_steps")

    analysis = data["analysis"]
    if not isinstance(analysis, str):
        raise ValueError("'analysis' must be a string.")
    analysis = analysis.strip()

    return {
        "findings": findings,
        "analysis": analysis,
        "final_steps": final_steps,
    }


def is_transient_gemini_error(value: object) -> bool:
    text = str(value).lower()
    transient_markers = (
        "503",
        "unavailable",
        "service unavailable",
        "overloaded",
        "429",
        "too many requests",
        "rate limit",
        "quota exceeded",
    )
    return any(marker in text for marker in transient_markers)


def extract_json_object_text(raw_text: str) -> str:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    if cleaned.startswith("{"):
        return cleaned

    start_index = cleaned.find("{")
    end_index = cleaned.rfind("}")
    if start_index != -1 and end_index != -1 and end_index > start_index:
        return cleaned[start_index : end_index + 1]

    return cleaned


def call_gemini_with_image(
    client: genai.Client,
    image_path: Path,
    prompt: str,
    model: str,
) -> str:
    image_bytes = image_path.read_bytes()

    max_attempts = GEMINI_MAX_RETRIES + 1
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    prompt,
                ],
            )
            raw_response_text = (response.text or "").strip()
            if is_transient_gemini_error(raw_response_text) and attempt < max_attempts:
                wait_time = GEMINI_RETRY_WAIT_SECONDS * (2 ** (attempt - 1))  # Exponential backoff
                print(f"Rate limited by Gemini. Waiting {wait_time}s before retry (attempt {attempt}/{max_attempts})...")
                time.sleep(wait_time)
                continue

            if is_transient_gemini_error(raw_response_text):
                raise GeminiAnalysisError(
                    "Gemini returned a transient service error after retries.",
                    raw_response_text,
                )

            if not raw_response_text:
                raise GeminiAnalysisError("Gemini returned an empty response.")

            return raw_response_text
        except Exception as error:
            if attempt < max_attempts and is_transient_gemini_error(error):
                wait_time = GEMINI_RETRY_WAIT_SECONDS * (2 ** (attempt - 1))  # Exponential backoff
                print(f"Gemini error, retrying in {wait_time}s (attempt {attempt}/{max_attempts})...")
                time.sleep(wait_time)
                continue
            raise

    raise GeminiAnalysisError("Gemini request failed after retries.")


def analyze_image(
    client: genai.Client,
    image_path: Path,
    prompt: str,
) -> tuple[str, dict[str, object]]:
    raw_response_text = call_gemini_with_image(
        client=client,
        image_path=image_path,
        prompt=prompt,
        model=ANALYST_MODEL,
    )

    try:
        parsed = json.loads(extract_json_object_text(raw_response_text))
        normalized = validate_analyst_response(parsed)
    except (json.JSONDecodeError, ValueError) as error:
        raise GeminiAnalysisError(str(error), raw_response_text) from error

    return raw_response_text, normalized


def analyze_commander(
    client: genai.Client,
    image_path: Path,
    history_of_analysts: dict[str, str],
    consultant_qas: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    prompt = build_commander_prompt(history_of_analysts, consultant_qas)

    try:
        raw_response_text = call_gemini_with_image(
            client=client,
            image_path=image_path,
            prompt=prompt,
            model=COMMANDER_MODEL,
        )
        parsed = json.loads(extract_json_object_text(raw_response_text))
        normalized = validate_commander_response(parsed)
        return {
            "model": COMMANDER_MODEL,
            "input_image_path": str(image_path),
            "prompt": prompt,
            "raw_response_text": raw_response_text,
            "validated_response": normalized,
            "status": "success",
            "error": None,
        }
    except Exception as error:
        raw_response_text = getattr(error, "raw_response_text", None)
        if raw_response_text is None and hasattr(error, "response"):
            response = getattr(error, "response", None)
            if response is not None:
                raw_response_text = str(response)

        return {
            "model": COMMANDER_MODEL,
            "input_image_path": str(image_path),
            "prompt": prompt,
            "raw_response_text": raw_response_text,
            "validated_response": None,
            "status": "error",
            "error": str(error),
        }


def extract_moondream_answer(result: object) -> str:
    if isinstance(result, dict):
        answer = result.get("answer", "")
    else:
        answer = getattr(result, "answer", "")

    return str(answer or "").strip()


def extract_moondream_request_id(result: object) -> str | None:
    if isinstance(result, dict):
        request_id = result.get("request_id")
    else:
        request_id = getattr(result, "request_id", None)

    if request_id is None:
        return None

    return str(request_id)


def build_compact_analysis_record(
    analyst_number: int,
    base_id: str,
    validated_response: dict[str, object],
) -> dict[str, Any]:
    return {
        "analyst_number": analyst_number,
        "base_id": base_id,
        "findings": validated_response["findings"],
        "analysis": validated_response["analysis"],
        "things_to_continue_analyzing": validated_response[
            "things_to_continue_analyzing"
        ],
        "action": validated_response["action"],
        "question": validated_response["question"],
        "answer": None,
    }


def update_compact_analysis_answer(
    compact_analyses: list[dict[str, Any]],
    asked_by_analyst: int,
    answer: object,
) -> None:
    for analysis_record in compact_analyses:
        if analysis_record["analyst_number"] == asked_by_analyst:
            analysis_record["answer"] = answer
            return


def build_compact_commander_record(commander_payload: dict[str, Any]) -> dict[str, Any]:
    validated_response = commander_payload.get("validated_response")
    if isinstance(validated_response, dict):
        return {
            "findings": validated_response.get("findings", []),
            "analysis": validated_response.get("analysis", ""),
            "final_steps": validated_response.get("final_steps", []),
        }

    return {
        "findings": [],
        "analysis": "",
        "final_steps": [],
    }


def ask_moondream(
    moondream_client: Any,
    image_path: Path,
    question: str,
    asked_by_analyst: int,
    answered_before_analyst: int | None,
    image_url: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": MOONDREAM_MODEL,
        "asked_by_analyst": asked_by_analyst,
        "answered_before_analyst": answered_before_analyst,
        "image_path": str(image_path),
        "image_url": image_url,
        "question": question,
        "answer": None,
        "request_id": None,
        "status": "error",
        "error": None,
    }

    try:
        with Image.open(image_path) as image:
            result = moondream_client.query(image, question)

        payload["answer"] = extract_moondream_answer(result)
        payload["request_id"] = extract_moondream_request_id(result)
        payload["status"] = "success"
    except Exception as error:
        payload["error"] = str(error)

    return payload


def hide_page_toolbars(driver: webdriver.Chrome) -> None:
    driver.execute_script(
        """
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;
        const candidates = Array.from(document.querySelectorAll("*"));

        for (const element of candidates) {
            if (["HTML", "BODY", "CANVAS"].includes(element.tagName)) {
                continue;
            }

            const style = window.getComputedStyle(element);
            if (!["fixed", "sticky"].includes(style.position)) {
                continue;
            }

            const rect = element.getBoundingClientRect();
            const isVisible = rect.width > 0 && rect.height > 0;
            const touchesEdge =
                rect.top <= 140 ||
                rect.left <= 80 ||
                rect.bottom >= viewportHeight - 80 ||
                rect.right >= viewportWidth - 80;

            if (isVisible && touchesEdge) {
                element.style.setProperty("visibility", "hidden", "important");
            }
        }
        """
    )


def wait_for_earth_view(driver: webdriver.Chrome) -> bytes:
    WebDriverWait(driver, PAGE_LOAD_WAIT_SECONDS).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )
    time.sleep(PAGE_LOAD_WAIT_SECONDS)

    deadline = time.time() + VISUAL_READY_TIMEOUT_SECONDS
    latest_screenshot = driver.get_screenshot_as_png()

    while time.time() < deadline:
        latest_screenshot = driver.get_screenshot_as_png()
        if not is_mostly_black(latest_screenshot):
            return latest_screenshot
        time.sleep(VISUAL_READY_POLL_SECONDS)

    raise TimeoutException("Google Earth view remained mostly black before timeout.")


def capture_current_view(driver: webdriver.Chrome, earth_url: str, output_path: Path) -> None:
    driver.get(earth_url)
    screenshot_bytes = wait_for_earth_view(driver)
    hide_page_toolbars(driver)
    time.sleep(1)
    screenshot_bytes = driver.get_screenshot_as_png()
    resize_and_save_as_jpeg(screenshot_bytes, output_path)


def build_error_response_text(message: str) -> str:
    return json.dumps({"status": "error", "error": message}, ensure_ascii=False)


def process_rows(
    rows: list[dict[str, str]],
    gemini_client: genai.Client,
    moondream_client: Any,
    persisted_data: dict[str, Any],
) -> dict[str, dict[str, str]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows_to_handle = rows[:ROWS_TO_PROCESS]
    history_of_analysts_by_row: dict[str, dict[str, str]] = {}

    if not rows_to_handle:
        print("No rows available to process.")
        return history_of_analysts_by_row

    driver = create_driver()
    try:
        for index, row in enumerate(rows_to_handle, start=1):
            base_id = (row.get("id") or "").strip()
            country = (row.get("country") or "").strip()
            initial_earth_url = (row.get("google_earth_link") or "").strip()

            if not base_id:
                print(f"Row {index}: missing id, skipping.")
                continue

            if base_id in persisted_data:
                print(f"Row {index} (id={base_id}): already exists in data.json, skipping.")
                continue

            if not initial_earth_url:
                print(f"Row {index} (id={base_id}): missing google_earth_link, skipping.")
                continue

            print(f"Processing row {index}/{len(rows_to_handle)} for base id {base_id}...")

            screenshot_folder = OUTPUT_DIR / f"base_{base_id}"
            screenshot_folder.mkdir(parents=True, exist_ok=True)

            current_url = initial_earth_url
            current_camera_state = parse_google_earth_url(current_url)
            history_of_analysts: dict[str, str] = {}
            history_of_analysts_by_row[base_id] = history_of_analysts
            analyst_steps: list[dict[str, Any]] = []
            consultant_qas: list[dict[str, Any]] = []
            compact_analyses: list[dict[str, Any]] = []
            pending_consultant_question: dict[str, Any] | None = None

            current_image_path: Path | None = None
            first_image_path: Path | None = None
            capture_new_image = True
            freeze_view = False

            for analyst_number in range(1, ANALYST_COUNT + 1):
                analyst_key = f"analyst_{analyst_number}"
                analyst_image_path = screenshot_folder / f"{analyst_key}.jpg"
                input_image_path = analyst_image_path
                input_url = current_url
                consultant_response_for_step: dict[str, Any] | None = None

                if capture_new_image:
                    try:
                        capture_current_view(driver, current_url, analyst_image_path)
                        print(f"Saved {analyst_image_path}")
                        current_image_path = analyst_image_path
                        if first_image_path is None:
                            first_image_path = analyst_image_path
                        capture_new_image = False
                    except (TimeoutException, WebDriverException, OSError) as error:
                        error_message = f"Screenshot capture failed: {error}"
                        raw_response_text = build_error_response_text(error_message)
                        history_of_analysts[analyst_key] = raw_response_text
                        analyst_steps.append(
                            {
                                "analyst_number": analyst_number,
                                "input_image_path": str(analyst_image_path),
                                "input_url": input_url,
                                "raw_response_text": raw_response_text,
                                "validated_response": None,
                                "consultant_response_before_analysis": None,
                                "question": None,
                                "action": None,
                                "next_url": current_url,
                                "status": "error",
                                "error": error_message,
                            }
                        )
                        print(f"Row {index} (id={base_id}): {error_message}")
                        continue
                elif current_image_path is not None and current_image_path != analyst_image_path:
                    shutil.copy2(current_image_path, analyst_image_path)
                    print(f"Reused image for {analyst_key}: {analyst_image_path}")
                    current_image_path = analyst_image_path

                if input_image_path is None or not input_image_path.exists():
                    error_message = "No screenshot available for Gemini analysis."
                    raw_response_text = build_error_response_text(error_message)
                    history_of_analysts[analyst_key] = raw_response_text
                    analyst_steps.append(
                        {
                            "analyst_number": analyst_number,
                            "input_image_path": str(input_image_path),
                            "input_url": input_url,
                            "raw_response_text": raw_response_text,
                            "validated_response": None,
                            "consultant_response_before_analysis": None,
                            "question": None,
                            "action": None,
                            "next_url": current_url,
                            "status": "error",
                            "error": error_message,
                        }
                    )
                    print(f"Row {index} (id={base_id}): {error_message}")
                    continue

                if pending_consultant_question is not None:
                    consultant_response_for_step = ask_moondream(
                        moondream_client=moondream_client,
                        image_path=input_image_path,
                        question=str(pending_consultant_question["question"]),
                        asked_by_analyst=int(pending_consultant_question["asked_by_analyst"]),
                        answered_before_analyst=analyst_number,
                        image_url=input_url,
                    )
                    consultant_qas.append(consultant_response_for_step)
                    update_compact_analysis_answer(
                        compact_analyses,
                        int(consultant_response_for_step["asked_by_analyst"]),
                        consultant_response_for_step.get("answer"),
                    )
                    pending_consultant_question = None
                    print(
                        f"Saved Moondream consultant response before analyst_{analyst_number}"
                    )

                prompt = build_analysis_prompt(
                    country=country,
                    history_of_analysts=history_of_analysts if analyst_number > 1 else None,
                    consultant_qas=consultant_qas,
                )

                try:
                    raw_response_text, validated_response = analyze_image(
                        gemini_client,
                        input_image_path,
                        prompt,
                    )
                    history_of_analysts[analyst_key] = raw_response_text
                    action = str(validated_response["action"])
                    question = str(validated_response["question"])
                    compact_analyses.append(
                        build_compact_analysis_record(
                            analyst_number,
                            base_id,
                            validated_response,
                        )
                    )

                    next_url = current_url
                    if not freeze_view:
                        if action == "finish":
                            freeze_view = True
                            capture_new_image = False
                        else:
                            current_camera_state = apply_action_to_camera_state(
                                current_camera_state,
                                action,
                            )
                            next_url = build_google_earth_url_from_state(current_camera_state)
                            current_url = next_url
                            capture_new_image = True

                    pending_consultant_question = {
                        "asked_by_analyst": analyst_number,
                        "question": question,
                    }

                    analyst_steps.append(
                        {
                            "analyst_number": analyst_number,
                            "input_image_path": str(input_image_path),
                            "input_url": input_url,
                            "raw_response_text": raw_response_text,
                            "validated_response": validated_response,
                            "consultant_response_before_analysis": consultant_response_for_step,
                            "question": question,
                            "action": action,
                            "next_url": next_url,
                            "status": "success",
                            "error": None,
                        }
                    )
                    print(f"Saved Gemini response for {analyst_key}")
                    time.sleep(GEMINI_RATE_LIMIT_DELAY_SECONDS)  # Rate limit delay between calls
                except Exception as error:
                    raw_response_text = getattr(error, "raw_response_text", None)
                    if raw_response_text is None and hasattr(error, "response"):
                        response = getattr(error, "response", None)
                        if response is not None:
                            raw_response_text = str(response)

                    error_message = str(error)
                    if raw_response_text is None:
                        raw_response_text = build_error_response_text(error_message)

                    history_of_analysts[analyst_key] = raw_response_text
                    analyst_steps.append(
                        {
                            "analyst_number": analyst_number,
                            "input_image_path": str(input_image_path),
                            "input_url": input_url,
                            "raw_response_text": raw_response_text,
                            "validated_response": None,
                            "consultant_response_before_analysis": consultant_response_for_step,
                            "question": None,
                            "action": None,
                            "next_url": current_url,
                            "status": "error",
                            "error": error_message,
                        }
                    )
                    print(f"Row {index} (id={base_id}): Gemini analysis failed: {error}")

            if pending_consultant_question is not None:
                final_consultant_image_path = screenshot_folder / "final_consultant.jpg"
                final_consultant_url = current_url
                final_consultant_error: str | None = None

                try:
                    if capture_new_image:
                        capture_current_view(driver, current_url, final_consultant_image_path)
                        current_image_path = final_consultant_image_path
                        capture_new_image = False
                        print(f"Saved {final_consultant_image_path}")
                    elif current_image_path is not None:
                        shutil.copy2(current_image_path, final_consultant_image_path)
                        current_image_path = final_consultant_image_path
                        print(f"Reused image for final consultant: {final_consultant_image_path}")
                    else:
                        final_consultant_error = "No screenshot available for final Moondream analysis."
                except (TimeoutException, WebDriverException, OSError) as error:
                    final_consultant_error = f"Final consultant screenshot capture failed: {error}"

                if final_consultant_error is None and final_consultant_image_path.exists():
                    final_consultant_response = ask_moondream(
                        moondream_client=moondream_client,
                        image_path=final_consultant_image_path,
                        question=str(pending_consultant_question["question"]),
                        asked_by_analyst=int(pending_consultant_question["asked_by_analyst"]),
                        answered_before_analyst=None,
                        image_url=final_consultant_url,
                    )
                    consultant_qas.append(final_consultant_response)
                    update_compact_analysis_answer(
                        compact_analyses,
                        int(final_consultant_response["asked_by_analyst"]),
                        final_consultant_response.get("answer"),
                    )
                    print("Saved final Moondream consultant response")
                else:
                    final_consultant_response = {
                        "model": MOONDREAM_MODEL,
                        "asked_by_analyst": int(
                            pending_consultant_question["asked_by_analyst"]
                        ),
                        "answered_before_analyst": None,
                        "image_path": str(final_consultant_image_path),
                        "image_url": final_consultant_url,
                        "question": str(pending_consultant_question["question"]),
                        "answer": None,
                        "request_id": None,
                        "status": "error",
                        "error": final_consultant_error,
                    }
                    consultant_qas.append(final_consultant_response)
                    update_compact_analysis_answer(
                        compact_analyses,
                        int(final_consultant_response["asked_by_analyst"]),
                        final_consultant_response.get("answer"),
                    )
                    print(f"Row {index} (id={base_id}): {final_consultant_error}")

                pending_consultant_question = None

            commander_payload = {
                "model": COMMANDER_MODEL,
                "input_image_path": str(first_image_path) if first_image_path is not None else None,
                "prompt": build_commander_prompt(history_of_analysts, consultant_qas),
                "raw_response_text": None,
                "validated_response": None,
                "status": "error",
                "error": "Commander analysis could not run because the first screenshot is missing.",
            }
            if first_image_path is not None and first_image_path.exists():
                commander_payload = analyze_commander(
                    gemini_client,
                    first_image_path,
                    history_of_analysts,
                    consultant_qas,
                )
                time.sleep(GEMINI_RATE_LIMIT_DELAY_SECONDS)  # Rate limit delay

            debug_file_path = DEBUG_OUTPUT_DIR / f"base_{base_id}_debug.json"
            debug_record = {
                "base_id": base_id,
                "country": country,
                "initial_google_earth_link": initial_earth_url,
                "final_google_earth_link": current_url,
                "screenshot_folder": str(screenshot_folder),
                "history_of_analysts": history_of_analysts,
                "consultant_qas": consultant_qas,
                "analyst_steps": analyst_steps,
                "commander": commander_payload,
            }
            save_debug_json(debug_file_path, debug_record)

            place_record = {
                "base_id": base_id,
                "country": country,
                "initial_google_earth_link": initial_earth_url,
                "final_google_earth_link": current_url,
                "screenshot_folder": str(screenshot_folder),
                "analyses": compact_analyses,
                "commander": build_compact_commander_record(commander_payload),
                "debug_file": str(debug_file_path),
            }

            persisted_data[base_id] = place_record
            save_data_json(persisted_data)
            print(f"Saved place {base_id} to data.json")
    finally:
        driver.quit()

    return history_of_analysts_by_row


def main() -> None:
    load_dotenv()
    rows = ensure_google_earth_links(CSV_PATH)
    persisted_data = load_data_json()
    gemini_client = create_gemini_client()
    moondream_client = create_moondream_client()
    process_rows(rows, gemini_client, moondream_client, persisted_data)


if __name__ == "__main__":
    main()
