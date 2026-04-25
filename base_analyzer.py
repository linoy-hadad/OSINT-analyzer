from __future__ import annotations

import csv
import json
import os
import time
from io import BytesIO
from pathlib import Path

from PIL import Image
from google import genai
from google.genai import types
from dotenv import load_dotenv
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


ROWS_TO_PROCESS = 1
CSV_PATH = Path("military_bases.csv")
OUTPUT_DIR = Path("screenshots")
ANALYSIS_OUTPUT_DIR = Path("analyses")
SCREENSHOT_WIDTH = 1024
PAGE_LOAD_WAIT_SECONDS = 6
VISUAL_READY_TIMEOUT_SECONDS = 45
VISUAL_READY_POLL_SECONDS = 2
WINDOW_SIZE = (1600, 900)
EARTH_VIEW_SUFFIX = ",10.04969521a,1825.78590766d,30.00000016y,-0h,0t,0r"
GEMINI_MODEL = "gemini-2.5-flash-lite"
GOOGLE_API_KEY_ENV_VAR = "GOOGLE_API_KEY"
ALLOWED_ACTIONS = {"zoom-in", "zoom-out", "move-left", "move-right", "finish"}
REQUIRED_GEMINI_KEYS = {
    "findings",
    "analysis",
    "things_to_continue_analyzing",
    "action",
}


class GeminiAnalysisError(Exception):
    def __init__(self, message: str, raw_response_text: str | None = None) -> None:
        super().__init__(message)
        self.raw_response_text = raw_response_text


def build_google_earth_url(latitude: str, longitude: str) -> str:
    return f"https://earth.google.com/web/@{latitude},{longitude}{EARTH_VIEW_SUFFIX}"


def build_analysis_prompt(country: str) -> str:
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
  "action": "One of: zoom-in, zoom-out, move-left, move-right, finish"
}}

Action rules:
- Choose "zoom-in" if you need to zoom in the image.
- Choose "zoom-out" if you need more context of the surrounding area or if you are zoomed in too much.
- Choose "move-left" or "move-right" if you suspect there are important features just outside the current view.
- Choose "finish" if you have a complete understanding of the location.

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
        latitude = (row.get("latitude") or "").strip()
        longitude = (row.get("longitude") or "").strip()
        existing_link = (row.get("google_earth_link") or "").strip()

        if not latitude or not longitude:
            continue

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


def create_gemini_client() -> genai.Client:
    api_key = os.getenv(GOOGLE_API_KEY_ENV_VAR, "").strip()
    if not api_key:
        raise ValueError(
            f"Missing {GOOGLE_API_KEY_ENV_VAR} environment variable. "
            "Set it before running Gemini analysis."
        )

    return genai.Client(api_key=api_key)


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


def validate_gemini_response(data: object) -> dict[str, object]:
    if not isinstance(data, dict):
        raise ValueError("Gemini response must be a JSON object.")

    missing_keys = REQUIRED_GEMINI_KEYS - data.keys()
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
        raise ValueError(
            f"'action' must be one of: {', '.join(sorted(ALLOWED_ACTIONS))}."
        )

    return {
        "findings": findings,
        "analysis": analysis,
        "things_to_continue_analyzing": continue_items,
        "action": action,
    }


def write_analysis_file(base_id: str, payload: dict[str, object]) -> None:
    ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ANALYSIS_OUTPUT_DIR / f"base_id_{base_id}.json"
    with output_path.open("w", encoding="utf-8") as analysis_file:
        json.dump(payload, analysis_file, indent=2, ensure_ascii=False)


def analyze_screenshot(
    client: genai.Client,
    row: dict[str, str],
    screenshot_path: Path,
) -> dict[str, object]:
    base_id = (row.get("id") or "").strip()
    country = (row.get("country") or "").strip()
    earth_url = (row.get("google_earth_link") or "").strip()
    prompt = build_analysis_prompt(country)

    image_bytes = screenshot_path.read_bytes()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            prompt,
        ],
    )

    raw_response_text = (response.text or "").strip()
    if not raw_response_text:
        raise GeminiAnalysisError("Gemini returned an empty response.")

    try:
        parsed = json.loads(raw_response_text)
        normalized = validate_gemini_response(parsed)
    except (json.JSONDecodeError, ValueError) as error:
        raise GeminiAnalysisError(str(error), raw_response_text) from error

    return {
        "status": "success",
        "base_id": base_id,
        "country": country,
        "image_path": str(screenshot_path),
        "google_earth_link": earth_url,
        "model": GEMINI_MODEL,
        "prompt": prompt,
        "result": normalized,
    }


def build_analysis_error_payload(
    row: dict[str, str],
    screenshot_path: Path | None,
    prompt: str,
    error: Exception,
    raw_response_text: str | None = None,
) -> dict[str, object]:
    return {
        "status": "error",
        "base_id": (row.get("id") or "").strip(),
        "country": (row.get("country") or "").strip(),
        "image_path": str(screenshot_path) if screenshot_path is not None else None,
        "google_earth_link": (row.get("google_earth_link") or "").strip(),
        "model": GEMINI_MODEL,
        "prompt": prompt,
        "error": str(error),
        "raw_response_text": raw_response_text,
    }


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


def process_rows(rows: list[dict[str, str]], gemini_client: genai.Client) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_to_handle = rows[:ROWS_TO_PROCESS]

    if not rows_to_handle:
        print("No rows available to process.")
        return

    driver = create_driver()
    try:
        for index, row in enumerate(rows_to_handle, start=1):
            base_id = (row.get("id") or "").strip()
            country = (row.get("country") or "").strip()
            earth_url = (row.get("google_earth_link") or "").strip()
            prompt = build_analysis_prompt(country)

            if not base_id:
                print(f"Row {index}: missing id, skipping.")
                continue

            if not earth_url:
                print(f"Row {index} (id={base_id}): missing google_earth_link, skipping.")
                continue

            print(f"Processing row {index}/{len(rows_to_handle)} for base id {base_id}...")

            try:
                driver.get(earth_url)
                screenshot_bytes = wait_for_earth_view(driver)
                hide_page_toolbars(driver)
                time.sleep(1)
                screenshot_bytes = driver.get_screenshot_as_png()
                output_path = OUTPUT_DIR / f"base_id_{base_id}.jpg"
                resize_and_save_as_jpeg(screenshot_bytes, output_path)
                print(f"Saved {output_path}")

                try:
                    analysis_payload = analyze_screenshot(gemini_client, row, output_path)
                except Exception as error:
                    raw_response_text = getattr(error, "raw_response_text", None)
                    if raw_response_text is None and hasattr(error, "response"):
                        response = getattr(error, "response", None)
                        if response is not None:
                            raw_response_text = str(response)

                    analysis_payload = build_analysis_error_payload(
                        row=row,
                        screenshot_path=output_path,
                        prompt=prompt,
                        error=error,
                        raw_response_text=raw_response_text,
                    )
                    print(f"Row {index} (id={base_id}): Gemini analysis failed: {error}")
                else:
                    print(f"Saved analysis for base id {base_id}")

                write_analysis_file(base_id, analysis_payload)
            except TimeoutException as error:
                print(f"Row {index} (id={base_id}): timed out waiting for map imagery: {error}")
            except WebDriverException as error:
                print(f"Row {index} (id={base_id}): browser error: {error}")
            except OSError as error:
                print(f"Row {index} (id={base_id}): image save error: {error}")
    finally:
        driver.quit()


def main() -> None:
    load_dotenv()
    rows = ensure_google_earth_links(CSV_PATH)
    gemini_client = create_gemini_client()
    process_rows(rows, gemini_client)


if __name__ == "__main__":
    main()
