from __future__ import annotations

import csv
import time
from io import BytesIO
from pathlib import Path

from PIL import Image
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.options import Options


ROWS_TO_PROCESS = 5
CSV_PATH = Path("military_bases.csv")
OUTPUT_DIR = Path("screenshots")
SCREENSHOT_WIDTH = 1024
PAGE_LOAD_WAIT_SECONDS = 8
WINDOW_SIZE = (1600, 900)


def build_google_earth_url(latitude: str, longitude: str) -> str:
    return f"https://earth.google.com/web/@{latitude},{longitude}"


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

        if not existing_link:
            row["google_earth_link"] = build_google_earth_url(latitude, longitude)
            updated = True

    if updated or "google_earth_link" not in reader.fieldnames:
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return rows


def create_driver() -> webdriver.Chrome:
    chrome_options = Options()
    chrome_options.add_argument(f"--window-size={WINDOW_SIZE[0]},{WINDOW_SIZE[1]}")
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument("--hide-scrollbars")
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


def process_rows(rows: list[dict[str, str]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_to_handle = rows[:ROWS_TO_PROCESS]

    if not rows_to_handle:
        print("No rows available to process.")
        return

    driver = create_driver()
    try:
        for index, row in enumerate(rows_to_handle, start=1):
            base_id = (row.get("id") or "").strip()
            earth_url = (row.get("google_earth_link") or "").strip()

            if not base_id:
                print(f"Row {index}: missing id, skipping.")
                continue

            if not earth_url:
                print(f"Row {index} (id={base_id}): missing google_earth_link, skipping.")
                continue

            print(f"Processing row {index}/{len(rows_to_handle)} for base id {base_id}...")

            try:
                driver.get(earth_url)
                time.sleep(PAGE_LOAD_WAIT_SECONDS)
                screenshot_bytes = driver.get_screenshot_as_png()
                output_path = OUTPUT_DIR / f"base_id_{base_id}.jpg"
                resize_and_save_as_jpeg(screenshot_bytes, output_path)
                print(f"Saved {output_path}")
            except WebDriverException as error:
                print(f"Row {index} (id={base_id}): browser error: {error}")
            except OSError as error:
                print(f"Row {index} (id={base_id}): image save error: {error}")
    finally:
        driver.quit()


def main() -> None:
    rows = ensure_google_earth_links(CSV_PATH)
    process_rows(rows)


if __name__ == "__main__":
    main()
