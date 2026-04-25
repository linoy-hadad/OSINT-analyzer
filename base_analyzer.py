from __future__ import annotations

import csv
import time
from io import BytesIO
from pathlib import Path

from PIL import Image
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


ROWS_TO_PROCESS = 5
CSV_PATH = Path("military_bases.csv")
OUTPUT_DIR = Path("screenshots")
SCREENSHOT_WIDTH = 1024
PAGE_LOAD_WAIT_SECONDS = 6
VISUAL_READY_TIMEOUT_SECONDS = 45
VISUAL_READY_POLL_SECONDS = 2
WINDOW_SIZE = (1600, 900)
EARTH_VIEW_SUFFIX = ",10.04969521a,1825.78590766d,30.00000016y,-0h,0t,0r"


def build_google_earth_url(latitude: str, longitude: str) -> str:
    return f"https://earth.google.com/web/@{latitude},{longitude}{EARTH_VIEW_SUFFIX}"


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
                screenshot_bytes = wait_for_earth_view(driver)
                hide_page_toolbars(driver)
                time.sleep(1)
                screenshot_bytes = driver.get_screenshot_as_png()
                output_path = OUTPUT_DIR / f"base_id_{base_id}.jpg"
                resize_and_save_as_jpeg(screenshot_bytes, output_path)
                print(f"Saved {output_path}")
            except TimeoutException as error:
                print(f"Row {index} (id={base_id}): timed out waiting for map imagery: {error}")
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
