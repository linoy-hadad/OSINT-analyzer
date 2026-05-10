# OSINT Analyzer

A Python-based tool for automated geospatial intelligence analysis using Google Earth imagery, Gemini multimodal analysis,Moondream visual consulting.

## What it does

- Reads military base coordinates from a military_bases csv file
- Ensures each row includes a Google Earth link
- Captures screenshots from Google Earth using Selenium
- Sends images to Gemini for analyst-style satellite imagery analysis
- Uses Moondream for focused visual object detection and consultant answers
- Aggregates analyst history and commander-level summaries
- Stores results in `data.json` and debug files under `analyses/`

## Requirements

- Python 3.11+ recommended
- Google Chrome installed
- Chromedriver compatible with the installed Chrome version available on `PATH`

## Dependencies

Install the Python requirements with:

```bash
pip install -r requirements.txt
```

## Environment variables

Create a `.env` file in the project root or set these variables in your shell:

```env
GOOGLE_API_KEY=your_google_api_key
MOONDREAM_API_KEY=your_moondream_api_key
```

## Files and folders

- `base_analyzer.py` - main automation script
- `military_bases.csv` - input base list with latitude, longitude, country, and optional `google_earth_link`
- `data.json` - generated output record for processed bases
- `analyses/` - debug JSON output for each base
- `screenshots/` - captured Google Earth screenshots per base
- `streamlit_app.py` - optional Streamlit interface (if applicable)

## Usage

Run the analyzer from the project folder:

```bash
python base_analyzer.py
```

The script will:

1. Read `military_bases.csv`
2. Create or update `google_earth_link` values
3. Capture imagery for the first `ROWS_TO_PROCESS` rows
4. Perform up to `ANALYST_COUNT` Gemini analyst rounds per base
5. Request visual consultant answers from Moondream
6. Generate a commander summary and save results in `data.json`

## Notes

- The script currently processes only the first `ROWS_TO_PROCESS` rows 
- If screenshots fail to capture, the script logs the error in debug output and continues.
- The output directories are created automatically if missing.

## Troubleshooting

- Ensure Chrome and Chromedriver versions match.
- Verify API keys are valid and have the required permissions.
- If Selenium fails, check that Chrome is not blocked by the system or running in a restricted environment.

## Extending the project

- Adjust `ROWS_TO_PROCESS` and `ANALYST_COUNT` in `base_analyzer.py` for more or fewer analysis rounds.
- Add new base rows to `military_bases.csv` with `id`, `country`, `latitude`, and `longitude`.
- Use `streamlit_app.py` to build a visual front-end if desired.
