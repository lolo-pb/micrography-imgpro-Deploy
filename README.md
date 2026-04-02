# micrography-imgpro

Micrograph image processing app for segmenting fibers, pores, resin, and undefined regions.

## What is included

- `app/`: Web UI.
- `app.py`: Streamlit web UI.
- `controller.py`: command-line batch runner.
- `getmeresults.py`, `getmefibers.py`, `getmeflashes.py`, `getmepores.py`: image processing pipeline modules.

## Requirements

- Python 3.10 or newer is recommended.
- Install the packages listed in [`requirements.txt`](requirements.txt).

## Install

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the app

Start the web UI with:

```bash
uvicorn app.main:app --reload
```

Then open `http://localhost:8000` in the browser.

The Streamlit UI can still be run with:

```powershell
streamlit run app.py
```

## Use the CLI

Run the batch processor from the terminal:

```powershell
python .\controller.py
```

Optional flags:

- `-f`: run fibers only
- `-fl`: run flashes only
- `-p`: run pores only

Example:

```powershell
python .\controller.py -f -p
```

Running without flags processes the full combined result pipeline and writes output files to `processed_results`.

## Build a single executable

This project includes [`run_app.py`](run_app.py) for packaging the Streamlit app with PyInstaller.

Already compiled in:
```
https://drive.google.com/drive/folders/19tp-wSOcunXumASR2pq6P5RpKWenxGwO?usp=drive_link
```

Install PyInstaller:

```powershell
pip install pyinstaller
```

Build the executable from the project root:

```powershell
pyinstaller --clean micrography-imgpro.spec
```

After the build completes, the executable will be created at:

```text
dist\micrography-imgpro.exe
```

Run it from a terminal the first time so any startup errors stay visible:

```powershell
.\dist\micrography-imgpro.exe
```

## Notes

- The packaged app still launches a local Streamlit server and opens in the browser.
- Do not commit `dist/` or `build/` outputs to GitHub unless you are using Git LFS.
