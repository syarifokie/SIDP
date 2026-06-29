# SIDP Laboratory Safety Monitoring
#change

A computer-vision-based laboratory safety monitoring system built with YOLO, OpenCV, Flask, and local speech alerts.
Testing 5

## Prerequisites

- Python 3.11 or newer
- Windows environment (required for `pyttsx3` and `pywin32` support)
- `git` if cloning the repository from a remote source

## Setup

1. Open PowerShell and navigate to the project folder:

```powershell
cd "c:\Users\Syarif Alan Taslim\SIDP\SIDP"
```

2. Create a Python virtual environment:

```powershell
python -m venv .venv
```

3. Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

4. Install Python dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the application

From the project root, start the system with:

```powershell
python main.py
```

Once the server is running, open the browser at the local dashboard URL shown in the terminal, for example:

```text
http://localhost:5000
```

## What this project does

- `main.py` is the application entrypoint and starts the Flask dashboard.
- `plugins/` contains the active safety detection modules.
- `configs/default.yaml` holds camera, server, and plugin settings.
- `model/` contains YOLO model weight files used for detection.
- `data/` stores generated log files.

## Notes

- The application uses `configs/default.yaml` to control enabled plugins and camera source.
- The project launches a web dashboard rather than showing an OpenCV window.
- If your camera source is invalid, update `configs/default.yaml`.

## Troubleshooting

- If `python` is not recognized, install Python and add it to your PATH.
- If dependency installation fails, confirm the virtual environment is activated.
- If the dashboard does not appear, verify the local URL and check the terminal for errors.
