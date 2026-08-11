# SIDP — Laboratory Safety Monitoring System

A computer-vision-based laboratory safety monitoring system designed to provide real-time monitoring of laboratory safety conditions. The system uses YOLO-based object detection, OpenCV, Flask, SQLite, and text-to-speech alerts to monitor **face-mask compliance, occupancy, and interpersonal proximity**.

## Key Features

* **Face-Mask Detection** — Detects whether people are wearing the required safety mask.
* **Occupancy Monitoring** — Monitors the number of people in the laboratory and detects occupancy-limit violations.
* **Proximity Monitoring** — Estimates the distance between detected individuals and identifies unsafe proximity conditions.
* **Safety Decision Logic** — Classifies detected conditions into appropriate safety states.
* **Audio Alarm System** — Provides voice notifications for warnings and confirmed safety violations.
* **Web Dashboard** — Provides real-time monitoring of system status and safety events.
* **Event Database** — Stores safety events for later review and reporting.
* **Video Recording** — Records relevant safety events for investigation and verification.

## System Architecture

```text
Camera Input
     ↓
Frame Pre-processing
     ↓
YOLO AI Detection
     ↓
┌─────────────┬────────────┬─────────────┐
│ Safety Mask │ Occupancy  │  Proximity  │
└─────────────┴────────────┴─────────────┘
              ↓
       Safety Decision Logic
              ↓
       ┌──────┴──────┐
       ↓             ↓
   Alarm System    Database
                       ↓
                   Dashboard
```

## Project Structure

```text
SIDP/
├── main.py
├── alarm.py
├── database.py
├── configs/
│   └── default.yaml
├── plugins/
│   ├── safety_mask.py
│   ├── occupancy.py
│   └── proximity.py
├── model/
│   └── YOLO model weights
├── data/
│   └── generated event data and recordings
├── requirements.txt
└── README.md
```

## Prerequisites

* Python 3.11 or newer
* Windows operating system
* Git, if cloning the repository
* A compatible camera or video source
* Required YOLO model weights

> **Note:** The current alarm implementation uses `pyttsx3` for local text-to-speech functionality on Windows.

## Installation

### 1. Clone the repository

```powershell
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd SIDP
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

### 3. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Configuration

System settings are managed through:

```text
configs/default.yaml
```

The configuration file can be used to specify settings such as:

* Camera source
* AI model paths
* Detection confidence thresholds
* Occupancy limit
* Proximity safety distance
* Warning/grace periods
* Enabled safety modules
* Server settings

This allows system parameters to be modified without changing the main application code.

## Running the System

From the project root, activate the virtual environment and run:

```powershell
python main.py
```

Once the application starts, the terminal will display the local dashboard address.

Open the displayed address in a web browser. For example:

```text
http://localhost:5000
```

## Safety Monitoring Functions

### Face-Mask Monitoring

The face-mask module detects people and associates detected mask regions with the corresponding person. A temporal majority-voting mechanism is used to reduce the effect of temporary detection errors before a safety violation is confirmed.

### Occupancy Monitoring

The occupancy module counts detected people in the monitored area and compares the count with the configured maximum occupancy limit. A stable-frame mechanism is used to reduce status changes caused by temporary detection fluctuations.

### Proximity Monitoring

The proximity module estimates the real-world position of detected people using a pinhole-camera model and calculates the distance between individuals. A warning is generated when the estimated distance falls below the configured safety threshold and is escalated if the condition persists.

## Alarm System

The alarm subsystem provides audio notifications using text-to-speech. It supports:

* Individual warning messages
* Individual alert messages
* Priority-based alarm handling
* Combined notifications for multiple simultaneous violations
* Alarm cooldown handling
* Automatic alarm clearing when violations are resolved

## Database and Dashboard

Safety events are stored locally using SQLite. The web dashboard provides real-time visibility of:

* Current safety status
* Active violations
* Detected occupancy
* Proximity violations
* Safety-mask status
* Event history
* Recorded safety incidents

## Troubleshooting

### Python is not recognized

Ensure Python is installed and added to the system PATH.

### Dependency installation fails

Verify that the virtual environment is activated:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then try:

```powershell
python -m pip install -r requirements.txt
```

### Dashboard does not load

Check the terminal output for errors and verify that the Flask server has started successfully.

### Camera is not detected

Check the camera source configured in:

```text
configs/default.yaml
```

### AI model cannot be loaded

Verify that the configured model path exists and that the required YOLO model weights are available.

## Project Status

The system integrates computer vision, safety decision logic, audio alarms, event storage, and a web-based dashboard into a single laboratory safety monitoring platform.
