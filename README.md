📌 SIDP Project
🚀 Overview

This project processes video input (webcam or video file) and runs inference using a configurable system.

⚙️ Setup Instructions
1. Clone the Repository
git clone <your-repo-link>
cd SIDP
2. Create Virtual Environment (Recommended)
python -m venv venv

Activate it:

Windows:
venv\Scripts\activate
Mac/Linux:
source venv/bin/activate
3. Install Requirements
pip install -r requirements.txt
⚙️ Configuration

Before running the program, edit the configuration file:

📄 default.yaml

📌 Camera Settings
camera:
  source: "Test2.mp4"   # 0 = webcam | "video.mp4" = video file
  infer_size: [640, 480]
  process_every_n: 2    # 1 = every frame, 2 = every 2 frames
  mjpeg_quality: 80
🎯 Explanation
source
0 → use webcam
"video.mp4" → use video file
infer_size
Resize input frame for faster processing
process_every_n
Controls performance vs accuracy
1 = process every frame (high accuracy, slower)
2 = process every 2 frames (faster)
mjpeg_quality
Output video stream quality (higher = better, slower)
▶️ Run the Project

After setup and configuration:

python main.py
