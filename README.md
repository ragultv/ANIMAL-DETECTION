# Animal Detection App

This repository contains two versions of the Animal Detection App:

- **CLI Version**: Command-line interface for animal detection and SMS notifications.
- **GUI Version**: Graphical user interface (Tkinter) for user-friendly interaction.

The app uses OpenCV and MobileNetSSD for real-time animal detection and can send alerts via SMS and play a siren sound when an animal is detected.

## Installation

### Clone the Repository:

```bash
git clone https://github.com/yourusername/animal-detection-app.git
cd animal-detection-app
```
## Install the Required Python Packages:

```bash
pip install numpy opencv-python imutils playsound twilio
```
## Download the Model Files:
- MobileNetSSD_deploy.prototxt.txt
- MobileNetSSD_deploy.caffemodel

Place the model files in the models/ directory:
    animal-detection-app/models/
      ├── MobileNetSSD_deploy.prototxt.txt
      ├── MobileNetSSD_deploy.caffemodel
      
## Place the Siren Sound File:

Add your siren file to the assets/ directory:
    animal-detection-app/assets/
      ├── Siren.wav

# Usage

## CLI Version:
Run the command-line version directly:

```bash
main_CLI.py
```
## GUI Version:
Run the graphical version with Tkinter:

```bash
main_GUI.py
```
## GUI Features:

- Select Video File: Use the file browser to choose a video file for processing.
- Start Detection: Click the "Start Detection" button to begin analyzing the video.
- Live Alerts: The app will highlight detected animals, play a siren sound, and send an SMS alert if configured.

# Customization

## Detection Classes:
Modify the REQ_CLASSES list in the code to detect different types of objects. Default:
```python
REQ_CLASSES = {"bird", "cat", "cow", "dog", "horse", "sheep"}
```
## Confidence Threshold:
Adjust the conf_thresh variable to set the minimum confidence level for detection.

## Alert Sound:
Replace Siren.wav with a different alert sound by updating the siren_path variable in the code.

## SMS Configuration:
Update the Twilio credentials in both versions for SMS functionality:

```python
TWILIO_ACCOUNT_SID = "your_twilio_account_sid"
TWILIO_AUTH_TOKEN = "your_twilio_auth_token"
TWILIO_PHONE_NUMBER = "+your_twilio_phone_number"
FARM_OWNER_NUMBER = "+recipient_phone_number"
```
# Acknowledgments
- OpenCV for providing powerful computer vision tools.
- MobileNetSSD for the pre-trained model used in this project.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
