Installation
Clone the repository:
git clone https://github.com/yourusername/animal-detection-app.git
cd animal-detection-app

Install the required Python packages:
pip install numpy opencv-python imutils playsound


Download the necessary model files:
MobileNetSSD_deploy.prototxt.txt
MobileNetSSD_deploy.caffemodel  

Place the model files in the appropriate directory:
C:\Users\yourusername\Downloads\animal-detection-using-ssd-algorithm-main\

Place the siren sound file in the appropriate directory:
C:\Users\yourusername\Downloads\animal-detection-using-ssd-algorithm-main\Siren.wav


Usage
Run the application:
python animal_detection_app.py

Use the GUI to select a video file for processing.
Click on "Start Detection" to begin the animal detection process.
The application will display the video with detected animals highlighted and trigger a siren sound if an animal is detected.

Customization
Detection Classes: Modify the REQ_CLASSES list in the start_detection function to detect different classes of objects.
Confidence Threshold: Adjust the conf_thresh variable to change the confidence threshold for detection.
Alert Sound: Change the siren_wav variable to use a different alert sound.
Acknowledgments
OpenCV for providing powerful computer vision tools.
MobileNetSSD for the pre-trained model used in this project.
License
This project is licensed under the MIT License - see the LICENSE file for details.

