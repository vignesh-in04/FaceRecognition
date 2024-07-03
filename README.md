# FaceRecognition

## Overview
  This repository showcases a face recognition project implemented in Python, using the powerful face_recognition library built on top of dlib, alongside OpenCV for real-time video processing. The project demonstrates the detection, encoding, and recognition of faces from a live video stream, making it ideal for applications like automated attendance systems and security systems.
  
## Features
-	Real-time Face Detection: Efficiently detects faces in live video using Histogram of Oriented Gradients (HOG) or Convolutional Neural Networks (CNN).
-	Face Encoding: Generates unique 128-dimensional encodings for each detected face using a deep Convolutional Neural Network.
-	Face Recognition: Matches live face encodings with pre-stored known face encodings using Euclidean distance.
-	Attendance Logging: Logs recognized faces with timestamps into a CSV file.
-	Sound Alert: Plays a sound when an unknown face is detected.

## Prerequisites
Ensure having the following libraries installed:

- OpenCV
- face_recognition
- numpy
- Pillow (PIL)
- winsound (Windows-specific)

## Key Components
- Loading Images and Encoding Faces: The script loads images from the Images directory, converts them to RGB, and generates face encodings.
- Real-Time Face Recognition: The system captures live video, detects faces, resizes and processes the images, and compares the encodings with known faces.
- Logging Attendance: Recognized faces are logged with the current date and time in Recognized_Faces.csv.

## Usage
1. Add images to the `images` folder, with a separate subdirectory for each person
2. Run the script: python FaceRecognition.py
3. Press 'q' to quit the application

## Configuration
- Adjust the face distance threshold in the script if needed
- Modify the `Mark_Attendance.csv` file format as required

## License
[MIT License](https://opensource.org/licenses/MIT)

