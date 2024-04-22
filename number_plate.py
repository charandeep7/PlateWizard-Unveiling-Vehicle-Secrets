import cv2
import os

class NumberPlateDetector:
    def __init__(self, cascade_file="haarcascade_russian_plate_number.xml", min_area=500):
        self.plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_file)
        self.min_area = min_area
        self.save_dir = "captured_number_plates"
        self.count = 0

        # Create directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)

    def detect_and_capture_plates(self, frame):
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        number_plates = self.plate_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=4)

        for (x, y, w, h) in number_plates:
            area = w * h
            if area > self.min_area:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, "NumberPlate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                img_roi = frame[y:y + h, x:x + w]
                cv2.imshow("Number Plate", img_roi)

        return frame

    def save_plate(self, frame):
        cv2.imwrite(os.path.join(self.save_dir, f"{self.count}.jpg"), frame)
        cv2.rectangle(frame, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, "Scan Saved", (15, 265), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
        cv2.imshow("Result", frame)
        cv2.waitKey(500)
        self.count += 1
