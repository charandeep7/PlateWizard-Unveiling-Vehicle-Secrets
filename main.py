import cv2
import number_plate

def main():
    frame_width = 1000
    frame_height = 480  

    camera = cv2.VideoCapture(0)
    camera.set(3, frame_width)
    camera.set(4, frame_height)
    camera.set(10, 150)

    plate_detector = number_plate.NumberPlateDetector()

    while True:
        success, img = camera.read()

        result_frame = plate_detector.detect_and_capture_plates(img)
        cv2.imshow("Result", result_frame)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            plate_detector.save_plate(result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()