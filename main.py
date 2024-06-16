import cv2
import argparse
from ultralytics import YOLO
#import supervision as sv

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution",
        default=[1280, 720],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args

def main():
    print("Begining ..")
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,frame_height)

    model = YOLO("yolov8l.pt")

    while True:
        ret, frame = cap.read()
        result = model(frame)
        #print("detected results, ", result)

        #cv2.imshow("yolov8", frame)

        # print(frame.shape)

        #if(cv2.waitKey(30) == 27):
        #    break

if __name__ == "__main__":
    main()

