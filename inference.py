import cv2
from ultralytics import YOLO

def inference_photo(model_path, img_path):
    model = YOLO(model_path)
    results = model(img_path)

    annotated_frame = results[0].plot()

    cv2.imshow("Inference", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

inference_photo("runs/detect/train-3/weights/best.pt", "img/2.jpg")