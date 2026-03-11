from ultralytics import YOLO
import argparse


def get_model_path(model_name):
    return f"./runs/detect/Acne_Detection/{model_name}/weights/best.pt"


def test_on_image(img_path, model_name, confidence):
    model_path = get_model_path(model_name)
    print(f"Loading model and testing on image: {img_path}...")
    model = YOLO(model_path)

    # Detect
    # conf=0.25 means it will only detect acne above 25% confidence
    resultados = model.predict(source=img_path, conf=confidence, save=True)

    # YOLO automatically saves predictions on "runs/detect/predict"
    print(f"\nDone! Detections saved on: {resultados[0].save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on images or webcam")
    parser.add_argument(
        "--model-name",
        type=str,
        default="yolo26s",
        help="Name of the model directory (default: yolo26s)",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default="acne-dataset/test/images/acne-355_jpeg.rf.a407ad30612ee0eea150efed8772694d.jpg",
        help="Name of image or folder to run the model prediction (default: acne-dataset/test/images/acne-355_jpeg.rf.a407ad30612ee0eea150efed8772694d.jpg)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.309,
        help="Threshold confidence where a spot is considered or not a detection (default: 0.309)",
    )
    args = parser.parse_args()

    # test on a folder of static photos or a img filename
    test_on_image(args.image_path, args.model_name, args.confidence)
