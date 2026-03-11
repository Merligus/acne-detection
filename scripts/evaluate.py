from ultralytics import YOLO
import logging
import argparse

# configure logging
logging.basicConfig(
    filename="test.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_model_path(model_name):
    return f"./runs/detect/Acne_Detection/{model_name}/weights/best.pt"


def evaluate_model(model_name):
    model_path = get_model_path(model_name)
    logging.info(f"Loading the trained model for final evaluation of {model_name}...")
    model = YOLO(model_path)

    # run validation explicitly on the 'test' split from your data.yaml
    logging.info(f"Evaluating on the TEST dataset...")
    metrics = model.val(split="test")

    # log the most important metrics directly to the terminal
    logging.info(f"--- FINAL REAL-WORLD SCORES ---")
    logging.info(f"mAP50 (Detection accuracy): {metrics.box.map50:.4f}")
    logging.info(f"mAP50-95 (Localization accuracy): {metrics.box.map:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO model")
    parser.add_argument(
        "--model-name",
        type=str,
        default="yolo26s",
        help="Name of the model directory (default: yolo26s)",
    )
    args = parser.parse_args()
    evaluate_model(args.model_name)
