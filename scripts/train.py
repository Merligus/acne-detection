from ultralytics import YOLO
import argparse


def train_model(model_name, batch_size, epochs, patience):
    model_weight = f"{model_name}.pt"
    # load yolo model
    model = YOLO(model_weight)

    print("Starting fine-tuning for acne detection...")

    # start training
    results = model.train(
        data="acne-dataset/data.yaml",  # Seu arquivo de configuração criado acima
        epochs=epochs,
        imgsz=640,  # default image size for yolo
        batch=batch_size,  # if low memory use a smaller batch size
        patience=patience,  # stop training if no evolution after 'patience' epochs
        device=0,  # use gpu 0. use 'cpu' if no gpu
        project="Acne_Detection",
        name=model_name,  # folder where the weights are going to be saved
    )

    print("Training finished. Best weights are saved on Acne_Detection/{model_name}/weights/best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a YOLO model for acne detection")
    parser.add_argument(
        "--model-name",
        type=str,
        default="yolo26s",
        help="Name of the YOLO model to train (default: yolo26s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training (default: 8)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs to train (default: 50)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Patience for early stopping (default: 15)",
    )
    args = parser.parse_args()
    train_model(args.model_name, args.batch_size, args.epochs, args.patience)
