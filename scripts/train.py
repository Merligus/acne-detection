from ultralytics import YOLO
import argparse


def train_model(model_name, epochs, patience, dataset):
    model_weight = f"{model_name}.pt"
    # load yolo model
    model = YOLO(model_weight)

    # start training
    results = model.train(
        data=f"{dataset}/data.yaml",
        epochs=epochs,
        imgsz=1280,  # higher resolution to preserve small acne lesions
        batch=2,  # small fixed batch to avoid system RAM OOM
        patience=patience,  # stop training if no evolution after 'patience' epochs
        device=0,  # use gpu 0. use 'cpu' if no gpu
        project="Acne_Detection",
        name=model_name,  # folder where the weights are going to be saved
        cos_lr=True,  # cosine LR decay for smoother convergence
        scale=0.9,  # aggressive scale augmentation for variable-resolution images
        translate=0.2,  # slightly more spatial variation
        mosaic=1.0,  # keeps mosaic (good for small objects)
        erasing=0.4,  # random erasing for regularization
        workers=2,  # keep low to manage RAM usage
    )

    print(f"Training finished. Best weights saved at Acne_Detection/{model_name}/weights/best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a YOLO model for acne detection")
    parser.add_argument(
        "--model-name",
        type=str,
        default="yolo26s",
        help="Name of the YOLO model to train (default: yolo26s)",
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
    parser.add_argument(
        "--dataset",
        type=str,
        default="acne-dataset",
        help="Dataset folder name (default: acne04-dataset)",
    )
    args = parser.parse_args()
    train_model(args.model_name, args.epochs, args.patience, args.dataset)
