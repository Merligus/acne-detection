# Acne Detection Analysis

Based on this dataset [Acne Dataset in YOLOv8 Format](https://www.kaggle.com/datasets/osmankagankurnaz/acne-dataset-in-yolov8-format?resource=download)

## Install

Install pytorch and other requirements.

- Change directory to `mtr-skin-analysis/AcneDetectionAnalysis/`;

- Create the environment in conda with python 3.11:
```bash
conda create -n AcneDetectionAnalysis python=3.11 -y
conda activate AcneDetectionAnalysis
```

```bash
pip install -r requirements.txt
```

## Run

In mtr-skin-analysis folder run:

```bash
gunicorn -w 1 -b 0.0.0.0:7871 AcneDetectionAnalysis:app --env origins="origins.txt" --env flask_key="morethanreal_acne_detection_analysis"
```

## Evaluate

Download [dataset](https://www.kaggle.com/datasets/osmankagankurnaz/acne-dataset-in-yolov8-format?resource=download).

To evaluate, download the dataset inside `AcneDetectionAnalysis/` folder, run the flask app with gunicorn command above and then run `evaluate.py` in `AcneDetectionAnalysis/` folder:

```bash
python evaluate.py
```

Output:

```
Number of samples tested: 0
Overall Accuracy: 00.00%

Per-Class Accuracy:
Class 0: 00.00% (0/0)
Class 1: 00.00% (0/0)
Class 2: 00.00% (0/0)
Class 3: 00.00% (0/0)
```

<img src="confusion_matrix.png" alt="Confusion Matrix" width="600">