import cv2
import os
import random
import glob


def visualizar_yolo_boxes(dataset_path, num_samples=3):
    img_dir = os.path.join(dataset_path, "train", "images")
    lbl_dir = os.path.join(dataset_path, "train", "labels")

    # Pegar todas as imagens disponíveis
    todas_imagens = glob.glob(os.path.join(img_dir, "*.jpg"))
    amostras = random.sample(todas_imagens, min(num_samples, len(todas_imagens)))

    for img_path in amostras:
        # Carregar imagem
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]

        # Encontrar o label correspondente
        nome_arquivo = os.path.basename(img_path)
        lbl_path = os.path.join(lbl_dir, nome_arquivo.replace(".jpg", ".txt"))

        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                linhas = f.readlines()

            for linha in linhas:
                partes = linha.strip().split()
                if len(partes) >= 5:
                    class_id = int(partes[0])
                    x_center, y_center, w, h = map(float, partes[1:5])

                    # Desnormalizar as coordenadas YOLO para Pixels
                    x_min = int((x_center - w / 2) * img_w)
                    y_min = int((y_center - h / 2) * img_h)
                    x_max = int((x_center + w / 2) * img_w)
                    y_max = int((y_center + h / 2) * img_h)

                    # Desenhar o retângulo verde e o texto
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(img, f"Acne", (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Salvar o resultado
        output_name = f"check_{nome_arquivo}"
        cv2.imwrite(output_name, img)
        print(f"Salvo: {output_name}")


# Execute a função apontando para a raiz do seu dataset
visualizar_yolo_boxes("acne04-dataset-512")
