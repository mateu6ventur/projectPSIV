"""
evaluate.py
-----------
Fase 1: Avalua el classificador binari ResNet-50 sobre el conjunt de test
        (matriu de confusió, accuracy, F1, report complet).

Fase 2: Per a les imatges classificades com a danyades, executa YOLOv8
        per localitzar i classificar el tipus de dany amb bounding boxes.

Ús:
    python Scripts/evaluate.py

    # Si el model és en un altre path:
    python Scripts/evaluate.py --model_path outputs/best_resnet50.pth

    # Per saltar la fase YOLO (si no està instal·lat):
    python Scripts/evaluate.py --skip_yolo
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

# ── Configuració ──────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
TEST_DIR   = BASE_DIR / "test"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

IMG_SIZE    = 224
BATCH_SIZE  = 32
NUM_CLASSES = 2
CLASS_NAMES = ["00-damage", "01-whole"]   # ordre que ImageFolder assigna
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — Classificació binària amb ResNet-50
# ══════════════════════════════════════════════════════════════════════════════

def load_model(model_path: Path, device):
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, NUM_CLASSES)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def get_test_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def run_classification(model, test_loader, device):
    all_preds, all_labels = [], []
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(labels, preds):
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Matriu de confusió — ResNet-50", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(path, dpi=150)
    print(f"  Matriu desada a: {path}")
    plt.show()


def phase1_evaluate(model_path: Path, device):
    print(f"\n{'='*50}")
    print("  FASE 1 — Classificació binària (ResNet-50)")
    print(f"{'='*50}")

    if not model_path.exists():
        print(f"\n  [ERROR] No s'ha trobat el model a: {model_path}")
        print("  Executa primer train_resnet.py per generar el model.")
        return None

    model = load_model(model_path, device)
    test_ds = datasets.ImageFolder(TEST_DIR, transform=get_test_transform())
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"\n  Imatges de test: {len(test_ds)}")
    labels, preds = run_classification(model, test_loader, device)

    acc = (labels == preds).mean() * 100
    print(f"\n  Accuracy: {acc:.2f}%")
    print(f"\n  Report complet:\n")
    print(classification_report(labels, preds, target_names=CLASS_NAMES))

    plot_confusion_matrix(labels, preds)

    # Retorna els paths de les imatges classificades com a danyades
    damaged_paths = [
        Path(test_ds.samples[i][0])
        for i in range(len(test_ds))
        if preds[i] == test_ds.class_to_idx["00-damage"]
    ]
    print(f"\n  Imatges classificades com a danyades: {len(damaged_paths)}")
    return damaged_paths


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — Localització i classificació del dany amb YOLOv8
# ══════════════════════════════════════════════════════════════════════════════

def check_yolo():
    """Comprova si ultralytics (YOLOv8) està instal·lat."""
    try:
        from ultralytics import YOLO
        return True
    except ImportError:
        return False


def run_yolo_on_images(damaged_paths: list, max_images: int = 10):
    """
    Executa YOLOv8 preentrenat sobre les imatges danyades.
    Mostra les deteccions amb bounding boxes.

    Nota: YOLOv8 preentrenat (COCO) detecta vehicles genèrics.
    Per detectar tipus de dany específics caldria fine-tuning
    amb un dataset anotat amb bounding boxes (p.ex. Roboflow COCO Car Damage).
    """
    from ultralytics import YOLO

    print(f"\n{'='*50}")
    print("  FASE 2 — Localització de danys (YOLOv8)")
    print(f"{'='*50}")
    print("  Carregant model YOLOv8n preentrenat...")

    # Carrega YOLOv8 nano (lleuger, adequat per a CPU/Colab sense GPU)
    model = YOLO("yolov8n.pt")

    sample_paths = damaged_paths[:max_images]
    print(f"  Processant {len(sample_paths)} imatges danyades...\n")

    cols = 2
    rows = (len(sample_paths) + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes

    for idx, img_path in enumerate(sample_paths):
        results = model(str(img_path), verbose=False)[0]
        img = Image.open(img_path).convert("RGB")
        ax = axes[idx] if len(sample_paths) > 1 else axes[0]
        ax.imshow(img)

        # Dibuixa les bounding boxes
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf  = box.conf[0].item()
            cls   = int(box.cls[0].item())
            label = f"{results.names[cls]} {conf:.2f}"

            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor="#e74c3c", facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, label, color="white", fontsize=8,
                    bbox=dict(facecolor="#e74c3c", alpha=0.7, pad=2, edgecolor="none"))

        ax.set_title(img_path.name, fontsize=9)
        ax.axis("off")

    # Amaga els eixos buits
    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Deteccions YOLOv8 sobre imatges danyades", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = OUTPUT_DIR / "yolo_detections.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Deteccions desades a: {path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Dispositiu: {device}")

    model_path    = Path(args.model_path)
    damaged_paths = phase1_evaluate(model_path, device)

    if damaged_paths and not args.skip_yolo:
        if check_yolo():
            run_yolo_on_images(damaged_paths, max_images=args.yolo_samples)
        else:
            print("\n  [INFO] YOLOv8 no està instal·lat.")
            print("  Instal·la'l amb:  pip install ultralytics")
            print("  Després torna a executar aquest script.")

    print("\n  Avaluació completada.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avalua ResNet-50 i executa YOLOv8 sobre les imatges danyades")
    parser.add_argument("--model_path",   type=str, default="outputs/best_resnet50.pth")
    parser.add_argument("--skip_yolo",    action="store_true", help="Salta la fase YOLO")
    parser.add_argument("--yolo_samples", type=int, default=10, help="Nombre d'imatges a processar amb YOLO")
    args = parser.parse_args()
    main(args)