"""
train_resnet.py
---------------
Entrena un classificador binari (danyat / sencer) basat en ResNet-50
amb transfer learning. Desa el millor model i les corbes d'entrenament.

Estructura esperada:
    train/
        00-damage/
        01-whole/
    test/
        00-damage/
        01-whole/

Ús:
    python Scripts/train_resnet.py

    # Si vols canviar hiperparàmetres:
    python Scripts/train_resnet.py --epochs 20 --batch_size 16 --lr 0.0001
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

# ── Configuració per defecte ──────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
TRAIN_DIR   = BASE_DIR / "train"
TEST_DIR    = BASE_DIR / "test"
OUTPUT_DIR  = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DEFAULT_EPOCHS      = 8     # Amb early stopping, normalment s'atura abans
DEFAULT_BATCH_SIZE  = 32
DEFAULT_LR          = 1e-4
VAL_SPLIT           = 0.2   # 20% del train per validació
IMG_SIZE            = 224
NUM_CLASSES         = 2
EARLY_STOPPING_PATIENCE = 3  # Atura si val_acc no millora en N èpoques
# En Windows amb CPU, num_workers > 0 pot causar problemes i és més lent
NUM_WORKERS = 0
# ─────────────────────────────────────────────────────────────────────────────


def get_transforms(augment: bool = True):
    """
    Retorna les transformacions per a train (amb augmentation)
    i per a validació/test (sense).
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    if augment:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            normalize,
        ])


def build_dataloaders(batch_size: int):
    """
    Carrega el dataset, separa un 20% del train per validació
    i retorna els DataLoaders i els noms de les classes.
    """
    full_train = datasets.ImageFolder(TRAIN_DIR, transform=get_transforms(augment=True))
    test_ds    = datasets.ImageFolder(TEST_DIR,  transform=get_transforms(augment=False))

    val_size   = int(len(full_train) * VAL_SPLIT)
    train_size = len(full_train) - val_size
    train_ds, val_ds = random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    # El conjunt de validació no hauria de tenir augmentation
    val_ds.dataset.transform = get_transforms(augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)

    class_names = full_train.classes  # ['00-damage', '01-whole']
    print(f"\n  Classes detectades: {class_names}")
    print(f"  Train: {train_size} | Val: {val_size} | Test: {len(test_ds)}")

    return train_loader, val_loader, test_loader, class_names


def build_model(device):
    """
    Carrega ResNet-50 preentrenat i adapta la capa final
    per a classificació binària.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # En CPU, congelem tot excepte la capa final per ser molt més ràpid.
    # El model preentrenat ja té prou capacitat representacional.
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False

    # Substitueix la capa final
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, NUM_CLASSES)
    )
    return model.to(device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += imgs.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += imgs.size(0)
    return running_loss / total, correct / total


def plot_curves(history: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["train_loss"], label="Train", color="#e74c3c")
    ax1.plot(history["val_loss"],   label="Val",   color="#3498db")
    ax1.set_title("Loss"); ax1.set_xlabel("Època"); ax1.legend(); ax1.grid(True)

    ax2.plot([a * 100 for a in history["train_acc"]], label="Train", color="#e74c3c")
    ax2.plot([a * 100 for a in history["val_acc"]],   label="Val",   color="#3498db")
    ax2.axhline(85, color="gray", linestyle="--", linewidth=1, label="Objectiu 85%")
    ax2.set_title("Accuracy (%)"); ax2.set_xlabel("Època"); ax2.legend(); ax2.grid(True)

    plt.suptitle("Corbes d'entrenament — ResNet-50", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = OUTPUT_DIR / "training_curves.png"
    plt.savefig(path, dpi=150)
    print(f"\n  Corbes desades a: {path}")
    plt.show()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Dispositiu: {device}")

    train_loader, val_loader, test_loader, class_names = build_dataloaders(args.batch_size)
    model     = build_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    epochs_no_improve = 0
    model_path = OUTPUT_DIR / "best_resnet50.pth"

    print(f"\n{'='*50}")
    print(f"  ENTRENAMENT  (màx {args.epochs} èpoques, early stopping={EARLY_STOPPING_PATIENCE})")
    print(f"{'='*50}")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        improved = vl_acc > best_val_acc
        flag = " ← millor" if improved else ""
        if improved:
            best_val_acc = vl_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
        else:
            epochs_no_improve += 1

        print(f"  Època {epoch:02d}/{args.epochs}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc*100:.1f}%  "
              f"val_loss={vl_loss:.4f}  val_acc={vl_acc*100:.1f}%{flag}")

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\n  Early stopping: val_acc no ha millorat en {EARLY_STOPPING_PATIENCE} èpoques.")
            break

    print(f"\n  Millor val_acc: {best_val_acc*100:.1f}%")
    print(f"  Model desat a: {model_path}")

    # Avaluació final sobre el conjunt de test
    model.load_state_dict(torch.load(model_path, map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\n  Test accuracy: {test_acc*100:.2f}%")

    plot_curves(history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena ResNet-50 per a detecció binària de danys")
    parser.add_argument("--epochs",     type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int,   default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=DEFAULT_LR)
    args = parser.parse_args()
    main(args)