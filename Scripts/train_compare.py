"""
train_compare.py
----------------
Entrena i compara 4 arquitectures per a classificació binària de danys:
  1. CustomCNN  — 2 capes conv + 2 FC (arquitectura pròpia)
  2. AlexNet    — preentrenat ImageNet, fine-tuning capa final
  3. ResNet-18  — preentrenat ImageNet, fine-tuning capa final
  4. ResNet-50  — preentrenat ImageNet, fine-tuning capa final

Desa els resultats a: outputs/comparison_results.json
Genera la taula comparativa a: outputs/comparison_table.png
Genera les corbes d'entrenament a: outputs/training_curves_compare.png

Ús:
    python Scripts/train_compare.py
    python Scripts/train_compare.py --epochs 10 --batch_size 16
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Configuració ──────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
TRAIN_DIR  = BASE_DIR / "train"
TEST_DIR   = BASE_DIR / "test"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

IMG_SIZE    = 224
NUM_CLASSES = 2
VAL_SPLIT   = 0.2
NUM_WORKERS = 0
EARLY_STOP  = 4
CLASS_NAMES = ["00-damage", "01-whole"]

DEFAULT_EPOCHS     = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR         = 1e-4
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORMACIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_transforms(augment: bool):
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
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])


def build_dataloaders(batch_size: int):
    full_train = datasets.ImageFolder(TRAIN_DIR, transform=get_transforms(augment=True))
    test_ds    = datasets.ImageFolder(TEST_DIR,  transform=get_transforms(augment=False))

    val_size   = int(len(full_train) * VAL_SPLIT)
    train_size = len(full_train) - val_size
    train_ds, val_ds = random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    val_ds.dataset.transform = get_transforms(augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    print(f"  Train: {train_size} | Val: {val_size} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader


# ══════════════════════════════════════════════════════════════════════════════
# DEFINICIÓ DE MODELS
# ══════════════════════════════════════════════════════════════════════════════

class CustomCNN(nn.Module):
    """
    Arquitectura pròpia: 2 blocs conv (Conv→BN→ReLU→MaxPool) + 2 capes FC.
    Entrada: 224×224×3
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            # Bloc 1: 3 → 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # → 112×112

            # Bloc 2: 32 → 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # → 56×56
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),  # → 64×7×7 = 3136
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_alexnet(num_classes: int, freeze_features: bool = True):
    """AlexNet preentrenat; substitueix el classificador final."""
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False
    # Substitueix la capa final del classificador
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    return model


def build_resnet18(num_classes: int):
    """ResNet-18 preentrenat; congela tot excepte la FC."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes)
    )
    return model


def build_resnet50(num_classes: int):
    """ResNet-50 preentrenat; congela tot excepte la FC."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes)
    )
    return model


MODEL_BUILDERS = {
    "CustomCNN": lambda: CustomCNN(NUM_CLASSES),
    "AlexNet":   lambda: build_alexnet(NUM_CLASSES),
    "ResNet-18": lambda: build_resnet18(NUM_CLASSES),
    "ResNet-50": lambda: build_resnet50(NUM_CLASSES),
}


def count_params(model: nn.Module):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN / EVAL
# ══════════════════════════════════════════════════════════════════════════════

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
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return running_loss / total, correct / total, f1, np.array(all_labels), np.array(all_preds)


def train_model(name, model, train_loader, val_loader, test_loader,
                criterion, device, args):
    """Entrena un model fins a convergència (early stopping) i retorna les mètriques."""
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    history = {"train_loss": [], "train_acc": [],
               "val_loss":   [], "val_acc":   []}
    best_val_acc = 0.0
    epochs_no_improve = 0
    best_state = None
    model_path = OUTPUT_DIR / f"best_{name.replace('-','').lower()}.pth"

    print(f"\n{'─'*55}")
    print(f"  Entrenament: {name}")
    print(f"{'─'*55}")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, vl_f1, _, _ = evaluate(model, val_loader, criterion, device)
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
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, model_path)
        else:
            epochs_no_improve += 1

        print(f"  Època {epoch:02d}/{args.epochs}  "
              f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc*100:.1f}%  "
              f"vl_loss={vl_loss:.4f}  vl_acc={vl_acc*100:.1f}%  "
              f"vl_f1={vl_f1:.3f}{flag}")

        if epochs_no_improve >= EARLY_STOP:
            print(f"  Early stopping a l'època {epoch}.")
            break

    train_time = time.time() - t0
    print(f"  Temps entrenament: {train_time:.1f}s")

    # Avaluació final sobre el test
    if best_state is not None:
        model.load_state_dict(best_state)
    te_loss, te_acc, te_f1, te_labels, te_preds = evaluate(
        model, test_loader, criterion, device
    )
    print(f"\n  Test accuracy: {te_acc*100:.2f}%  |  F1: {te_f1:.4f}")
    print(f"\n  Report complet:\n")
    print(classification_report(te_labels, te_preds, target_names=CLASS_NAMES))

    total_params, trainable_params = count_params(model)

    return {
        "name":             name,
        "test_acc":         round(te_acc * 100, 2),
        "val_acc_best":     round(best_val_acc * 100, 2),
        "test_f1":          round(te_f1, 4),
        "train_time_s":     round(train_time, 1),
        "epochs_trained":   len(history["train_loss"]),
        "total_params":     total_params,
        "trainable_params": trainable_params,
        "history":          history,
        "test_labels":      te_labels.tolist(),
        "test_preds":       te_preds.tolist(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# VISUALITZACIONS
# ══════════════════════════════════════════════════════════════════════════════

COLORS = {
    "CustomCNN": "#e74c3c",
    "AlexNet":   "#f39c12",
    "ResNet-18": "#2ecc71",
    "ResNet-50": "#3498db",
}


def plot_training_curves(all_results: list):
    """Corbes d'entrenament de totes les arquitectures en una mateixa figura."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for res in all_results:
        name  = res["name"]
        color = COLORS.get(name, "#95a5a6")
        h     = res["history"]
        epochs = range(1, len(h["train_acc"]) + 1)

        axes[0].plot(epochs, [l for l in h["train_loss"]], "--",
                     color=color, alpha=0.5, linewidth=1.2)
        axes[0].plot(epochs, [l for l in h["val_loss"]],   "-",
                     color=color, linewidth=2, label=name)

        axes[1].plot(epochs, [a * 100 for a in h["train_acc"]], "--",
                     color=color, alpha=0.5, linewidth=1.2)
        axes[1].plot(epochs, [a * 100 for a in h["val_acc"]],   "-",
                     color=color, linewidth=2, label=name)

    axes[0].set_title("Loss (-- train | — val)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Època")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].axhline(85, color="gray", linestyle=":", linewidth=1)
    axes[1].set_title("Accuracy % (-- train | — val)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Època")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Comparació corbes d'entrenament", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = OUTPUT_DIR / "training_curves_compare.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  Corbes desades a: {path}")
    plt.show()


def plot_confusion_matrices(all_results: list):
    """Matrius de confusió per a cada arquitectura."""
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, all_results):
        cm   = confusion_matrix(res["test_labels"], res["test_preds"])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{res['name']}\nAcc={res['test_acc']}%  F1={res['test_f1']}",
                     fontsize=10, fontweight="bold")

    plt.suptitle("Matrius de confusió — Test set", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = OUTPUT_DIR / "confusion_matrices_compare.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Matrius desades a: {path}")
    plt.show()


def plot_comparison_table(all_results: list):
    """
    Genera una taula visual comparativa amb les principals mètriques.
    """
    headers = [
        "Model",
        "Test Acc (%)",
        "Best Val Acc (%)",
        "F1 (weighted)",
        "Temps (s)",
        "Èpoques",
        "Params entrena.",
        "Params totals",
    ]

    rows = []
    for r in all_results:
        rows.append([
            r["name"],
            f"{r['test_acc']:.2f}",
            f"{r['val_acc_best']:.2f}",
            f"{r['test_f1']:.4f}",
            f"{r['train_time_s']:.1f}",
            str(r["epochs_trained"]),
            f"{r['trainable_params']:,}",
            f"{r['total_params']:,}",
        ])

    # Ressalta el millor valor de cada columna mètrica
    best_idxs = {}
    metric_cols = [1, 2, 3]   # Test Acc, Val Acc, F1 (com més alt millor)
    time_col    = [4]         # Temps (com més baix millor)
    for col in metric_cols:
        vals = [float(rows[i][col]) for i in range(len(rows))]
        best_idxs[(col,)] = vals.index(max(vals))
    for col in time_col:
        vals = [float(rows[i][col]) for i in range(len(rows))]
        best_idxs[(col,)] = vals.index(min(vals))

    fig, ax = plt.subplots(figsize=(14, 2 + len(rows) * 0.7))
    ax.axis("off")

    table_data = [headers] + rows
    col_widths = [0.14, 0.12, 0.16, 0.13, 0.10, 0.09, 0.16, 0.14]

    the_table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    the_table.scale(1.2, 1.8)

    # Estil capçalera
    for j in range(len(headers)):
        cell = the_table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    # Colors de fila per model
    row_colors = {
        "CustomCNN": "#fdecea",
        "AlexNet":   "#fef9e7",
        "ResNet-18": "#eafaf1",
        "ResNet-50": "#ebf5fb",
    }
    for i, row in enumerate(rows):
        color = row_colors.get(row[0], "#ffffff")
        for j in range(len(headers)):
            the_table[i + 1, j].set_facecolor(color)

    # Ressalta millors valors en negreta/verd
    for (col,), row_idx in best_idxs.items():
        cell = the_table[row_idx + 1, col]
        cell.set_facecolor("#27ae60")
        cell.set_text_props(color="white", fontweight="bold")

    plt.title("Taula comparativa d'arquitectures — Detecció de danys",
              fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    path = OUTPUT_DIR / "comparison_table.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Taula desada a: {path}")
    plt.show()


def plot_bar_comparison(all_results: list):
    """Gràfic de barres comparant Test Accuracy i F1."""
    names   = [r["name"] for r in all_results]
    accs    = [r["test_acc"] for r in all_results]
    f1s     = [r["test_f1"] * 100 for r in all_results]
    colors  = [COLORS.get(n, "#95a5a6") for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, accs, width, label="Test Accuracy (%)",
                   color=colors, edgecolor="white", alpha=0.9)
    bars2 = ax.bar(x + width / 2, f1s, width, label="F1 × 100",
                   color=colors, edgecolor="white", alpha=0.55, hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Valor (%)")
    ax.set_ylim(0, 110)
    ax.axhline(85, color="gray", linestyle="--", linewidth=1, label="Objectiu 85%")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                f"{h:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_title("Comparació Test Accuracy i F1 per arquitectura",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = OUTPUT_DIR / "bar_comparison.png"
    plt.savefig(path, dpi=150)
    print(f"  Barres desades a: {path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Dispositiu: {device}")

    train_loader, val_loader, test_loader = build_dataloaders(args.batch_size)
    criterion = nn.CrossEntropyLoss()

    all_results = []

    # Selecció de models a entrenar
    models_to_run = args.models if args.models else list(MODEL_BUILDERS.keys())

    for name in models_to_run:
        if name not in MODEL_BUILDERS:
            print(f"  [AVÍS] Model desconegut: {name}. Opcions: {list(MODEL_BUILDERS.keys())}")
            continue
        model = MODEL_BUILDERS[name]().to(device)
        total, trainable = count_params(model)
        print(f"\n  {name}: {total:,} params totals | {trainable:,} entrenables")
        result = train_model(name, model, train_loader, val_loader, test_loader,
                             criterion, device, args)
        all_results.append(result)

    if not all_results:
        print("  [ERROR] Cap model entrenat.")
        return

    # Desa resultats en JSON
    json_path = OUTPUT_DIR / "comparison_results.json"
    with open(json_path, "w") as f:
        json.dump(
            [{k: v for k, v in r.items() if k not in ("test_labels", "test_preds")}
             for r in all_results],
            f, indent=2
        )
    print(f"\n  Resultats JSON desats a: {json_path}")

    # Gràfics
    print("\n  Generant visualitzacions...")
    plot_training_curves(all_results)
    plot_confusion_matrices(all_results)
    plot_comparison_table(all_results)
    plot_bar_comparison(all_results)

    # Resum final a consola
    print(f"\n{'='*55}")
    print("  RESUM FINAL")
    print(f"{'='*55}")
    print(f"  {'Model':<12} {'Test Acc':>10} {'F1':>8} {'Temps(s)':>10} {'Èpoques':>8}")
    print(f"  {'─'*52}")
    for r in all_results:
        print(f"  {r['name']:<12} {r['test_acc']:>9.2f}% {r['test_f1']:>8.4f} "
              f"{r['train_time_s']:>9.1f}s {r['epochs_trained']:>7}")

    best = max(all_results, key=lambda r: r["test_acc"])
    print(f"\n  ★ Millor model: {best['name']}  ({best['test_acc']}% test acc)")
    print("\n  Comparació completada.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compara múltiples arquitectures per a detecció de danys"
    )
    parser.add_argument("--epochs",     type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int,   default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=DEFAULT_LR)
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Models a entrenar (per defecte tots). Ex: --models CustomCNN ResNet-18"
    )
    args = parser.parse_args()
    main(args)
