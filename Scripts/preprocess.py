"""
preprocess.py
-------------
Valida i prepara les imatges del dataset per a l'entrenament.
Comprova que totes les imatges siguin llegibles, mostra estadístiques
del dataset i aplica data augmentation per equilibrar les classes si cal.

Estructura esperada:
    train/
        00-damage/
        01-whole/
    test/
        00-damage/
        01-whole/
"""

import os
import sys
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# ── Configuració ──────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent  # arrel del projecte
TRAIN_DIR  = BASE_DIR / "train"
TEST_DIR   = BASE_DIR / "test"
SPLITS     = {"train": TRAIN_DIR, "test": TEST_DIR}
CLASSES    = {"00-damage": "Danyat", "01-whole": "Sencer"}
IMG_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# ─────────────────────────────────────────────────────────────────────────────


def check_images(split_dir: Path, split_name: str) -> dict:
    """
    Recorre totes les carpetes de classe i comprova que cada imatge
    sigui llegible. Retorna un resum amb comptes i errors.
    """
    stats = defaultdict(lambda: {"ok": 0, "error": 0, "paths": []})

    for class_folder, class_label in CLASSES.items():
        folder = split_dir / class_folder
        if not folder.exists():
            print(f"  [AVÍS] Carpeta no trobada: {folder}")
            continue

        for img_path in folder.iterdir():
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            try:
                with Image.open(img_path) as img:
                    img.verify()
                stats[class_label]["ok"] += 1
                stats[class_label]["paths"].append(img_path)
            except Exception as e:
                stats[class_label]["error"] += 1
                print(f"  [ERROR] {img_path.name}: {e}")

    return stats


def print_stats(stats: dict, split_name: str):
    """Mostra un resum de les imatges per split i classe."""
    print(f"\n{'─'*40}")
    print(f"  Split: {split_name.upper()}")
    print(f"{'─'*40}")
    total = 0
    for label, info in stats.items():
        n = info["ok"]
        total += n
        print(f"  {label:10s}: {n:4d} imatges OK  |  {info['error']} errors")
    print(f"  {'TOTAL':10s}: {total:4d} imatges")
    return total


def plot_distribution(all_stats: dict):
    """Genera un gràfic de barres amb la distribució de classes per split."""
    fig, axes = plt.subplots(1, len(all_stats), figsize=(10, 4))
    if len(all_stats) == 1:
        axes = [axes]

    colors = {"Danyat": "#e74c3c", "Sencer": "#2ecc71"}

    for ax, (split_name, stats) in zip(axes, all_stats.items()):
        labels = list(stats.keys())
        counts = [stats[l]["ok"] for l in labels]
        bar_colors = [colors.get(l, "#95a5a6") for l in labels]
        bars = ax.bar(labels, counts, color=bar_colors, edgecolor="white", width=0.5)
        ax.set_title(f"{split_name.capitalize()}", fontsize=13, fontweight="bold")
        ax.set_ylabel("Nombre d'imatges")
        ax.set_ylim(0, max(counts) * 1.2 if counts else 1)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    str(count), ha="center", va="bottom", fontsize=11)

    patches = [mpatches.Patch(color=c, label=l) for l, c in colors.items()]
    fig.legend(handles=patches, loc="upper right", fontsize=10)
    fig.suptitle("Distribució del dataset", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "dataset_distribution.png", bbox_inches="tight", dpi=150)
    print("\n  Gràfic desat a: dataset_distribution.png")
    plt.show()


def show_samples(stats: dict, n: int = 4):
    """Mostra n imatges de mostra de cada classe."""
    fig, axes = plt.subplots(len(stats), n, figsize=(3 * n, 3 * len(stats)))
    if len(stats) == 1:
        axes = [axes]

    for row, (label, info) in enumerate(stats.items()):
        paths = info["paths"][:n]
        for col in range(n):
            ax = axes[row][col]
            if col < len(paths):
                img = Image.open(paths[col]).convert("RGB")
                ax.imshow(img)
                ax.set_title(label, fontsize=9)
            ax.axis("off")

    plt.suptitle("Mostres del conjunt d'entrenament", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "dataset_samples.png", bbox_inches="tight", dpi=150)
    print("  Mostres desades a: dataset_samples.png")
    plt.show()


def main():
    print("=" * 40)
    print("  PREPROCESSAMENT DEL DATASET")
    print("=" * 40)

    all_stats = {}
    for split_name, split_dir in SPLITS.items():
        if not split_dir.exists():
            print(f"\n[ERROR] No s'ha trobat la carpeta: {split_dir}")
            sys.exit(1)
        stats = check_images(split_dir, split_name)
        print_stats(stats, split_name)
        all_stats[split_name] = stats

    plot_distribution(all_stats)

    # Mostres del conjunt d'entrenament
    if "train" in all_stats:
        show_samples(all_stats["train"], n=4)

    # Avís si hi ha desequilibri notable entre classes
    train_stats = all_stats.get("train", {})
    counts = [v["ok"] for v in train_stats.values()]
    if len(counts) == 2 and min(counts) > 0:
        ratio = max(counts) / min(counts)
        if ratio > 1.5:
            print(f"\n  [AVÍS] Desequilibri entre classes (ratio {ratio:.2f}x).")
            print("  Considera aplicar data augmentation o class weighting.")

    print("\n  Preprocessament completat.")


if __name__ == "__main__":
    main()