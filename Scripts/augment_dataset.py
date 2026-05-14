"""
augment_dataset.py
------------------
Genera 10 imatges augmentades per cada imatge original del dataset
i les desa al disc. Manté l'estructura de carpetes original.

Tècniques aplicades (combinades aleatòriament per cada nova imatge):
  · Flips horitzontal i vertical
  · Brillantor, contrast, saturació, tonalitat (ColorJitter)
  · Rotació (±30°)
  · Zoom / Random Crop
  · Gaussian blur
  · Soroll gaussià
  · Cutout / Random Erasing

Estructura esperada del dataset:
  train/
    00-damage/  img1.jpg  img2.jpg ...
    01-whole/   img1.jpg  img2.jpg ...

Ús:
  python augment_dataset.py
  python augment_dataset.py --input_dir train --output_dir train_augmented --n 10
  python augment_dataset.py --input_dir train --output_dir train_augmented --n 5 --seed 99
"""

import argparse
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms
from torchvision.transforms import functional as TF
import torch


# ══════════════════════════════════════════════════════════════════════════════
# TÈCNIQUES D'AUGMENTACIÓ
# ══════════════════════════════════════════════════════════════════════════════

def gaussian_noise(img: Image.Image, std: float = 0.05) -> Image.Image:
    """Afegeix soroll gaussià a la imatge."""
    arr = np.array(img).astype(np.float32) / 255.0
    noise = np.random.normal(0, std, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


def cutout(img: Image.Image, n_holes: int = 1, hole_size: float = 0.2) -> Image.Image:
    """Elimina rectangles aleatoris de la imatge (Cutout / Random Erasing)."""
    arr = np.array(img).copy()
    h, w = arr.shape[:2]
    hole_h = int(h * hole_size)
    hole_w = int(w * hole_size)
    for _ in range(n_holes):
        y = random.randint(0, h - hole_h)
        x = random.randint(0, w - hole_w)
        # Omple amb el color mitjà de la imatge per no introduir artefactes bruscos
        fill = arr.mean(axis=(0, 1)).astype(np.uint8)
        arr[y:y + hole_h, x:x + hole_w] = fill
    return Image.fromarray(arr)


# Pipeline complet d'augmentació: cada transformació s'aplica amb una probabilitat
# per generar variació entre les 10 còpies d'una mateixa imatge.
def build_augmentation_pipeline() -> transforms.Compose:
    return transforms.Compose([
        # ── Flips ──────────────────────────────────────────────────────────────
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),

        # ── Rotació ────────────────────────────────────────────────────────────
        transforms.RandomRotation(degrees=30, fill=0),

        # ── Zoom / Crop ────────────────────────────────────────────────────────
        transforms.RandomResizedCrop(
            size=224,
            scale=(0.7, 1.0),    # Zoom entre 70% i 100% de la imatge original
            ratio=(0.85, 1.15),  # Lleugera variació d'aspecte
        ),

        # ── Color: brillantor, contrast, saturació, tonalitat ──────────────────
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.3,
            hue=0.08,
        ),

        # ── Gaussian blur ──────────────────────────────────────────────────────
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))],
            p=0.3
        ),
    ])


def augment_image(img: Image.Image, pipeline: transforms.Compose) -> Image.Image:
    """Aplica el pipeline i després soroll + cutout amb probabilitat."""
    img = pipeline(img)
    if random.random() < 0.4:
        img = gaussian_noise(img, std=random.uniform(0.02, 0.06))
    if random.random() < 0.3:
        img = cutout(img, n_holes=random.randint(1, 2),
                     hole_size=random.uniform(0.1, 0.25))
    return img


# ══════════════════════════════════════════════════════════════════════════════
# LÒGICA PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def augment_dataset(input_dir: Path, output_dir: Path, n: int, seed: int,
                    copy_originals: bool) -> None:
    """
    Recorre totes les subcarpetes de `input_dir`, i per cada imatge genera
    `n` variants augmentades desades a `output_dir` mantenint l'estructura.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    pipeline = build_augmentation_pipeline()

    # Recull totes les imatges
    image_paths = [
        p for p in input_dir.rglob("*")
        if p.suffix.lower() in IMG_EXTS
    ]

    if not image_paths:
        print(f"[ERROR] No s'han trobat imatges a: {input_dir}")
        return

    print(f"\n{'═'*55}")
    print(f"  Dataset d'entrada : {input_dir}")
    print(f"  Dataset de sortida: {output_dir}")
    print(f"  Imatges originals : {len(image_paths)}")
    print(f"  Còpies per imatge : {n}")
    print(f"  Total esperat     : {len(image_paths) * (n + (1 if copy_originals else 0))}")
    print(f"  Seed              : {seed}")
    print(f"{'═'*55}\n")

    total_generated = 0
    errors = 0

    for idx, src_path in enumerate(image_paths, 1):
        # Replica l'estructura de carpetes a output_dir
        rel_path  = src_path.relative_to(input_dir)
        dst_class = output_dir / rel_path.parent
        dst_class.mkdir(parents=True, exist_ok=True)

        try:
            img = Image.open(src_path).convert("RGB")
        except Exception as e:
            print(f"  [ERROR] No es pot obrir {src_path.name}: {e}")
            errors += 1
            continue

        # Copia l'original si s'ha demanat
        if copy_originals:
            dst_orig = dst_class / src_path.name
            shutil.copy2(src_path, dst_orig)

        # Genera les n variants
        stem = src_path.stem
        ext  = src_path.suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png"}:
            ext = ".jpg"  # Normalitza formats menys comuns

        for i in range(1, n + 1):
            aug_img  = augment_image(img, pipeline)
            dst_name = f"{stem}_aug{i:02d}{ext}"
            dst_path = dst_class / dst_name
            aug_img.save(dst_path, quality=95)
            total_generated += 1

        # Progrés cada 50 imatges
        if idx % 50 == 0 or idx == len(image_paths):
            print(f"  [{idx:>4}/{len(image_paths)}] {rel_path.parent}/ — "
                  f"{total_generated} imatges generades fins ara")

    print(f"\n{'─'*55}")
    print(f"  ✓ Generades : {total_generated} imatges augmentades")
    print(f"  ✗ Errors    : {errors} imatges no processades")
    print(f"  Destí       : {output_dir.resolve()}")
    print(f"{'─'*55}\n")


# ══════════════════════════════════════════════════════════════════════════════
# NOTA SOBRE DROPOUT COM A TÈCNICA D'AUGMENTACIÓ
# ══════════════════════════════════════════════════════════════════════════════
#
# El Dropout (p=0.5 a CustomCNN) és una tècnica de REGULARITZACIÓ, no genera
# dades noves. Funcionalment, però, té un efecte similar:
#
#   · Durant l'entrenament, elimina aleatòriament el 50% de neurones en cada
#     forward pass → cada mini-batch veu una xarxa diferent → equival a entrenar
#     un ensemble de ~2^n xarxes compartint pesos.
#
# Si voleu l'equivalent real d'augmentació durant la INFERÈNCIA, useu TTA
# (Test-Time Augmentation): apliqueu múltiples transformacions a cada imatge
# de test i feu la mitjana de les prediccions. Exemple:
#
#   def predict_tta(model, img_tensor, n_aug=5):
#       preds = [model(augment(img_tensor)) for _ in range(n_aug)]
#       return torch.stack(preds).mean(0)
#
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Genera N imatges augmentades per cada foto del dataset"
    )
    parser.add_argument(
        "--input_dir", type=Path, default=Path("train"),
        help="Carpeta arrel del dataset original (default: train/)"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("train_augmented"),
        help="Carpeta de sortida (default: train_augmented/)"
    )
    parser.add_argument(
        "--n", type=int, default=10,
        help="Nombre d'imatges augmentades per original (default: 10)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed de reproducibilitat (default: 42)"
    )
    parser.add_argument(
        "--no_copy_originals", action="store_true",
        help="Si s'activa, NO copia les imatges originals al output_dir"
    )
    args = parser.parse_args()

    augment_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        n=args.n,
        seed=args.seed,
        copy_originals=not args.no_copy_originals,
    )


if __name__ == "__main__":
    main()
