"""
Entraînement Mask R-CNN pour segmentation des toitures cadastrales
Dataset: Images aériennes annotées avec CVAT (format COCO)
Classes: toiture_tole_ondulee, toiture_tole_bac, toiture_tuile, toiture_dalle

Modes d'entraînement:
  simple    : Mask R-CNN standard, hyperparamètres fixes
  attention : Mask R-CNN + CBAM (mécanisme d'attention), hyperparamètres fixes
  optimize  : Mask R-CNN standard + recherche bayésienne des hyperparamètres (Optuna)
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader, random_split
from collections import OrderedDict
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import v2 as T
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask_utils
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

def load_classes(yaml_path="classes.yaml"):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data['classes']

OPTUNA_CONFIG = {
    "n_trials": 30,
    "n_epochs_per_trial": 5,
    "study_name": "maskrcnn_cadastral",
    "output_dir": "./optuna_output",
}

CONFIG = {
    "images_dir": os.getenv("SEGMENTATION_DATASET_IMAGES_DIR"),
    "annotations_file": os.getenv("SEGMENTATION_DATASET_ANNOTATIONS_FILE"),
    "classes_file": os.getenv("CLASSES_FILE", "classes.yaml"),
    "output_dir": "./output",
    "classes": load_classes(os.getenv("CLASSES_FILE", "classes.yaml")),
    "num_epochs": 25,
    "batch_size": 2,
    "learning_rate": 0.005,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "lr_step_size": 8,
    "lr_gamma": 0.1,
    "train_split": 0.85,
    "num_workers": 2,
    "save_every": 5,
    # Paramètres CBAM par défaut (mode attention uniquement)
    "cbam_reduction": 16,
    "cbam_kernel_size": 7,
}


# =============================================================================
# UTILITAIRES TEMPS
# =============================================================================

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"


class TrainingTimer:
    def __init__(self, num_epochs):
        self.num_epochs = num_epochs
        self.start_time = None
        self.epoch_times = []
        self.epoch_start = None

    def start_training(self):
        self.start_time = time.time()
        self.training_start_datetime = datetime.now()

    def start_epoch(self):
        self.epoch_start = time.time()

    def end_epoch(self, epoch):
        epoch_time = time.time() - self.epoch_start
        self.epoch_times.append(epoch_time)
        total_elapsed = time.time() - self.start_time
        avg_epoch_time = np.mean(self.epoch_times)
        remaining_epochs = self.num_epochs - (epoch + 1)
        estimated_remaining = avg_epoch_time * remaining_epochs
        eta = datetime.now() + timedelta(seconds=estimated_remaining)
        return {
            'epoch_time': epoch_time,
            'total_elapsed': total_elapsed,
            'avg_epoch_time': avg_epoch_time,
            'estimated_remaining': estimated_remaining,
            'estimated_total': total_elapsed + estimated_remaining,
            'eta': eta,
            'progress_percent': ((epoch + 1) / self.num_epochs) * 100
        }

    def get_final_stats(self):
        total_time = time.time() - self.start_time
        return {
            'total_time': total_time,
            'total_time_formatted': format_time(total_time),
            'avg_epoch_time': np.mean(self.epoch_times),
            'avg_epoch_time_formatted': format_time(np.mean(self.epoch_times)),
            'min_epoch_time': np.min(self.epoch_times),
            'min_epoch_time_formatted': format_time(np.min(self.epoch_times)),
            'max_epoch_time': np.max(self.epoch_times),
            'max_epoch_time_formatted': format_time(np.max(self.epoch_times)),
            'std_epoch_time': np.std(self.epoch_times),
            'epoch_times': self.epoch_times,
            'start_datetime': self.training_start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            'end_datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }


# =============================================================================
# AUGMENTATION PAR CLASSE
# =============================================================================

def parse_aug_coeffs(aug_args, classes):
    """
    Parse les coefficients d'augmentation depuis les arguments CLI.
    Format: nom_partiel:coeff  (ex: --aug dalle:2 tole:3)
    Correspondance partielle insensible à la casse sur le nom de classe.
    Valeur par défaut: 1 (aucune augmentation).
    """
    real_classes = [c for c in classes if c != '__background__']
    coeffs = {cls: 1 for cls in real_classes}
    for item in (aug_args or []):
        if ':' not in item:
            print(f"⚠️  Format invalide (ignoré): {item!r} — attendu: classe:coeff")
            continue
        key, val = item.rsplit(':', 1)
        try:
            coeff = max(1, int(val))
        except ValueError:
            print(f"⚠️  Coefficient invalide (ignoré): {val!r}")
            continue
        matched = [c for c in real_classes if key.lower() in c.lower()]
        if not matched:
            print(f"⚠️  Classe non trouvée (ignorée): {key!r}  — classes dispo: {real_classes}")
            continue
        for cls in matched:
            coeffs[cls] = coeff
    return coeffs


def _count_class_stats(coco, image_ids, cat_id_to_name):
    """Retourne {nom_classe: {'images': n, 'annotations': n}} pour un ensemble d'IDs."""
    stats = {name: {'images': set(), 'annotations': 0} for name in cat_id_to_name.values()}
    for img_id in image_ids:
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=[img_id])):
            if ann.get('iscrowd', 0):
                continue
            cid = ann['category_id']
            if cid in cat_id_to_name:
                name = cat_id_to_name[cid]
                stats[name]['images'].add(img_id)
                stats[name]['annotations'] += 1
    return {name: {'images': len(s['images']), 'annotations': s['annotations']}
            for name, s in stats.items()}


def oversample_by_class(image_ids, coco, cat_id_to_name, aug_coeffs):
    """
    Duplique les IDs d'images selon le coefficient maximal de leurs classes.
    Chaque duplicate sera transformé différemment grâce aux augmentations aléatoires.
    Retourne (liste_augmentée, dict img_id -> coeff_appliqué).
    """
    img_max_coeff = {}
    for img_id in image_ids:
        max_c = 1
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=[img_id])):
            if ann.get('iscrowd', 0):
                continue
            cid = ann['category_id']
            if cid in cat_id_to_name:
                max_c = max(max_c, aug_coeffs.get(cat_id_to_name[cid], 1))
        img_max_coeff[img_id] = max_c

    augmented = []
    for img_id in image_ids:
        augmented.extend([img_id] * img_max_coeff[img_id])
    return augmented, img_max_coeff


def print_augmentation_report(coco, train_ids_before, train_ids_after,
                               cat_id_to_name, aug_coeffs):
    """Affiche le rapport d'augmentation avant/après dans le terminal."""
    before = _count_class_stats(coco, train_ids_before, cat_id_to_name)
    after  = _count_class_stats(coco, train_ids_after,  cat_id_to_name)

    print(f"\n{'='*70}")
    print(f"   RAPPORT D'AUGMENTATION DES DONNEES D'ENTRAINEMENT")
    print(f"{'='*70}")

    print(f"\n   AVANT AUGMENTATION  ({len(set(train_ids_before))} images uniques en train)")
    print(f"   {'─'*65}")
    print(f"   {'Classe':<38} {'Coeff':>5}  {'Images':>7}  {'Annot.':>7}")
    print(f"   {'─'*65}")
    total_ann_b = 0
    for cls_name, s in before.items():
        coeff  = aug_coeffs.get(cls_name, 1)
        marker = "  *" if coeff > 1 else ""
        print(f"   {cls_name:<38} x{coeff:>4}  {s['images']:>7}  {s['annotations']:>7}{marker}")
        total_ann_b += s['annotations']
    print(f"   {'─'*65}")
    print(f"   {'TOTAL':<38}       {len(set(train_ids_before)):>7}  {total_ann_b:>7}")

    print(f"\n   APRES AUGMENTATION  ({len(train_ids_after)} samples d'entrainement)")
    print(f"   {'─'*65}")
    print(f"   {'Classe':<38} {'Delta':>5}  {'Images':>7}  {'Annot.':>7}")
    print(f"   {'─'*65}")
    total_ann_a = 0
    for cls_name, sa in after.items():
        sb    = before[cls_name]
        delta = f"+{sa['images'] - sb['images']}" if sa['images'] != sb['images'] else "  ="
        print(f"   {cls_name:<38} {delta:>5}  {sa['images']:>7}  {sa['annotations']:>7}")
        total_ann_a += sa['annotations']
    print(f"   {'─'*65}")
    print(f"   {'TOTAL (avec duplicats)':<38}       {len(train_ids_after):>7}  {total_ann_a:>7}")
    ratio = len(train_ids_after) / max(len(train_ids_before), 1)
    print(f"\n   Ratio d'augmentation global: x{ratio:.2f} samples")
    print(f"{'='*70}\n")


# =============================================================================
# DATASET
# =============================================================================

class CadastralDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, annotations_file, transforms=None, image_ids=None):
        self.images_dir = images_dir
        self.transforms = transforms
        self.coco = COCO(annotations_file)
        self.image_ids = list(image_ids) if image_ids is not None else list(self.coco.imgs.keys())
        self.cat_ids = self.coco.getCatIds()
        self.cat_mapping = {cat_id: idx + 1 for idx, cat_id in enumerate(self.cat_ids)}
        print(f"Dataset chargé: {len(self.image_ids)} images")
        print(f"Catégories: {[self.coco.cats[c]['name'] for c in self.cat_ids]}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.imgs[img_id]
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes, labels, masks, areas = [], [], [], []
        for ann in anns:
            if ann.get('iscrowd', 0):
                continue
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_mapping[ann['category_id']])
            if 'segmentation' in ann:
                if isinstance(ann['segmentation'], list):
                    rles = coco_mask_utils.frPyObjects(
                        ann['segmentation'], img_info['height'], img_info['width']
                    )
                    rle = coco_mask_utils.merge(rles)
                    mask = coco_mask_utils.decode(rle)
                else:
                    mask = coco_mask_utils.decode(ann['segmentation'])
                masks.append(mask)
            areas.append(ann.get('area', w * h))
        if len(boxes) == 0:
            boxes  = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks  = torch.zeros((0, img_info['height'], img_info['width']), dtype=torch.uint8)
            areas  = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes  = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks  = torch.as_tensor(np.array(masks), dtype=torch.uint8)
            areas  = torch.as_tensor(areas, dtype=torch.float32)
        target = {
            "boxes": boxes, "labels": labels, "masks": masks,
            "image_id": torch.tensor([img_id]), "area": areas,
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64)
        }
        image = T.ToTensor()(image)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target


# =============================================================================
# TRANSFORMATIONS
# =============================================================================

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if np.random.random() < self.prob:
            image = torch.flip(image, [-1])
            if "boxes" in target and len(target["boxes"]) > 0:
                width = image.shape[-1]
                boxes = target["boxes"]
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                target["boxes"] = boxes
            if "masks" in target and len(target["masks"]) > 0:
                target["masks"] = torch.flip(target["masks"], [-1])
        return image, target


class RandomVerticalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if np.random.random() < self.prob:
            image = torch.flip(image, [-2])
            if "boxes" in target and len(target["boxes"]) > 0:
                height = image.shape[-2]
                boxes = target["boxes"]
                boxes[:, [1, 3]] = height - boxes[:, [3, 1]]
                target["boxes"] = boxes
            if "masks" in target and len(target["masks"]) > 0:
                target["masks"] = torch.flip(target["masks"], [-2])
        return image, target


def get_transforms(train=True):
    transforms = []
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
        transforms.append(RandomVerticalFlip(0.5))
    return Compose(transforms)


# =============================================================================
# MÉCANISME D'ATTENTION (CBAM) — utilisé uniquement en mode "attention"
# =============================================================================

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(),
            nn.Linear(mid, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
        nn.init.zeros_(self.fc[-1].weight)

    def forward(self, x):
        b, c = x.shape[:2]
        avg   = self.fc(self.avg_pool(x).view(b, c))
        mx    = self.fc(self.max_pool(x).view(b, c))
        scale = self.sigmoid(avg + mx).view(b, c, 1, 1)
        return x * scale


class SpatialAttention(nn.Module):
    """Spatial attention via channel-pooled convolution"""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        nn.init.zeros_(self.conv.weight)

    def forward(self, x):
        avg   = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        scale = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * scale


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel puis spatial)"""

    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial_att(self.channel_att(x))


class AttentionFPN(nn.Module):
    """Enveloppe FPN qui applique CBAM à chaque niveau de features"""

    def __init__(self, fpn, out_channels=256, cbam_reduction=16, cbam_kernel_size=7):
        super().__init__()
        self.fpn = fpn
        self.cbam_modules = nn.ModuleDict({
            '0':    CBAM(out_channels, cbam_reduction, cbam_kernel_size),
            '1':    CBAM(out_channels, cbam_reduction, cbam_kernel_size),
            '2':    CBAM(out_channels, cbam_reduction, cbam_kernel_size),
            '3':    CBAM(out_channels, cbam_reduction, cbam_kernel_size),
            'pool': CBAM(out_channels, cbam_reduction, cbam_kernel_size),
        })

    def forward(self, x):
        features = self.fpn(x)
        attended = OrderedDict()
        for key, feat in features.items():
            attended[key] = self.cbam_modules[key](feat) if key in self.cbam_modules else feat
        return attended


# =============================================================================
# MODÈLE
# =============================================================================

def _replace_heads(model, num_classes):
    """Remplace le classificateur et le prédicteur de masques."""
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)


def get_model_simple(num_classes):
    """Mask R-CNN standard sans attention."""
    model = maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
    _replace_heads(model, num_classes)
    return model


def get_model_attention(num_classes, cbam_reduction=16, cbam_kernel_size=7):
    """Mask R-CNN avec mécanisme d'attention CBAM sur le FPN."""
    model = maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
    fpn_out_channels = model.backbone.out_channels  # 256
    model.backbone.fpn = AttentionFPN(
        model.backbone.fpn, fpn_out_channels, cbam_reduction, cbam_kernel_size
    )
    _replace_heads(model, num_classes)
    return model


# =============================================================================
# OPTIMISATION BAYÉSIENNE (OPTUNA) — utilisé uniquement en mode "optimize"
# =============================================================================

def _run_optimization(device, train_loader, val_loader, num_classes):
    """Recherche bayésienne des hyperparamètres via Optuna (sans CBAM)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        lr           = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay",  1e-5, 1e-3, log=True)
        momentum     = trial.suggest_float("momentum",      0.80, 0.99)
        lr_step_size = trial.suggest_int  ("lr_step_size",  3,    15)

        model     = get_model_simple(num_classes)
        model.to(device)
        params    = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=0.1)

        best_val = float('inf')
        for epoch in range(OPTUNA_CONFIG["n_epochs_per_trial"]):
            train_one_epoch(model, optimizer, train_loader, device, epoch)
            val_loss = _evaluate(model, val_loader, device)
            scheduler.step()
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            best_val = min(best_val, val_loss)
        return best_val

    os.makedirs(OPTUNA_CONFIG["output_dir"], exist_ok=True)
    study = optuna.create_study(
        direction="minimize",
        study_name=OPTUNA_CONFIG["study_name"],
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )

    print(f"\n{'=' * 70}")
    print(f"   OPTIMISATION BAYÉSIENNE — {OPTUNA_CONFIG['n_trials']} essais")
    print(f"   {OPTUNA_CONFIG['n_epochs_per_trial']} epochs/essai | sampler: TPE | pruner: Median")
    print(f"{'=' * 70}\n")

    study.optimize(objective, n_trials=OPTUNA_CONFIG["n_trials"], show_progress_bar=True)

    best = study.best_trial
    print(f"\n{'=' * 70}")
    print(f"   MEILLEUR ESSAI #{best.number}  —  val_loss: {best.value:.4f}")
    print(f"{'=' * 70}")
    for k, v in best.params.items():
        print(f"   {k}: {v}")

    report = {
        "best_trial": best.number,
        "best_val_loss": best.value,
        "best_params": best.params,
        "all_trials": [
            {"number": t.number, "value": t.value, "params": t.params, "state": str(t.state)}
            for t in study.trials
        ],
    }
    report_path = os.path.join(OPTUNA_CONFIG["output_dir"], "optuna_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n   Rapport sauvegardé : {report_path}")

    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        values = [t.value for t in study.trials if t.value is not None]
        axes[0].plot(values, marker='o', linewidth=1.5)
        axes[0].set_xlabel("Essai")
        axes[0].set_ylabel("Val Loss")
        axes[0].set_title("Historique des essais Optuna")
        axes[0].grid(True, alpha=0.3)
        importances = optuna.importance.get_param_importances(study)
        axes[1].barh(list(importances.keys()), list(importances.values()))
        axes[1].set_xlabel("Importance relative")
        axes[1].set_title("Importance des hyperparamètres")
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OPTUNA_CONFIG["output_dir"], "optuna_results.png"), dpi=150)
        plt.close()
        print(f"   Graphiques : {OPTUNA_CONFIG['output_dir']}/optuna_results.png")
    except Exception:
        pass

    return best.params


# =============================================================================
# ENTRAÎNEMENT
# =============================================================================

def collate_fn(batch):
    return tuple(zip(*batch))


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = loss_classifier = loss_box_reg = 0
    loss_mask = loss_objectness = loss_rpn_box = 0

    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}")
    for images, targets in pbar:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses    = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        total_loss     += losses.item()
        loss_classifier += loss_dict.get('loss_classifier',    torch.tensor(0)).item()
        loss_box_reg    += loss_dict.get('loss_box_reg',       torch.tensor(0)).item()
        loss_mask       += loss_dict.get('loss_mask',          torch.tensor(0)).item()
        loss_objectness += loss_dict.get('loss_objectness',    torch.tensor(0)).item()
        loss_rpn_box    += loss_dict.get('loss_rpn_box_reg',   torch.tensor(0)).item()
        pbar.set_postfix({
            'loss': f"{losses.item():.4f}",
            'mask': f"{loss_dict.get('loss_mask', torch.tensor(0)).item():.4f}"
        })

    n = len(data_loader)
    return {
        'total':      total_loss     / n,
        'classifier': loss_classifier / n,
        'box_reg':    loss_box_reg    / n,
        'mask':       loss_mask       / n,
        'objectness': loss_objectness / n,
        'rpn_box':    loss_rpn_box    / n,
    }


@torch.no_grad()
def _evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    for images, targets in data_loader:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        model.train()
        loss_dict = model(images, targets)
        model.eval()
        total_loss += sum(loss for loss in loss_dict.values()).item()
    return total_loss / len(data_loader)


def save_checkpoint(model, optimizer, epoch, loss, path, time_stats=None, model_config=None):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    if time_stats:
        checkpoint['time_stats'] = time_stats
    if model_config:
        checkpoint['model_config'] = model_config
    torch.save(checkpoint, path)


# =============================================================================
# BOUCLE D'ENTRAÎNEMENT PRINCIPALE
# =============================================================================

def run_training(model, device, train_loader, val_loader, model_config):
    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=CONFIG["learning_rate"],
        momentum=CONFIG["momentum"],
        weight_decay=CONFIG["weight_decay"],
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=CONFIG["lr_step_size"], gamma=CONFIG["lr_gamma"]
    )

    history = {
        'train_loss': [], 'val_loss': [], 'lr': [],
        'epoch_times': [], 'cumulative_times': []
    }
    best_val_loss = float('inf')
    timer = TrainingTimer(CONFIG["num_epochs"])

    print("\n" + "=" * 70)
    print("   DEBUT DE L'ENTRAINEMENT")
    print(f"   Demarre le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Epochs: {CONFIG['num_epochs']} | Batch size: {CONFIG['batch_size']}")
    print("=" * 70)

    timer.start_training()

    for epoch in range(CONFIG["num_epochs"]):
        timer.start_epoch()
        train_losses = train_one_epoch(model, optimizer, train_loader, device, epoch)
        val_loss     = _evaluate(model, val_loader, device)
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        time_stats = timer.end_epoch(epoch)

        history['train_loss'].append(train_losses['total'])
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        history['epoch_times'].append(time_stats['epoch_time'])
        history['cumulative_times'].append(time_stats['total_elapsed'])

        print(f"\n{'─' * 70}")
        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']} | {time_stats['progress_percent']:.1f}%")
        print(f"   Train Loss: {train_losses['total']:.4f}  (mask: {train_losses['mask']:.4f})")
        print(f"   Val Loss:   {val_loss:.4f}")
        print(f"   LR:         {current_lr:.6f}")
        print(f"   Epoch:      {format_time(time_stats['epoch_time'])}  |  "
              f"Total: {format_time(time_stats['total_elapsed'])}  |  "
              f"Restant: {format_time(time_stats['estimated_remaining'])}  |  "
              f"ETA: {time_stats['eta'].strftime('%H:%M:%S')}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(CONFIG["output_dir"], "best_model.pth"),
                time_stats={'epoch_time': time_stats['epoch_time'],
                            'total_elapsed': time_stats['total_elapsed']},
                model_config=model_config,
            )
            print(f"   Meilleur modele sauvegarde!")

        if (epoch + 1) % CONFIG["save_every"] == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(CONFIG["output_dir"], f"checkpoint_epoch_{epoch+1}.pth"),
                time_stats={'epoch_time': time_stats['epoch_time'],
                            'total_elapsed': time_stats['total_elapsed']},
                model_config=model_config,
            )
            print(f"   Checkpoint epoch {epoch+1} sauvegarde")

    final_time_stats = timer.get_final_stats()
    history['time_stats'] = final_time_stats

    save_checkpoint(
        model, optimizer, CONFIG["num_epochs"] - 1, val_loss,
        os.path.join(CONFIG["output_dir"], "final_model.pth"),
        time_stats=final_time_stats,
        model_config=model_config,
    )

    with open(os.path.join(CONFIG["output_dir"], "history.json"), 'w') as f:
        json.dump(history, f, indent=2)

    _save_plots(history, final_time_stats)
    _print_final_report(history, best_val_loss, final_time_stats)
    _save_text_report(history, best_val_loss, final_time_stats)


def _save_plots(history, final_time_stats):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'],   label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Courbes de perte - Mask R-CNN Cadastral')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].bar(range(1, len(history['epoch_times']) + 1), history['epoch_times'],
                color='steelblue', alpha=0.7, label='Temps par epoch')
    axes[1].axhline(y=final_time_stats['avg_epoch_time'], color='red', linestyle='--',
                    linewidth=2,
                    label=f"Moyenne: {final_time_stats['avg_epoch_time_formatted']}")
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Temps (secondes)')
    axes[1].set_title('Temps par epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], "training_curves.png"), dpi=150)
    plt.close()


def _print_final_report(history, best_val_loss, fts):
    print("\n" + "=" * 70)
    print("   ENTRAINEMENT TERMINE")
    print("=" * 70)
    print(f"   Meilleure Val Loss: {best_val_loss:.4f}")
    print(f"   Train Loss finale:  {history['train_loss'][-1]:.4f}")
    print(f"   Val Loss finale:    {history['val_loss'][-1]:.4f}")
    print(f"   Temps total:        {fts['total_time_formatted']}")
    print(f"   Temps moyen/epoch:  {fts['avg_epoch_time_formatted']}")
    print(f"   Fichiers:           {CONFIG['output_dir']}/")
    print("=" * 70)


def _save_text_report(history, best_val_loss, fts):
    report_path = os.path.join(CONFIG["output_dir"], "training_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("   RAPPORT D'ENTRAINEMENT - MASK R-CNN CADASTRAL\n")
        f.write("=" * 70 + "\n\n")
        f.write("CONFIGURATION\n" + "-" * 50 + "\n")
        for key, value in CONFIG.items():
            f.write(f"   {key}: {value}\n")
        f.write("\nPERFORMANCES\n" + "-" * 50 + "\n")
        f.write(f"   Meilleure Val Loss: {best_val_loss:.4f}\n")
        f.write(f"   Train Loss finale:  {history['train_loss'][-1]:.4f}\n")
        f.write(f"   Val Loss finale:    {history['val_loss'][-1]:.4f}\n")
        f.write("\nTEMPS D'ENTRAINEMENT\n" + "-" * 50 + "\n")
        f.write(f"   Debut:              {fts['start_datetime']}\n")
        f.write(f"   Fin:                {fts['end_datetime']}\n")
        f.write(f"   Temps total:        {fts['total_time_formatted']}\n")
        f.write(f"   Temps moyen/epoch:  {fts['avg_epoch_time_formatted']}\n")
        f.write(f"   Epoch la + rapide:  {fts['min_epoch_time_formatted']}\n")
        f.write(f"   Epoch la + lente:   {fts['max_epoch_time_formatted']}\n")
        f.write(f"   Ecart-type:         {fts['std_epoch_time']:.2f}s\n")
        f.write("\nTEMPS PAR EPOCH\n" + "-" * 50 + "\n")
        for i, t in enumerate(fts['epoch_times']):
            f.write(f"   Epoch {i+1:3d}: {format_time(t)}\n")
    print(f"   Rapport sauvegarde: {report_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Mask R-CNN - Toitures Cadastrales",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes disponibles:
  simple    Mask R-CNN standard, hyperparamètres fixes du CONFIG
  attention Mask R-CNN + CBAM (mécanisme d'attention sur le FPN)
  optimize  Mask R-CNN standard + recherche bayésienne des hyperparamètres (Optuna)
        """
    )
    parser.add_argument(
        "--mode", choices=["simple", "attention", "optimize"],
        default="simple",
        help="Mode d'entraînement (défaut: simple)"
    )
    parser.add_argument(
        "--n-trials", type=int, default=OPTUNA_CONFIG["n_trials"],
        help=f"Nombre d'essais Optuna — mode optimize uniquement (défaut: {OPTUNA_CONFIG['n_trials']})"
    )
    parser.add_argument(
        "--n-epochs-trial", type=int, default=OPTUNA_CONFIG["n_epochs_per_trial"],
        help=f"Epochs par essai Optuna (défaut: {OPTUNA_CONFIG['n_epochs_per_trial']})"
    )
    parser.add_argument(
        "--cbam-reduction", type=int, default=CONFIG["cbam_reduction"],
        help=f"CBAM reduction — mode attention uniquement (défaut: {CONFIG['cbam_reduction']})"
    )
    parser.add_argument(
        "--cbam-kernel-size", type=int, default=CONFIG["cbam_kernel_size"],
        choices=[3, 5, 7],
        help=f"CBAM kernel size — mode attention uniquement (défaut: {CONFIG['cbam_kernel_size']})"
    )
    parser.add_argument(
        "--aug", nargs='*', default=[], metavar='CLASSE:COEFF',
        help="Coefficients d'augmentation par classe (ex: --aug dalle:2 tole:3)"
    )
    args = parser.parse_args()

    OPTUNA_CONFIG["n_trials"]          = args.n_trials
    OPTUNA_CONFIG["n_epochs_per_trial"] = args.n_epochs_trial

    print("=" * 70)
    print("   MASK R-CNN - Segmentation des Toitures Cadastrales")
    print(f"   Mode: {args.mode.upper()}")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"   GPU:  {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # ── Coefficients d'augmentation (depuis arguments CLI) ──────────────────
    aug_coeffs = parse_aug_coeffs(args.aug, CONFIG["classes"])

    print("\nChargement du dataset...")
    _coco_ref = COCO(CONFIG["annotations_file"])
    all_ids   = list(_coco_ref.imgs.keys())
    np.random.seed(42)
    np.random.shuffle(all_ids)
    split_idx     = int(CONFIG["train_split"] * len(all_ids))
    train_raw_ids = all_ids[:split_idx]
    val_ids       = all_ids[split_idx:]

    # ── Augmentation par classe ──────────────────────────────────────────────
    cat_id_to_name  = {cid: _coco_ref.cats[cid]['name'] for cid in _coco_ref.getCatIds()}
    train_aug_ids, _coeff_map = oversample_by_class(
        train_raw_ids, _coco_ref, cat_id_to_name, aug_coeffs
    )
    print_augmentation_report(
        _coco_ref, train_raw_ids, train_aug_ids, cat_id_to_name, aug_coeffs
    )

    train_dataset = CadastralDataset(
        CONFIG["images_dir"], CONFIG["annotations_file"],
        transforms=get_transforms(train=True), image_ids=train_aug_ids
    )
    val_dataset = CadastralDataset(
        CONFIG["images_dir"], CONFIG["annotations_file"],
        transforms=None, image_ids=val_ids
    )
    print(f"   Train: {len(train_dataset)} samples  ({len(set(train_aug_ids))} images uniques)")
    print(f"   Val:   {len(val_dataset)} images")

    pin = device.type == 'cuda'

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], shuffle=True,
        num_workers=CONFIG["num_workers"], collate_fn=collate_fn, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"], shuffle=False,
        num_workers=CONFIG["num_workers"], collate_fn=collate_fn, pin_memory=pin,
    )

    num_classes = len(CONFIG["classes"])

    # --- Sélection du mode ---

    if args.mode == "simple":
        print("\nArchitecture: Mask R-CNN ResNet50-FPN v2 (standard)")
        model        = get_model_simple(num_classes)
        model_config = {"mode": "simple"}

    elif args.mode == "attention":
        cbam_r = args.cbam_reduction
        cbam_k = args.cbam_kernel_size
        print(f"\nArchitecture: Mask R-CNN ResNet50-FPN v2 + CBAM")
        print(f"   cbam_reduction={cbam_r}, cbam_kernel_size={cbam_k}")
        model        = get_model_attention(num_classes, cbam_r, cbam_k)
        model_config = {"mode": "attention", "cbam_reduction": cbam_r, "cbam_kernel_size": cbam_k}

    else:  # optimize
        print("\nArchitecture: Mask R-CNN ResNet50-FPN v2 (standard)")
        print("Lancement de l'optimisation bayesienne des hyperparametres...")
        best_params = _run_optimization(device, train_loader, val_loader, num_classes)
        for key in ("learning_rate", "weight_decay", "momentum", "lr_step_size"):
            if key in best_params:
                CONFIG[key] = best_params[key]
        print(f"\nHyperparametres optimises appliques a l'entrainement.")
        model        = get_model_simple(num_classes)
        model_config = {"mode": "optimize", "best_params": best_params}

    model.to(device)
    print(f"   Classes: {CONFIG['classes']}")

    run_training(model, device, train_loader, val_loader, model_config)


if __name__ == "__main__":
    main()
