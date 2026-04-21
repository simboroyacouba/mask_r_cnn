"""
Inférence Mask R-CNN - Prédiction sur nouvelles images
Segmentation des toitures cadastrales

Fonctionnalités:
- Temps d'inférence par image
- Résumé global pour les dossiers
- Export des masques individuels
- Rapports JSON détaillés
"""

import os
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms import v2 as T
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
import time
import json
import yaml

# Charger les variables d'environnement
load_dotenv()
# =============================================================================
# CONFIGURATION
# =============================================================================

def load_classes(yaml_path=None):
    path = yaml_path or os.getenv("CLASSES_FILE", "classes.yaml")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)['classes']
    
    # Palette de couleurs auto-générée pour toutes les classes
_PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 165, 0),
    (128, 0, 128), (0, 255, 255), (255, 20, 147), (0, 128, 0),
]
CLASSES = load_classes()

COLORS = {
    cls: _PALETTE[i % len(_PALETTE)]
    for i, cls in enumerate(CLASSES[1:])  # on ignore __background__
}




CONFIG = {
    "model_path": os.getenv("SEGMENTATION_MODEL_PATH", "./output/best_model.pth"),
    "input_dir": os.getenv("SEGMENTATION_TEST_IMAGES_DIR", "./test_images"),
    "classes_file": os.getenv("CLASSES_FILE", "classes.yaml"),
    "output_dir": os.getenv("SEGMENTATION_OUTPUT_DIR", "./predictions"),
    "score_threshold": 0.5,
    "export_masks": False,
    "show_display": False,
}

# =============================================================================
# UTILITAIRES
# =============================================================================

def format_time(seconds):
    """Formater les secondes en format lisible"""
    if seconds < 1:
        return f"{seconds*1000:.1f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"


# =============================================================================
# MÉCANISME D'ATTENTION (CBAM) — doit correspondre à train.py
# =============================================================================

class ChannelAttention(nn.Module):
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

    def forward(self, x):
        b, c = x.shape[:2]
        avg = self.fc(self.avg_pool(x).view(b, c))
        mx  = self.fc(self.max_pool(x).view(b, c))
        return x * self.sigmoid(avg + mx).view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial_att(self.channel_att(x))


class AttentionFPN(nn.Module):
    def __init__(self, fpn, out_channels=256, cbam_reduction=16, cbam_kernel_size=7):
        super().__init__()
        self.fpn = fpn
        self.cbam_modules = nn.ModuleDict({
            '0': CBAM(out_channels, cbam_reduction, cbam_kernel_size),
            '1': CBAM(out_channels, cbam_reduction, cbam_kernel_size),
            '2': CBAM(out_channels, cbam_reduction, cbam_kernel_size),
            '3': CBAM(out_channels, cbam_reduction, cbam_kernel_size),
            'pool': CBAM(out_channels, cbam_reduction, cbam_kernel_size),
        })

    def forward(self, x):
        features = self.fpn(x)
        attended = OrderedDict()
        for key, feat in features.items():
            if key in self.cbam_modules:
                attended[key] = self.cbam_modules[key](feat)
            else:
                attended[key] = feat
        return attended


# =============================================================================
# MODÈLE
# =============================================================================

def get_model(num_classes, cbam_reduction=16, cbam_kernel_size=7):
    model = maskrcnn_resnet50_fpn_v2(weights=None)
    fpn_out_channels = model.backbone.out_channels
    model.backbone.fpn = AttentionFPN(
        model.backbone.fpn, fpn_out_channels, cbam_reduction, cbam_kernel_size
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config   = checkpoint.get('model_config', {})
    cbam_reduction   = model_config.get('cbam_reduction',   16)
    cbam_kernel_size = model_config.get('cbam_kernel_size',  7)
    model = get_model(len(CLASSES), cbam_reduction, cbam_kernel_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"✅ Modèle chargé: {checkpoint_path}")
    print(f"   CBAM: reduction={cbam_reduction}, kernel_size={cbam_kernel_size}")
    return model


# =============================================================================
# INFÉRENCE
# =============================================================================

def predict(model, image_path, device, score_threshold=0.5):
    image = Image.open(image_path).convert("RGB")
    image_tensor = T.ToTensor()(image)
    
    start_time = time.time()
    with torch.no_grad():
        predictions = model([image_tensor.to(device)])
    inference_time = time.time() - start_time
    
    pred = predictions[0]
    keep = pred['scores'] > score_threshold
    
    result = {
        'boxes': pred['boxes'][keep].cpu().numpy(),
        'labels': pred['labels'][keep].cpu().numpy(),
        'scores': pred['scores'][keep].cpu().numpy(),
        'masks': pred['masks'][keep].cpu().numpy(),
        'inference_time': inference_time,
    }
    return image, result


def calculate_surface(mask):
    return int(np.sum(mask > 0.5))


# =============================================================================
# VISUALISATION
# =============================================================================

def visualize_predictions(image, predictions, output_path=None, show=True):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    axes[0].imshow(image)
    axes[0].set_title("Image originale")
    axes[0].axis('off')
    
    axes[1].imshow(image)
    
    masks = predictions['masks']
    boxes = predictions['boxes']
    labels = predictions['labels']
    scores = predictions['scores']
    inference_time = predictions.get('inference_time', 0)
    
    overlay = np.zeros((*np.array(image).shape[:2], 4))
    
    for i, (mask, box, label, score) in enumerate(zip(masks, boxes, labels, scores)):
        class_name = CLASSES[label]
        color = COLORS.get(class_name, (128, 128, 128))
        color_normalized = [c/255 for c in color]
        
        mask_binary = mask[0] > 0.5
        overlay[mask_binary] = [*color_normalized, 0.5]
        
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2,
                                  edgecolor=color_normalized, facecolor='none')
        axes[1].add_patch(rect)
        
        surface = calculate_surface(mask[0])
        label_text = f"{class_name}\n{score:.2f} | {surface:,} px"
        axes[1].text(x1, y1-10, label_text, fontsize=8, color='white',
                     bbox=dict(boxstyle='round', facecolor=color_normalized, alpha=0.8))
    
    axes[1].imshow(overlay)
    axes[1].set_title(f"Prédictions ({len(masks)} objets) | ⏱️ {format_time(inference_time)}")
    axes[1].axis('off')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def export_masks(predictions, output_dir, image_name):
    os.makedirs(output_dir, exist_ok=True)
    for i, (mask, label, score) in enumerate(zip(predictions['masks'], predictions['labels'], predictions['scores'])):
        class_name = CLASSES[label]
        mask_binary = (mask[0] > 0.5).astype(np.uint8) * 255
        mask_path = os.path.join(output_dir, f"{image_name}_{i:02d}_{class_name}_{score:.2f}.png")
        Image.fromarray(mask_binary).save(mask_path)


def generate_report(predictions, image_name):
    report = {
        'image': image_name,
        'timestamp': datetime.now().isoformat(),
        'inference_time_ms': predictions.get('inference_time', 0) * 1000,
        'total_objects': len(predictions['labels']),
        'surfaces_by_class': {},
        'details': []
    }
    
    for class_name in CLASSES[1:]:
        report['surfaces_by_class'][class_name] = {'count': 0, 'total_surface_px': 0}
    
    for i, (mask, label, score, box) in enumerate(zip(
        predictions['masks'], predictions['labels'], predictions['scores'], predictions['boxes']
    )):
        class_name = CLASSES[label]
        surface = calculate_surface(mask[0])
        report['surfaces_by_class'][class_name]['count'] += 1
        report['surfaces_by_class'][class_name]['total_surface_px'] += surface
        report['details'].append({
            'id': i, 'class': class_name, 'score': float(score),
            'surface_px': int(surface), 'bbox': box.tolist()
        })
    return report


# =============================================================================
# RÉSUMÉ GLOBAL
# =============================================================================

def generate_summary(all_reports, output_dir, total_processing_time):
    summary = {
        'timestamp': datetime.now().isoformat(),
        'model': 'Mask R-CNN',
        'total_images': len(all_reports),
        'total_processing_time_s': total_processing_time,
        'avg_inference_time_ms': 0,
        'total_objects': 0,
        'objects_by_class': {c: 0 for c in CLASSES[1:]},
        'surfaces_by_class': {c: 0 for c in CLASSES[1:]},
        'per_image_stats': []
    }
    
    total_inference_time = 0
    for report in all_reports:
        total_inference_time += report['inference_time_ms']
        summary['total_objects'] += report['total_objects']
        for class_name, data in report['surfaces_by_class'].items():
            summary['objects_by_class'][class_name] += data['count']
            summary['surfaces_by_class'][class_name] += data['total_surface_px']
        summary['per_image_stats'].append({
            'image': report['image'],
            'objects': report['total_objects'],
            'inference_time_ms': report['inference_time_ms']
        })
    
    summary['avg_inference_time_ms'] = total_inference_time / len(all_reports) if all_reports else 0
    
    # Sauvegarder JSON
    with open(os.path.join(output_dir, "summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Sauvegarder TXT
    total_surface = sum(summary['surfaces_by_class'].values())
    with open(os.path.join(output_dir, "summary.txt"), 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("   RÉSUMÉ D'INFÉRENCE - MASK R-CNN CADASTRAL\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"📅 Date: {summary['timestamp']}\n")
        f.write(f"🖼️  Images traitées: {summary['total_images']}\n")
        f.write(f"⏱️  Temps total: {format_time(summary['total_processing_time_s'])}\n")
        f.write(f"⏱️  Temps moyen/image: {summary['avg_inference_time_ms']:.1f} ms\n")
        f.write(f"🎯 Total objets: {summary['total_objects']}\n\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Classe':<25} {'Objets':>10} {'Surface (px)':>15} {'%':>10}\n")
        f.write("-" * 70 + "\n")
        for class_name in CLASSES[1:]:
            count = summary['objects_by_class'][class_name]
            surface = summary['surfaces_by_class'][class_name]
            pct = (surface / total_surface * 100) if total_surface > 0 else 0
            f.write(f"{class_name:<25} {count:>10} {surface:>15,} {pct:>9.1f}%\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'TOTAL':<25} {summary['total_objects']:>10} {total_surface:>15,} {'100.0%':>10}\n")
        f.write("\n" + "-" * 70 + "\n")
        f.write("DÉTAILS PAR IMAGE\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Image':<40} {'Objets':>10} {'Temps (ms)':>15}\n")
        f.write("-" * 70 + "\n")
        for stat in summary['per_image_stats']:
            img_name = stat['image'][:38] + '..' if len(stat['image']) > 40 else stat['image']
            f.write(f"{img_name:<40} {stat['objects']:>10} {stat['inference_time_ms']:>15.1f}\n")
        f.write("=" * 70 + "\n")
    
    return summary


def print_summary(summary):
    print("\n" + "=" * 70)
    print("   📊 RÉSUMÉ GLOBAL - MASK R-CNN")
    print("=" * 70)
    print(f"\n   🖼️  Images traitées:     {summary['total_images']}")
    print(f"   ⏱️  Temps total:          {format_time(summary['total_processing_time_s'])}")
    print(f"   ⏱️  Temps moyen/image:    {summary['avg_inference_time_ms']:.1f} ms")
    print(f"   🎯 Total objets:         {summary['total_objects']}")
    
    total_surface = sum(summary['surfaces_by_class'].values())
    print(f"\n   📋 Par classe:")
    print(f"   {'-'*50}")
    for class_name in CLASSES[1:]:
        count = summary['objects_by_class'][class_name]
        surface = summary['surfaces_by_class'][class_name]
        pct = (surface / total_surface * 100) if total_surface > 0 else 0
        if count > 0:
            print(f"      • {class_name}: {count} objets | {surface:,} px ({pct:.1f}%)")
    print("\n" + "=" * 70)


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def process_directory(model, input_dir, output_dir, device, score_threshold=0.5, export_masks_flag=False, show_display=False):
    os.makedirs(output_dir, exist_ok=True)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
    image_paths = sorted([p for p in Path(input_dir).iterdir() if p.suffix.lower() in image_extensions])
    
    if not image_paths:
        print(f"❌ Aucune image trouvée dans {input_dir}")
        return []
    
    print(f"\n🖼️  {len(image_paths)} images à traiter\n")
    
    all_reports = []
    start_total = time.time()
    
    for idx, img_path in enumerate(image_paths, 1):
        print(f"[{idx}/{len(image_paths)}] 🔍 {img_path.name}")
        
        image, predictions = predict(model, str(img_path), device, score_threshold)
        
        output_path = os.path.join(output_dir, f"{img_path.stem}_pred.png")
        visualize_predictions(image, predictions, output_path, show=show_display)
        
        if export_masks_flag and len(predictions['masks']) > 0:
            export_masks(predictions, os.path.join(output_dir, "masks", img_path.stem), img_path.stem)
        
        report = generate_report(predictions, img_path.name)
        all_reports.append(report)
        print(f"   ✅ {report['total_objects']} objets | ⏱️ {report['inference_time_ms']:.1f} ms")
    
    total_processing_time = time.time() - start_total
    
    # Sauvegarder les rapports
    with open(os.path.join(output_dir, "reports.json"), 'w', encoding='utf-8') as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False)
    
    # Générer le résumé
    summary = generate_summary(all_reports, output_dir, total_processing_time)
    print_summary(summary)
    
    print(f"\n📁 Résultats sauvegardés dans: {output_dir}")
    print(f"   ├── *_pred.png (visualisations)")
    print(f"   ├── reports.json (rapports détaillés)")
    print(f"   ├── summary.json (résumé JSON)")
    print(f"   └── summary.txt (résumé lisible)")
    
    return all_reports


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Configuration depuis variables d'environnement
    model_path = CONFIG["model_path"]
    input_dir = CONFIG["input_dir"]
    output_dir = CONFIG["output_dir"]
    score_threshold = CONFIG["score_threshold"]
    export_masks_flag = CONFIG["export_masks"]
    show_display = CONFIG["show_display"]
    
    # Vérifications
    if not os.path.exists(model_path):
        print(f"❌ Modèle non trouvé: {model_path}")
        print(f"   Définissez SEGMENTATION_MODEL_PATH")
        return
    
    if not os.path.exists(input_dir):
        print(f"❌ Dossier d'images non trouvé: {input_dir}")
        print(f"   Définissez SEGMENTATION_TEST_IMAGES_DIR")
        return
    
    print("=" * 70)
    print("   🚀 INFÉRENCE MASK R-CNN CADASTRAL")
    print("=" * 70)
    print(f"\n📂 Configuration:")
    print(f"   • Modèle:      {model_path}")
    print(f"   • Images:      {input_dir}")
    print(f"   • Sortie:      {output_dir}")
    print(f"   • Seuil:       {score_threshold}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   • Device:      {device}")
    
    model = load_model(model_path, device)
    
    input_path = Path(input_dir)
    
    if input_path.is_dir():
        process_directory(model, str(input_path), output_dir, device, score_threshold, export_masks_flag, show_display)
    else:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n🔍 Traitement: {input_path.name}")
        
        image, predictions = predict(model, str(input_path), device, score_threshold)
        
        output_path = os.path.join(output_dir, f"{input_path.stem}_pred.png")
        visualize_predictions(image, predictions, output_path, show=show_display)
        
        if export_masks_flag and len(predictions['masks']) > 0:
            export_masks(predictions, os.path.join(output_dir, "masks"), input_path.stem)
        
        report = generate_report(predictions, input_path.name)
        print(f"\n{'='*60}")
        print(f"📊 RAPPORT - {report['image']}")
        print(f"{'='*60}")
        print(f"   ⏱️  Temps d'inférence: {report['inference_time_ms']:.1f} ms")
        print(f"   🎯 Objets détectés: {report['total_objects']}")
        for class_name, data in report['surfaces_by_class'].items():
            if data['count'] > 0:
                print(f"      • {class_name}: {data['count']} objets, {data['total_surface_px']:,} px")
        print(f"{'='*60}")
        
        with open(os.path.join(output_dir, f"{input_path.stem}_report.json"), 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
