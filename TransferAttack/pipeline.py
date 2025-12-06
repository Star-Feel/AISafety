import argparse
import json
import logging
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as tv_models
import torchvision.transforms as T
from PIL import Image
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

import transferattack
from transferattack.utils import EnsembleModel, save_images, wrap_model
from robustbench.utils import load_model as load_robust_model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# LATEST_ATTACKS = ['mef', 'anda', 'mumodig', 'gaa', 'foolmix', 'bsr', 'ops', 'bfa', 'p2fa', 'ana', 'll2s', 'smer']
LATEST_ATTACKS = [ 'bsr', 'ops', 'bfa', 'p2fa', 'ana', 'll2s', 'smer']
SINGLE_MODEL_ONLY_ATTACKS = {'bfa', 'p2fa'}


def split_attack_sequence(name: str) -> List[str]:
    if '->' not in name:
        return [name]
    return [token.strip() for token in name.split('->') if token.strip()]


def attack_requires_single_model(name: str) -> bool:
    return any(part in SINGLE_MODEL_ONLY_ATTACKS for part in split_attack_sequence(name))


def attack_dir_slug(name: str) -> str:
    sanitized = re.sub(r'[^A-Za-z0-9._+\-]+', '_', name)
    sanitized = sanitized.strip('_')
    return sanitized or 'attack'


class CIFAR10CleanDataset(Dataset):
    """Minimal dataloader for datasets/CIFAR10_clean."""

    def __init__(
        self,
        root: str,
        resize_to: int = 224,
        targeted: bool = False,
        target_file: Optional[str] = None,
        adv_dir: Optional[str] = None,
    ) -> None:
        self.root = Path(root)
        self.img_dir = Path(adv_dir) if adv_dir else self.root / 'images'
        self.resize = T.Resize((resize_to, resize_to))
        self.to_tensor = T.ToTensor()
        self.targeted = targeted
        self.samples = self._load_labels(self.root / 'label.txt')
        self.target_map = self._load_targets(target_file) if target_file else None

    def _load_labels(self, label_path: Path) -> List[Tuple[str, int]]:
        samples: List[Tuple[str, int]] = []
        with open(label_path, 'r') as handle:
            for line in handle:
                row = line.strip()
                if not row:
                    continue
                fname, label = row.split()
                samples.append((fname, int(label)))
        return samples

    def _load_targets(self, target_file: str) -> Dict[str, int]:
        target_map: Dict[str, int] = {}
        with open(target_file, 'r') as handle:
            for line in handle:
                row = line.strip()
                if not row:
                    continue
                fname, label = row.split()
                target_map[fname] = int(label)
        return target_map

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        filename, label = self.samples[index]
        image_path = self.img_dir / filename
        # if not image_path.exists() and self.img_dir != self.root / 'images':
        #     print(f'[warn] Missing adversarial image {filename} in {self.img_dir}, falling back to clean images')
        #     fallback_dir = self.root / 'images'
        #     fallback_path = fallback_dir / filename
        #     if fallback_path.exists():
        #         image_path = fallback_path
        #     else:
        #         raise FileNotFoundError(
        #             f"Missing image {filename} in {self.img_dir} and fallback clean directory {fallback_dir}"
        #         )
        with Image.open(image_path) as img:
            image = img.convert('RGB')
        image = self.resize(image)
        image = self.to_tensor(image)
        if self.targeted:
            if self.target_map is None:
                raise ValueError('target_file must be provided for targeted attacks')
            if filename not in self.target_map:
                raise KeyError(f'Missing target label for {filename}')
            return image, [label, self.target_map[filename]], filename
        return image, label, filename


def parse_args():
    parser = argparse.ArgumentParser(
        description='Unified CIFAR10 -> CIFAR10_clean pipeline for evaluating latest TransferAttack methods.'
    )
    parser.add_argument('--clean_dir', default='datasets/CIFAR10_clean', type=str,
                        help='Path to datasets/CIFAR10_clean (expects images/ & label.txt)')
    parser.add_argument('--cifar_root', default='datasets/CIFAR10', type=str,
                        help='Root folder that already contains cifar-10-batches-py')
    parser.add_argument('--checkpoint_dir', default='checkpoints/cifar10', type=str,
                        help='Directory to store fine-tuned backbones')
    parser.add_argument('--attack_out', default='attackout', type=str,
                        help='Directory to store generated adversarial images')
    parser.add_argument('--log_dir', default='log', type=str, help='Directory for pipeline logs/metrics')
    parser.add_argument('--results_file', default='log/pipeline_results.json', type=str,
                        help='File to append aggregated JSON metrics')

    parser.add_argument('--surrogates', nargs='+', default=[],
                        help='Backbones to fine-tune on CIFAR10 and use for attacks')# default=['resnet50', 'vit_base_patch16_224']
    parser.add_argument('--eval_models', nargs='+', default=None,
                        help='Models used for evaluation (defaults to surrogates)')
    parser.add_argument('--attacks', nargs='+', default=LATEST_ATTACKS,
                        help='Attack names (see README). Defaults to the newest ones.')
    parser.add_argument('--attack_ensembles', nargs='*', default=None,
                        help='Optional ensembles, e.g. "resnet50+vit_base_patch16_224"')

    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batchsize', default=64, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--val_split', default=0.1, type=float)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--skip_finetune', action='store_true',
                        help='Skip fine-tuning if checkpoints already exist')
    parser.add_argument('--force_finetune', action='store_true',
                        help='Always re-train, even when checkpoints exist')

    parser.add_argument('--eps', default=16 / 255, type=float)
    parser.add_argument('--alpha', default=1.6 / 255, type=float)
    parser.add_argument('--attack_steps', default=10, type=int)
    parser.add_argument('--momentum', default=1.0, type=float)
    parser.add_argument('--random_start', action='store_true')
    parser.add_argument('--targeted', action='store_true')
    parser.add_argument('--target_file', default=None, type=str,
                        help='Optional target label file for targeted evaluation')
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--save_size', default=32, type=int,
                        help='Resolution for saving adversarial images (set None to keep model input size)')
    parser.add_argument('--max_batches', default=None, type=int,
                        help='Debug option: limit number of attack batches')
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--robust_models', nargs='*', default=None,
                        help='Optional RobustBench model names (e.g., Bartoldson2024Adversarial_WRN-94-16) to include during evaluation')
    parser.add_argument('--robust_dataset', default='cifar10', type=str,
                        help='Dataset flag passed to RobustBench loader (default: cifar10)')
    parser.add_argument('--robust_threat_model', default='Linf', type=str,
                        help='Threat model passed to RobustBench loader (default: Linf)')
    parser.add_argument('--robust_input_size', default=32, type=int,
                        help='Spatial size fed into RobustBench backbones (default 32 for CIFAR10 models)')
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(log_dir: Path) -> Tuple[logging.Logger, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = log_dir / f'pipeline_{timestamp}.log'
    logger = logging.getLogger('transferattack.pipeline')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger, log_path


def build_transforms(size: int) -> Tuple[T.Compose, T.Compose]:
    train_tf = T.Compose([
        T.RandomResizedCrop(size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = T.Compose([
        T.Resize(size),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


def split_indices(total: int, val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(total)
    rng.shuffle(indices)
    val_size = int(total * val_ratio)
    val_idx = indices[:val_size].tolist()
    train_idx = indices[val_size:].tolist()
    return train_idx, val_idx


def _reset_classifier(model: nn.Module, num_classes: int) -> nn.Module:
    if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    if hasattr(model, 'classifier'):
        classifier = model.classifier
        if isinstance(classifier, nn.Linear):
            model.classifier = nn.Linear(classifier.in_features, num_classes)
        elif isinstance(classifier, nn.Sequential):
            layers = list(classifier)
            for idx in range(len(layers) - 1, -1, -1):
                if isinstance(layers[idx], nn.Linear):
                    in_features = layers[idx].in_features
                    layers[idx] = nn.Linear(in_features, num_classes)
                    break
            model.classifier = nn.Sequential(*layers)
        return model
    if hasattr(model, 'heads') and isinstance(model.heads, nn.Sequential):
        layers = list(model.heads)
        for idx in range(len(layers) - 1, -1, -1):
            if isinstance(layers[idx], nn.Linear):
                in_features = layers[idx].in_features
                layers[idx] = nn.Linear(in_features, num_classes)
                break
        model.heads = nn.Sequential(*layers)
        return model
    if hasattr(model, 'head') and isinstance(model.head, nn.Linear):
        model.head = nn.Linear(model.head.in_features, num_classes)
    return model


def create_backbone(model_name: str, num_classes: int) -> nn.Module:
    if model_name in tv_models.__dict__:
        model = tv_models.__dict__[model_name](weights='DEFAULT')
        model = _reset_classifier(model, num_classes)
        return model
    if model_name in timm.list_models():
        return timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    raise ValueError(f'Model {model_name} not supported by torchvision/timm')


def train_one_epoch(model, loader, criterion, optimizer, device) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc='Train', leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
    return running_loss / total, correct / total


def get_checkpoint_path(model_name: str, ckpt_dir: Path) -> Path:
    return ckpt_dir / f'{model_name}_cifar10.pth'


def finetune_model(model_name: str, args, device: torch.device, logger: logging.Logger) -> Path:
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = get_checkpoint_path(model_name, ckpt_dir)
    if ckpt_path.exists() and args.skip_finetune and not args.force_finetune:
        logger.info(f'[skip] Found existing checkpoint for {model_name}: {ckpt_path}')
        return ckpt_path
    if ckpt_path.exists() and not args.force_finetune:
        logger.info(f'[reuse] Using existing checkpoint for {model_name}: {ckpt_path}')
        return ckpt_path

    logger.info(f'==> Fine-tuning {model_name} on CIFAR10')
    train_tf, val_tf = build_transforms(args.input_size)
    dataset = torchvision.datasets.CIFAR10(
        root=args.cifar_root,
        train=True,
        download=False,
        transform=train_tf,
    )
    full_dataset = torchvision.datasets.CIFAR10(
        root=args.cifar_root,
        train=True,
        download=False,
        transform=val_tf,
    )
    train_idx, val_idx = split_indices(len(dataset), args.val_split, args.seed)
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=args.batchsize, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=args.batchsize, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = create_backbone(model_name, args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    best_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        logger.info(
            f'Epoch {epoch+1}/{args.epochs} | train_acc={train_acc:.3f} | val_acc={val_acc:.3f} | '
            f'train_loss={train_loss:.4f} | val_loss={val_loss:.4f}'
        )
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f'  [best] Saved checkpoint -> {ckpt_path}')

    return ckpt_path


def finetune_backbones(model_names: Sequence[str], args, device, logger) -> Dict[str, Path]:
    ckpt_map: Dict[str, Path] = {}
    for name in model_names:
        ckpt_map[name] = finetune_model(name, args, device, logger)
    return ckpt_map


def build_attack_model(model_name: str, ckpt_path: Path, num_classes: int, device: torch.device) -> nn.Module:
    model = create_backbone(model_name, num_classes)
    state = torch.load(ckpt_path, map_location='cpu')
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'[warn] Missing keys for {model_name}: {missing}')
    if unexpected:
        print(f'[warn] Unexpected keys for {model_name}: {unexpected}')
    model = wrap_model(model.eval().to(device))
    return model


def inject_backbones(attacker, ensemble: List[str], ckpt_map: Dict[str, Path], num_classes: int,
                     device: torch.device, args, robust_model_cache: Dict[str, nn.Module],
                     robust_names: Sequence[str]):
    robust_name_set = set(robust_names)

    def _load_member(name: str) -> nn.Module:
        if name in ckpt_map:
            return build_attack_model(name, ckpt_map[name], num_classes, device)
        if name in robust_name_set:
            return build_robust_attack_model(name, args, device, robust_model_cache)
        raise KeyError(f'Unknown surrogate backbone {name}. Ensure it is either fine-tuned or provided via --robust_models.')

    if len(ensemble) == 1:
        attacker.model = _load_member(ensemble[0])
        attacker.device = device
        if hasattr(attacker, 'propagate_model_to_sub_attacks'):
            attacker.propagate_model_to_sub_attacks()
        return
    models = [_load_member(name) for name in ensemble]
    attacker.model = EnsembleModel(models)
    attacker.device = attacker.model.device
    if hasattr(attacker, 'propagate_model_to_sub_attacks'):
        attacker.propagate_model_to_sub_attacks()


def compute_batch_ssim(clean_batch: torch.Tensor, adv_batch: torch.Tensor) -> List[float]:
    clean_np = clean_batch.detach().cpu().permute(0, 2, 3, 1).numpy()
    adv_np = adv_batch.detach().cpu().permute(0, 2, 3, 1).numpy()
    scores: List[float] = []
    for clean_img, adv_img in zip(clean_np, adv_np):
        score = structural_similarity(clean_img, adv_img, data_range=1.0, channel_axis=2)
        scores.append(float(score))
    return scores


def save_adv_batch(output_dir: Path, tensors: torch.Tensor, filenames: Sequence[str], save_size: Optional[int]):
    adv = tensors.detach().cpu()
    if save_size is not None:
        adv = F.interpolate(adv, size=(save_size, save_size), mode='bilinear', align_corners=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_images(str(output_dir), adv, filenames)


def run_attack(attack_name: str, ensemble: List[str], attacker, loader: DataLoader, args,
               logger: logging.Logger) -> Tuple[float, Path]:
    ensemble_tag = '+'.join(ensemble)
    logger.info(f'==> Running {attack_name.upper()} with surrogate [{ensemble_tag}]')
    output_dir = Path(args.attack_out) / attack_dir_slug(attack_name) / ensemble_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    ssim_scores: List[float] = []
    for batch_idx, (images, labels, filenames) in enumerate(tqdm(loader, desc=f'{attack_name.upper()}:{ensemble_tag}')):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break
        perturbations = attacker(images, labels)
        adversaries = torch.clamp(images + perturbations.cpu(), 0.0, 1.0)
        ssim_scores.extend(compute_batch_ssim(images, adversaries))
        save_adv_batch(output_dir, adversaries, filenames, args.save_size)
    avg_ssim = float(np.mean(ssim_scores)) if ssim_scores else 0.0
    logger.info(f'[{attack_name.upper()}|{ensemble_tag}] Mean SSIM={avg_ssim:.4f}')
    return avg_ssim, output_dir


def load_eval_model(model_name: str, ckpt_map: Dict[str, Path], num_classes: int,
                    device: torch.device) -> nn.Module:
    if model_name not in ckpt_map:
        raise KeyError(f'Missing checkpoint for eval model {model_name}')
    model = create_backbone(model_name, num_classes)
    state = torch.load(ckpt_map[model_name], map_location='cpu')
    model.load_state_dict(state, strict=False)
    model = wrap_model(model.eval().to(device))
    return model


def load_robustbench_backbone(model_name: str, dataset: str, threat_model: str,
                              device: torch.device) -> nn.Module:
    model = load_robust_model(model_name=model_name, dataset=dataset, threat_model=threat_model)
    return model.eval().to(device)


class RobustBenchPreprocess(nn.Module):

    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] == self.size and x.shape[-1] == self.size:
            return x
        return F.interpolate(x, size=(self.size, self.size), mode='bilinear', align_corners=False)


def build_robust_attack_model(model_name: str, args, device: torch.device,
                              robust_model_cache: Dict[str, nn.Module]) -> nn.Module:
    if model_name not in robust_model_cache:
        backbone = load_robustbench_backbone(
            model_name,
            dataset=args.robust_dataset,
            threat_model=args.robust_threat_model,
            device=device,
        )
        resize_to = getattr(args, 'robust_input_size', None) or args.input_size or 32
        preprocess = RobustBenchPreprocess(resize_to)
        model = nn.Sequential(preprocess, backbone).to(device).eval()
        robust_model_cache[model_name] = model
    return robust_model_cache[model_name]


def compute_arc(acc: float, targeted: bool) -> float:
    return acc if targeted else (1.0 - acc)


def evaluate_attack(ensemble_tag: str, attack_name: str, adv_dir: Path, avg_ssim: float,
                    eval_models: Sequence[str], robust_models: Sequence[str], ckpt_map: Dict[str, Path],
                    args, logger: logging.Logger, device: torch.device,
                    robust_model_cache: Dict[str, nn.Module]):
    dataset = CIFAR10CleanDataset(
        root=args.clean_dir,
        resize_to=args.input_size,
        targeted=args.targeted,
        target_file=args.target_file,
        adv_dir=str(adv_dir),
    )
    loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    metrics: List[Dict] = []

    def _evaluate_model(model: nn.Module, model_name: str) -> None:
        correct, total = 0, 0
        for images, labels, _ in loader:
            target = labels[1] if args.targeted else labels
            logits = model(images.to(device, non_blocking=True))
            preds = logits.argmax(dim=1).detach().cpu()
            correct += (preds == target).sum().item()
            total += target.shape[0]
        acc = correct / total if total else 0.0
        arc = compute_arc(acc, args.targeted)
        score = 100.0 * avg_ssim * arc
        metrics.append({
            'attack': attack_name,
            'ensemble': ensemble_tag,
            'model': model_name,
            'acc': acc,
            'arc': arc,
            'ssim': avg_ssim,
            'score': score,
        })
        logger.info(
            f'[{attack_name.upper()}|{ensemble_tag}] Eval on {model_name}: ' f'acc={acc*100:.2f}% | ARC={arc*100:.2f}% | metric=100*ARC*SSIM={score:.2f}'
        )

    for model_name in eval_models:
        model = load_eval_model(model_name, ckpt_map, args.num_classes, device)
        _evaluate_model(model, model_name)

    for robust_name in robust_models:
        model = build_robust_attack_model(robust_name, args, device, robust_model_cache)
        _evaluate_model(model, robust_name)

    return metrics


def parse_ensembles(specs: Optional[Sequence[str]], surrogates: Sequence[str]) -> List[List[str]]:
    if not specs:
        return [surrogates]
    ensembles: List[List[str]] = []
    for spec in specs:
        members = [token.strip() for token in spec.split('+') if token.strip()]
        if members:
            ensembles.append(members)
    return ensembles


def adjust_ensembles_for_attack(attack_name: str, ensembles: List[List[str]], logger: logging.Logger) -> List[List[str]]:
    if not attack_requires_single_model(attack_name):
        return ensembles
    adjusted: List[List[str]] = []
    seen = set()
    for members in ensembles:
        if len(members) == 1:
            key = tuple(members)
            if key not in seen:
                seen.add(key)
                adjusted.append(members)
            continue
        joined = ' + '.join(members)
        logger.warning(
            f'Attack {attack_name.upper()} only supports single-model surrogates; splitting {joined}'
        )
        for member in members:
            key = (member,)
            if key not in seen:
                seen.add(key)
                adjusted.append([member])
    return adjusted


def summarize_results(all_metrics: List[Dict]) -> Dict:
    if not all_metrics:
        return {}
    grouped: Dict[Tuple[str, str], List[float]] = {}
    for entry in all_metrics:
        key = (entry['attack'], entry['ensemble'])
        grouped.setdefault(key, []).append(entry['score'])
    best_key = max(grouped.items(), key=lambda kv: np.mean(kv[1]))[0]
    best_metrics = [m for m in all_metrics if (m['attack'], m['ensemble']) == best_key]
    summary = {
        'best_attack': best_key[0],
        'best_ensemble': best_key[1],
        'mean_score': float(np.mean(grouped[best_key])),
        'details': best_metrics,
    }
    return summary


def append_results(results_path: Path, payload: Dict) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'a') as f:
        f.write(json.dumps(payload) + '\n')


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)
    logger, log_path = setup_logger(Path(args.log_dir))
    logger.info(f'Logs -> {log_path}')
    eval_models = args.eval_models or args.surrogates
    robust_models = args.robust_models or []
    robust_model_cache: Dict[str, nn.Module] = {}

    ckpt_map = finetune_backbones(args.surrogates, args, device, logger)
    attack_dataset = CIFAR10CleanDataset(
        root=args.clean_dir,
        resize_to=args.input_size,
        targeted=args.targeted,
        target_file=args.target_file,
    )
    attack_loader = DataLoader(attack_dataset, batch_size=args.batchsize, shuffle=False,
                               num_workers=args.workers, pin_memory=True)

    base_ensembles = parse_ensembles(args.attack_ensembles, args.surrogates)
    if args.attack_ensembles is None and robust_models:
        existing = {tuple(members) for members in base_ensembles}
        for name in robust_models:
            key = (name,)
            if key not in existing:
                base_ensembles.append([name])
                existing.add(key)
    all_metrics: List[Dict] = []
    for attack_name in args.attacks:
        attack_ensembles = adjust_ensembles_for_attack(attack_name, base_ensembles, logger)
        if not attack_ensembles:
            logger.warning(f'Skipping {attack_name.upper()} because no valid surrogate combinations are available')
            continue
        for ensemble in attack_ensembles:
            attacker = transferattack.load_attack_class(attack_name)(
                model_name=ensemble if len(ensemble) > 1 else ensemble[0],
                epsilon=args.eps,
                alpha=args.alpha,
                epoch=args.attack_steps,
                decay=args.momentum,
                targeted=args.targeted,
                random_start=args.random_start,
            )
            inject_backbones(
                attacker,
                ensemble,
                ckpt_map,
                args.num_classes,
                device,
                args,
                robust_model_cache,
                robust_models,
            )
            avg_ssim, adv_dir = run_attack(attack_name, ensemble, attacker, attack_loader, args, logger)
            metrics = evaluate_attack('+'.join(ensemble), attack_name, adv_dir, avg_ssim,
                                      eval_models, robust_models, ckpt_map, args, logger, device,
                                      robust_model_cache)
            all_metrics.extend(metrics)

    summary = summarize_results(all_metrics)
    if summary:
        summary['timestamp'] = datetime.now().isoformat()
        summary['log_path'] = str(log_path)
        append_results(Path(args.results_file), summary)
        logger.info('==== Final Summary ====')
        logger.info(json.dumps(summary, indent=2))
    else:
        logger.warning('No metrics recorded. Please check the pipeline configuration.')


if __name__ == '__main__':
    main()
