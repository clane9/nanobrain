import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

GRID_SHAPE = (192, 240, 192)
PATCH_GRID_SHAPE = (24, 30, 24)
CV_SEED = 0
N_BOOTSTRAP = 1000
LOGISTIC_CS = np.logspace(-4, 4, 9)
RIDGE_ALPHAS = np.logspace(-2, 6, 9)
SEGMENTATION_ALPHAS = (1e1, 1e2, 1e3, 1e4, 1e5)
SEGMENTATION_THRESHOLDS = torch.logspace(-3, -0.1, 30)


class Encoder(nn.Module):
    def global_embed(
        self,
        images: Float[Tensor, "B X Y Z"],
        mask: Float[Tensor, "B X Y Z"],
    ) -> Float[Tensor, "B D"]: ...

    def dense_embed(
        self,
        images: Float[Tensor, "B X Y Z"],
        mask: Float[Tensor, "B X Y Z"],
    ) -> Float[Tensor, "B Gx Gy Gz C"]: ...


def probe_binary_classification(
    model: Encoder,
    dataset: Dataset,
    image_key: str,
    label_key: str,
    mask_key: str = "mask",
    n_folds: int = 5,
    batch_size: int = 8,
    num_workers: int = 8,
    device: str = "cuda",
    amp: bool = True,
) -> dict[str, Any]:
    """
    Linear probe for binary classification on the global embedding. Logistic regression with
    the penalty tuned by inner CV. Scores pooled out-of-fold predictions with bootstrap CIs.
    """
    device_type = torch.device(device).type
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    embeddings = []
    labels = []
    embed_seconds = 0.0
    with torch.inference_mode(), torch.autocast(device_type, dtype=torch.bfloat16, enabled=amp):
        for batch in loader:
            images = batch[image_key].to(device).float()
            masks = batch[mask_key].to(device).float()
            assert images.shape[1:] == GRID_SHAPE, f"unexpected image shape {images.shape}"
            start = time.perf_counter()
            embedding = model.global_embed(images, masks).float().cpu()
            embed_seconds += time.perf_counter() - start
            assert embedding.ndim == 2 and len(embedding) == len(images), (
                f"unexpected global embedding shape {embedding.shape}"
            )
            embeddings.append(embedding)
            labels.append(batch[label_key])
    features = torch.cat(embeddings).numpy()
    labels = torch.cat(labels).numpy()
    assert set(np.unique(labels)) == {0, 1}, "expected binary labels"

    n_samples = len(labels)
    probabilities = np.zeros(n_samples)
    fold_Cs = []
    folds = StratifiedKFold(n_folds, shuffle=True, random_state=CV_SEED)
    for train_ids, test_ids in folds.split(features, labels):
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegressionCV(
                Cs=LOGISTIC_CS,
                l1_ratios=(0.0,),
                scoring="neg_log_loss",
                # so that 0.5 is a sensible cutoff for balanced accuracy
                class_weight="balanced",
                max_iter=1000,
                use_legacy_attributes=False,
            ),
        )
        classifier.fit(features[train_ids], labels[train_ids])
        probabilities[test_ids] = classifier.predict_proba(features[test_ids])[:, 1]
        fold_Cs.append(float(classifier[-1].C_))
    predictions = probabilities > 0.5

    rng = np.random.default_rng(0)
    bootstrap_auroc = []
    bootstrap_balanced_accuracy = []
    for _ in range(N_BOOTSTRAP):
        ids = rng.integers(0, n_samples, n_samples)
        # auroc is undefined when a resample has only one class
        if len(np.unique(labels[ids])) < 2:
            continue
        bootstrap_auroc.append(roc_auc_score(labels[ids], probabilities[ids]))
        bootstrap_balanced_accuracy.append(balanced_accuracy_score(labels[ids], predictions[ids]))

    return {
        "auroc": roc_auc_score(labels, probabilities),
        "auroc_ci": np.percentile(bootstrap_auroc, [2.5, 97.5]).tolist(),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "balanced_accuracy_ci": np.percentile(bootstrap_balanced_accuracy, [2.5, 97.5]).tolist(),
        "fold_C": fold_Cs,
        "labels": labels.tolist(),
        "probabilities": probabilities.tolist(),
        "n_samples": n_samples,
        "embed_dim": features.shape[1],
        "n_params": sum(p.numel() for p in model.parameters()),
        "embed_seconds": embed_seconds,
    }


def probe_regression(
    model: Encoder,
    dataset: Dataset,
    image_key: str,
    target_key: str,
    mask_key: str = "mask",
    n_folds: int = 5,
    batch_size: int = 8,
    num_workers: int = 8,
    device: str = "cuda",
    amp: bool = True,
) -> dict[str, Any]:
    """
    Linear probe for regression on the global embedding. Ridge regression with the penalty
    tuned by leave-one-out. Scores pooled out-of-fold predictions with bootstrap CIs.
    """
    device_type = torch.device(device).type
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    embeddings = []
    targets = []
    embed_seconds = 0.0
    with torch.inference_mode(), torch.autocast(device_type, dtype=torch.bfloat16, enabled=amp):
        for batch in loader:
            images = batch[image_key].to(device).float()
            masks = batch[mask_key].to(device).float()
            assert images.shape[1:] == GRID_SHAPE, f"unexpected image shape {images.shape}"
            start = time.perf_counter()
            embedding = model.global_embed(images, masks).float().cpu()
            embed_seconds += time.perf_counter() - start
            assert embedding.ndim == 2 and len(embedding) == len(images), (
                f"unexpected global embedding shape {embedding.shape}"
            )
            embeddings.append(embedding)
            targets.append(batch[target_key])
    features = torch.cat(embeddings).numpy()
    targets = torch.cat(targets).double().numpy()

    n_samples = len(targets)
    predictions = np.zeros(n_samples)
    fold_alphas = []
    folds = KFold(n_folds, shuffle=True, random_state=CV_SEED)
    for train_ids, test_ids in folds.split(features):
        regressor = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
        regressor.fit(features[train_ids], targets[train_ids])
        predictions[test_ids] = regressor.predict(features[test_ids])
        fold_alphas.append(float(regressor[-1].alpha_))

    rng = np.random.default_rng(0)
    bootstrap_mae = []
    bootstrap_r = []
    bootstrap_r2 = []
    for _ in range(N_BOOTSTRAP):
        ids = rng.integers(0, n_samples, n_samples)
        bootstrap_mae.append(mean_absolute_error(targets[ids], predictions[ids]))
        bootstrap_r.append(np.corrcoef(targets[ids], predictions[ids])[0, 1])
        bootstrap_r2.append(r2_score(targets[ids], predictions[ids]))

    return {
        "mae": mean_absolute_error(targets, predictions),
        "mae_ci": np.percentile(bootstrap_mae, [2.5, 97.5]).tolist(),
        "r": np.corrcoef(targets, predictions)[0, 1],
        "r_ci": np.percentile(bootstrap_r, [2.5, 97.5]).tolist(),
        "r2": r2_score(targets, predictions),
        "r2_ci": np.percentile(bootstrap_r2, [2.5, 97.5]).tolist(),
        "fold_alpha": fold_alphas,
        "targets": targets.tolist(),
        "predictions": predictions.tolist(),
        "n_samples": n_samples,
        "embed_dim": features.shape[1],
        "n_params": sum(p.numel() for p in model.parameters()),
        "embed_seconds": embed_seconds,
    }


def probe_binary_segmentation(
    model: Encoder,
    dataset: Dataset,
    image_key: str,
    label_key: str,
    mask_key: str = "mask",
    n_folds: int = 5,
    n_inner_folds: int = 5,
    batch_size: int = 4,
    num_workers: int = 8,
    device: str = "cuda",
    amp: bool = True,
) -> dict[str, Any]:
    """
    Linear probe for binary segmentation on the dense embedding. Each patch embedding predicts
    the 8x8x8 voxel labels inside its patch, so predictions are at full 1mm resolution. The
    penalty and threshold are tuned by inner CV. Scores per-subject dice and average precision
    with bootstrap CIs.
    """
    device_type = torch.device(device).type
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    patch_features = []
    patch_labels = []
    embed_seconds = 0.0
    with torch.inference_mode(), torch.autocast(device_type, dtype=torch.bfloat16, enabled=amp):
        for batch in loader:
            images = batch[image_key].to(device).float()
            masks = batch[mask_key].to(device).float()
            labels = batch[label_key].to(device)
            assert images.shape[1:] == GRID_SHAPE, f"unexpected image shape {images.shape}"
            assert labels.max() <= 1, "expected binary labels"
            start = time.perf_counter()
            embedding = model.dense_embed(images, masks).float()
            if device_type == "cuda":
                torch.cuda.synchronize()
            embed_seconds += time.perf_counter() - start
            assert embedding.ndim == 5 and embedding.shape[:4] == (
                len(images),
                *PATCH_GRID_SHAPE,
            ), f"unexpected dense embedding shape {embedding.shape}"

            embedding = embedding.flatten(1, 3)
            mask_patches = patchify3d(masks)
            label_patches = patchify3d(labels)
            # keep only patches that overlap the mask, assuming no labels outside it
            for ii in range(len(images)):
                in_mask = mask_patches[ii].any(dim=1)
                patch_features.append(embedding[ii, in_mask])
                patch_labels.append(label_patches[ii, in_mask])
    embed_dim = patch_features[0].shape[1]

    n_subjects = len(patch_features)
    n_thresholds = len(SEGMENTATION_THRESHOLDS)
    subject_dice = np.zeros(n_subjects)
    subject_average_precision = np.zeros(n_subjects)
    fold_alphas = []
    fold_thresholds = []
    folds = KFold(n_folds, shuffle=True, random_state=CV_SEED)
    for train_ids, test_ids in folds.split(np.arange(n_subjects)):
        inner_dice = torch.zeros(len(SEGMENTATION_ALPHAS), n_subjects, n_thresholds)
        inner_folds = KFold(n_inner_folds, shuffle=True, random_state=CV_SEED)
        for inner_train, inner_val in inner_folds.split(train_ids):
            inner_train_ids = train_ids[inner_train]
            inner_val_ids = train_ids[inner_val]
            for alpha_id, alpha in enumerate(SEGMENTATION_ALPHAS):
                probabilities = fit_predict_segmentation(
                    patch_features, patch_labels, inner_train_ids, inner_val_ids, alpha
                )
                for ii, subject_probabilities in zip(inner_val_ids, probabilities):
                    inner_dice[alpha_id, ii] = dice_by_threshold(
                        subject_probabilities, patch_labels[ii]
                    )
        # pick the alpha and threshold with the best mean dice over inner validation subjects
        mean_inner_dice = inner_dice[:, train_ids].mean(dim=1)
        alpha_id, threshold_id = np.unravel_index(
            mean_inner_dice.argmax().item(), (len(SEGMENTATION_ALPHAS), n_thresholds)
        )
        alpha = SEGMENTATION_ALPHAS[alpha_id]
        fold_alphas.append(alpha)
        fold_thresholds.append(SEGMENTATION_THRESHOLDS[threshold_id].item())

        probabilities = fit_predict_segmentation(
            patch_features, patch_labels, train_ids, test_ids, alpha
        )
        for ii, subject_probabilities in zip(test_ids, probabilities):
            dice = dice_by_threshold(subject_probabilities, patch_labels[ii])
            subject_dice[ii] = dice[threshold_id].item()
            subject_average_precision[ii] = average_precision(
                subject_probabilities, patch_labels[ii]
            )

    rng = np.random.default_rng(0)
    bootstrap_dice = []
    bootstrap_average_precision = []
    for _ in range(N_BOOTSTRAP):
        ids = rng.integers(0, n_subjects, n_subjects)
        bootstrap_dice.append(subject_dice[ids].mean())
        bootstrap_average_precision.append(np.nanmean(subject_average_precision[ids]))

    return {
        "dice": subject_dice.mean(),
        "dice_ci": np.percentile(bootstrap_dice, [2.5, 97.5]).tolist(),
        "average_precision": np.nanmean(subject_average_precision),
        "average_precision_ci": np.nanpercentile(bootstrap_average_precision, [2.5, 97.5]).tolist(),
        "fold_alpha": fold_alphas,
        "fold_threshold": fold_thresholds,
        "subject_dice": subject_dice.tolist(),
        "subject_average_precision": subject_average_precision.tolist(),
        "n_samples": n_subjects,
        "embed_dim": embed_dim,
        "n_params": sum(p.numel() for p in model.parameters()),
        "embed_seconds": embed_seconds,
    }


def fit_predict_segmentation(
    patch_features: list[Tensor],
    patch_labels: list[Tensor],
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    alpha: float,
) -> list[Tensor]:
    """Fit on the train subjects' patches, return per-patch probabilities for each test subject."""
    train_features = torch.cat([patch_features[ii] for ii in train_ids])
    train_labels = torch.cat([patch_labels[ii] for ii in train_ids]).float()
    mean = train_features.mean(dim=0)
    std = train_features.std(dim=0, correction=0).clamp_min(1e-6)
    coef, intercept = fit_logistic((train_features - mean) / std, train_labels, alpha)

    probabilities = []
    for ii in test_ids:
        logits = (patch_features[ii] - mean) / std @ coef + intercept
        probabilities.append(torch.sigmoid(logits))
    return probabilities


def dice_by_threshold(probabilities: Tensor, labels: Tensor) -> Tensor:
    """Dice at each of SEGMENTATION_THRESHOLDS. Empty prediction and empty label counts as 1."""
    thresholds = SEGMENTATION_THRESHOLDS.to(probabilities.device)
    predicted = probabilities.flatten()[None, :] >= thresholds[:, None]
    overlap = (predicted & labels.flatten().bool()[None, :]).sum(dim=1)
    denominator = predicted.sum(dim=1) + labels.sum()
    dice = torch.where(denominator > 0, 2 * overlap / denominator.clamp_min(1), 1.0)
    return dice.cpu()


def average_precision(probabilities: Tensor, labels: Tensor) -> float:
    """Voxel average precision, NaN when there are no positives. Ties are not merged."""
    order = probabilities.flatten().argsort(descending=True)
    sorted_labels = labels.flatten()[order].float()
    n_positive = sorted_labels.sum().item()
    if n_positive == 0:
        return np.nan
    ranks = torch.arange(1, len(sorted_labels) + 1, device=sorted_labels.device)
    precision = sorted_labels.cumsum(dim=0) / ranks
    return (precision * sorted_labels).sum().item() / n_positive


def fit_logistic(
    features: Tensor,
    targets: Tensor,
    alpha: float,
    max_iter: int = 1000,
) -> tuple[Tensor, Tensor]:
    """
    L2 penalized logistic regression fit with L-BFGS. Each target column is a separate binary
    problem sharing the same features. Targets can be soft (between 0 and 1).
    """
    n, d = features.shape
    assert targets.ndim == 2 and len(targets) == n, (
        f"targets {tuple(targets.shape)} do not match {n} samples of {d} features"
    )
    n_outputs = targets.shape[1]

    coef = torch.zeros(d, n_outputs, device=features.device, dtype=features.dtype)
    intercept = torch.logit(targets.mean(dim=0).clamp(1e-6, 1 - 1e-6))
    coef.requires_grad_(True)
    intercept.requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        [coef, intercept], max_iter=max_iter, history_size=10, line_search_fn="strong_wolfe"
    )

    def closure() -> Tensor:
        optimizer.zero_grad()
        logits = features @ coef + intercept
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        loss = loss + alpha * coef.square().sum() / (n * n_outputs)
        loss.backward()
        return loss

    optimizer.step(closure)

    # l-bfgs stops on its own gradient and step tolerances; exhausting the budget means neither met
    n_iter = optimizer.state[coef]["n_iter"]
    assert n_iter < max_iter, f"l-bfgs used all {max_iter} iterations without converging"
    return coef.detach(), intercept.detach()


def patchify3d(x: Tensor, patch_size: int = 8) -> Tensor:
    p = patch_size
    x = rearrange(x, "b (gx px) (gy py) (gz pz) -> b (gx gy gz) (px py pz)", px=p, py=p, pz=p)
    return x
