"""Detached measurements for foreground collapse; never alter model outputs."""
import torch


def tensor_stats(value):
    x = value.detach().float()
    return dict(shape=list(x.shape), finite=bool(torch.isfinite(x).all()),
                minimum=float(x.min()), maximum=float(x.max()),
                mean=float(x.mean()), std=float(x.std(unbiased=False)),
                rms=float(x.square().mean().sqrt()))


def binary_report(logits, mask):
    """Measure the actual two-class argmax, with undefined ratios as None."""
    logits, target = logits.detach().float(), mask.detach()[:, 0] > .5
    probability = logits.softmax(1)[:, 1]
    prediction = logits.argmax(1) == 1
    tp, fp = int((prediction & target).sum()), int((prediction & ~target).sum())
    fn, tn = int((~prediction & target).sum()), int((~prediction & ~target).sum())
    ratio = lambda a, b: a / b if b else None
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, iou=ratio(tp, tp + fp + fn),
                precision=ratio(tp, tp + fp), recall=ratio(tp, tp + fn),
                false_positive_rate=ratio(fp, fp + tn),
                predicted_fraction=float(prediction.float().mean()),
                target_fraction=float(target.float().mean()),
                foreground_probability=(float(probability[target].mean()) if target.any() else None),
                background_probability=(float(probability[~target].mean()) if (~target).any() else None),
                probability_min=float(probability.min()), probability_max=float(probability.max()),
                saturated_fraction=float(((probability < 1e-6) | (probability > 1 - 1e-6)).float().mean()),
                logit_margin=tensor_stats(logits[:, 1] - logits[:, 0]))


def pooled_reports(reports):
    counts = {key: sum(r[key] for r in reports) for key in ('tp', 'fp', 'fn', 'tn')}
    tp, fp, fn, tn = (counts[k] for k in ('tp', 'fp', 'fn', 'tn'))
    ratio = lambda a, b: a / b if b else None
    return dict(**counts, iou=ratio(tp, tp + fp + fn), precision=ratio(tp, tp + fp),
                recall=ratio(tp, tp + fn), false_positive_rate=ratio(fp, fp + tn))


def gradient_norms(model):
    squares = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            if parameter.grad is not None:
                raise RuntimeError('Frozen parameter has a gradient: ' + name)
        elif parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                raise FloatingPointError('Nonfinite gradient: ' + name)
            group = '.'.join(name.split('.')[:2])
            squares[group] = squares.get(group, 0.) + float(parameter.grad.float().square().sum())
    return {k: v ** .5 for k, v in squares.items()}
