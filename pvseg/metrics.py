"""Micro foreground metrics from cumulative TP/FP/FN; empty union -> undefined."""


class BinaryCounts:
    def __init__(self):
        self.tp = self.fp = self.fn = self.images = self.pixels = 0
        self.empty_targets = self.false_positive_images = 0
        self.negative_fp_pixels = self.negative_pixels = self.missed_positive_images = 0

    def update(self, logits, targets):
        pred, truth = logits.detach().float().sigmoid() >= .5, targets >= .5
        self.tp += int((pred & truth).sum().item())
        self.fp += int((pred & ~truth).sum().item())
        self.fn += int((~pred & truth).sum().item())
        self.images += len(targets)
        self.pixels += targets.numel()
        empty = ~truth.flatten(1).any(1)
        self.empty_targets += int(empty.sum().item())
        self.false_positive_images += int((empty & pred.flatten(1).any(1)).sum().item())
        self.negative_fp_pixels += int(pred[empty].sum().item())
        self.negative_pixels += int(targets[empty].numel())
        self.missed_positive_images += int((~empty & ~(pred & truth).flatten(1).any(1)).sum().item())

    def report(self):
        def divide(n, d):
            return n / d if d else None
        return {'tp': self.tp, 'fp': self.fp, 'fn': self.fn,
                'iou': divide(self.tp, self.tp + self.fp + self.fn),
                'f1': divide(2 * self.tp, 2 * self.tp + self.fp + self.fn),
                'precision': divide(self.tp, self.tp + self.fp),
                'recall': divide(self.tp, self.tp + self.fn),
                'images': self.images, 'empty_targets': self.empty_targets,
                'negative_false_positive_images': self.false_positive_images,
                'negative_false_positive_pixels': self.negative_fp_pixels,
                'negative_false_positive_pixel_ratio': divide(self.negative_fp_pixels, self.negative_pixels),
                'negative_false_positive_image_ratio': divide(self.false_positive_images, self.empty_targets),
                'positive_images_without_true_positive_pixels': self.missed_positive_images,
                'foreground_false_negative_ratio': divide(self.fn, self.tp + self.fn),
                'false_positive_pixel_ratio': divide(self.fp, self.pixels)}
