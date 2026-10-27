"""Single 896x896 RGB image inference; local exact-hash backbone only, no downloads."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True, help='Mounted model.safetensors with adjacent config.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    from PIL import Image
    import torch
    from pvseg.model import FrozenSATSegDINO, resolve_weights
    from pvseg.runtime import Precision, seed_everything
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if state['format'] not in ['pvseg_decoder_v1', 'pvseg_inference_decoder_v1']:
        raise ValueError('Unsupported checkpoint')
    path, config, source = resolve_weights(args.weights, download=False)
    if source['sha256'] != state['model']['weight_source']['sha256']:
        raise ValueError('Backbone hash differs from checkpoint')
    seed_everything(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = FrozenSATSegDINO(path, config, source, 896).to(device).eval()
    for key in ['backbone_id', 'normalization', 'decoder', 'indices_zero_based', 'intermediate_norm']:
        if model.metadata[key] != state['model'][key]:
            raise ValueError(f'Metadata mismatch: {key}')
    model.decoder.load_state_dict(state['decoder_state'], strict=True)
    model.requires_grad_(False)
    with Image.open(args.image) as image:
        if image.mode != 'RGB' or image.size != (896, 896):
            raise ValueError('Expected original 896x896 RGB image; no automatic resizing')
        rgb = np.array(image)
    normalization = model.metadata['normalization']
    x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255)
    x = (x-torch.tensor(normalization['mean'])[:, None, None])/torch.tensor(normalization['std'])[:, None, None]
    with torch.inference_mode(), Precision(device).context():
        logits = model(x[None].to(device))
    if not torch.isfinite(logits).all().item() or list(logits.shape) != [1, 1, 896, 896]:
        raise RuntimeError('Invalid logits')
    mask = (logits[0, 0].float().sigmoid().cpu().numpy() >= .5).astype('uint8')*255
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(args.output)
    print(f'Inference only, threshold 0.5; saved {args.output}')


if __name__ == '__main__':
    main()
