import argparse
import torch
from cleanfid.features import InceptionV3W

from modelsearch.feature_stats import FeatureStats
from modelsearch.feature_stats import CLIPModel, DINOModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--model_image_folder', required=True)
    parser.add_argument('--folder', default='my_model_features')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_samples', type=int, default=50000)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    feat_folder = f'{args.folder}/raw_features'
    stat_folder = f'{args.folder}/statistics'

    print("Setting up feature extractors...")
    feature_extractors = {}
    feature_extractors['clip_normalized'] = CLIPModel('cpu').eval()
    feature_extractors['dino'] = DINOModel('cpu').eval()
    feature_extractors['inception'] = InceptionV3W("/tmp", download=True, resize_inside=False).to('cpu').eval()

    for feat_type in ['clip_normalized', 'dino', 'inception']:
        normalize = feat_type == 'clip_normalized'
        preprocess = 'clean' if feat_type == 'inception' else 'legacy_tensorflow'
        feat_model = feature_extractors[feat_type]
        feat_stats = FeatureStats(feat_type, preprocess, feat_model, feat_folder, stat_folder, batch_size=args.batch_size, num_samples=args.num_samples, device=args.device)
        feat_stats.get_features_and_statistics(args.model_image_folder, args.model_name, normalize=normalize, overwrite=args.overwrite)
