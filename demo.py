import os
import clip
import pickle
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image
from cleanfid.features import InceptionV3W

from modelsearch.retrieval import estimate_score
from modelsearch.feature_stats import CLIPModel, DINOModel


def parse_args():
    parser = argparse.ArgumentParser("Demo code to retrieve images."
                                     "Text example: python demo.py --query_type text --input 'African animals'"
                                     "Image example: python demo.py --query_type image --input examples/elephant.png")
    parser.add_argument("--query_type", required=True, choices=['text', 'image', 'sketch'])
    parser.add_argument("--input", required=True, type=str, help='path to given image')
    parser.add_argument("--feat_type", default='clip_normalized', choices=['clip_normalized', 'dino', 'inception'])
    parser.add_argument("--method", default='first_moment', choices=['monte_carlo', 'gaussian_density', 'first_moment', 'first_and_second_moment'])
    parser.add_argument("--topk", type=int, default=5, help='top k models to return')
    parser.add_argument("--device", default='cuda', help="which device to use")
    return parser.parse_args()


def main(args):
    device = args.device
    query = args.input
    query_type = args.query_type
    feat_type = args.feat_type
    method = args.method
    topk = args.topk

    # gather the model features and statistics
    model_feats = []
    model_names = sorted([n.replace('.pkl', '') for n in os.listdir('./model_features/statistics')])
    print(f"Found {len(model_names)} models in total!")
    for m_name in tqdm(model_names):
        stat_path = f'./model_features/statistics/{m_name}.pkl'
        feat_path = f'./model_features/raw_features/{m_name}.npz'
        assert os.path.exists(stat_path), stat_path + " does not exist!"
        assert os.path.exists(feat_path), feat_path + " does not exist!"

        if method == 'monte_carlo':
            assert 'clip' in feat_type, "monte carlo method only supports CLIP features"
            raw_feats = np.load(feat_path)
            item = {'samples': raw_feats[feat_type]}
        else:
            with open(stat_path, 'rb') as f:
                feat_stats = pickle.load(f)
            s = feat_stats[feat_type]
            mu, sigma = s['mu'], s['sigma']
            item = {'mu': mu, 'sigma': sigma}
        model_feats.append(item)

    # feature extractors
    if feat_type == 'clip_normalized':
        feat_extractor = CLIPModel(device).eval()
    elif feat_type == 'dino':
        feat_extractor = DINOModel(device).eval()
    elif feat_type == 'inception':
        feat_extractor = InceptionV3W("/tmp", download=True, resize_inside=False).to(device).eval()

    # text query
    if query_type == 'text':
        token = clip.tokenize([query]).to(device)
        with torch.no_grad():
            query_feat = feat_extractor.model.encode_text(token)

    # image or sketch query
    elif query_type == 'image' or query_type == 'sketch':
        query = Image.open(query)
        query = query.convert('RGB')

        # inception needs resizing
        if feat_type == 'inception':
            query = query.resize((299, 299), resample=Image.LANCZOS)

        query = np.asarray(query).copy()
        # convert image query into features
        query = torch.from_numpy(query).float().permute(2, 0, 1).unsqueeze(0).to(device)
        query_feat = feat_extractor(query).detach()

    if feat_type == 'clip_normalized':
        query_feat = F.normalize(query_feat, p=2, dim=1)

    scores = [estimate_score(m_feat, query_feat, method) for m_feat in model_feats]
    scores = np.concatenate(scores)
    sorted_indices = np.argsort(scores)[::-1]
    scores = np.sort(scores)[::-1]

    models = [model_names[i] for i in sorted_indices[:topk]]
    scores = scores[:topk]

    return models, scores

if __name__ == "__main__":
    args = parse_args()
    models, scores = main(args)

    print("Top ranked models and the matching scores:")
    for m, s in zip(models, scores):
        print(f"{m}: {s}")
