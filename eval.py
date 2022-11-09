import os
import json
import argparse
import pickle
import numpy as np
import pandas as pd
import clip
import torch
import torch.nn.functional as F
from tqdm import tqdm
from cleanfid.features import InceptionV3W
from PIL import Image
from collections import OrderedDict

import metrics
from modelsearch.retrieval import estimate_score
from modelsearch.feature_stats import CLIPModel, DINOModel


def get_result_entry(mean_average_precision, top_k_accuracies, stat_type, eval_kwargs):
    entry = OrderedDict()
    entry['stat_type'] = stat_type
    entry.update(eval_kwargs)

    if top_k_accuracies is not None:
        entry['Top-1 Accuracy'] = top_k_accuracies[0]
        entry['Top-5 Accuracy'] = top_k_accuracies[4]
        entry['Top-10 Accuracy'] = top_k_accuracies[9]

    entry['mAP@5'] = mean_average_precision[0]
    entry['mAP@10'] = mean_average_precision[1]
    entry['mAP@all'] = mean_average_precision[2]
    return entry


def get_topk_accuracies(model_indices, predictions):
    total_correct = np.zeros((predictions.shape[0], predictions.shape[1] - 1), dtype=float)
    for i in range(1, predictions.shape[1]):
        if i == 1:
            total_correct[:, i - 1] = (model_indices == predictions[:, i - 1]) * 1
        else:
            total_correct[:, i - 1] = total_correct[:, i - 2] + (model_indices == predictions[:, i - 1]) * 1
    topk_accuracies = total_correct.sum(0) / total_correct.shape[0]
    return topk_accuracies


def get_mean_average_precision(model_indices, predictions, label, label_rev, model_names):
    predictions_dict = {}
    retrieval_sol = {}
    for i in range(predictions.shape[0]):
        predictions_dict[i] = predictions[i].tolist()
        model_index = model_indices[i]
        class_names = label_rev[model_names[model_index]]

        gt_retrieve_model_names = []
        for c in class_names:
            gt_retrieve_model_names += label[c]
        retrieval_sol[i] = list(set([model_names.index(x) for x in gt_retrieve_model_names]))

    max_pred_list = [5, 10, float('inf')]
    map_list = [metrics.MeanAveragePrecision(predictions_dict, retrieval_sol, max_predictions=max_pred) for max_pred in max_pred_list]
    return map_list


def get_eval_data(model_names, stat_type, feat_extractor, query_json, device='cuda'):
    query_type = query_json["query_type"]
    cache_path = query_json['cache_path']
    query_dict = query_json['data']

    assert query_type != 'text' or 'clip' in stat_type, "text query must use CLIP features."

    full_query_data = {}
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            full_query_data = pickle.load(f)

    if stat_type in full_query_data:
        d = full_query_data[stat_type]
        feats = d['feats']
        model_indices = d['model_indices']
        feats = torch.from_numpy(feats).to(device).float()
        print(f'found {stat_type} query features cache with {len(model_indices)} {query_type} queries!!')
    else:
        (f'extracting {stat_type} features from {query_type} queries...')
        feats = []
        model_indices = []
        for i, model in enumerate(model_names):
            # text query
            if query_type == 'text':
                query = query_dict[model]
                token = clip.tokenize([query]).to(device)
                with torch.no_grad():
                    feat = feat_extractor.model.encode_text(token)
                feats.append(feat.detach())
                model_indices.append(i)

            # image or sketch query
            elif query_type == 'image' or query_type == 'sketch':
                folder_name = query_dict[model]

                for fname in os.listdir(folder_name):
                    query = Image.open(os.path.join(folder_name, fname))
                    query = query.convert('RGB')

                    # inception needs resizing
                    if stat_type == 'inception':
                        query = query.resize((299, 299), resample=Image.LANCZOS)

                    query = np.asarray(query).copy()
                    # convert image query into inception features
                    query = torch.from_numpy(query).float().permute(2, 0, 1).unsqueeze(0).to(device)
                    feat = feat_extractor(query).detach()
                    feats.append(feat)
                    model_indices.append(i)

        print(f"Total number of {query_type} queries: {len(model_indices)}")

        feats = torch.cat(feats, 0)
        if stat_type == 'clip_normalized':
            feats = F.normalize(feats, p=2, dim=1)
        feats = feats.float()

        full_query_data[stat_type] = {'feats': feats.cpu().numpy(), 'model_indices': model_indices}
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(full_query_data, f)

    return feats, model_indices, query_type


def evaluate(feat_extractor, model_names, model_feats, stat_type, device='cuda', **eval_kwargs):
    # load the labels
    label_data = open('./query_data/label.csv', 'r')
    label_data = label_data.readlines()
    label = {}
    label_rev = {}
    for each in label_data:
        model_name = each.strip().split(',')[0]
        classes = each.strip().split(',')[1:]
        classes = [x.strip('"').strip('\'').strip() for x in classes]
        label_rev[model_name] = classes
        for cl in classes:
            if cl not in label:
                label[cl] = [model_name]
            else:
                label[cl].append(model_name)

    # get features and label of each query in the dataset
    query_path = eval_kwargs['query_path']
    with open(query_path, 'r') as f:
        query_json = json.load(f)
    query_feats, model_indices, query_type = get_eval_data(model_names, stat_type, feat_extractor, query_json, device=device)

    # get score of all the queries for each model
    print('getting scores...')
    scores = [estimate_score(m_feat, query_feats, **eval_kwargs) for m_feat in model_feats]
    scores = np.stack(scores, axis=1)

    # evaluate metrics
    sorted_indices = np.argsort(scores, axis=1)[:, ::-1]
    scores = np.sort(scores, axis=1)[:, ::-1]
    topk_accuracies = get_topk_accuracies(model_indices, sorted_indices) if query_type != 'text' else None
    mean_ap = get_mean_average_precision(model_indices, sorted_indices, label, label_rev, model_names)
    entry = get_result_entry(mean_ap, topk_accuracies, stat_type, eval_kwargs)

    return entry


def main(tasks, output, device):
    # gather the model features and statistics
    print("Gathering model features and statistics from ./model_features/")
    all_model_feats = {k: [] for k in ['clip_normalized', 'inception', 'dino']}
    model_names = sorted([n.replace('.pkl', '') for n in os.listdir('./model_features/statistics')])
    print(f"Found {len(model_names)} models in total!")
    for m_name in tqdm(model_names):
        stat_path = f'./model_features/statistics/{m_name}.pkl'
        feat_path = f'./model_features/raw_features/{m_name}.npz'
        assert os.path.exists(stat_path), stat_path + " does not exist!"
        assert os.path.exists(feat_path), feat_path + " does not exist!"

        with open(stat_path, 'rb') as f:
            feat_stats = pickle.load(f)

        for stat_type in feat_stats.keys():
            if stat_type not in all_model_feats:
                continue

            s = feat_stats[stat_type]
            mu, sigma = s['mu'], s['sigma']
            item = {'mu': mu, 'sigma': sigma}

            if 'clip' in stat_type:
                raw_feats = np.load(feat_path)
                item['samples'] = raw_feats[stat_type]

            all_model_feats[stat_type].append(item)

    # feature extractors
    print("Setting up feature extractors...")
    feature_extractors = {}
    feature_extractors['clip_normalized'] = CLIPModel('cpu').eval()
    feature_extractors['dino'] = DINOModel('cpu').eval()
    feature_extractors['inception'] = InceptionV3W("/tmp", download=True, resize_inside=False).to('cpu').eval()

    # evaluate all the tasks
    print("Evaluation starts...")
    results = []
    for task in tasks:
        stat_type = task['stat_type']
        model_feats = all_model_feats[stat_type]
        feat_extractor = feature_extractors[stat_type]

        # overwrite temperature if it's None
        if 'inv_temperature' in task and task['inv_temperature'] is None:
            assert 'clip' in stat_type, "temperature only applies to CLIP models" 
            task['inv_temperature'] = feat_extractor.model.logit_scale.detach().exp().item()

        feat_extractor.to(device)
        entry = evaluate(feat_extractor, model_names, model_feats, device=device, **task)
        feat_extractor.to('cpu')

        print(entry)
        results.append(entry)

    table = pd.DataFrame(results)
    table.to_csv(output, na_rep='--')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', default='result.csv')
    parser.add_argument('--device', '-d', default='cuda')
    args = parser.parse_args()

    # list out all the tasks
    tasks = [
        {"query_path": "query_data/query_text.json",
         "stat_type": "clip_normalized",
         "method": "monte_carlo",
         "inv_temperature": None,
         "mc_sample_size": 50000},

        {"query_path": "query_data/query_text.json",
         "stat_type": "clip_normalized",
         "method": "monte_carlo",
         "inv_temperature": None,
         "mc_sample_size": 1000},

        {"query_path": "query_data/query_text.json",
         "stat_type": "clip_normalized",
         "method": "first_moment"},

        {"query_path": "query_data/query_text.json",
         "stat_type": "clip_normalized",
         "method": "first_and_second_moment"},

        {"query_path": "query_data/query_image.json",
         "stat_type": "clip_normalized",
         "method": "monte_carlo",
         "inv_temperature": None,
         "mc_sample_size": 50000},
    
        {"query_path": "query_data/query_image.json",
         "stat_type": "clip_normalized",
         "method": "first_moment"},

        {"query_path": "query_data/query_image.json",
         "stat_type": "clip_normalized",
         "method": "gaussian_density"},

        {"query_path": "query_data/query_image.json",
         "stat_type": "dino",
         "method": "gaussian_density"},

        {"query_path": "query_data/query_image.json",
         "stat_type": "inception",
         "method": "gaussian_density"},

        {"query_path": "query_data/query_sketch.json",
         "stat_type": "clip_normalized",
         "method": "monte_carlo",
         "inv_temperature": None,
         "mc_sample_size": 50000},
    
        {"query_path": "query_data/query_sketch.json",
         "stat_type": "clip_normalized",
         "method": "first_moment"},

        {"query_path": "query_data/query_sketch.json",
         "stat_type": "clip_normalized",
         "method": "gaussian_density"},

        {"query_path": "query_data/query_sketch.json",
         "stat_type": "dino",
         "method": "gaussian_density"},

        {"query_path": "query_data/query_sketch.json",
         "stat_type": "inception",
         "method": "gaussian_density"},
    ]

    main(tasks, args.output, args.device)
