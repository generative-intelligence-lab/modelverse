import os
import pickle
import clip
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from cleanfid import fid
import torchvision


class FeatureStats(nn.Module):
    """
    Base class to gather mean and covariance of model features from a generative model.
    """
    def __init__(self, feat_type, preprocess, feat_model, feat_folder, stat_folder, batch_size=10, num_samples=50000, device='cuda'):
        nn.Module.__init__(self)
        self.device = device
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.feat_type = feat_type
        self.preprocess = preprocess
        self.feat_model = feat_model.to(device)
        os.makedirs(feat_folder, exist_ok=True)
        os.makedirs(stat_folder, exist_ok=True)
        self.feat_folder = feat_folder
        self.stat_folder = stat_folder

    @torch.no_grad()
    def get_features_and_statistics(self, image_folder, name, normalize=False, overwrite=False):
        """
        Gather mean and covariance of the features.
        The statistics will be cached in `self.stat_folder`.
        The raw features will be cached in `self.feat_folder`.
        Input:
        - folder: folder containing the images.
        - name: filename to save to
        - normalize: normalize the features or not
        - overwrite: overwrite the cache if this is True.
        """
        assert len(os.listdir(image_folder)) >= self.num_samples, "Not enough images in the folder."

        feat_path = f'{self.feat_folder}/{name}.npz'
        stat_path = f'{self.stat_folder}/{name}.pkl'

        all_feats, all_stats = {}, {}
        # load features and stats if exists
        if os.path.exists(feat_path):
            all_feats = np.load(feat_path)
            all_feats = dict(all_feats)

        if os.path.exists(stat_path):
            with open(stat_path, 'rb') as f:
                all_stats = pickle.load(f)

        # skip if everything is done already
        if not overwrite and self.feat_type in all_stats and self.feat_type in all_feats:
            return

        # calculate stats
        feats = fid.get_folder_features(image_folder,
                                        self.feat_model,
                                        mode=self.preprocess,
                                        batch_size=self.batch_size,
                                        device=self.device)
        feats = feats[:self.num_samples]

        if normalize:
            feats = feats / np.linalg.norm(feats, ord=2, axis=-1, keepdims=True)

        mu = np.mean(feats, axis=0)
        sigma = np.cov(feats, rowvar=False)

        # save updated features and stats to file
        all_feats[self.feat_type] = feats
        np.savez(feat_path, **all_feats)

        stats = {'mu': mu, 'sigma': sigma}
        all_stats[self.feat_type] = stats
        with open(stat_path, 'wb') as f:
            pickle.dump(all_stats, f)

    def to(self, device):
        """
        Change which device to use for the model.
        Input:
        - device: which device to switch to.
        """
        self.device = device
        self.feat_model.to(device)


class CLIPModel(torch.nn.Module):
    """
    Wrapper class to run CLIP on the clean-fid framework.
    """
    def __init__(self, device='cuda'):
        super().__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)

    def forward(self, images):
        device = images.device
        # images = ((images + 1) * 127.5).clip(0, 255).permute(0, 2, 3, 1).to(torch.uint8).detach().cpu().numpy()
        images = images.permute(0, 2, 3, 1).to(torch.uint8).detach().cpu().numpy() # assume image range [0 - 255]
        images = [self.preprocess(Image.fromarray(im)) for im in images]
        images = torch.stack(images, dim=0).to(device)
        return self.model.encode_image(images)

    def get_text_feats(self, text, device='cuda'):
        text = clip.tokenize([text]).to(device)
        with torch.no_grad():
            return self.model.encode_text(text)


class DINOModel(torch.nn.Module):
    """
    Wrapper class to run DINO on the clean-fid framework.
    """
    def __init__(self, device='cuda'):
        super().__init__()
        self.preprocess = torchvision.transforms.Compose(  
                [torchvision.transforms.Resize(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073], std=[0.229, 0.224, 0.225])]
             )
        self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16').to(device)

    def forward(self, images):
        device = images.device
        # images = ((images + 1) * 127.5).clip(0, 255).permute(0, 2, 3, 1).to(torch.uint8).detach().cpu().numpy()
        images = images.permute(0, 2, 3, 1).to(torch.uint8).detach().cpu().numpy() # assume image range [0 - 255]
        images = [self.preprocess(Image.fromarray(im)) for im in images]
        images = torch.stack(images, dim=0).to(device)
        return self.model(images)
