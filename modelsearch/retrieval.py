import numpy as np
import torch
import torch.distributions as tdist
import torch.nn.functional as F


# function to try different kind of score estimate
def estimate_score(model_stat, query_feat, method, **kwargs):
    if method == 'gaussian_density':
        return gaussian_density_estimate(model_stat, query_feat, **kwargs)

    elif method == 'monte_carlo':
        return monte_carlo_estimate(model_stat, query_feat, **kwargs)

    elif method == 'first_moment':
        return first_moment_estimate(model_stat, query_feat, **kwargs)

    elif method == 'first_and_second_moment':
        return first_and_second_moment_estimate(model_stat, query_feat, **kwargs)


@torch.no_grad()
def gaussian_density_estimate(model_stat, query_feat, **kwargs):
    f_dim = query_feat.size(-1)
    device = query_feat.device

    mu = model_stat['mu']
    mu = torch.from_numpy(mu).float().reshape(1, -1).to(device)
    sigma = torch.from_numpy(model_stat['sigma'] + np.eye(model_stat['sigma'].shape[0]) * 0.001)
    sigma = sigma.float().to(device)

    distribution = tdist.MultivariateNormal(mu, sigma)

    distances = distribution.log_prob(query_feat.reshape(-1, f_dim))
    return distances.cpu().numpy()


@torch.no_grad()
def monte_carlo_estimate(model_stat, query_feat, inv_temperature=100, mc_sample_size=50000, **kwargs):
    f_dim = query_feat.size(-1)
    query_feat = F.normalize(query_feat.reshape(-1, f_dim), 2, -1).float()                              # query_feat: N_queries x f_dim
    samples = torch.from_numpy(model_stat['samples'][:mc_sample_size]).to(query_feat.device).float()    # samples: N_samples_per_model x f_dim
    scores = query_feat @ samples.T                                                                     # scores: N_queries x N_samples_per_model

    # apply temperature
    if inv_temperature == float('inf'):  # take the maximum when temperature is infinity
        average_score_log = torch.max(scores, dim=1)                # exp is strictly increasing, so take max first
        average_score_log = average_score_log.double()              # more numerical accuracy for exp()
        average_score = torch.exp(average_score_log).values         # average_score: N_queries
    else:
        inv_temperature = torch.as_tensor(inv_temperature).double()
        scores = scores.double()                                    # more numerical accuracy for exp()
        scores = torch.exp(inv_temperature * scores)
        average_score = torch.mean(scores, dim=1)                   # average_score: N_queries
    return average_score.cpu().numpy()


@torch.no_grad()
def first_moment_estimate(model_stat, query_feat, **kwargs):
    f_dim = query_feat.size(-1)
    mu = model_stat['mu']
    mu = F.normalize(torch.from_numpy(mu).to(query_feat.device).reshape(1, -1), 2, -1).float()          # mu: 1 x f_dim
    query_feat = F.normalize(query_feat.reshape(-1, f_dim), 2, -1).float()                              # query_feat: N_queries x f_dim
    cos = torch.einsum('bi,bi->b', query_feat.reshape(query_feat.size(0), -1).float(), mu.expand(query_feat.size(0), f_dim))
    return torch.exp(cos).cpu().numpy()                                                                  # output shape: N_queries


@torch.no_grad()
def first_and_second_moment_estimate(model_stat, query_feat, inv_temperature=100, **kwargs):
    f_dim = query_feat.size(-1)
    mu, sigma = model_stat['mu'], model_stat['sigma']

    mu = torch.from_numpy(mu).to(query_feat.device).reshape(1, -1).float()
    sigma = torch.from_numpy(sigma).to(query_feat.device).float()
    query_feat = F.normalize(query_feat.reshape(-1, f_dim), 2, -1).float()
    cosine_term = torch.squeeze(query_feat @ mu.T)                             # (N_query, f_dim) x (f_dim, 1) -> (N_query)
    spread_term = (query_feat @ sigma)                                         # (N_query, f_dim) x (f_dim, f_dim) -> (N_query, f_dim)
    spread_term *= query_feat                                                  # (N_query, f_dim) * (N_query, f_dim) -> (N_query, f_dim)
    spread_term = torch.sum(spread_term, dim=1)                                # (N_query)

    if inv_temperature == float('inf'):
        scores = spread_term
    else:
        scores = 0.5 * (inv_temperature ** 2) * spread_term + inv_temperature * cosine_term
    return scores.cpu().numpy()
