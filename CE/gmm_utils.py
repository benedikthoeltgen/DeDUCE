
### utils for computations with latent GMMs (code from Mukhoti et al. 2021)

import torch
from tqdm import tqdm

DOUBLE_INFO = torch.finfo(torch.double)
JITTERS = [0, DOUBLE_INFO.tiny] + [10 ** exp for exp in range(-308, 0, 1)]


def centered_cov_torch(x):
    n = x.shape[0]
    res = 1 / (n - 1) * x.t().mm(x)
    return res


def get_embeddings(net, loader: torch.utils.data.DataLoader, num_dim: int, dtype, device, storage_device):
    num_samples = len(loader.dataset)
    embeddings = torch.empty((num_samples, num_dim), dtype=dtype, device=storage_device)
    labels = torch.empty(num_samples, dtype=torch.int, device=storage_device)

    with torch.no_grad():
        start = 0
        for data, label in tqdm(loader):
            data = data.to(device)
            label = label.to(device)
            out = net(data)
            out = net.feature

            end = start + len(data)
            embeddings[start:end].copy_(out, non_blocking=True)
            labels[start:end].copy_(label, non_blocking=True)
            start = end

    return embeddings, labels


def gmm_forward(net, gaussians_model, data_B_X):
    features_B_Z = net(data_B_X)
    features_B_Z = net.feature

    log_probs_B_Y = gaussians_model.log_prob(features_B_Z[:, None, :])

    return log_probs_B_Y


def gmm_evaluate(net, gaussians_model, loader, device, num_classes, storage_device):
    num_samples = len(loader.dataset)
    logits_N_C = torch.empty((num_samples, num_classes), dtype=torch.float, device=storage_device)
    labels_N = torch.empty(num_samples, dtype=torch.int, device=storage_device)

    with torch.no_grad():
        start = 0
        for data, label in tqdm(loader):
            data = data.to(device)
            label = label.to(device)

            logit_B_C = gmm_forward(net, gaussians_model, data)

            end = start + len(data)
            logits_N_C[start:end].copy_(logit_B_C, non_blocking=True)
            labels_N[start:end].copy_(label, non_blocking=True)
            start = end

    return logits_N_C, labels_N


def gmm_get_logits(gmm, embeddings):
    log_probs_B_Y = gmm.log_prob(embeddings[:, None, :])

    return log_probs_B_Y


GMM_TYPES = {"gda", "lda", "lda_limited_covariance", "gmm"}


def gmm_fit_ex(*, embeddings, labels, num_classes, gmm_type: str):
    with torch.no_grad():
        classwise_mean_features = torch.stack([torch.mean(embeddings[labels == c], dim=0) for c in range(num_classes)])
        if gmm_type == "lda":
            classwise_cov_features = centered_cov_torch(
                torch.cat(
                    [embeddings[labels == c] - torch.mean(embeddings[labels == c], dim=0) for c in range(num_classes)]
                )
            )
        elif gmm_type == "lda_limited_covariance":
            sub_slice = slice(None, len(embeddings) // num_classes)
            sub_embeddings = embeddings[sub_slice]
            sub_labels = labels[sub_slice]
            classwise_cov_features = centered_cov_torch(
                torch.cat(
                    [
                        sub_embeddings[sub_labels == c] - classwise_mean_features[c]
                        for c in range(num_classes)
                    ]
                )
            )
        elif gmm_type == "gda":
            classwise_cov_features = torch.stack(
                [centered_cov_torch(embeddings[labels == c] - classwise_mean_features[c]) for c in range(num_classes)]
            )
        elif gmm_type == "gmm":
            import sklearn.mixture
            gmm = sklearn.mixture.GaussianMixture(n_components=num_classes, reg_covar=0)
            gmm.fit(embeddings.cpu().numpy())
            classwise_mean_features = torch.as_tensor(gmm.means_, device=embeddings.device)
            classwise_cov_features = torch.as_tensor(gmm.covariances_, device=embeddings.device)
        else:
            raise Exception(f"Unknown GMM type {gmm_type}!")

    with torch.no_grad():
        for jitter_eps in JITTERS:
            try:
                jitter = jitter_eps * torch.eye(
                    classwise_cov_features.shape[1], device=classwise_cov_features.device
                ).unsqueeze(0)
                gmm = torch.distributions.MultivariateNormal(
                    loc=classwise_mean_features, covariance_matrix=(classwise_cov_features + jitter)
                )
            except RuntimeError as e:
                if "cholesky" in str(e):
                    continue
            except ValueError as e:
                if "invalid" in str(e):
                    continue
            break

    return gmm, jitter_eps