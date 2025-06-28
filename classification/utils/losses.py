import torch
import torch.nn as nn


class Entropy(nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()

    def __call__(self, logits):
        return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


class I2TLoss(nn.Module):
    def __init__(self):
        super(I2TLoss, self).__init__()

    def __call__(self, logits, img_feats, text_norm_feats):
        labels = torch.argmax(logits.softmax(1), dim=1)
        loss = 0.0
        for l in torch.unique(labels, sorted = True).tolist():
            img_idx_embeddings = img_feats[labels == l]
            mean_feats = img_idx_embeddings.mean(0).type(text_norm_feats.dtype)
            dist = torch.matmul(mean_feats.unsqueeze(0), text_norm_feats[l].unsqueeze(0).t()).mean()
            loss += dist
        return loss / len(torch.unique(labels))
    
class InterMeanLoss(nn.Module):
    def __init__(self):
        super(InterMeanLoss, self).__init__()
        
    def __call__(self, logits, img_feats):
        labels = torch.argmax(logits.softmax(1), dim=1)
        mean_feats = []
        for l in torch.unique(labels, sorted = True).tolist():
            img_idx_embeddings = img_feats[labels == l]
            mean = img_idx_embeddings.mean(0)
            mean_feats.append(mean / mean.norm())

        cosine_sim_matrix = torch.matmul(torch.stack(mean_feats), torch.stack(mean_feats).t())
        loss = 1 - cosine_sim_matrix
        loss.fill_diagonal_(0)
        return loss.sum()


class SymmetricCrossEntropy(nn.Module):
    def __init__(self, alpha=0.5):
        super(SymmetricCrossEntropy, self).__init__()
        self.alpha = alpha

    def __call__(self, x, x_ema):
        return -(1-self.alpha) * (x_ema.softmax(1) * x.log_softmax(1)).sum(1) - self.alpha * (x.softmax(1) * x_ema.log_softmax(1)).sum(1)


class AugCrossEntropy(nn.Module):
    def __init__(self, alpha=0.5):
        super(AugCrossEntropy, self).__init__()
        self.alpha = alpha

    def __call__(self, x, x_aug, x_ema):
        return -(1-self.alpha) * (x.softmax(1) * x_ema.log_softmax(1)).sum(1) \
                  - self.alpha * (x_aug.softmax(1) * x_ema.log_softmax(1)).sum(1)


class SoftLikelihoodRatio(nn.Module):
    def __init__(self, clip=0.99, eps=1e-5):
        super(SoftLikelihoodRatio, self).__init__()
        self.eps = eps
        self.clip = clip

    def __call__(self, logits):
        probs = logits.softmax(1)
        probs = torch.clamp(probs, min=0.0, max=self.clip)
        return - (probs * torch.log((probs / (torch.ones_like(probs) - probs)) + self.eps)).sum(1)


class GeneralizedCrossEntropy(nn.Module):
    """ Paper: https://arxiv.org/abs/1805.07836 """
    def __init__(self, q=0.8):
        super(GeneralizedCrossEntropy, self).__init__()
        self.q = q

    def __call__(self, logits, targets=None):
        probs = logits.softmax(1)
        if targets is None:
            targets = probs.argmax(dim=1)
        probs_with_correct_idx = probs.index_select(-1, targets).diag()
        return (1.0 - probs_with_correct_idx ** self.q) / self.q
