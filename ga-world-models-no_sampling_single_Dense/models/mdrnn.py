"""
Define MDRNN model, supposed to be used as a world model
on the latent space.
"""
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.normal import Normal

def gmm_loss(batch, mus, sigmas, logpi, true_reward, prob_reward, reduce=True): # pylint: disable=too-many-arguments
    """ Computes the gmm loss.

    Compute minus the log probability of batch under the GMM model described
    by mus, sigmas, pi. Precisely, with bs1, bs2, ... the sizes of the batch
    dimensions (several batch dimension are useful when you have both a batch
    axis and a time step axis), gs the number of mixtures and fs the number of
    features.

    :args batch: (bs1, bs2, *, fs) torch tensor
    :args mus: (bs1, bs2, *, gs, fs) torch tensor
    :args sigmas: (bs1, bs2, *, gs, fs) torch tensor
    :args logpi: (bs1, bs2, *, gs) torch tensor
    :args reduce: if not reduce, the mean in the following formula is ommited

    :returns:
    loss(batch) = - mean_{i1=0..bs1, i2=0..bs2, ...} log(
        sum_{k=1..gs} pi[i1, i2, ..., k] * N(
            batch[i1, i2, ..., :] | mus[i1, i2, ..., k, :], sigmas[i1, i2, ..., k, :]))

    NOTE: The loss is not reduced along the feature dimension (i.e. it should scale ~linearily
    with fs).
    """
    # 報酬損失：2値交差エントロピー
    reward_loss = - (true_reward * torch.log(prob_reward + 1e-5) + (1 - true_reward) * torch.log(1 - prob_reward + 1e-5))

    batch = batch.unsqueeze(-2)
    normal_dist = Normal(mus, sigmas)
    g_log_probs = normal_dist.log_prob(batch)
    g_log_probs = logpi + torch.sum(g_log_probs, dim=-1)
    max_log_probs = torch.max(g_log_probs, dim=-1, keepdim=True)[0]
    g_log_probs = g_log_probs - max_log_probs

    g_probs = torch.exp(g_log_probs)
    probs = torch.sum(g_probs, dim=-1)

    log_prob = max_log_probs.squeeze() + torch.log(probs)
    if reduce:
        return - torch.mean(log_prob) + torch.mean(reward_loss)
    return - log_prob + reward_loss

# def gmm_loss(batch, mus, sigmas, logpi, reduce=True): # pylint: disable=too-many-arguments
#     """ Computes the gmm loss.

#     Compute minus the log probability of batch under the GMM model described
#     by mus, sigmas, pi. Precisely, with bs1, bs2, ... the sizes of the batch
#     dimensions (several batch dimension are useful when you have both a batch
#     axis and a time step axis), gs the number of mixtures and fs the number of
#     features.

#     :args batch: (bs1, bs2, *, fs) torch tensor
#     :args mus: (bs1, bs2, *, gs, fs) torch tensor
#     :args sigmas: (bs1, bs2, *, gs, fs) torch tensor
#     :args logpi: (bs1, bs2, *, gs) torch tensor
#     :args reduce: if not reduce, the mean in the following formula is ommited

#     :returns:
#     loss(batch) = - mean_{i1=0..bs1, i2=0..bs2, ...} log(
#         sum_{k=1..gs} pi[i1, i2, ..., k] * N(
#             batch[i1, i2, ..., :] | mus[i1, i2, ..., k, :], sigmas[i1, i2, ..., k, :]))

#     NOTE: The loss is not reduced along the feature dimension (i.e. it should scale ~linearily
#     with fs).
#     """
#     batch = batch.unsqueeze(-2)
#     normal_dist = Normal(mus, sigmas)
#     g_log_probs = normal_dist.log_prob(batch)
#     g_log_probs = logpi + torch.sum(g_log_probs, dim=-1)
#     max_log_probs = torch.max(g_log_probs, dim=-1, keepdim=True)[0]
#     g_log_probs = g_log_probs - max_log_probs

#     g_probs = torch.exp(g_log_probs)
#     probs = torch.sum(g_probs, dim=-1)

#     log_prob = max_log_probs.squeeze() + torch.log(probs)
#     if reduce:
#         return - torch.mean(log_prob)
#     return - log_prob

def reward_loss(batch, prob_reward, reduce=True):
    # 2値交差エントロピー
    reward_loss = - (batch * torch.log(prob_reward + 1e-5) + (1 - batch) * torch.log(1 - prob_reward + 1e-5))
    if reduce:
        return torch.mean(reward_loss)
    return reward_loss

def terminal_loss(batch, prob_terminal, reduce=True):
    # 2値交差エントロピー
    terminal_loss = - (batch * torch.log(prob_terminal + 1e-5) + (1 - batch) * torch.log(1 - prob_terminal + 1e-5))
    if reduce:
        return torch.mean(terminal_loss)
    return terminal_loss

def loss_fn(batch, mus, sigmas, logpi, rs, ds, reduce=True, one_step=True):
    """ Compute the VAE loss function.

    Computes the VAE loss function, which is composed as the reconstruction
    loss (the negative log likelihood of the data under the specified Gaussian
    mixture model), augmented by a KL divergence term.

    :args mus: (SEQ_LEN, BSIZE, N_GAUSS, LSIZE) torch tensor
    :args sigmas: (SEQ_LEN, BSIZE, N_GAUSS, LSIZE) torch tensor
    :args logpi: (SEQ_LEN, BSIZE, N_GAUSS) torch tensor
    :args rs: (SEQ_LEN, BSIZE) torch tensor
    :args ds: (SEQ_LEN, BSIZE) torch tensor
    :args true_reward: (SEQ_LEN, BSIZE) torch tensor
    :args prob_reward: (SEQ_LEN, BSIZE) torch tensor
    :args prob_terminal: (SEQ_LEN, BSIZE) torch tensor
    :args reduce: if not reduce, the mean in the following formula is ommited

    :returns: the loss
    """
    if one_step:
        states = batch[:, :-2]
        rewards = batch[:, -2]
        terminals = batch[:, -1]
    else:
        states = batch[:, :-2]
        rewards = batch[:, :, -2]
        terminals = batch[:, :, -1]
    
    gmm_losses = gmm_loss(states, mus, sigmas, logpi, reduce=reduce)
    reward_losses = reward_loss(rewards, rs, reduce=reduce)
    terminal_losses = terminal_loss(terminals, ds, reduce=reduce)

    return gmm_losses + reward_losses + terminal_losses

class _MDRNNBase(nn.Module):
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__()
        self.latents = latents
        self.actions = actions
        self.hiddens = hiddens
        self.gaussians = gaussians

        self.gmm_linear = nn.Linear(
            hiddens, (2 * latents + 1) * gaussians + 2)

    def forward(self, *inputs):
        pass

class MDRNN(_MDRNNBase):
    """ MDRNN model for multi steps forward """
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__(latents, actions, hiddens, gaussians)
        self.rnn = nn.LSTM(latents + actions, hiddens)

    def forward(self, actions, latents): # pylint: disable=arguments-differ
        """ MULTI STEPS forward.

        :args actions: (SEQ_LEN, BSIZE, ASIZE) torch tensor
        :args latents: (SEQ_LEN, BSIZE, LSIZE) torch tensor

        :returns: mu_nlat, sig_nlat, pi_nlat, rs, ds, parameters of the GMM
        prediction for the next latent, gaussian prediction of the reward and
        logit prediction of terminality.
            - mu_nlat: (SEQ_LEN, BSIZE, N_GAUSS, LSIZE) torch tensor
            - sigma_nlat: (SEQ_LEN, BSIZE, N_GAUSS, LSIZE) torch tensor
            - logpi_nlat: (SEQ_LEN, BSIZE, N_GAUSS) torch tensor
            - rs: (SEQ_LEN, BSIZE) torch tensor
            - ds: (SEQ_LEN, BSIZE) torch tensor
        """
        seq_len, bs = actions.size(0), actions.size(1)

        ins = torch.cat([actions, latents], dim=-1)
        outs, _ = self.rnn(ins)
        gmm_outs = self.gmm_linear(outs)

        stride = self.gaussians * self.latents

        mus = gmm_outs[:, :, :stride]
        mus = mus.view(seq_len, bs, self.gaussians, self.latents)

        sigmas = gmm_outs[:, :, stride:2 * stride]
        sigmas = sigmas.view(seq_len, bs, self.gaussians, self.latents)
        sigmas = torch.exp(sigmas)

        pi = gmm_outs[:, :, 2 * stride: 2 * stride + self.gaussians]
        pi = pi.view(seq_len, bs, self.gaussians)
        logpi = f.log_softmax(pi, dim=-1)

        rs = gmm_outs[:, :, -2]

        ds = gmm_outs[:, :, -1]

        return mus, sigmas, logpi, rs, ds

class MDRNNCell(_MDRNNBase):
    """ MDRNN model for one step forward """
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__(latents, actions, hiddens, gaussians)
        self.rnn = nn.LSTMCell(latents + actions, hiddens)

    def forward(self, action, latent, hidden): # pylint: disable=arguments-differ
        """ ONE STEP forward.

        :args actions: (BSIZE, ASIZE) torch tensor
        :args latents: (BSIZE, LSIZE) torch tensor
        :args hidden: (BSIZE, RSIZE) torch tensor

        :returns: mu_nlat, sig_nlat, pi_nlat, r, d, next_hidden, parameters of
        the GMM prediction for the next latent, gaussian prediction of the
        reward, logit prediction of terminality and next hidden state.
            - mu_nlat: (BSIZE, N_GAUSS, LSIZE) torch tensor
            - sigma_nlat: (BSIZE, N_GAUSS, LSIZE) torch tensor
            - logpi_nlat: (BSIZE, N_GAUSS) torch tensor
            - rs: (BSIZE) torch tensor
            - ds: (BSIZE) torch tensor
        """
        in_al = torch.cat([action, latent], dim=1)

        next_hidden = self.rnn(in_al, hidden)
        out_rnn = next_hidden[0]

        out_full = self.gmm_linear(out_rnn)

        stride = self.gaussians * self.latents

        mus = out_full[:, :stride]
        mus = mus.view(-1, self.gaussians, self.latents)

        sigmas = out_full[:, stride:2 * stride]
        sigmas = sigmas.view(-1, self.gaussians, self.latents)
        sigmas = torch.exp(sigmas)

        pi = out_full[:, 2 * stride:2 * stride + self.gaussians]
        pi = pi.view(-1, self.gaussians)
        logpi = f.log_softmax(pi, dim=-2)

        r = out_full[:, -2]

        d = out_full[:, -1]

        return mus, sigmas, logpi, r, d, next_hidden
