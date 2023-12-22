import torch
import torch.nn.functional as F
from torch import nn

from ..nn import soft_target_update, reset_network, copy_network, BeamEnsemble
from ..nn import beam_weights_initializer, freeze_network_params, free_network_params
from ..ssl.beam_ssl import BeamSSL
from ..utils import as_numpy
from ..nn import BeamOptimizer

from torch.nn.utils import spectral_norm


class BeamBarlowTwins(BeamSSL):

    def __init__(self, hparams, networks=None, optimizers=None, schedulers=None, **kwargs):

        if networks is None:
            networks = {}
        h = self.h_dim
        p = self.p_dim

        self.n_ensembles = hparams.n_ensembles

        networks['projection'] = nn.Sequential(nn.Linear(h, h), nn.BatchNorm1d(h),
                                               nn.ReLU(), nn.Linear(h, h), nn.BatchNorm1d(h),
                                               nn.ReLU(), nn.Linear(h, p))

        networks['discriminator'] = nn.Sequential(spectral_norm(nn.Linear(h, h)),
                                   nn.ReLU(), spectral_norm(nn.Linear(h, h)), nn.ReLU(), nn.Linear(h, 1))

        super().__init__(hparams, networks=networks, optimizers=optimizers, schedulers=schedulers, **kwargs)

        ensemble = BeamEnsemble(self.generate_encoder, n_ensembles=self.n_ensembles)
        ensemble.set_optimizers(BeamOptimizer.prototype(dense_args={'lr': self.hparams.lr_dense,
                                                                    'weight_decay': self.hparams.weight_decay,
                                                                    'betas': (self.hparams.momentum, self.hparams.beta2),
                                                                    'eps': self.hparams.eps}))

        self.add_components(networks=ensemble, name='encoder', build_optimizers=False)
        beam_weights_initializer(self.networks['discriminator'], method='orthogonal')

    def train_iteration(self, sample=None, results=None, subset=None, counter=None, training=True, **kwargs):

        x_aug1, x_aug2 = sample['augmentations']

        encoder = self.networks['encoder']

        projection = self.networks['projection']
        opt_p = self.optimizers['projection']

        discriminator = self.networks['discriminator']
        opt_d = self.optimizers['discriminator']

        freeze_network_params(encoder, projection)
        free_network_params(discriminator)

        index = torch.randperm(encoder.n_ensembles)
        h = encoder(x_aug1, index=index[0])
        r = torch.randn_like(h)

        d_h = discriminator(h)
        d_r = discriminator(r)

        loss_d = F.softplus(-d_h) + F.softplus(d_r)
        # loss_d = -d_h + d_r
        loss_d = self.apply(loss_d, training=training, optimizers=[opt_d], name='discriminator')
        results['scalar']['loss_d'].append(float(loss_d))
        results['scalar']['stats_mu'].append(float(h.mean()))
        results['scalar']['stats_std'].append(float(h.std()))

        if not counter % self.hparams.n_discriminator_steps:
            free_network_params(encoder, projection)
            freeze_network_params(discriminator)

            index = torch.randperm(encoder.n_ensembles)

            ind1 = index[0]
            ind2 = index[min(len(index)-1, 1)]
            opt_e1 = encoder.optimizers[ind1]
            opt_e2 = encoder.optimizers[ind2]

            h1 = encoder(x_aug1, index=index[0])
            h2 = encoder(x_aug2, index=index[min(len(index)-1, 1)])

            d1 = discriminator(h1)
            d2 = discriminator(h2)

            z1 = projection(h1)
            z2 = projection(h2)

            z1 = (z1 - z1.mean(dim=0, keepdim=True)) / (z1.std(dim=0, keepdim=True) + 1e-6)
            z2 = (z2 - z2.mean(dim=0, keepdim=True)) / (z2.std(dim=0, keepdim=True) + 1e-6)

            b, d = z1.shape
            corr = (z1.T @ z2) / b

            I = torch.eye(d, device=corr.device)
            corr_diff = (corr - I) ** 2

            invariance = torch.diag(corr_diff)
            redundancy = (corr_diff * (1 - I)).sum(dim=-1)
            discrimination = F.softplus(d1) + F.softplus(d2)

            opts = [opt_e1, opt_p] if ind1 == ind2 else [opt_e1, opt_e2, opt_p]
            loss = self.apply(invariance, self.hparams.lambda_twins * redundancy,
                              self.hparams.lambda_disc * discrimination, training=training,
                              optimizers=opts, name='encoder')

            # add scalar measurements
            results['scalar']['loss'].append(float(loss))
            results['scalar']['invariance'].append(float(invariance.mean()))
            results['scalar']['redundancy'].append(float(redundancy.mean()))
            results['scalar']['discrimination'].append(float(discrimination.mean()))

        return results


class BarlowTwins(BeamSSL):

    def __init__(self, hparams, networks=None, optimizers=None, schedulers=None, **kwargs):

        if networks is None:
            networks = {}
        h = self.h_dim
        p = self.p_dim

        networks['projection'] = nn.Sequential(nn.Linear(h, h), nn.BatchNorm1d(h),
                                               nn.ReLU(), nn.Linear(h, h), nn.BatchNorm1d(h),
                                               nn.ReLU(), nn.Linear(h, p))
        super().__init__(hparams, networks=networks, optimizers=optimizers, schedulers=schedulers, **kwargs)

    def train_iteration(self, sample=None, results=None, subset=None, counter=None, training=True, **kwargs):

        x_aug1, x_aug2 = sample['augmentations']

        encoder = self.networks['encoder']
        opt_e = self.optimizers['encoder']

        projection = self.networks['projection']
        opt_p = self.optimizers['projection']

        z1 = projection(encoder(x_aug1))
        z2 = projection(encoder(x_aug2))

        z1 = (z1 - z1.mean(dim=0, keepdim=True)) / (z1.std(dim=0, keepdim=True) + 1e-6)
        z2 = (z2 - z2.mean(dim=0, keepdim=True)) / (z2.std(dim=0, keepdim=True) + 1e-6)

        b, d = z1.shape
        corr = (z1.T @ z2) / b

        I = torch.eye(d, device=corr.device)
        corr_diff = (corr - I) ** 2

        invariance = torch.diag(corr_diff)
        redundancy = (corr_diff * (1 - I)).sum(dim=-1)

        loss = invariance + self.hparams.lambda_twins * redundancy
        loss = self.apply(loss, training=training, optimizers=[opt_e, opt_p])

        # add scalar measurements
        results['scalar']['loss'].append(float(loss))
        results['scalar']['invariance'].append(float(invariance.mean()))
        results['scalar']['redundancy'].append(float(redundancy.mean()))

        return results


class BeamVICReg(BeamSSL):

    def __init__(self, hparams, networks=None, optimizers=None, schedulers=None, **kwargs):

        if networks is None:
            networks = {}
        h = self.h_dim
        p = self.p_dim

        networks['projection'] = nn.Sequential(nn.Linear(h, h), nn.BatchNorm1d(h),
                                               nn.ReLU(), nn.Linear(h, h), nn.BatchNorm1d(h),
                                               nn.ReLU(), nn.Linear(h, p))
        super().__init__(hparams, networks=networks, optimizers=optimizers, schedulers=schedulers, **kwargs)

    def train_iteration(self, sample=None, results=None, subset=None, counter=None, training=True, **kwargs):

        x_aug1, x_aug2 = sample['augmentations']

        encoder = self.networks['encoder']
        opt_e = self.optimizers['encoder']

        projection = self.networks['projection']
        opt_p = self.optimizers['projection']

        h1 = encoder(x_aug1)
        h2 = encoder(x_aug2)

        z1 = projection(h1)
        z2 = projection(h2)

        sim_loss = F.mse_loss(z1, z2, reduction='none').mean(dim=0)

        # mu1_h = h1.mean(dim=0, keepdim=True)
        # mu2_h = h2.mean(dim=0, keepdim=True)

        mu1 = z1.mean(dim=0, keepdim=True)
        mu2 = z2.mean(dim=0, keepdim=True)

        mean_loss = mu1.pow(2) + mu2.pow(2)

        std1 = torch.sqrt(z1.var(dim=0) + self.hparams.var_eps)
        std2 = torch.sqrt(z2.var(dim=0) + self.hparams.var_eps)

        std_loss = F.relu(1 - std1) + F.relu(1 - std2)

        z1 = (z1 - mu1)
        z2 = (z2 - mu2)

        b, d = z1.shape

        corr1 = (z1.T @ z1) / (b - 1)
        corr2 = (z2.T @ z2) / (b - 1)

        I = torch.eye(d, device=corr1.device)
        cov_loss = (corr1 * (1 - I)).pow(2).sum(dim=0) + (corr2 * (1 - I)).pow(2).sum(dim=0)

        self.apply({'sim_loss': sim_loss, 'std_loss': std_loss,
                           'cov_loss': cov_loss, 'mean_loss': mean_loss, },
                          weights={'sim_loss': self.hparams.lambda_vicreg,
                                   'std_loss': self.hparams.mu_vicreg,
                                   'cov_loss': self.hparams.nu_vicreg,
                                   'mean_loss': self.hparams.lambda_mean_vicreg,}, results=results,
                          training=training, optimizers=[opt_e, opt_p])

        # add scalar measurements
        results['scalar']['h_mean'].append(as_numpy(h1.mean(dim=0).flatten()))
        results['scalar']['h_std'].append(as_numpy(h1.std(dim=0).flatten()))
        results['scalar']['z_mean'].append(as_numpy(mu1.flatten()))
        results['scalar']['z_std'].append(as_numpy(z1.std(dim=0).flatten()))

        return results


class VICReg(BeamSSL):

    def __init__(self, hparams, networks=None, optimizers=None, schedulers=None, **kwargs):

        if networks is None:
            networks = {}
        h = self.h_dim
        p = self.p_dim

        networks['projection'] = nn.Sequential(nn.Linear(h, h), nn.BatchNorm1d(h),
                                               nn.ReLU(), nn.Linear(h, h), nn.BatchNorm1d(h),
                                               nn.ReLU(), nn.Linear(h, p))

        super().__init__(hparams, networks=networks, optimizers=optimizers, schedulers=schedulers, **kwargs)

    def train_iteration(self, sample=None, results=None, subset=None, counter=None, training=True, **kwargs):

        x_aug1, x_aug2 = sample['augmentations']

        encoder = self.networks['encoder']
        opt_e = self.optimizers['encoder']

        projection = self.networks['projection']
        opt_p = self.optimizers['projection']

        h1 = encoder(x_aug1)
        h2 = encoder(x_aug2)

        z1 = projection(h1)
        z2 = projection(h2)

        sim_loss = F.mse_loss(z1, z2, reduction='mean')

        mu1 = z1.mean(dim=0, keepdim=True)
        mu2 = z2.mean(dim=0, keepdim=True)

        std1 = torch.sqrt(z1.var(dim=0) + self.hparams.var_eps)
        std2 = torch.sqrt(z2.var(dim=0) + self.hparams.var_eps)
        std_loss = torch.mean(F.relu(1 - std1)) + torch.mean(F.relu(1 - std2))

        z1 = (z1 - mu1)
        z2 = (z2 - mu2)

        b, d = z1.shape
        corr1 = (z1.T @ z1) / (b - 1)
        corr2 = (z2.T @ z2) / (b - 1)

        I = torch.eye(d, device=corr1.device)
        cov_loss = (corr1 * (1 - I)).pow(2).sum() / d + (corr2 * (1 - I)).pow(2).sum() / d

        loss = self.hparams.lambda_vicreg * sim_loss + self.hparams.mu_vicreg * std_loss + self.hparams.nu_vicreg * cov_loss
        loss = self.apply(loss, training=training, optimizers=[opt_e, opt_p])

        # add scalar measurements
        results['scalar']['loss'].append(as_numpy(loss))
        results['scalar']['sim_loss'].append(as_numpy(sim_loss))
        results['scalar']['std_loss'].append(as_numpy(std_loss))
        results['scalar']['cov_loss'].append(as_numpy(cov_loss))
        results['scalar']['stats_mu'].append(as_numpy(h1.mean(dim=0)))
        results['scalar']['stats_std'].append(as_numpy(h1.std(dim=0)))

        return results


class SimCLR(BeamSSL):

    def __init__(self, hparams, networks=None, optimizers=None, schedulers=None, **kwargs):

        if networks is None:
            networks = {}
        h = self.h_dim
        p = self.p_dim

        networks['projection'] = nn.Sequential(nn.Linear(h, h), nn.BatchNorm1d(h),
                                                   nn.ReLU(), nn.Linear(h, h), nn.BatchNorm1d(h),
                                                   nn.ReLU(), nn.Linear(h, p))

        super().__init__(hparams, networks=networks, optimizers=optimizers, schedulers=schedulers, **kwargs)

    def train_iteration(self, sample=None, results=None, subset=None, counter=None, training=True, **kwargs):

        x_aug1, x_aug2 = sample['augmentations']

        encoder = self.networks['encoder']
        opt_e = self.optimizers['encoder']

        projection = self.networks['projection']
        opt_p = self.optimizers['projection']

        z1 = projection(encoder(x_aug1))
        z2 = projection(encoder(x_aug2))

        b, h = z1.shape
        z = torch.cat([z1, z2], dim=1).view(-1, h)

        z_norm = torch.norm(z, dim=1, keepdim=True)

        s = (z @ z.T) / (z_norm @ z_norm.T)
        s = s * (1 - torch.eye(2 * b, 2 * b, device=s.device)) / self.hparams.temperature

        logsumexp = torch.logsumexp(s[::2], dim=1)
        s_couple = torch.diag(s, diagonal=1)[::2]

        loss = - s_couple + logsumexp
        loss = self.apply(loss, training=training, optimizers=[opt_e, opt_p])

        # add scalar measurements
        results['scalar']['loss'].append(float(loss))
        results['scalar']['acc'].append(float((s_couple >= s[::2].max(dim=1).values).float().mean()))

        return results


class SimSiam(BeamSSL):

    def __init__(self, hparams, networks=None, optimizers=None, schedulers=None, **kwargs):

        if networks is None:
            networks = {}
        h = self.h_dim
        p = self.p_dim

        networks['projection'] = nn.Sequential(nn.Linear(h, h), nn.BatchNorm1d(h),
                                               nn.ReLU(), nn.Linear(h, h), nn.BatchNorm1d(h),
                                               nn.ReLU(), nn.Linear(h, p))

        networks['prediction'] = nn.Sequential(nn.Linear(p, p), nn.BatchNorm1d(p), nn.ReLU(), nn.Linear(p, p))
        super().__init__(hparams, networks=networks, optimizers=optimizers, schedulers=schedulers, **kwargs)

    @staticmethod
    def simsiam_loss(p, z):

        z = z.detach()
        z = z / torch.norm(z, dim=1, keepdim=True)
        p = p / torch.norm(p, dim=1, keepdim=True)
        return 2 - (z * p).sum(dim=1)

    def train_iteration(self, sample=None, results=None, subset=None, counter=None, training=True, **kwargs):

        x_aug1, x_aug2 = sample['augmentations']

        encoder = self.networks['encoder']
        opt_e = self.optimizers['encoder']

        projection = self.networks['projection']
        opt_proj = self.optimizers['projection']

        prediction = self.networks['prediction']
        opt_pred = self.optimizers['prediction']

        z1 = projection(encoder(x_aug1))
        z2 = projection(encoder(x_aug2))

        p1 = prediction(z1)
        p2 = prediction(z2)

        d1 = SimSiam.simsiam_loss(p1, z2)
        d2 = SimSiam.simsiam_loss(p2, z1)

        loss = (d1 + d2) / 2
        loss = self.apply(loss, training=training, optimizers=[opt_e, opt_proj, opt_pred])

        # add scalar measurements
        results['scalar']['loss'].append(float(loss))

        return results


class BYOL(BeamSSL):

    def __init__(self, hparams, networks=None, optimizers=None, schedulers=None, **kwargs):

        if networks is None:
            networks = {}
        h = self.h_dim
        p = self.p_dim

        networks['target_encoder'] = self.generate_encoder(pretrained=False)
        reset_network(networks['target_encoder'])

        networks['projection'] = nn.Sequential(nn.Linear(h, h), nn.BatchNorm1d(h),
                                               nn.ReLU(), nn.Linear(h, h), nn.BatchNorm1d(h),
                                               nn.ReLU(), nn.Linear(h, p))

        networks['target_projection'] = copy_network(networks['projection'])
        reset_network(networks['target_projection'])

        networks['prediction'] = nn.Sequential(nn.Linear(p, p), nn.BatchNorm1d(p), nn.ReLU(), nn.Linear(p, p))
        super().__init__(hparams, networks=networks, optimizers=optimizers, schedulers=schedulers, **kwargs)

    def train_iteration(self, sample=None, results=None, subset=None, counter=None, training=True, **kwargs):

        x_aug1, x_aug2 = sample['augmentations']

        encoder = self.networks['encoder']
        projection = self.networks['projection']
        prediction = self.networks['prediction']

        opt_e = self.optimizers['encoder']
        opt_proj = self.optimizers['projection']
        opt_pred = self.optimizers['prediction']

        z1 = projection(encoder(x_aug1))
        p1 = prediction(z1)

        target_encoder = self.networks['target_encoder']
        target_projection = self.networks['target_projection']

        with torch.no_grad():
            z2 = target_projection(target_encoder(x_aug2))

        z2 = z2 / torch.norm(z2, dim=1, keepdim=True)
        p1 = p1 / torch.norm(p1, dim=1, keepdim=True)

        loss = torch.pow(p1 - z2, 2).sum(dim=1)
        loss = self.apply(loss, training=training, optimizers=[opt_e, opt_proj, opt_pred])

        if training:

            soft_target_update(encoder, target_encoder, self.hparams.tau)
            soft_target_update(projection, target_projection, self.hparams.tau)

        # add scalar measurements
        results['scalar']['loss'].append(float(loss))

        return results