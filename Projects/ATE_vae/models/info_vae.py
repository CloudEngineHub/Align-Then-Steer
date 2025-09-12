from typing import List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor, nn
from torch.distributions.distribution import Distribution

from .cross_attention import (
    SkipTransformerEncoder,
    SkipTransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)

from .base import PositionEmbeddingLearned1D
from .base import lengths_to_mask

class KLLoss:
    def __init__(self, reverse):
        self.reverse = reverse

    def __call__(self, q, p):
        if not self.reverse:
            div = torch.distributions.kl_divergence(q, p)
        else:
            div = torch.distributions.kl_divergence(p, q)
        return div.mean()
    
    def __repr__(self):
        return "KLLoss()"

class Mldloss(object):
    def __init__(self, reverse):
        self.loss1 = torch.nn.SmoothL1Loss(
                    reduction='mean')
        self.loss2 = KLLoss(reverse)
        self.ratio2 = 0.000075
        self.ratio3 = 1
        self.ratio1 = 1.0
        self.z_var = 2.

    def __call__(self, feats):
        loss1 = self.loss1(feats['m_rst'], feats['m_ref'])
        loss2 = self.loss2(feats['dist_m'], feats['dist_ref'])
        z = torch.squeeze(feats['z'])
        mmd_loss = self.compute_mmd(z)
        
        weighted_loss = self.ratio1 * loss1
        weighted_loss += self.ratio2 * loss2
        weighted_loss += self.ratio3 * mmd_loss

        return weighted_loss, loss1, loss2, mmd_loss
    
    def compute_mmd(self, z: Tensor) -> Tensor:
        # Sample from prior (Gaussian) distribution
        prior_z = torch.randn_like(z)

        prior_z__kernel  = self.compute_kernel(prior_z, prior_z)
        z__kernel        = self.compute_kernel(z, z)
        priorz_z__kernel = self.compute_kernel(prior_z, z)

        mmd = prior_z__kernel.mean() + z__kernel.mean() - 2 * priorz_z__kernel.mean()
        return mmd

    def compute_kernel(self,
                       x1: Tensor,
                       x2: Tensor) -> Tensor:
        # Convert the tensors into row and column vectors
        D = x1.size(1)
        N = x1.size(0)

        x1 = x1.unsqueeze(0) # Make it into a column tensor
        x2 = x2.unsqueeze(1) # Make it into a row tensor

        x1 = x1.expand(N, N, D)
        x2 = x2.expand(N, N, D)

        result = self.compute_inv_mult_quad(x1, x2)
        return result

    def compute_inv_mult_quad(self,
                               x1: Tensor,
                               x2: Tensor,
                               eps: float = 1e-7) -> Tensor:
        """
        Computes the Inverse Multi-Quadratics Kernel between x1 and x2,
        given by

                k(x_1, x_2) = \sum \frac{C}{C + \|x_1 - x_2 \|^2}
        :param x1: (Tensor)
        :param x2: (Tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        C = 2 * z_dim * self.z_var
        result = C / (eps + C + (x1 - x2).pow(2).mean(-1))

        return result

    def compute_(self,
            x1: Tensor,
            x2: Tensor,
            dim: int,
            eps: float = 1e-7,
            ):
        
        kernel_input = (x1 - x2).pow(2).mean(2) / float(dim)
        return torch.exp(-kernel_input)


    def compute_rbf(self,
                x1: Tensor,
                x2: Tensor,
                eps: float = 1e-7) -> Tensor:
        """
        Computes the RBF Kernel between x1 and x2.
        :param x1: (Tensor)
        :param x2: (Tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        sigma = 2. * z_dim * self.z_var

        result = torch.exp(-((x1 - x2).pow(2).mean(-1) / sigma))
        return result


class InfoVAE(nn.Module):

    def __init__(self,
                 nfeats: int = 128,
                 latent_dim: list = [1, 256],
                 ff_size: int = 1024,
                 num_layers: int = 9,
                 num_heads: int = 4,
                 dropout: float = 0.0,
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 **kwargs) -> None:

        super().__init__()

        self.latent_size = latent_dim[0]
        self.latent_dim = latent_dim[-1]
        input_feats = nfeats
        output_feats = nfeats

        self.query_pos_encoder = PositionEmbeddingLearned1D(self.latent_dim)
        self.query_pos_decoder = PositionEmbeddingLearned1D(self.latent_dim)

        encoder_layer = TransformerEncoderLayer(
            self.latent_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
        )

        encoder_norm = nn.LayerNorm(self.latent_dim)
        self.encoder = SkipTransformerEncoder(encoder_layer, num_layers,
                                              encoder_norm)

        decoder_layer = TransformerDecoderLayer(
            self.latent_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
        )

        decoder_norm = nn.LayerNorm(self.latent_dim)
        self.decoder = SkipTransformerDecoder(decoder_layer, num_layers,
                                                  decoder_norm)

        self.global_motion_token = nn.Parameter(
            torch.randn(self.latent_size * 2, self.latent_dim))

        self.skel_embedding = nn.Linear(input_feats, self.latent_dim)
    
        self.final_layer = nn.Linear(self.latent_dim, output_feats)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)


    def encode(
            self,
            features: Tensor,
            lengths: Optional[List[int]] = None
    ) -> Union[Tensor, Distribution]:
        if lengths is None:
            lengths = [len(feature) for feature in features]

        device = features.device
        
        bs, nframes, nfeats = features.shape
        #! AKA:
        # Need input: [batchsize, 64, 128]
        mask = lengths_to_mask(lengths, device)

        x = features

        # Embed each human poses into latent vectors
        x = self.skel_embedding(x)
        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]
        # Each batch has its own set of tokens
        dist = torch.tile(self.global_motion_token[:, None, :], (1, bs, 1))

        # create a bigger mask, to allow attend to emb
        dist_masks = torch.ones((bs, dist.shape[0]),
                                dtype=bool,
                                device=x.device)
        aug_mask = torch.cat((dist_masks, mask), 1)

        # adding the embedding token for all sequences
        xseq = torch.cat((dist, x), 0)
        
        xseq = self.query_pos_encoder(xseq)
        dist = self.encoder(xseq,
                            src_key_padding_mask=~aug_mask)[:dist.shape[0]]

        # content distribution
        # self.latent_dim => 2*self.latent_dim
        mu = dist[0:self.latent_size, ...]
        logvar = dist[self.latent_size:, ...]

        # reparameterize
        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        latent = dist.rsample()
        return latent, dist, mu, std

    def decode(self, z: Tensor, lengths: List[int]):
        mask = lengths_to_mask(lengths, z.device)
        bs, nframes = mask.shape

        queries = torch.zeros(nframes, bs, self.latent_dim, device=z.device)

        # Pass through the transformer decoder
        # with the latent vector for memory
        queries = self.query_pos_decoder(queries)
        output = self.decoder(
            tgt=queries,
            memory=z,
            tgt_key_padding_mask=~mask,
        ).squeeze(0)
        output = self.final_layer(output)

        # zero for padded area
        output[~mask.T] = 0
        # Pytorch Transformer: [Sequence, Batch size, ...]
        feats = output.permute(1, 0, 2)

        return feats