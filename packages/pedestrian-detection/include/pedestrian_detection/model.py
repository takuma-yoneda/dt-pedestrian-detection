#!/usr/bin/env python3
import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional
from pedestrian_detection.modules import MLP, BasicBlock, DecoderNet, TinyResNet


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet = TinyResNet(BasicBlock, [2, 2, 2, 2], 1024)

    def forward(self, x):
        # Input shape: (batch, 112, 112, 3)
        assert x.shape[1:] == (112, 112, 3)


class SegmentModel:
    def __init__(self, encoder: nn.Module, segm_head: nn.Module, encoder_opt: torch.optim.Optimizer) -> None:
        self.encoder = encoder
        self.encoder_opt = encoder_opt
        self.segm_head = segm_head
        self.segm_opt = torch.optim.Adam(self.segm_head.parameters())

    def predict(self, img):
        h, x1, x2, x3 = self.encoder(img, unet=True)
        segm = self.segm_head(h, x1, x2, x3)
        return segm

    def to(self, device):
        self.encoder = self.encoder.to(device)
        self.segm_head = self.segm_head.to(device)
        return self

    @staticmethod
    def compute_loss(segm_pred, segm_label):
        loss = F.cross_entropy(segm_pred, segm_label)
        return loss

    def step(self, latent, x1, x2, x3, label, retain_graph=False):
        import wandb
        # self.encoder_opt.zero_grad(set_to_none=True)
        self.segm_opt.zero_grad(set_to_none=True)
        segm = self.segm_head(latent, x1, x2, x3)
        loss = self.compute_loss(segm, label)
        loss.backward(retain_graph=retain_graph)
        self.segm_opt.step()
        # self.encoder_opt.step()

        wandb.log({'train/loss/segm': loss.item()})

        return loss


class LaneposModel:
    def __init__(self, encoder: nn.Module, offset_head: nn.Module, phi_head: nn.Module, encoder_opt: torch.optim.Optimizer) -> None:
        self.encoder = encoder
        self.encoder_opt = encoder_opt
        self.offset_head = offset_head
        self.phi_head = phi_head
        self.offset_opt = torch.optim.Adam(offset_head.parameters())
        self.phi_opt = torch.optim.Adam(phi_head.parameters())

    def predict(self, img):
        h = self.encoder(img)
        offset = self.offset_head(h)
        phi = self.phi_head(h)
        return offset, phi

    def to(self, device):
        self.offset_head = self.offset_head.to(device)
        self.phi_head = self.phi_head.to(device)
        self.encoder = self.encoder.to(device)
        return self

    @staticmethod
    def compute_loss(offset_pred, phi_pred, label, verbose=False):
        offset_label = label[:, 2].unsqueeze(1)
        phi_label = label[:, 3].unsqueeze(1)
        offset_loss = F.mse_loss(offset_pred, offset_label)
        phi_loss = F.mse_loss(phi_pred, phi_label)
        loss = offset_loss + phi_loss
        if verbose:
            return loss, offset_loss, phi_loss
        return loss

    def step(self, latent, label, retain_graph=False):
        import wandb
        # Zero grad
        # self.encoder_opt.zero_grad(set_to_none=True)
        self.offset_opt.zero_grad(set_to_none=True)
        self.phi_opt.zero_grad(set_to_none=True)

        offset = self.offset_head(latent)
        phi = self.phi_head(latent)
        loss, offset_loss, phi_loss = self.compute_loss(offset, phi, label, verbose=True)

        loss.backward(retain_graph=retain_graph)

        # Step optimizers
        # self.encoder_opt.step()
        self.offset_opt.step()
        self.phi_opt.step()

        wandb.log({'train/loss/lanepos-offset': offset_loss.item(),
                   'train/loss/lanepos-phi': phi_loss.item()})


        return loss


class PedestModel:
    def __init__(self, encoder: nn.Module, dist_head: nn.Module, lane_head: nn.Module, encoder_opt: torch.optim.Optimizer) -> None:
        self.encoder = encoder
        self.dist_head = dist_head
        self.lane_head = lane_head
        self.dist_opt = torch.optim.Adam(dist_head.parameters())
        self.lane_opt = torch.optim.Adam(lane_head.parameters())
        self.encoder_opt = encoder_opt

    def predict(self, img):
        h = self.encoder(img)
        dist = self.dist_head(h)
        lane = self.lane_head(h)
        return dist, lane

    def to(self, device):
        self.dist_head = self.dist_head.to(device)
        self.leane_head = self.lane_head.to(device)
        self.encoder = self.encoder.to(device)
        return self

    @staticmethod
    def compute_loss(lane_pred, label):
        # dist_label = label[:, 0].unsqueeze(1)
        lane_label = label[:, 1].unsqueeze(1)

        # dist_loss = F.mse_loss(dist_pred, dist_label)
        # lane_loss = F.binary_cross_entropy_with_logits(lane_pred, lane_label)
        lane_loss = F.binary_cross_entropy(lane_pred, lane_label)

        # loss = dist_loss + lane_loss
        # if verbose:
        #     return loss, dist_loss, lane_loss
        loss = lane_loss
        return loss

    def step(self, latent, label, retain_graph=False):
        import wandb
        # Zero grad
        # self.encoder_opt.zero_grad(set_to_none=True)
        # self.dist_opt.zero_grad(set_to_none=True)
        self.lane_opt.zero_grad(set_to_none=True)

        # dist = self.dist_head(latent)
        lane = self.lane_head(latent)
        # loss, dist_loss, lane_loss = self.compute_loss(dist, lane, label, verbose=True)
        loss = self.compute_loss(lane, label)
        loss.backward(retain_graph=retain_graph)

        # Step optimizers
        # self.encoder_opt.step()
        # self.dist_opt.step()
        self.lane_opt.step()

        # wandb.log({'train/loss/pedest-dist': dist_loss.item(),
        #            'train/loss/pedest-lane': lane_loss.item()})

        wandb.log({'train/loss/pedest-lane': loss.item()})

        return loss


class JointModel:
    def __init__(self, pedest_model, lanepos_model, segment_model) -> None:
        self.pedest = pedest_model
        self.lanepos = lanepos_model
        self.segment = segment_model
        self.encoder = pedest_model.encoder
        self.encoder_opt = pedest_model.encoder_opt
        assert id(self.pedest.encoder) == id(self.lanepos.encoder) == id(self.segment.encoder)
        assert id(self.pedest.encoder_opt) == id(self.lanepos.encoder_opt) == id(self.segment.encoder_opt)

    def to(self, device):
        self.pedest = self.pedest.to(device)
        self.lanepos = self.lanepos.to(device)
        self.segment = self.segment.to(device)
        return self

    def step(self, img, segm_label, label):
        self.encoder_opt.zero_grad(set_to_none=True)

        h, x1, x2, x3 = self.encoder(img, unet=True)
        pedest_loss = self.pedest.step(h, label, retain_graph=True)
        lanepos_loss = self.lanepos.step(h, label, retain_graph=True)
        segm_loss = self.segment.step(h, x1, x2, x3, segm_label, retain_graph=False)

        self.encoder_opt.step()

        loss = pedest_loss + lanepos_loss + segm_loss

        # wandb.log({'train/loss/pedest': pedest_loss.item(),
        #            'train/loss/segm': segm_loss.item(),
        #            'train/loss/lanepos': lanepos_loss.item()})
        return loss

# class Model(nn.Module):
#     def __init__(self,
#                  encoder: nn.Module,
#                  segm_head: nn.Module,
#                  pedest_dist_head: nn.Module,
#                  pedest_lane_head: nn.Module,
#                  lanepos_offset_head: nn.Module,
#                  lanepos_phi_head: nn.Module,
#                  ) -> None:
#         super().__init__()
#         self.encoder = encoder
#         self.segm_head = segm_head
#         self.pedest_dist_head = pedest_dist_head
#         self.pedest_lane_head = pedest_lane_head
#         self.lanepos_offset_head = lanepos_offset_head
#         self.lanepos_phi_head = lanepos_phi_head

#     def segment(self, x):
#         h, x1, x2, x3 = self.encoder(x, unet=True)
#         return self.segm_head(h, x1, x2, x3)

#     def no_segment(self, x):
#         h = self.encoder(x, unet=False)
#         pedest = self.pedest_head(h)
#         lanepos = self.lanepos_head(h)
#         return pedest, lanepos

#     def forward(self, x):
#         h, x1, x2, x3 = self.encoder(x, unet=True)
#         segm = self.segm_head(h, x1, x2, x3)
#         pedest = self.pedest_dist_head(h)
#         lanepos = self.lanepos_head(h)
#         return segm, pedest, lanepos


def make_model(hidden_dim=512):
    encoder = TinyResNet(BasicBlock, [2, 2, 2, 2], hidden_dim)
    encoder_opt = torch.optim.Adam(encoder.parameters(), lr=1e-4)
    pedest = PedestModel(encoder,
                         MLP(hidden_dim, 1, 256),
                         MLP(hidden_dim, 1, 256, out_sigmoid=True),
                         encoder_opt)
    lanepos = LaneposModel(encoder,
                           MLP(hidden_dim, 1, 256),
                           MLP(hidden_dim, 1, 256),
                           encoder_opt)
    segm = SegmentModel(encoder,
                        DecoderNet(num_classes=7),
                        encoder_opt)
    return JointModel(pedest, lanepos, segm)
