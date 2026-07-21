# Costruisce il modello di riparazione dal config.
#
# NCA e U-Net condividono l'interfaccia (stato -> stato della stessa forma), quindi
# pool, trainer e valutazione non sanno quale dei due stanno usando: basta scegliere
# la classe qui. 'arch' nel config del modello seleziona l'architettura.
from __future__ import annotations

from src.models.nca import NCA
from src.models.unet import UNet


def build_model(model_cfg):
    """Istanzia il modello secondo model_cfg.arch ('nca' oppure 'unet')."""
    arch = model_cfg.get("arch", "nca")
    if arch == "nca":
        return NCA(hidden_channels=model_cfg.hidden_channels,
                   mlp_hidden=model_cfg.mlp_hidden,
                   update_prob=model_cfg.update_prob,
                   use_laplacian=model_cfg.use_laplacian)
    if arch == "unet":
        return UNet(hidden_channels=model_cfg.hidden_channels,
                    base=model_cfg.get("base", 32),
                    iterative=model_cfg.get("iterative", True))
    raise ValueError(f"architettura sconosciuta: {arch!r} (usa 'nca' o 'unet')")