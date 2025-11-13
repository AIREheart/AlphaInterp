import torch
from openfold.model.model import AlphaFold
from openfold.config import model_config
from openfold.utils.script_utils import load_checkpoint

class AlphaFoldFeatureExtractor:
    def __init__(self, ckpt_path, device="cuda"):
        cfg = model_config("model_1")
        self.model = AlphaFold(cfg)
        load_checkpoint(self.model, ckpt_path)
        self.model.eval().to(device)
        self.device = device

    @torch.no_grad()
    def extract_embeddings(self, batch):
        # Forward pass until Evoformer
        evoformer_out = self.model.evoformer(batch)
        return evoformer_out["msa"], evoformer_out["pair"]
