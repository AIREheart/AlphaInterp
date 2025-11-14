# models/openfold_wrapper.py
import os
import torch
from pathlib import Path
import numpy as np

# OpenFold import
from openfold.model.model import AlphaFold
from openfold.model.modules import ExtraMSAStack
from openfold.config import model_config
from openfold.utils import protein as openfold_protein
from openfold.utils.script_utils import load_pretrained, load_checkpoint

# helper: convert OpenFold outputs to numpy and save
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)

class OpenFoldExtractor:
    def __init__(self, ckpt_path, model_name="model_1", device="cuda"):
        cfg = model_config(model_name)
        self.model = AlphaFold(cfg)
        # load weights - try load_pretrained or load_checkpoint
        if ckpt_path is not None and os.path.exists(ckpt_path):
            load_checkpoint(self.model, ckpt_path, map_location=device)
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def run_inference_and_extract(self, features):
        """
        features: dict prepared as OpenFold expects (MSA, a3m, residue_index, sequence, etc.)
        returns dict with evoformer MSA/pair/per-residue embeddings and attention maps if available
        """
        # forward through the model and retrieve intermediate activations. OpenFold's AlphaFold returns
        # a dict of outputs via self.model(features)
        out = self.model(features)

        # typical keys in OpenFold: 'predicted_lddt', 'final_atom_positions', 'evoformer_output' (depending on model)
        # We try to extract MSA/pair embeddings if present
        embeddings = {}
        # common names vary — inspect out printed keys if something is missing
        for k in ["representations", "evoformer_output", "msa"]:
            if k in out:
                embeddings[k] = to_numpy(out[k])
        # also capture final residue representations if present
        if "representations" in out and "single" in out["representations"]:
            embeddings["single"] = to_numpy(out["representations"]["single"])  # [N_res, C]
        # some versions return 'msa_first_row' or 'pair' inside 'representations'
        if "pair" in out.get("representations", {}):
            embeddings["pair"] = to_numpy(out["representations"]["pair"])
        # predicted LDDT
        if "predicted_lddt" in out:
            embeddings["plddt"] = to_numpy(out["predicted_lddt"])
        # return metadata + embeddings
        return embeddings

    @staticmethod
    def save_embeddings(embeddings, out_path):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **embeddings)
