# models/feature_prep.py
import os
from openfold.data.tools import hhsearch
from openfold.utils import protein as openfold_protein
from openfold.data.feature_pipeline import FeaturePipeline
from openfold.config import model_config

def prepare_features(fasta_path, output_dir, msa_mode="mmseqs"):
    """
    Uses OpenFold's FeaturePipeline to construct features dict.
    Requires preinstalled MMseqs2 and databases (or use small test a3m).
    """
    cfg = model_config("model_1")
    feature_pipeline = FeaturePipeline(cfg, use_templates=False)

    # read sequence
    seq = open(fasta_path).read().strip().splitlines()
    seq = "".join([line.strip() for line in seq if not line.startswith(">")])
    # FeaturePipeline expects a dict of inputs; see OpenFold doc
    feature_dict = feature_pipeline.process_fasta(fasta_path)
    return feature_dict
