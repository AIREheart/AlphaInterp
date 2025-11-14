# scripts/extract_embeddings.py
import argparse
import torch
from models.openfold_wrapper import OpenFoldExtractor
from models.feature_prep import prepare_features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True, help="input fasta file")
    parser.add_argument("--ckpt", required=True, help="OpenFold checkpoint path")
    parser.add_argument("--out", required=True, help="output .npz path")
    parser.add_argument("--device", default="cuda", help="device")
    args = parser.parse_args()

    features = prepare_features(args.fasta, None)
    extractor = OpenFoldExtractor(args.ckpt, device=args.device)
    embeddings = extractor.run_inference_and_extract(features)
    extractor.save_embeddings(embeddings, args.out)
    print("Saved embeddings to", args.out)

if __name__ == "__main__":
    main()
