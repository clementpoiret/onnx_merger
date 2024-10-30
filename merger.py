import argparse
from pathlib import Path

import onnx
from onnx.compose import merge_models
from onnxsim import simplify


def parse_arg(path_str: str) -> Path:
    path = Path(path_str)
    if not path.exists():
        raise ValueError("File not found:", path_str)

    return path


def main():
    parser = argparse.ArgumentParser(
        description="Simple merger of (1) a backbone, with (2) its head."
    )
    parser.add_argument(
        "backbone",
        type=str,
        help="Path to the backbone onnx file.",
    )
    parser.add_argument("head", type=str, help="Path to the head onnx file.")
    parser.add_argument(
        "-o",
        "--out_node_name",
        type=str,
        help="Output name of the backbone",
        default="Identity_1:0",
    )
    parser.add_argument(
        "-i",
        "--in_node_name",
        type=str,
        help="Input name of the head",
        default="args_tf_0",
    )
    args = parser.parse_args()

    backbone_path = parse_arg(args.backbone)
    head_path = parse_arg(args.head)
    output_path = backbone_path.parent / f"{backbone_path.stem}_{head_path.stem}.onnx"

    print("Loading models...")
    backbone = onnx.load(backbone_path)
    head = onnx.load(head_path)

    print("Merging models...")
    model = merge_models(
        m1=backbone,
        m2=head,
        io_map=[[args.out_node_name, args.in_node_name]],
        prefix1="backbone_",
        prefix2="head_",
    )

    print("Simplifying the final graph...")
    model_simp, check = simplify(model)
    assert check, "An error occured while checking the model."

    print("Saving...")
    onnx.save(model_simp, output_path)

    print("Model successfully saved at:", output_path)


if __name__ == "__main__":
    main()
