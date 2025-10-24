# python scripts/03_test_model_gpu.py --batch_size 32 --input_height 256 --input_width 128 --model_names tf_efficientnet_b0_ns,tf_efficientnet_b1_ns,tf_efficientnet_b2_ns,tf_efficientnet_b3_ns,tf_efficientnet_b4_ns,resnet18,resnet34,resnet50,resnet101,inception_v3,inception_v4

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

if bool(os.environ.get("KAGGLE_URL_BASE", "")):
    sys.path.insert(0, "/kaggle/input/hsm-source-files")
else:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.base_cnn import BaseCNN


def test_model_memory(model_name, batch_size=32, img_size=(128, 256), in_channels=4, num_classes=6):
    device = torch.device("cuda")

    results = {
        "model_name": model_name,
        "success": False,
        "error": None,
        "peak_memory_mb": None,
        "allocated_memory_mb": None,
    }

    try:
        print(f"\nTesting {model_name}...")
        print("-" * 60)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        print("  Loading model...")
        model = BaseCNN(model_name, pretrained=True, num_classes=num_classes)
        model.to(device)

        # create dummy inputwith shape: (batch_size, in_channels, height, width)
        dummy_input = torch.randn(batch_size, in_channels, img_size[0], img_size[1]).to(device)
        dummy_target = torch.randn(batch_size, num_classes).to(device)
        dummy_target = F.softmax(dummy_target, dim=1)

        print("  Running forward pass...")
        model.train()
        outputs = model(dummy_input)

        print("  Calculating loss...")
        log_probs = F.log_softmax(outputs, dim=1)
        loss_fn = nn.KLDivLoss(reduction="batchmean")
        loss = loss_fn(log_probs, dummy_target)

        print("  Running backward pass...")
        loss.backward()

        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        allocated_memory = torch.cuda.memory_allocated() / 1024**2
        results["peak_memory_mb"] = round(peak_memory, 2)
        results["allocated_memory_mb"] = round(allocated_memory, 2)
        print(f"  Peak GPU memory: {peak_memory:.2f} MB")
        print(f"  Allocated GPU memory: {allocated_memory:.2f} MB")

        results["success"] = True
        print(f"  SUCCESS: {model_name} can run on {device}")

    except Exception as e:
        results["error"] = str(e)
        print(f"  FAILED: {model_name}")
        print(f"  Error: {str(e)}")

    finally:
        torch.cuda.empty_cache()

    print("-" * 60)
    return results


def test_all_models(models_list, batch_size=32, img_size=(128, 256)):
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {img_size}")
    print("=" * 60)

    results = []
    for model_name in models_list:
        result = test_model_memory(model_name, batch_size=batch_size, img_size=img_size)
        results.append(result)

    print("\n" + "=" * 60)

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"Successful: {len(successful)}/{len(results)}")
    print(f"Failed: {len(failed)}/{len(results)}")

    if failed:
        print("\nModels that failed:")
        for r in failed:
            print(f"  - {r['model_name']}")
            print(f"    Error: {r['error']}")

    return results


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Please run this script on a machine with a GPU.")
        return

    parser = argparse.ArgumentParser(description="Test GPU memory requirements for CNN architectures")

    parser.add_argument(
        "--model_names",
        type=str,
        required=True,
        help="Comma separated list of model names to test",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size to test with (default: 32)",
    )

    parser.add_argument(
        "--input_height",
        type=int,
        default=128,
        help="Image height (default: 128)",
    )

    parser.add_argument(
        "--input_width",
        type=int,
        default=256,
        help="Image width (default: 256)",
    )

    args = parser.parse_args()

    models = args.model_names.split(",")
    models = [m.strip() for m in models if m.strip()]

    img_size = (args.input_height, args.input_width)

    test_all_models(models, batch_size=args.batch_size, img_size=img_size)


if __name__ == "__main__":
    main()
