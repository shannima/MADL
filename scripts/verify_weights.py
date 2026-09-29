import argparse

from madl.weights import verify_weight_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the MADL Hugging Face release manifest.")
    parser.add_argument("weight_root")
    args = parser.parse_args()
    failures = verify_weight_manifest(args.weight_root)
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        return 1
    print("All files match MANIFEST.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
