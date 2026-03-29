import importlib
import sys


REQUIRED_MODULES = [
    "torch",
    "numpy",
    "scipy",
    "sklearn",
    "pandas",
    "ipdb",
]


def main():
    all_ok = True
    print("Checking DRL environment...")
    for module_name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "unknown")
            print(f"[OK] {module_name}: {version}")
        except Exception as exc:
            all_ok = False
            print(f"[FAIL] {module_name}: {exc}")

    try:
        import torch

        print(f"Python: {sys.version.split()[0]}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        print(f"MPS available: {mps_available}")
    except Exception:
        all_ok = False

    if not all_ok:
        raise SystemExit(1)

    print("Environment OK")


if __name__ == "__main__":
    main()
