import pathlib
import os
import sys

def main():
    root = pathlib.Path(__file__).resolve().parent
    src = os.path.join(root, "symMatmulRoutines.f90")
    module_name = "symMatmulRoutines"

    # Optimization flags
    flags = {
        "safe": "-O3 -march=native -mtune=native -fno-unsafe-math-optimizations -fno-fast-math -funroll-loops -fomit-frame-pointer -flto -fstrict-aliasing",
        "fast": "-Ofast -march=native -mtune=native -ffast-math -fno-math-errno -funroll-loops -fomit-frame-pointer -flto -fstrict-aliasing -fno-trapping-math -fno-signaling-nans",
        "ultimate": "-Ofast -march=native -mtune=native -funroll-all-loops -ffast-math -fassociative-math -freciprocal-math -fno-signed-zeros -fno-trapping-math -fno-math-errno -fno-protect-parens -fno-rounding-math -fno-signaling-nans -fomit-frame-pointer -frename-registers -fvect-cost-model=cheap -ftree-vectorize -falign-loops=64 -falign-functions=64 -fno-stack-protector -fno-semantic-interposition -flto -fno-plt -Wno-maybe-uninitialized"
    }

    # User input
    blas = input("Input BLAS linking flag (default: -lopenblas): ").strip() or "-lopenblas"
    choice = input("Choose optimization level (S-safe, F-fast, U-ultimate), default safe: ").strip().lower()

    if choice in ("f", "fast"):
        level = "fast"
    elif choice in ("u", "ultimate"):
        level = "ultimate"
    else:
        level = "safe"

    os.environ["FFLAGS"] = flags[level]
    os.environ["FCFLAGS"] = flags[level]

    print(f"Using {level} optimization flags:")
    print("FFLAGS:", os.environ["FFLAGS"])
    print("FCFLAGS:", os.environ["FCFLAGS"])

    # Build command
    cmd = f"{sys.executable} -m numpy.f2py -c -m {module_name} {src} {blas}"

    print("Running:", cmd)

    # Run inside the folder so the .so lands here
    os.chdir(root)
    os.system(cmd)

    print("\nFortran extension built successfully.")
    print(f"Stored in: {root}")

if __name__ == "__main__":
    main()