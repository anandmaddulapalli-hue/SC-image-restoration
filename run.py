import sys
from pathlib import Path

import numpy as np
import torch


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
MODEL_PATH = ROOT_DIR / "models" / "hf_residual_best.pth"

# Allow importing model_hf_residual.py from src/
sys.path.insert(0, str(SRC_DIR))

from model_hf_residual import HFResidualSRNet


# ============================================================
# SELF-ENSEMBLE
# ============================================================

def self_ensemble_predict(model, image):

    # 1. Original
    pred_original = model(image)

    # 2. Horizontal flip
    image_h = torch.flip(
        image,
        dims=[3]
    )

    pred_h = model(image_h)

    pred_h = torch.flip(
        pred_h,
        dims=[3]
    )

    # 3. Vertical flip
    image_v = torch.flip(
        image,
        dims=[2]
    )

    pred_v = model(image_v)

    pred_v = torch.flip(
        pred_v,
        dims=[2]
    )

    # 4. Horizontal + vertical flip
    image_hv = torch.flip(
        image,
        dims=[2, 3]
    )

    pred_hv = model(image_hv)

    pred_hv = torch.flip(
        pred_hv,
        dims=[2, 3]
    )

    # Average all predictions
    prediction = (
        pred_original
        + pred_h
        + pred_v
        + pred_hv
    ) / 4.0

    return prediction


# ============================================================
# LOAD INPUT IMAGE
# ============================================================

def load_input_image(file_path):

    image = np.load(file_path)

    # --------------------------------------------------------
    # Accept:
    #
    # (H, W)
    # OR
    # (H, W, 1)
    # --------------------------------------------------------

    if image.ndim == 3:

        if image.shape[-1] == 1:
            image = image[..., 0]

        else:
            raise ValueError(
                f"{file_path.name}: "
                f"expected grayscale image, "
                f"got shape {image.shape}"
            )

    if image.ndim != 2:

        raise ValueError(
            f"{file_path.name}: "
            f"expected 2D grayscale array, "
            f"got shape {image.shape}"
        )

    # Float32
    image = image.astype(
        np.float32,
        copy=False
    )

    # Prevent invalid input values from propagating
    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    return image


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Required command:
    #
    # python run.py <input-dir> <output-dir>
    # --------------------------------------------------------

    if len(sys.argv) != 3:

        print(
            "Usage:\n"
            "python run.py <input-dir> <output-dir>"
        )

        sys.exit(1)


    input_dir = Path(
        sys.argv[1]
    ).resolve()

    output_dir = Path(
        sys.argv[2]
    ).resolve()


    # ========================================================
    # CHECK INPUT DIRECTORY
    # ========================================================

    if not input_dir.exists():

        raise FileNotFoundError(
            f"Input directory does not exist:\n"
            f"{input_dir}"
        )

    if not input_dir.is_dir():

        raise NotADirectoryError(
            f"Input path is not a directory:\n"
            f"{input_dir}"
        )


    # ========================================================
    # FIND .NPY FILES
    # ========================================================

    input_files = sorted(
        input_dir.glob("*.npy")
    )

    if len(input_files) == 0:

        raise RuntimeError(
            f"No .npy files found in:\n"
            f"{input_dir}"
        )


    # ========================================================
    # CREATE OUTPUT DIRECTORY
    # ========================================================

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # CHECK MODEL WEIGHTS
    # ========================================================

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model weights not found:\n"
            f"{MODEL_PATH}"
        )


    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print("=" * 70)
    print(
        "HF RESIDUAL + 4-WAY SELF-ENSEMBLE"
    )
    print("=" * 70)

    print(
        f"Input directory : {input_dir}"
    )

    print(
        f"Output directory: {output_dir}"
    )

    print(
        f"Images found    : {len(input_files)}"
    )

    print(
        f"Device          : {device}"
    )

    if device.type == "cuda":

        print(
            "GPU             : "
            f"{torch.cuda.get_device_name(0)}"
        )


    # ========================================================
    # LOAD FINAL MODEL
    # ========================================================

    model = HFResidualSRNet(
        channels=96,
        num_blocks=8
    ).to(device)


    state_dict = torch.load(
        MODEL_PATH,
        map_location=device
    )


    model.load_state_dict(
        state_dict
    )

    model.eval()


    print(
        "Model           : "
        "HFResidualSRNet "
        "(96 channels, 8 blocks)"
    )

    print(
        "Ensemble        : "
        "4-way self-ensemble"
    )

    print()


    # ========================================================
    # INFERENCE
    # ========================================================

    with torch.inference_mode():

        for index, file_path in enumerate(
            input_files,
            start=1
        ):

            # ------------------------------------------------
            # Load image
            # ------------------------------------------------

            image = load_input_image(
                file_path
            )

            input_height = image.shape[0]
            input_width = image.shape[1]


            # ------------------------------------------------
            # NumPy -> Tensor
            #
            # (H, W)
            # ->
            # (1, 1, H, W)
            # ------------------------------------------------

            tensor = torch.from_numpy(
                image
            )

            tensor = (
                tensor
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )


            # ------------------------------------------------
            # HF Residual + 4-way ensemble
            # ------------------------------------------------

            prediction = self_ensemble_predict(
                model,
                tensor
            )


            # ------------------------------------------------
            # Tensor -> NumPy
            #
            # Output:
            # (2H, 2W)
            # ------------------------------------------------

            restored = (
                prediction
                .squeeze(0)
                .squeeze(0)
                .cpu()
                .numpy()
                .astype(np.float32)
            )


            # ------------------------------------------------
            # Sanitize output
            # ------------------------------------------------

            restored = np.nan_to_num(
                restored,
                nan=0.0,
                posinf=1.0,
                neginf=0.0
            )


            restored = np.clip(
                restored,
                0.0,
                1.0
            ).astype(
                np.float32,
                copy=False
            )


            # ------------------------------------------------
            # Validate output shape
            # ------------------------------------------------

            expected_shape = (
                input_height * 2,
                input_width * 2
            )


            if restored.shape != expected_shape:

                raise RuntimeError(
                    f"{file_path.name}: "
                    f"unexpected output shape "
                    f"{restored.shape}; "
                    f"expected {expected_shape}"
                )


            # ------------------------------------------------
            # Validate numerical values
            # ------------------------------------------------

            if not np.isfinite(
                restored
            ).all():

                raise RuntimeError(
                    f"{file_path.name}: "
                    "output contains NaN or Inf."
                )


            if (
                restored.min() < 0.0
                or restored.max() > 1.0
            ):

                raise RuntimeError(
                    f"{file_path.name}: "
                    "output is outside [0,1]."
                )


            # ------------------------------------------------
            # Save SAME filename
            # ------------------------------------------------

            output_path = (
                output_dir
                / file_path.name
            )


            np.save(
                output_path,
                restored
            )


            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                index == 1
                or index % 25 == 0
                or index == len(input_files)
            ):

                print(
                    f"Processed "
                    f"{index}/{len(input_files)}"
                )


    # ========================================================
    # FINAL VERIFICATION
    # ========================================================

    output_files = sorted(
        output_dir.glob("*.npy")
    )


    # Only count filenames belonging to current input set
    expected_names = {
        f.name
        for f in input_files
    }

    generated_names = {
        f.name
        for f in output_files
        if f.name in expected_names
    }


    if generated_names != expected_names:

        missing = (
            expected_names
            - generated_names
        )

        raise RuntimeError(
            "Not all input files produced outputs. "
            f"Missing: {sorted(missing)}"
        )


    print()
    print("=" * 70)
    print("INFERENCE COMPLETE")
    print("=" * 70)

    print(
        f"Inputs processed : "
        f"{len(input_files)}"
    )

    print(
        f"Outputs generated: "
        f"{len(generated_names)}"
    )

    print(
        "Output format    : "
        "float32 .npy"
    )

    print(
        "Value range      : "
        "[0, 1]"
    )

    print(
        "Upscaling        : "
        "2x"
    )

    print()
    print("All checks passed.")


if __name__ == "__main__":
    main()