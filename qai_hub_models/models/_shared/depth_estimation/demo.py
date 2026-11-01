# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


import glob
import os

from PIL import Image

from qai_hub_models.models._shared.depth_estimation.app import DepthEstimationApp
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.display import display_or_save_image


# The demo will display a heatmap of the estimated depth at each point in the image.
def depth_estimation_demo(
    model_cls: type[BaseModel],
    model_id: str,
    default_image: CachedWebModelAsset,
    is_test: bool = False,
) -> None:
    parser = get_model_cli_parser(model_cls)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=default_image,
        help="image file path, image file folder or URL",
    )
    args = parser.parse_args([] if is_test else None)
    model = demo_model_from_cli_args(model_cls, model_id, args)
    validate_on_device_demo_args(args, model_id)

    # Load image
    (_, _, height, width) = model_cls.get_input_spec()["image"][0]
    print("Model Loaded")

    if os.path.isfile(args.image):
        filenames = [args.image]
    else:
        filenames = glob.glob(os.path.join(args.image, '**/*'), recursive=True)

    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')

        image = load_image(filename)

        app = DepthEstimationApp(
            model,  # type: ignore[arg-type]
            height,
            width,
        )
        heatmap_image = app.estimate_depth(image, raw_output=True)
        assert isinstance(heatmap_image, Image.Image)

        if not is_test:
            # Resize / unpad annotated image
            basename = os.path.basename(filename)
            outname = os.path.splitext(basename)[0] + ".png"
            display_or_save_image(
                heatmap_image, args.output_dir, outname, "heatmap"
            )
