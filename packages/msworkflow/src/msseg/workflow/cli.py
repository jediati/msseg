"""Command-line runner for the generic MSSeg JSON workflow.

    msworkflow <workflow.json> <input.tif[f]> <output_labels.tif[f]>

Reads a 2D/3D TIFF volume, runs the JSON-described workflow over the core
pipeline, and writes an int32 label volume. (The native `msworkflow` C++ CLI
covers the .raw + .dat sidecar path.)
"""
import argparse
import sys

import numpy as np

from . import msworkflow_py as _ext


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("workflow", help="workflow JSON spec")
    ap.add_argument("input", help="input volume (.tif/.tiff)")
    ap.add_argument("output", help="output int32 label volume (.tif/.tiff)")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)

    import tifffile
    vol = np.ascontiguousarray(np.squeeze(np.asarray(tifffile.imread(args.input))), dtype=np.float32)
    if vol.ndim == 2:
        vol = vol[None, ...]
    if vol.ndim != 3:
        print(f"error: expected a 2D/3D volume, got shape {vol.shape}", file=sys.stderr)
        return 2
    with open(args.workflow, encoding="utf-8") as fh:
        spec = fh.read()
    labels = np.asarray(_ext.run(vol, spec), dtype=np.int32)
    tifffile.imwrite(args.output, labels)
    print(f"wrote {args.output} shape {labels.shape} labels {int(labels.max()) + 1}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
