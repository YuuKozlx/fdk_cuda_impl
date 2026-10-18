#!/usr/bin/env python3
"""Pack SpekPy legacy/v1 tables and its NIST spline into one float32 file."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline


MAGIC = b"SPEKDATA"
def flatten(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        for item in value:
            yield from flatten(item)
    else:
        yield float(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tables", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--precision", choices=("float32", "float64"),
                        default="float32")
    args = parser.parse_args()
    version = 3 if args.precision == "float32" else 2
    scalar_size = 4 if args.precision == "float32" else 8
    scalar_code = "f" if args.precision == "float32" else "d"

    names = ("pe.dat", "ne.dat", "erange.dat", "lines.dat",
             "nist_brem.dat", "nist_mu.dat", "nist_muen_air.dat")
    data = {name: json.loads((args.tables / name).read_text(encoding="utf-8"))
            for name in names}

    sections = []
    sections.append(("pe", list(flatten(data["pe.dat"]["pe"]))))
    sections.append(("ne", list(flatten(data["ne.dat"]["ne"]))))
    sections.append(("range_e", list(flatten(data["erange.dat"]["E0"]))))
    sections.append(("range_r", list(flatten(data["erange.dat"]["range_csda"]))))
    for shell in ("L_1", "L_2", "L_3", "K"):
        line = data["lines.dat"][shell]
        sections.append((f"{shell}_edge", [line["Edge"]]))
        sections.append((f"{shell}_lines", list(flatten(line["Lines"]))))
        sections.append((f"{shell}_weights", list(flatten(line["W"]))))
    brem = data["nist_brem.dat"]
    brem_spline = RectBivariateSpline(
        np.asarray(brem["ebr"], dtype=float) * 1000.0,
        np.asarray(brem["ubr"], dtype=float),
        np.asarray(brem["xbr"], dtype=float), kx=3, ky=3)
    brem_tx, brem_ty, brem_coeff = brem_spline.tck
    sections.extend((
        ("brem_e", list(flatten(brem["ebr"]))),
        ("brem_u", list(flatten(brem["ubr"]))),
        ("brem_x", list(flatten(brem["xbr"]))),
        ("brem_tx", list(flatten(brem_tx))),
        ("brem_ty", list(flatten(brem_ty))),
        ("brem_coeff", list(flatten(brem_coeff))),
    ))
    mu = data["nist_mu.dat"]
    for z, (energies, values) in enumerate(
            zip(mu["photon energy"], mu["mu_over_rho"]), start=1):
        sections.append((f"mu{z}_e", list(flatten(energies))))
        sections.append((f"mu{z}_v", list(flatten(values))))
    muen = data["nist_muen_air.dat"]
    sections.extend((
        ("muen_e", list(flatten(muen["photon energy"]))),
        ("muen_v", list(flatten(muen["muen_over_rho_air"]))),
    ))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    header_size = 8 + 4 + 4 + len(sections) * (24 + 8 + 8)
    offset = header_size
    with args.output.open("wb") as stream:
        stream.write(MAGIC)
        stream.write(struct.pack("<II", version, len(sections)))
        for name, values in sections:
            encoded = name.encode("ascii")
            if len(encoded) >= 24:
                raise ValueError(f"section name too long: {name}")
            stream.write(encoded.ljust(24, b"\0"))
            stream.write(struct.pack("<QQ", offset, len(values)))
            offset += scalar_size * len(values)
        for _, values in sections:
            stream.write(struct.pack(f"<{len(values)}{scalar_code}", *values))
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
