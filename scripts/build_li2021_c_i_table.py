#!/usr/bin/env python3
"""Build data/atomic/li2021_c_i_e1.csv.xz from Li et al. (2021) supplementary Table A2.

Li, Amarsi, Papoulia, Ekman & Jonsson (2021, MNRAS 502, 3780; arXiv:2101.09114)
give MCDHF/RCI E1 line strengths of C I.  Their online supplementary file
``tableA2_C_I.tex`` lists, per transition, the upper and lower state, the
transition energy, the vacuum wavelength (``*`` marks a predicted one), the
line strength S, gf, A and dT, the relative difference between the
length- and velocity-gauge values.

Each state is written as the Stout C I level label it denotes
(``2s^22p4p~^1D_{2}`` -> ``2s2.2p.(2Po).4p.(1D<2>)``), so that the table is
matched to the bundled line list by exact level identity at run time.  The
authors write the 2s2p3 states either ``2s2p^3`` or ``2s2p^32p``; both are the
``2s.2p3`` configuration.  4f states are given in LS coupling, which has no
counterpart among the jK-coupled Stout 4f levels, and are kept with their LS
label (they do not match).

usage: build_li2021_c_i_table.py TABLE_A2_TEX [OUTPUT]
"""

import hashlib
import lzma
import re
import sys
from pathlib import Path

_STATE = re.compile(
    r"\$(?P<cfg>[^~]+)~\^(?P<mult>\d)(?P<L>[SPDFGHI])_\{\s*(?P<J>\d+)\s*\}(?P<odd>\^o)?\$"
)
COLUMNS = (
    "upper_label,lower_label,vacuum_wavelength_A,predicted_wavelength,gf,einstein_a_s,"
    "gauge_difference"
)


def stout_label(state: str) -> str | None:
    match = _STATE.fullmatch(state.strip())
    if match is None:
        return None
    configuration = match["cfg"]
    if configuration == "2s^22p^2":
        prefix = "2s2.2p2"
    elif configuration in ("2s2p^3", "2s2p^32p"):
        prefix = "2s.2p3"
    else:
        orbital = re.fullmatch(r"2s\^22p(\d+)([spdfg])", configuration)
        if orbital is None:
            return None
        prefix = f"2s2.2p.(2Po).{orbital[1]}{orbital[2]}"
    parity = "o" if match["odd"] else ""
    return f"{prefix}.({match['mult']}{match['L']}{parity}<{match['J']}>)"


def main() -> None:
    source = Path(sys.argv[1])
    output = (
        Path(sys.argv[2]) if len(sys.argv) > 2
        else Path(__file__).resolve().parents[1] / "src/wd_spectra/data/atomic/li2021_c_i_e1.csv.xz"
    )
    text = source.read_bytes()
    rows = []
    for line in text.decode("ascii").splitlines()[1:]:
        cells = [cell.strip() for cell in line.split("&")]
        if len(cells) < 8:
            continue
        upper, lower = stout_label(cells[0]), stout_label(cells[1])
        if upper is None or lower is None:
            raise ValueError(f"unparsed state in {line!r}")
        wavelength = cells[3]
        rows.append(
            f"{upper},{lower},{float(wavelength.rstrip('*')):.3f},{int(wavelength.endswith('*'))},"
            f"{float(cells[5]):.4e},{float(cells[6]):.4e},{float(cells[7].rstrip(chr(92)).strip()):.3f}"
        )
    header = [
        "# Li, Amarsi, Papoulia, Ekman & Jonsson 2021, MNRAS 502, 3780 (arXiv:2101.09114):",
        "# MCDHF/RCI E1 transition data of C I, supplementary Table A2 (tableA2_C_I.tex),",
        f"# SHA256 {hashlib.sha256(text).hexdigest()}; {len(rows)} transitions.",
        "# States are written as Stout level labels; gf and A as tabulated by the authors,",
        "# gauge_difference is their dT; predicted_wavelength = 1 marks a calculated wavelength.",
        "# Built by scripts/build_li2021_c_i_table.py.",
        COLUMNS,
    ]
    with lzma.open(output, "wt", encoding="ascii") as stream:
        stream.write("\n".join(header + rows) + "\n")
    print(f"wrote {output} ({len(rows)} transitions)")


if __name__ == "__main__":
    main()
