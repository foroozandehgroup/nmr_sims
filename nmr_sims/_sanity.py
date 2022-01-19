# _sanity.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Sun 16 Jan 2022 15:56:59 GMT

from dataclasses import dataclass
import re
from typing import Any, Iterable, Tuple, Union
import numpy as np
from nmr_sims import nuclei


def is_multiple_of_one_half(x):
    return round(x, 10) % 0.5 == 0


def is_an_iterable_of_spins(obj):
    try:
        iter(obj)
    except TypeError:
        return False
    return all([is_multiple_of_one_half(x) for x in obj])


def check_dict_with_int_keys(
    obj: Any, varname: str, consecutive: bool = False, max_: Union[int, None] = None,
    forbidden: Union[Iterable[int], None] = None,
) -> None:
    errmsg = f"`{varname}` should be a dict, and it's keys should be<1>ints<2>.<3>"
    if consecutive:
        errmsg = errmsg.replace("<1>", " consecutive ")
        errmsg = errmsg.replace("<2>", ", starting at 1")
        errmsg = errmsg.replace("<3>", "")
    elif isinstance(max_, int):
        errmsg = errmsg.replace("<1>", " positive ")
        errmsg = errmsg.replace("<2>", f", that are no greater than {max_}")
        if forbidden is not None:
            if len(forbidden) == 1:
                errmsg = errmsg.replace("<3>", f" {forbidden[0]} is not permitted.")
            else:
                errmsg = errmsg.replace(
                    "<3>", " " + ", ".join(forbidden) + " are not permitted.",
                )
        else:
            errmsg = errmsg.replace("<3>", "")

    else:
        errmsg = errmsg.replace("<1>", " ")
        errmsg = errmsg.replace("<2>", "")
        errmsg = errmsg.replace("<3>", "")

    if not isinstance(obj, dict):
        raise TypeError(errmsg)
    keys = list(obj.keys())
    if any([not isinstance(key, int) for key in keys]):
        raise ValueError(errmsg)
    if consecutive and (sorted(keys) != list(range(1, len(keys) + 1))):
        raise ValueError(errmsg)
    if isinstance(max_, int) and any([key > max_ for key in keys]):
        raise ValueError(errmsg)
    if forbidden is not None and any([key in forbidden for key in keys]):
        raise ValueError(errmsg)


def process_spins(spins: Any, default_nucleus: Any) -> Iterable[dict]:
    check_dict_with_int_keys(spins, "spins", consecutive=True)
    nspins = len(spins)
    spin_dict = {}
    couplings = np.zeros((nspins, nspins))
    default_nucleus = process_nucleus(default_nucleus, "default_nucleus")

    for i, spin in spins.items():
        # Process nucleus
        if "nucleus" in spin.keys():
            nucleus = process_nucleus(spin["nucleus"], f"spins[{i}][\"nucelus\"]")
        else:
            nucleus = default_nucleus

        if "shift" not in spin.keys():
            raise ValueError(
                "Each value in `spins` should be a dict with the keys "
                "\"shift\" and (optional) \"couplings\", \"nucleus\". "
                f"This is not satisfied by spin {i}."
            )

        if not isinstance(spin["shift"], (int, float)):
            raise TypeError(
                "\"shift\" entries should be scalar values. This is not "
                f"satisfied by spin {i}."
            )

        shift = float(spin["shift"])

        if "couplings" in spin.keys():
            check_dict_with_int_keys(
                spin["couplings"], f"spins[{i}][\"couplings\"]", max_=nspins,
                forbidden=[i],
            )

            for j, coupling in spin["couplings"].items():
                current_value = couplings[i - 1, j - 1]
                if float(current_value) != 0.:
                    if coupling != current_value:
                        raise ValueError(
                            f"Contradictory couplings given between spins {j} and "
                            f"{i}: {float(coupling)} and {current_value}."
                        )
                else:
                    couplings[i - 1, j - 1] = coupling
                    couplings[j - 1, i - 1] = coupling

        spin_dict[i] = Spin(
            nucleus,
            float(shift),
            {
                j + 1: couplings[i - 1, j] for j in range(i, nspins)
            },
        )

    return spin_dict


def process_channel(
    obj: Any, field: float
) -> Tuple[nuclei.Nucleus, float, float, int]:
    keys = ("nucleus", "sweep_width", "offset", "points")
    if not isinstance(obj, dict) or any([k not in obj for k in keys]):
        raise ValueError(
            "Channels should be dicts with keys:\n" +
            ", ".join([f"\"{k}\"" for k in keys])
        )

    nucleus = process_nucleus(obj["nucleus"], "nucleus")
    sweep_width = process_sweep_width(obj["sweep_width"], nucleus, field)
    offset = process_offset(obj["offset"], nucleus, field)
    points = process_points(obj["points"])
    return nucleus, sweep_width, offset, points


def process_nucleus(nucleus: Any, varname: str) -> nuclei.Nucleus:
    if isinstance(nucleus, nuclei.Nucleus):
        return nucleus
    elif nucleus in nuclei.supported_nuclei:
        return nuclei.supported_nuclei[nucleus]
    else:
        raise ValueError(
            "`{varname}` specified is not recognised. Either provide a "
            "`nuclei.Nucleus instance, or one of the following\n:" +
            ", ".join([f"\"{n}\"" for n in nuclei.supported_nuclei])
        )


def process_value(
        value: Any, varname: str, regex: str, can_be_negative: bool
) -> Tuple[float, str]:
    errmsg = (
        f"`{varname}` should be a<POS>scalar, or a string satifying \"{regex}\""
    )
    if can_be_negative:
        errmsg = errmsg.replace("<POS>", " ")
    else:
        errmsg = errmsg.replace("<POS>", " positive ")

    if isinstance(value, (int, float)):
        if can_be_negative:
            return value, None
        else:
            if value > 0:
                return value, None
            else:
                raise ValueError(errmsg)

    if not isinstance(value, str):
        raise ValueError(errmsg)
    print(regex)
    match = re.match(regex, value, re.IGNORECASE)
    if match is None:
        raise ValueError(errmsg)
    else:
        value = float(match.group(1))
        unit = match.group(2).lower()
        return value, unit


def process_temperature(temperature: Any) -> float:
    temp, unit = process_value(
        temperature, "temperature", r"^(\d*\.?\d*)(C|K)$", False,
    )
    if unit is None or unit == "k":
        return temp
    elif unit == "c":
        return temp + 273.15


def process_field(field: Any):
    field, unit = process_value(field, "field", r"^(\d*\.?\d*)(T|MHz)$", False)
    if unit is None or unit == "t":
        return field
    elif unit == "mhz":
        return 2e6 * np.pi * field / nuclei.supported_nuclei["1H"].gamma


def process_sweep_width(
    sweep_width: Any, nucleus: nuclei.Nucleus, field: float,
) -> float:
    sweep_width, unit = process_value(
        sweep_width, "sweep_width", r"^(\d*\.?\d*)(Hz|ppm)$", False,
    )
    if unit is None or unit == "hz":
        return sweep_width
    elif unit == "ppm":
        return sweep_width * field * nucleus.gamma / (2e6 * np.pi)


def process_offset(
    offset: Any, nucleus: nuclei.Nucleus, field: float,
) -> float:
    offset, unit = process_value(
        offset, "offset", r"(-?\d*\.?\d*)(Hz|ppm)", True,
    )
    if unit is None or unit == "hz":
        return offset
    elif unit == "ppm":
        return offset * field * nucleus.gamma / (2e6 * np.pi)


def process_points(points: Any) -> int:
    if isinstance(points, int) and points > 0:
        return points
    else:
        raise ValueError("`points` should be a positive int.")


@dataclass
class Spin:
    nucleus: nuclei.Nucleus
    shift: float
    couplings: dict[int, float]
