# experimental.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Thu 13 Jan 2022 12:08:19 GMT

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Union
from nmr_sims import _sanity, nuclei


class Experimental:
    boltzmann = 1.380649e-23
    hbar = 1.054571817e-34

    def __init__(
        self, channels: Union[dict, None] = None,
        temperature: Union[int, float, str] = "298K",
        field: Union[int, float, str] = "500MHz",

    ) -> None:
        self.temperature = _sanity.process_temperature(temperature)
        self.field = _sanity.process_field(field)

        if channels is None:
            channels = {
                1: {
                    "nucleus": "1H",
                    "sweep_width": "10ppm",
                    "offset": "4ppm",
                    "points": 2048,
                },
            }
        _sanity.check_dict_with_int_keys(channels, "channels", consecutive=True)
        self.channels = OrderedDict()
        for key, channel in channels.items():
            self.channels[key] = Channel(*_sanity.process_channel(channel, self.field))

    @property
    def inverse_temperature(self):
        return self.hbar / (self.boltzmann * self.temperature)

    @property
    def boltzmann_factor(self) -> Iterable[float]:
        return [self.inverse_temperature * channel.nucleus.gamma * self.field
                for channel in self.channels.values()]



@dataclass
class Channel:
    nucleus: nuclei.Nucleus
    sweep_width: float
    offset: float
    points: int
