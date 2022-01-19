# experimental.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Mon 17 Jan 2022 19:42:09 GMT

from typing import Iterable, Union
from nmr_sims import _sanity, nuclei


class Parameters:
    def __init__(
        self,
        channels: Iterable[nuclei.Nucleus],
        sweep_widths: Iterable[Union[float, int, str]],
        points: Iterable[int],
        offsets: Union[Iterable[Union[float, int, str]], None] = None,
    ) -> None:
        """Sweep widths in Hz. Offset in Hz. Will make more sophisticated at some
        point."""
        self.channels = []
        for channel in channels:
            self.channels.append(_sanity.process_nucleus(channel, "Item in channels"))
        self.sweep_widths = list(sweep_widths)
        self.points = list(points)
        if offsets is None:
            self.offsets = [0 for _ in range(len(self.sweep_widths))]
        else:
            self.offsets = list(offsets)
