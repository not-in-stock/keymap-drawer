"""
Module containing class and methods to help with fetching
and drawing SVG glyphs.
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from random import random
from time import sleep
from typing import Iterable
from urllib.error import HTTPError
from urllib.request import urlopen

from platformdirs import user_cache_dir

from keymap_drawer.config import DrawConfig
from keymap_drawer.keymap import KeymapData, LayoutKey

logger = logging.getLogger(__name__)


@dataclass
class LegendSegment:
    """A segment of a legend - either text or a glyph."""

    content: str
    is_glyph: bool

    @property
    def glyph_name(self) -> str | None:
        """Return glyph name if this is a glyph segment."""
        return self.content if self.is_glyph else None

FETCH_WORKERS = 8
FETCH_TIMEOUT = 10
N_RETRY = 5
CACHE_GLYPHS_PATH = Path(user_cache_dir("keymap-drawer", False)) / "glyphs"


class GlyphMixin:
    """Mixin that handles SVG glyphs for KeymapDrawer."""

    # Pattern to find all glyphs in a string: $$glyph_name$$
    _glyph_pattern_re = re.compile(r"\$\$(?P<glyph>.*?)\$\$")
    _view_box_dimensions_re = re.compile(
        r'<svg.*viewbox="(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)".*>',
        flags=re.IGNORECASE | re.ASCII | re.DOTALL,
    )
    _scrub_dims_re = re.compile(r' (width|height)=".*?"')

    # initialized in KeymapDrawer
    cfg: DrawConfig
    keymap: KeymapData

    @classmethod
    def parse_legend_segments(cls, legend: str) -> list[LegendSegment]:
        """
        Parse a legend string into segments of text and glyphs.
        Example: "hello$$mdi:icon1$$$$mdi:icon2$$world"
                 -> [Text("hello"), Glyph("mdi:icon1"), Glyph("mdi:icon2"), Text("world")]
        """
        segments: list[LegendSegment] = []
        last_end = 0

        for match in cls._glyph_pattern_re.finditer(legend):
            # Add text before this glyph (if any)
            if match.start() > last_end:
                text = legend[last_end : match.start()]
                if text:
                    segments.append(LegendSegment(content=text, is_glyph=False))

            # Add the glyph
            glyph_name = match.group("glyph")
            segments.append(LegendSegment(content=glyph_name, is_glyph=True))
            last_end = match.end()

        # Add remaining text after last glyph (if any)
        if last_end < len(legend):
            text = legend[last_end:]
            if text:
                segments.append(LegendSegment(content=text, is_glyph=False))

        return segments

    @classmethod
    def legend_has_glyphs(cls, legend: str) -> bool:
        """Check if a legend contains any glyphs."""
        return bool(cls._glyph_pattern_re.search(legend))

    @classmethod
    def extract_glyph_names(cls, legend: str) -> set[str]:
        """Extract all glyph names from a legend string."""
        return {match.group("glyph") for match in cls._glyph_pattern_re.finditer(legend)}

    def init_glyphs(self) -> None:
        """Preprocess all glyphs in the keymap to get their name to SVG mapping."""

        def find_key_glyph_names(key: LayoutKey) -> set[str]:
            names: set[str] = set()
            for field in (key.tap, key.hold, key.shifted, key.left, key.right, key.tl, key.tr, key.bl, key.br):
                names |= self.extract_glyph_names(field)
            return names

        # find all named glyphs in the keymap
        names = set()
        for layer in self.keymap.layers.values():
            for key in layer:
                names |= find_key_glyph_names(key)
        for combo in self.keymap.combos:
            names |= find_key_glyph_names(combo.key)

        # get the ones defined in draw_config.glyphs
        self.name_to_svg = {name: glyph for name in names if (glyph := self.cfg.glyphs.get(name))}
        logger.debug("found glyphs %s in draw_config.glyphs", list(self.name_to_svg))
        rest = names - set(self.name_to_svg)

        # try to fetch the rest using draw_config.glyph_urls
        if rest:
            self.name_to_svg |= self._fetch_glyphs(rest)
        if rest := rest - set(self.name_to_svg):
            raise ValueError(
                f'Glyphs "{rest}" are not defined in draw_config.glyphs or fetchable using draw_config.glyph_urls'
            )

        for name, svg in self.name_to_svg.items():
            if not self._view_box_dimensions_re.match(svg):
                raise ValueError(f'Glyph definition for "{name}" does not have the required "viewbox" property')

    def _fetch_glyphs(self, names: Iterable[str]) -> dict[str, str]:
        names = list(names)
        urls = []
        for name in names:
            if ":" in name:  # templated source:ID format
                source, glyph_id = name.split(":", maxsplit=1)
                if templated_url := self.cfg.glyph_urls.get(source):
                    if source in ("phosphor", "fa"):  # special case to handle variants
                        assert "/" in glyph_id, "phosphor/fa glyphs should be in `$$<source>:<type>/<id>$$` format"
                        sub_type, sub_id = glyph_id.split("/", maxsplit=1)
                        sub_type = sub_type.lower()
                        glyph_id = f"{sub_type}/{sub_id}"
                        if source == "phosphor" and sub_type != "regular":
                            glyph_id += f"-{sub_type}"
                    urls.append(templated_url.format(glyph_id))
            if url := self.cfg.glyph_urls.get(name):  # source only
                urls.append(url)

        with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as p:
            fetch_fn = partial(_fetch_svg_url, use_local_cache=self.cfg.use_local_cache)
            return dict(zip(names, p.map(fetch_fn, names, urls, timeout=N_RETRY * (FETCH_TIMEOUT + 1))))

    def get_validated_segments(self, legend: str) -> list[LegendSegment] | None:
        """
        Parse legend and validate that all glyphs exist.
        Returns segments if legend contains glyphs and all are valid, None otherwise.
        """
        if not self.legend_has_glyphs(legend):
            return None
        segments = self.parse_legend_segments(legend)
        # Validate all glyphs exist
        for seg in segments:
            if seg.is_glyph and seg.content not in self.name_to_svg:
                return None
        return segments

    def get_glyph_defs(self) -> str:
        """Return an SVG defs block with all glyph SVG definitions to be referred to later on."""
        if not self.name_to_svg:
            return ""

        defs = "<defs>/* start glyphs */\n"
        for name, svg in sorted(self.name_to_svg.items()):
            defs += f'<svg id="{name}">\n'
            defs += self._scrub_dims_re.sub("", svg)
            defs += "\n</svg>\n"
        defs += "</defs>/* end glyphs */\n"
        return defs

    def get_glyph_dimensions(self, name: str, legend_type: str) -> tuple[float, float, float, float]:
        """Given a glyph name, calculate and return its width, height and y-offset for drawing."""
        view_box = self._view_box_dimensions_re.match(self.name_to_svg[name])
        assert view_box is not None
        _, _, w, h = (float(v) for v in view_box.groups())

        # set dimensions and offsets from center
        match legend_type:
            case "tap":
                height = self.cfg.glyph_tap_size
                width = w * height / h
                d_x = 0.5 * width
                d_y = 0.5 * height
            case "hold":
                height = self.cfg.glyph_hold_size
                width = w * height / h
                d_x = 0.5 * width
                d_y = height
            case "shifted":
                height = self.cfg.glyph_shifted_size
                width = w * height / h
                d_x = 0.5 * width
                d_y = 0
            case "left":
                height = self.cfg.glyph_shifted_size
                width = w * height / h
                d_x = 0
                d_y = 0.5 * height
            case "right":
                height = self.cfg.glyph_shifted_size
                width = w * height / h
                d_x = width
                d_y = 0.5 * height
            case "tl":  # top-left corner
                height = self.cfg.glyph_shifted_size
                width = w * height / h
                d_x = 0
                d_y = 0
            case "tr":  # top-right corner
                height = self.cfg.glyph_shifted_size
                width = w * height / h
                d_x = width
                d_y = 0
            case "bl":  # bottom-left corner
                height = self.cfg.glyph_shifted_size
                width = w * height / h
                d_x = 0
                d_y = height
            case "br":  # bottom-right corner
                height = self.cfg.glyph_shifted_size
                width = w * height / h
                d_x = width
                d_y = height
            case _:
                raise ValueError("Unsupported legend_type for glyph")

        return width, height, d_x, d_y


@lru_cache(maxsize=128)
def _fetch_svg_url(name: str, url: str, use_local_cache: bool = False) -> str:
    """Get an SVG glyph definition from url, using the local cache for reading and writing if enabled."""
    cache_path = CACHE_GLYPHS_PATH / f"{name.replace('/', '@')}.svg"
    if use_local_cache and cache_path.is_file():
        logger.debug('found glyph "%s" in local cache', name)
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()

    logger.debug('fetching glyph "%s" from %s', name, url)
    try:
        for _ in range(N_RETRY):
            try:
                sleep(0.2 * random())
                with urlopen(url, timeout=FETCH_TIMEOUT) as f:
                    content = f.read().decode("utf-8")
                break
            except TimeoutError:
                logger.warning("request timed out while trying to fetch SVG from %s", url)
        else:
            raise RuntimeError(f"Failed to fetch SVG in {N_RETRY} tries")
        if use_local_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f_out:
                f_out.write(content)
        return content
    except (HTTPError, RuntimeError) as exc:
        raise RuntimeError(f'Could not fetch SVG from URL "{url}"') from exc
