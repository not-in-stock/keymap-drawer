"""Module containing lower-level SVG drawing utils, to be used as a mixin."""

import re
import string
from html import escape
from io import StringIO
from textwrap import TextWrapper
from typing import Literal, Sequence

from keymap_drawer.config import DrawConfig
from keymap_drawer.draw.glyph import GlyphMixin, LegendSegment
from keymap_drawer.physical_layout import Point

LegendType = Literal["tap", "hold", "shifted", "left", "right", "tl", "tr", "bl", "br"]


class UtilsMixin(GlyphMixin):
    """Mixin that adds low-level SVG drawing methods for KeymapDrawer."""

    # initialized in KeymapDrawer
    cfg: DrawConfig
    layer_names: set[str]
    out: StringIO

    @staticmethod
    def _str_to_id(val: str) -> str:
        if not val:
            return "o_o"
        val = val.replace(" ", "-")
        while val[0] not in string.ascii_letters:
            val = val[1:]
            if not val:
                return "x_x"
        allowed = string.ascii_letters + string.digits + "-_:."
        return "".join([c for c in val if c in allowed])

    @staticmethod
    def _to_class_str(classes: Sequence[str]) -> str:
        return (' class="' + " ".join(c for c in classes if c) + '"') if classes else ""

    def _split_text(self, text: str, truncate: int = 0, line_width: int = 0) -> list[str]:
        if self.legend_has_glyphs(text):
            return [text]

        # do not split on double spaces, but do split on single
        lines = [word.replace("\x00", " ") for word in text.replace("  ", "\x00").split()]

        # wrap on word boundaries if a line is too long
        if line_width > 0 and len(lines) < truncate:
            tw = TextWrapper(width=line_width, break_long_words=False, break_on_hyphens=False)

            wrapped: list[str] = []
            for i, line in enumerate(lines):
                if len(line) > line_width:
                    wrapped_line = tw._wrap_chunks(re.split(r"(?<!^.)\b", line))  # pylint: disable=protected-access

                    # if we are going to exceed the max line limit, give up here and do not modify lines
                    new_total_lines = len(wrapped) + len(wrapped_line) - 1 + len(lines) - i
                    if (diff := new_total_lines - truncate) > 0:
                        if diff < len(wrapped_line):  # salvage part of this line as much as we can
                            wrapped += wrapped_line[: -diff - 1] + ["".join(wrapped_line[-diff - 1 :])]
                        else:
                            wrapped.append(line)
                        wrapped += lines[i + 1 :]
                        break
                    wrapped += wrapped_line
                else:
                    wrapped.append(line)
            lines = wrapped

        # truncate number of lines if requested
        if truncate and len(lines) > truncate:
            lines = lines[: truncate - 1] + ["…"]
        return lines

    def _draw_rect(self, p: Point, dims: Point, radii: Point, classes: Sequence[str]) -> None:
        self.out.write(
            f'<rect rx="{round(radii.x)}" ry="{round(radii.y)}"'
            f' x="{round(p.x - dims.x / 2)}" y="{round(p.y - dims.y / 2)}" '
            f'width="{round(dims.x)}" height="{round(dims.y)}"{self._to_class_str(classes)}/>\n'
        )

    def _draw_key(self, dims: Point, classes: Sequence[str]) -> None:
        if self.cfg.draw_key_sides:
            # draw side rectangle
            self._draw_rect(
                Point(0.0, 0.0),
                dims,
                Point(self.cfg.key_rx, self.cfg.key_ry),
                classes=[*classes, "side"],
            )
            # draw internal rectangle
            self._draw_rect(
                Point(-self.cfg.key_side_pars.rel_x, -self.cfg.key_side_pars.rel_y),
                dims - Point(self.cfg.key_side_pars.rel_w, self.cfg.key_side_pars.rel_h),
                Point(self.cfg.key_side_pars.rx, self.cfg.key_side_pars.ry),
                classes=classes,
            )
        else:
            # default key style
            self._draw_rect(
                Point(0.0, 0.0),
                dims,
                Point(self.cfg.key_rx, self.cfg.key_ry),
                classes=classes,
            )

    def _get_scaling(self, width: int) -> str:
        if not self.cfg.shrink_wide_legends or width <= self.cfg.shrink_wide_legends:
            return ""
        return f' style="font-size: {max(60.0, 100 * self.cfg.shrink_wide_legends / width):.0f}%"'

    def _truncate_word(self, word: str) -> str:
        if not self.cfg.shrink_wide_legends or len(word) <= (limit := int(1.7 * self.cfg.shrink_wide_legends)):
            return word
        return word[: limit - 1] + "…"

    def _draw_text(self, p: Point, word: str, classes: Sequence[str]) -> None:
        if not word:
            return
        word = self._truncate_word(word)
        self.out.write(f'<text x="{round(p.x)}" y="{round(p.y)}"{self._to_class_str(classes)}>')
        self.out.write(
            f"<tspan{scale}>{escape(word)}</tspan>" if (scale := self._get_scaling(len(word))) else escape(word)
        )
        self.out.write("</text>\n")

    def _draw_textblock(self, p: Point, words: Sequence[str], classes: Sequence[str], shift: float = 0) -> None:
        words = [self._truncate_word(word) for word in words]
        self.out.write(f'<text x="{round(p.x)}" y="{round(p.y)}"{self._to_class_str(classes)}>\n')
        dy_0 = (len(words) - 1) * (self.cfg.line_spacing * (1 + shift / 2) / 2)
        scaling = self._get_scaling(max(len(w) for w in words))
        self.out.write(f'<tspan x="{round(p.x)}" dy="-{round(dy_0, 2)}em"{scaling}>{escape(words[0])}</tspan>')
        for word in words[1:]:
            self.out.write(f'<tspan x="{round(p.x)}" dy="{self.cfg.line_spacing}em"{scaling}>{escape(word)}</tspan>')
        self.out.write("\n</text>\n")

    def _draw_glyph(self, p: Point, name: str, legend_type: LegendType, classes: Sequence[str]) -> None:
        width, height, d_x, d_y = self.get_glyph_dimensions(name, legend_type)

        classes = [*classes, "glyph", name]
        self.out.write(
            f'<use href="#{name}" xlink:href="#{name}" x="{round(p.x - d_x)}" y="{round(p.y - d_y)}" '
            f'height="{height}" width="{width}"{self._to_class_str(classes)}/>\n'
        )

    def _draw_segments(
        self,
        p: Point,
        segments: list[LegendSegment],
        legend_type: LegendType,
        classes: Sequence[str],
        debug: bool = False,
    ) -> None:
        """Draw a sequence of text and glyph segments, centered at point p."""
        # Get settings from config
        font_size = self.cfg.composed_glyph_text_size
        char_width = font_size * 0.5  # approximate character width for monospace
        glyph_glyph_gap = self.cfg.adjacent_icon_gap  # negative to compensate for glyph internal padding
        text_gap = 2  # gap when text is involved

        # Calculate width of each segment and gaps
        segment_widths: list[float] = []
        segment_gaps: list[float] = []  # gap after each segment (except last)
        glyph_height = 0.0

        for i, seg in enumerate(segments):
            if seg.is_glyph:
                width, height, _, _ = self.get_glyph_dimensions(seg.content, legend_type)
                segment_widths.append(width)
                glyph_height = max(glyph_height, height)
            else:
                segment_widths.append(len(seg.content) * char_width)

            # Calculate gap after this segment
            if i < len(segments) - 1:
                next_seg = segments[i + 1]
                # No gap between two glyphs, otherwise use text_gap
                if seg.is_glyph and next_seg.is_glyph:
                    segment_gaps.append(glyph_glyph_gap)
                else:
                    segment_gaps.append(text_gap)

        # Total width including gaps
        total_width = sum(segment_widths) + sum(segment_gaps)

        # Starting x position (left edge of first segment)
        current_x = p.x - total_width / 2

        # Get glyph dimensions for vertical positioning (use first glyph or default)
        if glyph_height == 0:
            _, glyph_height, _, _ = self.get_glyph_dimensions(segments[0].content, legend_type)

        # Draw each segment
        for i, seg in enumerate(segments):
            seg_width = segment_widths[i]

            if seg.is_glyph:
                # Draw glyph
                width, height, _, _ = self.get_glyph_dimensions(seg.content, legend_type)
                glyph_classes = [*classes, "glyph", seg.content]
                glyph_y = p.y - height / 2

                # Debug: draw glyph bounding box
                if debug:
                    self.out.write(
                        f'<rect x="{round(current_x)}" y="{round(glyph_y)}" '
                        f'width="{width}" height="{height}" '
                        f'fill="none" stroke="red" stroke-width="1"/>\n'
                    )

                self.out.write(
                    f'<use href="#{seg.content}" xlink:href="#{seg.content}" '
                    f'x="{round(current_x)}" y="{round(glyph_y)}" '
                    f'height="{height}" width="{width}"{self._to_class_str(glyph_classes)}/>\n'
                )
            else:
                # Draw text
                text_classes = [*classes, "glyph-text"]
                text_x = current_x + seg_width / 2  # center of text segment
                text_height = font_size

                # Debug: draw text bounding box
                if debug:
                    self.out.write(
                        f'<rect x="{round(current_x)}" y="{round(p.y - text_height / 2)}" '
                        f'width="{seg_width}" height="{text_height}" '
                        f'fill="none" stroke="blue" stroke-width="1"/>\n'
                    )

                self.out.write(
                    f'<text x="{round(text_x)}" y="{round(p.y)}" '
                    f'text-anchor="middle" dominant-baseline="middle" font-size="{font_size:.1f}px"'
                    f'{self._to_class_str(text_classes)}>{escape(seg.content)}</text>\n'
                )

            # Move to next segment position
            current_x += seg_width
            if i < len(segment_gaps):
                current_x += segment_gaps[i]

    def _draw_legend(
        self, p: Point, words: Sequence[str], classes: Sequence[str], legend_type: LegendType, shift: float = 0
    ) -> None:
        if not words:
            return

        is_layer = self.cfg.style_layer_activators and (layer_name := " ".join(words)) in self.layer_names

        classes = [*classes, legend_type]
        if is_layer:
            classes.append("layer-activator")

        if len(words) == 1:
            if segments := self.get_validated_segments(words[0]):
                # Check if it's a single glyph (use original centered drawing)
                if len(segments) == 1 and segments[0].is_glyph:
                    self._draw_glyph(p, segments[0].content, legend_type, classes)
                else:
                    self._draw_segments(p, segments, legend_type, classes)
                return

        if is_layer:
            self.out.write(f'<a href="#{self._str_to_id(layer_name)}">\n')

        if len(words) == 1:
            self._draw_text(p, words[0], classes)
        else:
            self._draw_textblock(p, words, classes, shift)

        if is_layer:
            self.out.write("</a>")
