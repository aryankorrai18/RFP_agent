"""Plain SVG charts (no plotting library): grouped bars with 95% whiskers, and lines. Drawn on a white card so they read the
same in a light or dark viewer."""

from __future__ import annotations

from html import escape

PALETTE = ("#64748b", "#d97706", "#0f766e", "#2563eb", "#7c3aed", "#be123c")
INK, MUTED, GRID = "#17201f", "#5a6866", "#e3e8e3"


def _frame(width: int, height: int, title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" '
        f'aria-label="{escape(title)}" font-family="system-ui, -apple-system, Segoe UI, Roboto, sans-serif">',
        f'<rect width="{width}" height="{height}" rx="14" fill="#ffffff" stroke="#d8ddd8"/>',
        f'<text x="24" y="34" font-size="18" font-weight="700" fill="{INK}">{escape(title)}</text>',
        f'<text x="24" y="54" font-size="12.5" fill="{MUTED}">{escape(subtitle)}</text>',
    ]


def bar_chart(title: str, subtitle: str, groups: list[str], series: dict[str, list[tuple[float, float, float]]],
              width: int = 920, height: int = 380, y_label: str = "share of deals") -> str:
    """`series[name][i]` is (value, low, high) for group i, all in 0..1."""
    left, right, top, bottom = 56, 20, 78, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    out = _frame(width, height, title, subtitle)
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        y = top + plot_h * (1 - tick)
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="{GRID}"/>')
        out.append(f'<text x="{left - 8}" y="{y + 4:.1f}" font-size="11" text-anchor="end" fill="{MUTED}">{int(tick * 100)}%</text>')
    out.append(f'<text x="14" y="{top + plot_h / 2:.0f}" font-size="11" fill="{MUTED}" transform="rotate(-90 14 {top + plot_h / 2:.0f})" '
               f'text-anchor="middle">{escape(y_label)}</text>')
    names = list(series)
    group_w = plot_w / max(1, len(groups))
    bar_w = min(46.0, (group_w - 18) / max(1, len(names)))
    for gi, group in enumerate(groups):
        x0 = left + gi * group_w + (group_w - bar_w * len(names)) / 2
        for si, name in enumerate(names):
            value, low, high = series[name][gi]
            x = x0 + si * bar_w
            y = top + plot_h * (1 - value)
            out.append(f'<rect x="{x + 2:.1f}" y="{y:.1f}" width="{bar_w - 4:.1f}" height="{max(0.0, plot_h * value):.1f}" '
                       f'fill="{PALETTE[si % len(PALETTE)]}" rx="3"><title>{escape(name)}: {value:.0%} ({low:.0%} to {high:.0%})</title></rect>')
            cx = x + bar_w / 2
            out.append(f'<line x1="{cx:.1f}" y1="{top + plot_h * (1 - high):.1f}" x2="{cx:.1f}" y2="{top + plot_h * (1 - low):.1f}" stroke="{INK}" stroke-width="1.4"/>')
            for edge in (high, low):
                ey = top + plot_h * (1 - edge)
                out.append(f'<line x1="{cx - 4:.1f}" y1="{ey:.1f}" x2="{cx + 4:.1f}" y2="{ey:.1f}" stroke="{INK}" stroke-width="1.4"/>')
            out.append(f'<text x="{cx:.1f}" y="{top + plot_h * (1 - high) - 8:.1f}" font-size="10.5" text-anchor="middle" fill="{INK}">{value:.0%}</text>')
        out.append(f'<text x="{left + gi * group_w + group_w / 2:.1f}" y="{top + plot_h + 20}" font-size="12" text-anchor="middle" fill="{INK}">{escape(group)}</text>')
    lx = left
    for si, name in enumerate(names):
        out.append(f'<rect x="{lx}" y="{height - 30}" width="12" height="12" rx="3" fill="{PALETTE[si % len(PALETTE)]}"/>')
        out.append(f'<text x="{lx + 18}" y="{height - 20}" font-size="12" fill="{INK}">{escape(name)}</text>')
        lx += 28 + 7.2 * len(name)
    out.append("</svg>")
    return "\n".join(line for line in out if line)


def line_chart(title: str, subtitle: str, xs: list[float], series: dict[str, list[tuple[float, float, float]]],
               width: int = 920, height: int = 360, x_label: str = "closed deals in memory", y_label: str = "share of deals") -> str:
    left, right, top, bottom = 56, 24, 78, 72
    plot_w, plot_h = width - left - right, height - top - bottom
    out = _frame(width, height, title, subtitle)
    lo_x, hi_x = min(xs), max(xs)
    px = lambda x: left + plot_w * (x - lo_x) / ((hi_x - lo_x) or 1)  # noqa: E731
    py = lambda v: top + plot_h * (1 - v)  # noqa: E731
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{left}" y1="{py(tick):.1f}" x2="{width - right}" y2="{py(tick):.1f}" stroke="{GRID}"/>')
        out.append(f'<text x="{left - 8}" y="{py(tick) + 4:.1f}" font-size="11" text-anchor="end" fill="{MUTED}">{int(tick * 100)}%</text>')
    for x in xs:
        out.append(f'<text x="{px(x):.1f}" y="{top + plot_h + 18}" font-size="11.5" text-anchor="middle" fill="{INK}">{int(x)}</text>')
    out.append(f'<text x="{left + plot_w / 2:.0f}" y="{top + plot_h + 38}" font-size="11.5" text-anchor="middle" fill="{MUTED}">{escape(x_label)}</text>')
    out.append(f'<text x="14" y="{top + plot_h / 2:.0f}" font-size="11" fill="{MUTED}" transform="rotate(-90 14 {top + plot_h / 2:.0f})" text-anchor="middle">{escape(y_label)}</text>')
    for si, (name, points) in enumerate(series.items()):
        colour = PALETTE[si % len(PALETTE)]
        band = [(px(x), py(hi)) for x, (_v, _lo, hi) in zip(xs, points, strict=True)] + [(px(x), py(lo)) for x, (_v, lo, _hi) in reversed(list(zip(xs, points, strict=True)))]
        out.append(f'<polygon points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in band)}" fill="{colour}" opacity="0.13"/>')
        out.append(f'<polyline fill="none" stroke="{colour}" stroke-width="2.4" points="{" ".join(f"{px(x):.1f},{py(v):.1f}" for x, (v, _l, _h) in zip(xs, points, strict=True))}"/>')
        for x, (v, lo, hi) in zip(xs, points, strict=True):
            out.append(f'<circle cx="{px(x):.1f}" cy="{py(v):.1f}" r="3.6" fill="{colour}"><title>{escape(name)} at {int(x)}: {v:.0%} ({lo:.0%} to {hi:.0%})</title></circle>')
    lx = left
    for si, name in enumerate(series):
        out.append(f'<rect x="{lx}" y="{height - 26}" width="12" height="12" rx="3" fill="{PALETTE[si % len(PALETTE)]}"/>')
        out.append(f'<text x="{lx + 18}" y="{height - 16}" font-size="12" fill="{INK}">{escape(name)}</text>')
        lx += 28 + 7.2 * len(name)
    out.append("</svg>")
    return "\n".join(out)
