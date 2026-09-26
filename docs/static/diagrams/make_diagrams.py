"""Generate the architecture diagrams as inline-SVG HTML fragments.

Run ``python docs/static/diagrams/make_diagrams.py`` after editing; the
``*.svg.html`` files are included into the architecture pages with
``.. raw:: html``. Styles live in ``docs/static/custom.css``.
"""
from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parent


def box(x, y, w, h, title, subs=(), cls="", href=None, plain=False, label=None):
    pill = "pillar" in cls.split()
    tcls = "title" + (" plain" if plain else "") + (" pillar-text" if pill else "")
    scls = "sub" + (" pillar-text" if pill else "")
    lines = len(subs)
    top = y + h / 2 - (lines * 14) / 2 + 4
    parts = [f'<rect class="box {cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="6" ry="6"/>',
             f'<text class="{tcls}" x="{x + w / 2}" y="{top}" text-anchor="middle">{escape(title)}</text>']
    for i, s in enumerate(subs):
        parts.append(f'<text class="{scls}" x="{x + w / 2}" y="{top + 16 + i * 14}" text-anchor="middle">{escape(s)}</text>')
    body = "\n".join(parts)
    if href:
        aria = escape(label or title)
        return f'<a href="{href}" aria-label="{aria}"><title>{aria}</title>\n{body}\n</a>'
    return body


def arrow(points, cls="", label=None, lx=None, ly=None, anchor="middle"):
    d = "M " + " L ".join(f"{x} {y}" for x, y in points)
    marker = "arrow-pillar" if "pillar" in cls else "arrow"
    out = f'<path class="edge {cls}" d="{d}" marker-end="url(#{marker})"/>'
    if label:
        lcls = "edge-label" + (" pillar" if "pillar" in cls else "")
        out += f'\n<text class="{lcls}" x="{lx}" y="{ly}" text-anchor="{anchor}">{escape(label)}</text>'
    return out


def svg(width, height, body, title, desc):
    defs = """<defs>
<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrowhead" d="M 0 0 L 10 5 L 0 10 z"/></marker>
<marker id="arrow-pillar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrowhead pillar" d="M 0 0 L 10 5 L 0 10 z"/></marker>
</defs>"""
    return (f'<div class="cdiag"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'role="img" aria-labelledby="t-{title[0]} d-{title[0]}">\n'
            f'<title id="t-{title[0]}">{escape(title[1])}</title><desc id="d-{title[0]}">{escape(desc)}</desc>\n'
            f'{defs}\n{body}\n</svg></div>\n')


# ---------------------------------------------------------------- layers
def layers():
    W = 900
    bx, bw, gap = 180, 705, 15
    bands = [
        ("Your code", "host Python", "user", [
            ("Model classes", ["fields · processes · hooks"], "../concepts/models.html", True),
            ("Configured objects", ["sweeps · sources · captures"], "../concepts/models.html#where-values-come-from", True),
            ("Experiment(...)", ["replications · window · seed"], "../concepts/experiments.html", True),
            ("Observed data", ["logs · histories · panels"], "inputs.html", True),
        ]),
        ("Public API", "stable surface", "", [
            ("cimba", ["the facade"], "packages.html#arch-pkg-cimba", False),
            ("cimba.inputs", ["sources · fit"], "packages.html#arch-pkg-inputs", False),
            ("cimba.analysis", ["statistics · checks"], "packages.html#arch-pkg-analysis", False),
            ("cimba.random", ["in-model draws"], "packages.html#arch-pkg-random", False),
            ("cimba.diagrams", ["structure"], "packages.html#arch-pkg-diagrams", False),
        ]),
        ("Model core", "pure Python", "", [
            ("modeling", ["the language"], "packages.html#arch-pkg-modeling", False),
            ("schema", ["ClassSchema · Assembly"], "packages.html#arch-pkg-schema", False),
            ("layout", ["records · trial image"], "packages.html#arch-pkg-layout", False),
            ("results", ["immutable outcomes"], "packages.html#arch-pkg-results", False),
        ]),
        ("Compile & run", "Numba · ctypes", "", [
            ("compiler", ["views · handles · per-class njit"], "packages.html#arch-pkg-compiler", False),
            ("experiments", ["design · rows · bind · run"], "packages.html#arch-pkg-experiments", False),
            ("engine", ["ctypes · ABI · symbols"], "packages.html#arch-pkg-engine", False),
        ]),
        ("Native runtime", "libcimba_py (C)", "native", [
            ("trial.c", ["lifecycle · spawn · capture"], "native.html#arch-native-trial", False),
            ("inputs.c", ["streams · series · exhaustion"], "native.html#arch-native-inputs", False),
            ("verbs.c · nbshim.c", ["entity & process verbs"], "native.html#arch-native-verbs", False),
            ("Cimba C engine", ["coroutines · events · entities"], "native.html#arch-native-engine", False),
        ]),
    ]
    pillar = {"cimba.inputs", "inputs.c"}
    y, bh, out = 10, 84, []
    for name, note, kind, boxes in bands:
        out.append(f'<rect class="band {kind}" x="10" y="{y}" width="{W - 20}" height="{bh}" rx="8" ry="8"/>')
        out.append(f'<text class="band-label" x="24" y="{y + 36}">{escape(name)}</text>')
        out.append(f'<text class="band-note" x="24" y="{y + 54}">{escape(note)}</text>')
        n = len(boxes)
        w = (bw - gap * (n - 1)) / n
        for i, (title, subs, href, plain) in enumerate(boxes):
            cls = "pillar" if title in pillar else ("user" if kind == "user" else ("native" if kind == "native" else ""))
            out.append(box(round(bx + i * (w + gap), 1), y + 13, round(w, 1), 58, title, subs, cls, href, plain))
        y += bh + 10
    out.append(f'<text class="band-note" x="{W - 20}" y="{y + 8}" text-anchor="end">Each layer builds on the layers below it. Click a box for details.</text>')
    return svg(W, y + 16, "\n".join(out), ("layers", "Cimba Python system layers"),
               "Five layers from user code to the native runtime. The input-modeling pillar, "
               "cimba.inputs and inputs.c, is highlighted in orange.")


# ---------------------------------------------------------------- pipeline
def pipeline():
    W, H = 900, 452
    xs = [15, 190, 365, 540, 715]
    w, h = 165, 66
    ys = [40, 190, 340]
    out = []
    out.append('<text class="band-label" x="15" y="22">Class path — compiled once per class</text>')
    out.append('<text class="band-label" x="15" y="172">Instance path — laid out per experiment</text>')
    out.append('<text class="band-label" x="15" y="322">Input path — rows per replication</text>')
    B = lambda c, r, t, s, cls="", href=None, plain=False: box(xs[c], ys[r], w, h, t, s, cls, href, plain)
    out += [
        B(0, 0, "Model classes", ["class Shop(cb.Model)"], "user", "../concepts/models.html", True),
        B(1, 0, "schema", ["ClassSchema.of(cls)"], "", "packages.html#arch-pkg-schema"),
        B(2, 0, "compiler", ["views · handles · njit"], "", "packages.html#arch-pkg-compiler"),
        B(3, 0, "CompiledClass", ["callbacks + dispatch table"], "", "pipeline.html#compile-classes-bind-instances"),
        B(0, 1, "Configured objects", ["shop = Shop(); shop.x = …"], "user", "../concepts/models.html#where-values-come-from", True),
        B(1, 1, "schema", ["Assembly.of(root)"], "", "packages.html#arch-pkg-schema"),
        B(2, 1, "layout", ["TrialImageLayout"], "", "packages.html#arch-pkg-layout"),
        B(3, 1, "experiments", ["design · seeds · bind · run"], "", "packages.html#arch-pkg-experiments"),
        B(4, 1, "libcimba_py", ["trial blocks · all cores"], "native", "native.html"),
        B(0, 2, "Observed data", ["arrays · panels"], "user", "inputs.html", True),
        B(1, 2, "cimba.inputs", ["sources: dist · trace · …"], "pillar", "inputs.html"),
        B(2, 2, "row generation", ["vectorized, per replication"], "pillar-soft", "inputs.html#where-values-come-from"),
        B(4, 2, "results → analysis", ["read through objects"], "", "../concepts/results.html"),
    ]
    mid = lambda r: ys[r] + h / 2
    out += [
        arrow([(xs[0] + w, mid(0)), (xs[1] - 2, mid(0))]),
        arrow([(xs[1] + w, mid(0)), (xs[2] - 2, mid(0))]),
        arrow([(xs[2] + w, mid(0)), (xs[3] - 2, mid(0))]),
        arrow([(xs[0] + w, mid(1)), (xs[1] - 2, mid(1))]),
        arrow([(xs[1] + w, mid(1)), (xs[2] - 2, mid(1))]),
        arrow([(xs[2] + w, mid(1)), (xs[3] - 2, mid(1))]),
        arrow([(xs[3] + w, mid(1)), (xs[4] - 2, mid(1))]),
        arrow([(xs[3] + w / 2, ys[0] + h), (xs[3] + w / 2, ys[1] - 2)], label="callback addresses", lx=xs[3] + w / 2 + 8, ly=(ys[0] + h + ys[1]) / 2 + 4, anchor="start"),
        arrow([(xs[1] + w / 2, ys[0] + h), (xs[2] + 30, ys[1] - 2)], "dashed", label="RecordLayout", lx=xs[1] + w / 2 + 52, ly=(ys[0] + h + ys[1]) / 2 - 2, anchor="start"),
        arrow([(xs[0] + w, mid(2)), (xs[1] - 2, mid(2))], "pillar"),
        arrow([(xs[1] + w, mid(2)), (xs[2] - 2, mid(2))], "pillar"),
        arrow([(xs[2] + w, mid(2)), (xs[3] + 40, ys[1] + h + 2)], "pillar", label="rows (shared by all points)", lx=xs[3] + 48, ly=ys[2] + 22, anchor="start"),
        arrow([(xs[4] + w / 2, ys[1] + h), (xs[4] + w / 2, ys[2] - 2)], label="outputs · captures · cursors", lx=xs[4] + w / 2 - 6, ly=(ys[1] + h + ys[2]) / 2 + 4, anchor="end"),
    ]
    out.append(f'<text class="band-note" x="{W - 15}" y="{H - 8}" text-anchor="end">Compiled code never sees instances or sources; structure and data never trigger recompilation.</text>')
    return svg(W, H, "\n".join(out), ("pipeline", "From classes and objects to trials"),
               "Three paths converge in experiments: model classes are compiled via schema and compiler; "
               "configured objects are laid out via schema and layout; observed data becomes sources whose rows "
               "are generated per replication.")


# ---------------------------------------------------------------- input pillar
def pillar():
    W, H = 900, 470
    out = []
    sx, sw, sh = 15, 250, 76
    sources = [
        ("inputs.dist.*", ["distribution: drawn inside the trial,", "in C, on the input's own stream"], "../reference/inputs.html#distributions-inputs-dist"),
        ("inputs.trace(data)", ["recorded: replayed exactly as recorded,", "shared read-only by all trials"], "../reference/inputs.html#recorded-data"),
        ("inputs.bootstrap.*", ["resample: rows generated in Python", "once per replication"], "../reference/inputs.html#bootstrap-resampling-inputs-bootstrap"),
        ("inputs.fitted.*", ["fitted model: rows generated in Python", "once per replication"], "../reference/inputs.html#fitted-time-series-models-inputs-fitted"),
    ]
    ys = [40, 135, 230, 325]
    out.append('<text class="band-label" x="15" y="24">Four interchangeable sources</text>')
    for (t, s, href), y in zip(sources, ys):
        out.append(box(sx, y, sw, sh, t, s, "pillar-soft", href))
    # row buffer
    rx, ry, rw, rh = 305, 135, 190, 266
    out.append(box(rx, ry, rw, rh, "row buffers", ["read-only memory", "", "one row per", "(replication, source)", "", "shared by every", "design point (CRN)", "", "regenerated 2× longer", "on exhaustion"], "", "inputs.html#where-values-come-from"))
    # slot
    px, py, pw, ph = 540, 110, 190, 210
    out.append(box(px, py, pw, ph, "inputs.c", ["one InputSlot per input field", "in the trial record", "", "kind · policy · parameters", "stream state · data · cursor", "series step · origin · history", "", "own random stream", "exhaustion policy"], "pillar", "native.html#arch-native-inputs"))
    # model code
    mx, mw = 770, 118
    out.append(box(mx, 150, mw, 130, "model code", ["x.next()", "d.now()  d.at(t)", "", "identical for", "every source"], "", "../concepts/inputs.html#declaring-inputs"))
    # arrows
    out.append(arrow([(sx + sw, ys[0] + sh / 2), (px - 2, py + 40)], "pillar", label="parameters", lx=420, ly=ys[0] + 42))
    for y in ys[1:]:
        out.append(arrow([(sx + sw, y + sh / 2), (rx - 2, y + sh / 2)], "pillar"))
    out.append(arrow([(rx + rw, 250), (px - 2, 250)], "pillar", label="data*", lx=(rx + rw + px) / 2, ly=242))
    out.append(arrow([(px + pw, 215), (mx - 2, 215)], "", label="1 C call", lx=(px + pw + mx) / 2, ly=205))
    # exhaustion note
    out.append(box(540, 345, 190, 56, "on exhaustion", ["fail · wrap · end_trial · extend"], "", "inputs.html#exhaustion-and-extension", True))
    out.append(arrow([(px + pw / 2, py + ph), (px + pw / 2, 343)], "dashed"))
    out.append(f'<text class="band-note" x="{W - 12}" y="{H - 10}" text-anchor="end">Swap or sweep sources without touching the model or recompiling. Click a box for details.</text>')
    return svg(W, H, "\n".join(out), ("pillar", "The input-modeling pillar"),
               "Distribution, trace, bootstrap and fitted sources all feed the same input field. Distributions "
               "are drawn in the trial; traces are replayed; bootstrap and fitted rows are generated once per "
               "replication and shared by every design point. inputs.c gives each input its own random stream "
               "and applies its exhaustion policy.")


for name, text in (("layers", layers()), ("pipeline", pipeline()), ("input_pillar", pillar())):
    (OUT / f"{name}.svg.html").write_text(text)
    print(name, len(text))
