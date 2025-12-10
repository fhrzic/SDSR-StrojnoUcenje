import numpy as np
from ipywidgets import IntSlider, Button, HBox, VBox, Output, Layout, Label
from IPython.display import display, HTML

def create_convtranspose1d_widget():
    """
    Interaktivni widget za 1D transposed (de)konvoluciju.

    Parametri:
      - Duljina ulaza
      - Širina jezgre
      - Korak (stride)
      - Popuna (padding)
      - Output padding

    Račun:
      out_len = (in_len - 1)*stride - 2*padding + kernel_size + output_padding

    Vizualizacija:
      - Gornji okvir:
          * istaknuti trenutni ulazni element x[n]
          * jezgru w
          * doprinos ovog elementa x[n] na cijeli izlaz
      - Donji okvir:
          * cijeli izlaz y (zbroj doprinosa svih ulaznih elemenata)
      - Prev/Next se kreću po ulaznim indeksima n.
    """

    # -------------------------
    # Kontrole
    # -------------------------
    _in_len = IntSlider(
        description="Duljina ulaza",
        min=2, max=20, step=1, value=5,
        continuous_update=False,
        layout=Layout(width="260px"),
        style={'description_width': 'initial'}
    )
    _k_size = IntSlider(
        description="Širina jezgre",
        min=1, max=9, step=1, value=3,
        continuous_update=False,
        layout=Layout(width="260px"),
        style={'description_width': 'initial'}
    )
    _stride = IntSlider(
        description="Korak (stride)",
        min=1, max=5, step=1, value=2,
        continuous_update=False,
        layout=Layout(width="260px"),
        style={'description_width': 'initial'}
    )
    _padding = IntSlider(
        description="Popuna (padding)",
        min=0, max=5, step=1, value=1,
        continuous_update=False,
        layout=Layout(width="260px"),
        style={'description_width': 'initial'}
    )
    _out_pad = IntSlider(
        description="Output padding",
        min=0, max=5, step=1, value=0,
        continuous_update=False,
        layout=Layout(width="260px"),
        style={'description_width': 'initial'}
    )

    _btn_generate = Button(
        description="Generiraj slučajne podatke",
        button_style="info",
        layout=Layout(width="260px")
    )
    _btn_prev = Button(
        description="Prethodni",
        button_style="",
        layout=Layout(width="100px")
    )
    _btn_next = Button(
        description="Sljedeći",
        button_style="",
        layout=Layout(width="100px")
    )

    _status_label = Label(value="")

    _out_step = Output(
        layout=Layout(
            border="1px solid #ccc",
            padding="8px",
            min_height="160px"
        )
    )
    _out_result = Output(
        layout=Layout(
            border="1px solid #ccc",
            padding="8px",
            min_height="80px"
        )
    )

    # -------------------------
    # Stanje
    # -------------------------
    state = {
        "x": None,        # ulaz (1D)
        "w": None,        # jezgra
        "y": None,        # izlaz (1D)
        "out_len": 0,
        "current_n": 0    # trenutačni ulazni indeks
    }

    # -------------------------
    # Pomoćne funkcije
    # -------------------------
    def _compute_output():
        """Izračunaj izlaz y za zadane parametre 1D transposed conv."""
        N = _in_len.value
        K = _k_size.value
        s = _stride.value
        p = _padding.value
        op = _out_pad.value

        # generiraj ulaz ako treba
        if state["x"] is None or len(state["x"]) != N:
            state["x"] = np.random.randint(1, 11, size=N)

        # nasumična jezgra
        state["w"] = np.random.randint(1, 11, size=K)

        out_len = (N - 1) * s - 2 * p + K + op
        if out_len <= 0:
            state["y"] = None
            state["out_len"] = 0
            _status_label.value = "Parametri daju nevaljanu duljinu izlaza."
            return

        y = np.zeros(out_len, dtype=float)

        # standardni "stamp" pogled: za svaki ulazni element doda kernel
        for n in range(N):
            for k in range(K):
                o = n * s - p + k
                if 0 <= o < out_len:
                    y[o] += state["x"][n] * state["w"][k]

        state["y"] = y
        state["out_len"] = out_len
        state["current_n"] = 0

        _status_label.value = (
            f"in_len={N}, out_len={out_len} | "
            f"stride={s}, padding={p}, k={K}, out_pad={op}"
        )
        _update_nav()

    def _update_nav():
        N = _in_len.value
        if state["y"] is None or state["out_len"] == 0:
            _btn_prev.disabled = True
            _btn_next.disabled = True
            return
        _btn_prev.disabled = (state["current_n"] == 0)
        _btn_next.disabled = (state["current_n"] == N - 1)

    def _render_step():
        """Prikaži doprinos trenutnog ulaznog elementa x[n] na izlaz."""
        with _out_step:
            _out_step.clear_output()

            if state["y"] is None or state["out_len"] == 0:
                print("Nema valjanog izlaza. Podesi parametre ili generiraj podatke.")
                return

            x = state["x"]
            w = state["w"]
            n = state["current_n"]
            N = len(x)
            K = len(w)
            s = _stride.value
            p = _padding.value
            op = _out_pad.value
            out_len = state["out_len"]

            # doprinos samo ovog x[n]
            contrib = np.zeros(out_len, dtype=float)
            terms = []
            for k in range(K):
                o = n * s - p + k
                if 0 <= o < out_len:
                    val = x[n] * w[k]
                    contrib[o] += val
                    terms.append(f"y[{o}] += x[{n}]*w[{k}] = {x[n]}×{w[k]}")

            # HTML za ulaz, istaknut x[n]
            input_cells = []
            for i, v in enumerate(x):
                if i == n:
                    cell = (
                        '<span style="display:inline-block; padding:2px 5px; margin:1px; '
                        'border-radius:4px; background-color:#ffe0e0; font-weight:bold;">'
                        f'{v}</span>'
                    )
                else:
                    cell = (
                        '<span style="display:inline-block; padding:2px 5px; margin:1px; '
                        'border-radius:4px; background-color:#f4f4f4;">'
                        f'{v}</span>'
                    )
                input_cells.append(cell)
            input_html = " ".join(input_cells)

            # HTML za jezgru
            kernel_cells = []
            for v in w:
                cell = (
                    '<span style="display:inline-block; padding:2px 5px; margin:1px; '
                    'border-radius:4px; background-color:#e0f0ff; font-weight:bold;">'
                    f'{v}</span>'
                )
                kernel_cells.append(cell)
            kernel_html = " ".join(kernel_cells)

            # HTML za doprinos (samo od x[n])
            contrib_cells = []
            for i, v in enumerate(contrib):
                if v != 0:
                    cell = (
                        '<span style="display:inline-block; padding:2px 5px; margin:1px; '
                        'border-radius:4px; background-color:#e8ffe8; font-weight:bold;">'
                        f'{v:.2f}</span>'
                    )
                else:
                    cell = (
                        '<span style="display:inline-block; padding:2px 5px; margin:1px; '
                        'border-radius:4px; background-color:#f4f4f4;">'
                        f'{v:.2f}</span>'
                    )
                contrib_cells.append(cell)
            contrib_html = " ".join(contrib_cells)

            term_str = "<br>".join(terms) if terms else "Nema doprinosa (sve izvan granica izlaza)."

            header = f"<b>Ulazni indeks n = {n} / {N-1}</b>"

            html = f"""
            <div style="font-family: monospace; font-size: 13px;">
              {header}<br><br>
              <b>Ulazni vektor x:</b><br>
              {input_html}<br><br>
              <b>Jezgra w:</b><br>
              {kernel_html}<br><br>
              <b>Doprinos x[{n}] na izlaz y:</b><br>
              {contrib_html}<br><br>
              <b>Pravila ažuriranja (transposed conv):</b><br>
              out_len = (in_len - 1)*stride - 2*padding + kernel_size + output_padding<br>
              y[o] += x[n] * w[k], gdje je<br>
              o = n*stride - padding + k<br><br>
              <b>Konkretno za x[{n}]:</b><br>
              {term_str}
            </div>
            """
            display(HTML(html))

    def _render_full_result():
        """Prikaži puni izlaz y (zbroj doprinosa svih x[n])."""
        with _out_result:
            _out_result.clear_output()
            if state["y"] is None or state["out_len"] == 0:
                print("Nema izlaza konvolucije za prikaz.")
                return
            y = state["y"]
            cells = [
                '<span style="display:inline-block; padding:2px 5px; margin:1px; '
                'border-radius:4px; background-color:#d8fdd8;">'
                f'{v:.2f}</span>'
                for v in y
            ]
            y_html = " ".join(cells)
            html = f"""
            <div style="font-family: monospace; font-size: 13px;">
              <b>Cijeli izlaz transposed konvolucije y:</b><br>
              {y_html}
            </div>
            """
            display(HTML(html))

    # -------------------------
    # Callback funkcije
    # -------------------------
    def _on_generate_clicked(b):
        N = _in_len.value
        state["x"] = np.random.randint(1, 11, size=N)
        _compute_output()
        _render_step()
        _render_full_result()

    def _on_param_changed(change):
        if change["name"] == "value":
            _compute_output()
            _render_step()
            _render_full_result()

    def _on_prev_clicked(b):
        if state["y"] is None:
            return
        if state["current_n"] > 0:
            state["current_n"] -= 1
            _update_nav()
            _render_step()

    def _on_next_clicked(b):
        if state["y"] is None:
            return
        N = _in_len.value
        if state["current_n"] < N - 1:
            state["current_n"] += 1
            _update_nav()
            _render_step()

    # -------------------------
    # Spajanje callbackova
    # -------------------------
    _btn_generate.on_click(_on_generate_clicked)
    _btn_prev.on_click(_on_prev_clicked)
    _btn_next.on_click(_on_next_clicked)

    _in_len.observe(_on_param_changed, names="value")
    _k_size.observe(_on_param_changed, names="value")
    _stride.observe(_on_param_changed, names="value")
    _padding.observe(_on_param_changed, names="value")
    _out_pad.observe(_on_param_changed, names="value")

    # -------------------------
    # Inicijalno
    # -------------------------
    _on_generate_clicked(None)

    controls_top = HBox([_in_len, _k_size, _stride])
    controls_mid = HBox([_padding, _out_pad, _btn_generate])
    nav_box = HBox([_btn_prev, _btn_next, _status_label],
                   layout=Layout(justify_content="flex-start", gap="10px"))

    ui = VBox([
        controls_top,
        controls_mid,
        _out_step,
        nav_box,
        _out_result
    ])

    return ui

def create_convtranspose2d_widget():
    """
    Interaktivni widget za 2D transposed (de)konvoluciju.

    Parametri:
      - Visina i širina ulaza (H, W)
      - Visina i širina jezgre (Kh, Kw)
      - Korak (stride) – isti u oba smjera
      - Popuna (padding) – isti u oba smjera
      - Output padding – isti u oba smjera

    Izlaz:
      - Visina:  (H - 1)*stride - 2*padding + Kh + output_padding
      - Širina:  (W - 1)*stride - 2*padding + Kw + output_padding

    Vizualizacija:
      - Prev/Next prolazi kroz sve ulazne pozicije (i, j)
      - Gornji okvir:
          * ulaz x (matrica), istaknuti element x[i,j]
          * jezgra w
          * doprinos ovog x[i,j] na cijeli izlaz
      - Donji okvir:
          * cijeli izlaz y (Oh × Ow), zbroj svih doprinosa
    """

    # -------------------------
    # Kontrole
    # -------------------------
    _in_h = IntSlider(
        description="Visina ulaza (H)",
        min=2, max=10, step=1, value=3,
        continuous_update=False,
        layout=Layout(width="260px"),
        style={'description_width': 'initial'}
    )
    _in_w = IntSlider(
        description="Širina ulaza (W)",
        min=2, max=10, step=1, value=3,
        continuous_update=False,
        layout=Layout(width="260px"),
        style={'description_width': 'initial'}
    )
    _k_h = IntSlider(
        description="Visina jezgre (Kh)",
        min=1, max=7, step=1, value=3,
        continuous_update=False,
        layout=Layout(width="260px"),
        style={'description_width': 'initial'}
    )
    _k_w = IntSlider(
        description="Širina jezgre (Kw)",
        min=1, max=7, step=1, value=3,
        continuous_update=False,
        layout=Layout(width="260px"),
        style={'description_width': 'initial'}
    )
    _stride = IntSlider(
        description="Korak (stride)",
        min=1, max=4, step=1, value=2,
        continuous_update=False,
        layout=Layout(width="260px"),
        style={'description_width': 'initial'}
    )
    _padding = IntSlider(
        description="Popuna (padding)",
        min=0, max=5, step=1, value=1,
        continuous_update=False,
        layout=Layout(width="260px"),
        style={'description_width': 'initial'}
    )
    _out_pad = IntSlider(
        description="Output padding",
        min=0, max=5, step=1, value=0,
        continuous_update=False,
        layout=Layout(width="260px"),
        style={'description_width': 'initial'}
    )

    _btn_generate = Button(
        description="Generiraj slučajne podatke",
        button_style="info",
        layout=Layout(width="260px")
    )
    _btn_prev = Button(
        description="Prethodni",
        button_style="",
        layout=Layout(width="100px")
    )
    _btn_next = Button(
        description="Sljedeći",
        button_style="",
        layout=Layout(width="100px")
    )

    _status_label = Label(value="")

    _out_step = Output(
        layout=Layout(
            border="1px solid #ccc",
            padding="8px",
            min_height="220px"
        )
    )
    _out_result = Output(
        layout=Layout(
            border="1px solid #ccc",
            padding="8px",
            min_height="80px"
        )
    )

    # -------------------------
    # Stanje
    # -------------------------
    state = {
        "x": None,        # ulaz (H×W)
        "w": None,        # jezgra (Kh×Kw)
        "y": None,        # izlaz (Oh×Ow)
        "H": 0,
        "W": 0,
        "Oh": 0,
        "Ow": 0,
        "current_idx": 0  # linearni indeks ulaza (i*W + j)
    }

    # -------------------------
    # Pomoćne funkcije
    # -------------------------
    def _compute_output():
        """Izračunaj izlaz za zadane parametre 2D transposed conv."""
        H = _in_h.value
        W = _in_w.value
        Kh = _k_h.value
        Kw = _k_w.value
        s = _stride.value
        p = _padding.value
        op = _out_pad.value

        # generiraj ulaz ako treba
        if state["x"] is None or state["x"].shape != (H, W):
            state["x"] = np.random.randint(1, 11, size=(H, W))

        # jezgra
        state["w"] = np.random.randint(1, 11, size=(Kh, Kw))

        Oh = (H - 1) * s - 2 * p + Kh + op
        Ow = (W - 1) * s - 2 * p + Kw + op

        if Oh <= 0 or Ow <= 0:
            state["y"] = None
            state["Oh"], state["Ow"] = 0, 0
            _status_label.value = "Parametri daju nevaljanu veličinu izlaza."
            return

        y = np.zeros((Oh, Ow), dtype=float)

        x = state["x"]
        w = state["w"]

        # transposed conv: za svaki ulazni piksel "pečatiramo" jezgru
        for i in range(H):
            for j in range(W):
                for ki in range(Kh):
                    for kj in range(Kw):
                        oi = i * s - p + ki
                        oj = j * s - p + kj
                        if 0 <= oi < Oh and 0 <= oj < Ow:
                            y[oi, oj] += x[i, j] * w[ki, kj]

        state["y"] = y
        state["H"], state["W"] = H, W
        state["Oh"], state["Ow"] = Oh, Ow
        state["current_idx"] = 0

        _status_label.value = (
            f"Ulaz: {H}×{W}, Izlaz: {Oh}×{Ow} | "
            f"stride={s}, padding={p}, Kh={Kh}, Kw={Kw}, out_pad={op}"
        )
        _update_nav()

    def _update_nav():
        if state["y"] is None:
            _btn_prev.disabled = True
            _btn_next.disabled = True
            return
        H, W = state["H"], state["W"]
        max_idx = H * W - 1
        _btn_prev.disabled = (state["current_idx"] == 0)
        _btn_next.disabled = (state["current_idx"] == max_idx)

    def _render_step():
        """Prikaži doprinos jednog ulaznog elementa x[i,j] na cijeli izlaz."""
        with _out_step:
            _out_step.clear_output()

            if state["y"] is None:
                print("Nema valjanog izlaza. Podesi parametre ili generiraj podatke.")
                return

            x = state["x"]
            w = state["w"]
            y = state["y"]
            H, W = state["H"], state["W"]
            Oh, Ow = state["Oh"], state["Ow"]
            s = _stride.value
            p = _padding.value

            idx = state["current_idx"]
            i = idx // W
            j = idx % W

            Kh, Kw = w.shape

            # doprinos samo ovog x[i,j]
            contrib = np.zeros_like(y)
            for ki in range(Kh):
                for kj in range(Kw):
                    oi = i * s - p + ki
                    oj = j * s - p + kj
                    if 0 <= oi < Oh and 0 <= oj < Ow:
                        contrib[oi, oj] += x[i, j] * w[ki, kj]

            # HTML tablica za ulaz (x) s istaknutim elementom
            rows_x = []
            for r in range(H):
                row_cells = []
                for c in range(W):
                    if r == i and c == j:
                        cell = (
                            '<td style="padding:3px 6px; '
                            'background-color:#ffe0e0; font-weight:bold; '
                            'border:1px solid #cccccc; text-align:center;">'
                            f'{x[r, c]}</td>'
                        )
                    else:
                        cell = (
                            '<td style="padding:3px 6px; '
                            'background-color:#f4f4f4; '
                            'border:1px solid #dddddd; text-align:center;">'
                            f'{x[r, c]}</td>'
                        )
                    row_cells.append(cell)
                rows_x.append("<tr>" + "".join(row_cells) + "</tr>")
            x_table_html = "<table style='border-collapse:collapse;'>" + "".join(rows_x) + "</table>"

            # HTML za jezgru w
            rows_w = []
            for r in range(Kh):
                row_cells = []
                for c in range(Kw):
                    cell = (
                        '<td style="padding:3px 6px; '
                        'background-color:#e0f0ff; font-weight:bold; '
                        'border:1px solid #cccccc; text-align:center;">'
                        f'{w[r, c]}</td>'
                    )
                    row_cells.append(cell)
                rows_w.append("<tr>" + "".join(row_cells) + "</tr>")
            w_table_html = "<table style='border-collapse:collapse;'>" + "".join(rows_w) + "</table>"

            # HTML tablica doprinosa
            rows_c = []
            for r in range(Oh):
                row_cells = []
                for c in range(Ow):
                    val = contrib[r, c]
                    if val != 0:
                        cell = (
                            '<td style="padding:3px 6px; '
                            'background-color:#e8ffe8; font-weight:bold; '
                            'border:1px solid #cccccc; text-align:center;">'
                            f'{val:.2f}</td>'
                        )
                    else:
                        cell = (
                            '<td style="padding:3px 6px; '
                            'background-color:#f4f4f4; '
                            'border:1px solid #dddddd; text-align:center;">'
                            f'{val:.2f}</td>'
                        )
                    row_cells.append(cell)
                rows_c.append("<tr>" + "".join(row_cells) + "</tr>")
            contrib_table_html = "<table style='border-collapse:collapse;'>" + "".join(rows_c) + "</table>"

            header = (
                f"<b>Ulazni element x[{i}, {j}] (korak {idx} od {H*W - 1})</b>"
            )

            html = f"""
            <div style="font-family: monospace; font-size: 13px;">
              {header}<br><br>
              <b>Ulazna matrica x:</b><br>
              {x_table_html}<br><br>
              <b>Jezgra w:</b><br>
              {w_table_html}<br><br>
              <b>Doprinos ovog elementa x[{i}, {j}] na izlaz y:</b><br>
              {contrib_table_html}<br><br>
              <b>Pravilo (transposed conv 2D):</b><br>
              Za svaki ulaz x[i,j]:<br>
              y[i*stride - padding + ki, j*stride - padding + kj] += x[i,j] * w[ki,kj]
            </div>
            """
            display(HTML(html))

    def _render_full_result():
        """Prikaži cijeli izlaz y."""
        with _out_result:
            _out_result.clear_output()
            if state["y"] is None:
                print("Nema izlaza konvolucije za prikaz.")
                return
            y = state["y"]
            Oh, Ow = y.shape

            rows_y = []
            for r in range(Oh):
                row_cells = []
                for c in range(Ow):
                    cell = (
                        '<td style="padding:3px 6px; '
                        'background-color:#d8fdd8; '
                        'border:1px solid #cccccc; text-align:center;">'
                        f'{y[r, c]:.2f}</td>'
                    )
                    row_cells.append(cell)
                rows_y.append("<tr>" + "".join(row_cells) + "</tr>")
            y_table_html = "<table style='border-collapse:collapse;'>" + "".join(rows_y) + "</table>"

            html = f"""
            <div style="font-family: monospace; font-size: 13px;">
              <b>Cijeli izlaz transposed konvolucije y (dimenzije {Oh}×{Ow}):</b><br>
              {y_table_html}
            </div>
            """
            display(HTML(html))

    # -------------------------
    # Callback funkcije
    # -------------------------
    def _on_generate_clicked(b):
        H = _in_h.value
        W = _in_w.value
        state["x"] = np.random.randint(1, 11, size=(H, W))
        _compute_output()
        _render_step()
        _render_full_result()

    def _on_param_changed(change):
        if change["name"] == "value":
            _compute_output()
            _render_step()
            _render_full_result()

    def _on_prev_clicked(b):
        if state["y"] is None:
            return
        if state["current_idx"] > 0:
            state["current_idx"] -= 1
            _update_nav()
            _render_step()

    def _on_next_clicked(b):
        if state["y"] is None:
            return
        H, W = state["H"], state["W"]
        max_idx = H * W - 1
        if state["current_idx"] < max_idx:
            state["current_idx"] += 1
            _update_nav()
            _render_step()

    # -------------------------
    # Spajanje callbackova
    # -------------------------
    _btn_generate.on_click(_on_generate_clicked)
    _btn_prev.on_click(_on_prev_clicked)
    _btn_next.on_click(_on_next_clicked)

    _in_h.observe(_on_param_changed, names="value")
    _in_w.observe(_on_param_changed, names="value")
    _k_h.observe(_on_param_changed, names="value")
    _k_w.observe(_on_param_changed, names="value")
    _stride.observe(_on_param_changed, names="value")
    _padding.observe(_on_param_changed, names="value")
    _out_pad.observe(_on_param_changed, names="value")

    # -------------------------
    # Inicijalno
    # -------------------------
    _on_generate_clicked(None)

    controls_top = HBox([_in_h, _in_w, _stride])
    controls_mid = HBox([_k_h, _k_w, _padding])
    controls_bot = HBox([_out_pad, _btn_generate])
    nav_box = HBox([_btn_prev, _btn_next, _status_label],
                   layout=Layout(justify_content="flex-start", gap="10px"))

    ui = VBox([
        controls_top,
        controls_mid,
        controls_bot,
        _out_step,
        nav_box,
        _out_result
    ])

    return ui
