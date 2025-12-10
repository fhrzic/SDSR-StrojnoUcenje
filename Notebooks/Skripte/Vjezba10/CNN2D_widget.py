import numpy as np
from ipywidgets import (
    IntSlider, Button, HBox, VBox, Output, Layout, Label
)
from IPython.display import display, HTML

def create_conv2d_widget():
    """
    Interaktivni widget koji simulira 2D konvoluciju.

    Kontrole:
      - Visina i širina ulazne matrice
      - Visina i širina jezgre (konvolucijskog prozora)
      - Korak (stride) – isti u oba smjera
      - Popuna (padding, simetrična, s nulama)
      - Dilatacija (razmak između elemenata jezgre, isti u oba smjera)

    Značajke:
      - Slučajni cijelobrojni ulaz (1–10) i jezgra (1–10)
      - Korak-po-korak prikaz:
          * istaknuti aktivni elementi u popunjenom ulazu
          * prikaz jezgre
          * prikaz izračuna za trenutni izlazni element
      - Navigacija preko gumba Prethodni / Sljedeći
      - Cijeli izlaz konvolucije (2D) uvijek prikazan ispod
    """

    # -------------------------
    # Kontrole
    # -------------------------
    _in_h = IntSlider(
        description="Visina ulaza",
        min=2, max=20, step=1, value=5,
        continuous_update=False,
        layout=Layout(width="260px"),
        style={'description_width': 'initial'}
    )
    _in_w = IntSlider(
        description="Širina ulaza",
        min=2, max=20, step=1, value=5,
        continuous_update=False,
        layout=Layout(width="260px"),
        style={'description_width': 'initial'}
    )
    _k_h = IntSlider(
        description="Visina jezgre",
        min=1, max=9, step=1, value=3,
        continuous_update=False,
        layout=Layout(width="260px"),
        style={'description_width': 'initial'}
    )
    _k_w = IntSlider(
        description="Širina jezgre",
        min=1, max=9, step=1, value=3,
        continuous_update=False,
        layout=Layout(width="260px"),
        style={'description_width': 'initial'}
    )
    _stride = IntSlider(
        description="Korak (stride)",
        min=1, max=5, step=1, value=1,
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
    _dilation = IntSlider(
        description="Dilatacija",
        min=1, max=4, step=1, value=1,
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

    # Gornji prozor: trenutni prozor konvolucije
    _out_step = Output(
        layout=Layout(
            border="1px solid #ccc",
            padding="8px",
            min_height="180px"
        )
    )
    # Donji prozor: puni izlaz konvolucije
    _out_result = Output(
        layout=Layout(
            border="1px solid #ccc",
            padding="8px",
            min_height="80px"
        )
    )

    # -------------------------
    # Interno stanje
    # -------------------------
    state = {
        "input": None,        # izvorni ulaz (H x W)
        "kernel": None,       # jezgra (Kh x Kw)
        "padded": None,       # ulaz s popunom
        "positions": [],      # lista (start_row, start_col)
        "outputs": None,      # izlaz kao matrica (Oh x Ow)
        "Oh": 0,
        "Ow": 0,
        "current_idx": 0      # indeks trenutnog prozora (linearno)
    }

    # -------------------------
    # Pomoćne funkcije
    # -------------------------
    def _compute_windows():
        """
        Izračunaj popunjeni ulaz, početne pozicije prozora
        i izlaz konvolucije za trenutne parametre.
        Jezgra se nasumično generira pri svakoj promjeni.
        """
        H = _in_h.value
        W = _in_w.value
        Kh = _k_h.value
        Kw = _k_w.value
        stride = _stride.value
        pad = _padding.value
        dilation = _dilation.value

        # Ograniči jezgru da ne bude veća od ulaza (inače je teško vizualno)
        Kh_eff_max = H + 2 * pad
        Kw_eff_max = W + 2 * pad
        eff_Kh = (Kh - 1) * dilation + 1
        eff_Kw = (Kw - 1) * dilation + 1
        if eff_Kh > Kh_eff_max:
            # smanji Kh
            Kh_new = max(1, (Kh_eff_max - 1) // dilation + 1)
            _k_h.value = Kh_new
            Kh = Kh_new
            eff_Kh = (Kh - 1) * dilation + 1
        if eff_Kw > Kw_eff_max:
            # smanji Kw
            Kw_new = max(1, (Kw_eff_max - 1) // dilation + 1)
            _k_w.value = Kw_new
            Kw = Kw_new
            eff_Kw = (Kw - 1) * dilation + 1

        # generiraj novi ulaz ako ga nema ili se dimenzije promijene
        if state["input"] is None or state["input"].shape != (H, W):
            state["input"] = np.random.randint(1, 11, size=(H, W))

        # nasumična jezgra
        state["kernel"] = np.random.randint(1, 11, size=(Kh, Kw))

        # popunjeni ulaz (0-padding)
        padded = np.pad(
            state["input"],
            pad_width=((pad, pad), (pad, pad)),
            mode="constant",
            constant_values=0
        )
        state["padded"] = padded

        Hp, Wp = padded.shape

        positions = []
        # broj koraka po visini/širini
        if Hp >= eff_Kh and Wp >= eff_Kw:
            Oh = (Hp - eff_Kh) // stride + 1
            Ow = (Wp - eff_Kw) // stride + 1
        else:
            Oh, Ow = 0, 0

        outputs = None
        if Oh > 0 and Ow > 0:
            outputs = np.zeros((Oh, Ow), dtype=float)
            for i_out in range(Oh):
                for j_out in range(Ow):
                    sr = i_out * stride
                    sc = j_out * stride
                    positions.append((sr, sc))

                    r_idx = sr + dilation * np.arange(Kh)
                    c_idx = sc + dilation * np.arange(Kw)
                    window = padded[np.ix_(r_idx, c_idx)]
                    outputs[i_out, j_out] = float(np.sum(window * state["kernel"]))

        state["positions"] = positions
        state["outputs"] = outputs
        state["Oh"] = Oh
        state["Ow"] = Ow
        state["current_idx"] = 0

        # Ažuriranje statusa
        if Oh == 0 or Ow == 0:
            _status_label.value = "Nema valjanih prozora. Podesi parametre."
        else:
            _status_label.value = f"Izlaz: {Oh} × {Ow} (broj prozora = {Oh * Ow})"

        _update_nav_buttons()
        _render_full_result()

    def _update_nav_buttons():
        """Uključi/isključi gumbe Prethodni/Sljedeći ovisno o indeksu i broju prozora."""
        n = len(state["positions"])
        if n == 0:
            _btn_prev.disabled = True
            _btn_next.disabled = True
            return
        _btn_prev.disabled = (state["current_idx"] == 0)
        _btn_next.disabled = (state["current_idx"] == n - 1)

    def _render_current_window():
        """Prikaz jednog prozora s istaknutim elementima i detaljem izračuna."""
        with _out_step:
            _out_step.clear_output()

            if len(state["positions"]) == 0 or state["outputs"] is None:
                print("Nema valjanih prozora. Pokušaj promijeniti parametre ili klikni 'Generiraj slučajne podatke'.")
                return

            idx = state["current_idx"]
            Oh = state["Oh"]
            Ow = state["Ow"]
            Kh, Kw = state["kernel"].shape
            dilation = _dilation.value
            padded = state["padded"]
            kernel = state["kernel"]

            # mapiraj linearni indeks -> (i_out, j_out)
            i_out = idx // Ow
            j_out = idx % Ow
            sr, sc = state["positions"][idx]

            r_idx = sr + dilation * np.arange(Kh)
            c_idx = sc + dilation * np.arange(Kw)
            window_vals = padded[np.ix_(r_idx, c_idx)]

            # HTML tablica za popunjeni ulaz
            rows_html = []
            for i in range(padded.shape[0]):
                row_cells = []
                for j in range(padded.shape[1]):
                    if (i in r_idx) and (j in c_idx):
                        cell = (
                            '<td style="padding:3px 6px; '
                            'background-color:#ffe0e0; '
                            'border:1px solid #cccccc; '
                            'text-align:center; font-weight:bold;">'
                            f'{padded[i, j]}</td>'
                        )
                    else:
                        cell = (
                            '<td style="padding:3px 6px; '
                            'background-color:#f4f4f4; '
                            'border:1px solid #dddddd; '
                            'text-align:center;">'
                            f'{padded[i, j]}</td>'
                        )
                    row_cells.append(cell)
                rows_html.append("<tr>" + "".join(row_cells) + "</tr>")
            padded_table_html = "<table style='border-collapse:collapse;'>" + "".join(rows_html) + "</table>"

            # HTML tablica za jezgru
            k_rows_html = []
            for i in range(Kh):
                row_cells = []
                for j in range(Kw):
                    cell = (
                        '<td style="padding:3px 6px; '
                        'background-color:#e0f0ff; '
                        'border:1px solid #cccccc; '
                        'text-align:center; font-weight:bold;">'
                        f'{kernel[i, j]}</td>'
                    )
                    row_cells.append(cell)
                k_rows_html.append("<tr>" + "".join(row_cells) + "</tr>")
            kernel_table_html = "<table style='border-collapse:collapse;'>" + "".join(k_rows_html) + "</table>"

            # Izračun: suma x_ij * k_ij
            terms = []
            for i in range(Kh):
                for j in range(Kw):
                    x_val = window_vals[i, j]
                    k_val = kernel[i, j]
                    terms.append(f"{x_val}×{k_val}")
            term_str = " + ".join(terms)
            out_val = state["outputs"][i_out, j_out]

            header = (
                f"<b>Prozor {idx+1} / {len(state['positions'])}</b> "
                f"(izlazni indeks = y[{i_out}, {j_out}])"
            )

            html = f"""
            <div style="font-family: monospace; font-size: 13px;">
              {header}<br><br>
              <b>Ulaz (s popunom):</b><br>
              {padded_table_html}<br>
              <b>Jezgra:</b><br>
              {kernel_table_html}<br>
              <b>Indeksi u popunjenom ulazu:</b><br>
              redovi = {list(r_idx)}, stupci = {list(c_idx)}<br><br>
              <b>Izračun za y[{i_out}, {j_out}]:</b><br>
              y[{i_out}, {j_out}] = {term_str} = <b>{out_val:.2f}</b>
            </div>
            """
            display(HTML(html))

    def _render_full_result():
        """Prikaz kompletnog izlaza konvolucije (2D)."""
        with _out_result:
            _out_result.clear_output()
            if state["outputs"] is None or state["Oh"] == 0 or state["Ow"] == 0:
                print("Nema izlaza konvolucije za prikaz.")
                return

            y = state["outputs"]
            Oh, Ow = y.shape

            rows_html = []
            for i in range(Oh):
                row_cells = []
                for j in range(Ow):
                    cell = (
                        '<td style="padding:3px 6px; '
                        'background-color:#e8ffe8; '
                        'border:1px solid #cccccc; '
                        'text-align:center;">'
                        f'{y[i, j]:.2f}</td>'
                    )
                    row_cells.append(cell)
                rows_html.append("<tr>" + "".join(row_cells) + "</tr>")
            y_table_html = "<table style='border-collapse:collapse;'>" + "".join(rows_html) + "</table>"

            html = f"""
            <div style="font-family: monospace; font-size: 13px;">
              <b>Cijeli izlaz konvolucije y (dimenzije {Oh} × {Ow}):</b><br>
              {y_table_html}
            </div>
            """
            display(HTML(html))

    # -------------------------
    # Callback funkcije
    # -------------------------
    def _on_generate_clicked(b):
        # Generiraj novi slučajni ulaz zadane visine i širine i ponovo izračunaj
        H = _in_h.value
        W = _in_w.value
        state["input"] = np.random.randint(1, 11, size=(H, W))
        _compute_windows()
        _render_current_window()

    def _on_input_shape_changed(change):
        # Promjena visine/širine ulaza -> generiraj novi ulaz i ponovo izračunaj
        if change["name"] == "value":
            _on_generate_clicked(None)

    def _on_param_changed(change):
        # Promjena jezgre / koraka / popune / dilatacije -> ponovni izračun s istim ulazom
        if change["name"] == "value":
            _compute_windows()
            _render_current_window()

    def _on_prev_clicked(b):
        if len(state["positions"]) == 0:
            return
        if state["current_idx"] > 0:
            state["current_idx"] -= 1
            _update_nav_buttons()
            _render_current_window()

    def _on_next_clicked(b):
        if len(state["positions"]) == 0:
            return
        if state["current_idx"] < len(state["positions"]) - 1:
            state["current_idx"] += 1
            _update_nav_buttons()
            _render_current_window()

    # -------------------------
    # Povezivanje callbackova
    # -------------------------
    _btn_generate.on_click(_on_generate_clicked)
    _btn_prev.on_click(_on_prev_clicked)
    _btn_next.on_click(_on_next_clicked)

    _in_h.observe(_on_input_shape_changed, names="value")
    _in_w.observe(_on_input_shape_changed, names="value")
    _k_h.observe(_on_param_changed, names="value")
    _k_w.observe(_on_param_changed, names="value")
    _stride.observe(_on_param_changed, names="value")
    _padding.observe(_on_param_changed, names="value")
    _dilation.observe(_on_param_changed, names="value")

    # -------------------------
    # Inicijalno postavljanje
    # -------------------------
    _on_generate_clicked(None)  # generiraj početne podatke i prikaži

    controls_top = HBox([_in_h, _in_w, _stride])
    controls_mid = HBox([_k_h, _k_w, _padding])
    controls_bot = HBox([_dilation, _btn_generate])
    nav_box = HBox(
        [_btn_prev, _btn_next, _status_label],
        layout=Layout(justify_content="flex-start", gap="10px")
    )

    ui = VBox([
        controls_top,
        controls_mid,
        controls_bot,
        _out_step,
        nav_box,
        _out_result
    ])

    return ui


# U Jupyter Lab-u:
# widget2d = create_conv2d_widget()
# display(widget2d)
