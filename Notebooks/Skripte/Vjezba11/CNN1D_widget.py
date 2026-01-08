import numpy as np
from ipywidgets import (
    IntSlider, Button, HBox, VBox, Output, Layout, Label
)
from IPython.display import display, HTML

def create_conv1d_widget():
    """
    Interaktivni widget koji simulira 1D konvoluciju.

    Kontrole:
      - Duljina ulaznog niza
      - Širina jezgre (konvolucijskog prozora)
      - Korak (stride)
      - Popuna (padding, simetrična, s nulama)
      - Dilatacija (razmak između elemenata jezgre)

    Značajke:
      - Slučajni cijelobrojni ulaz (1–10) i jezgra (1–10)
      - Korak-po-korak prikaz:
          * istaknuti aktivni elementi u ulazu i jezgri
          * prikaz izračuna za svaki izlazni element
      - Navigacija preko gumba Prethodni / Sljedeći
      - Konačni izlaz konvolucije uvijek prikazan ispod
    """

    # -------------------------
    # Kontrole
    # -------------------------
    _array_len = IntSlider(
        description="Duljina niza",
        min=3, max=30, step=1, value=8,
        continuous_update=False,
        layout=Layout(width="260px"),
        style={'description_width': 'initial'}
    )
    _kernel_size = IntSlider(
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

    # Prozor za prikaz pojedinog koraka
    _out_step = Output(
        layout=Layout(
            border="1px solid #ccc",
            padding="8px",
            min_height="150px"
        )
    )
    # Prozor za prikaz konačnog rezultata (uvijek vidljiv)
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
        "input": None,        # izvorni ulaz (bez popune)
        "kernel": None,       # jezgra
        "padded": None,       # ulaz s popunom
        "positions": [],      # lista početnih indeksa prozora
        "outputs": [],        # izlazi konvolucije za svaki prozor
        "current_idx": 0      # indeks trenutnog prozora
    }

    # -------------------------
    # Pomoćne funkcije
    # -------------------------
    def _compute_windows():
        """
        Izračunaj popunjeni ulaz, početne pozicije prozora
        i izlaze konvolucije za trenutne parametre.
        Jezgra se nasumično generira pri svakoj promjeni.
        """
        N = _array_len.value
        K = _kernel_size.value
        stride = _stride.value
        pad = _padding.value
        dilation = _dilation.value

        if state["input"] is None or len(state["input"]) != N:
            # ako nema ulaza ili je duljina promijenjena, generiraj novi
            state["input"] = np.random.randint(1, 11, size=N)

        # nasumična jezgra (1–10)
        state["kernel"] = np.random.randint(1, 11, size=K)

        # efektivna širina jezgre s dilatacijom
        eff_kernel = (K - 1) * dilation + 1

        # popunjeni ulaz
        padded = np.pad(
            state["input"],
            pad_width=(pad, pad),
            mode="constant",
            constant_values=0
        )
        state["padded"] = padded

        # svi mogući početni indeksi prozora
        Np = len(padded)
        positions = []
        outputs = []

        if Np >= eff_kernel:
            num_windows = (Np - eff_kernel) // stride + 1
            for i in range(num_windows):
                start = i * stride
                idxs = start + dilation * np.arange(K)
                vals = padded[idxs]
                out_val = float(np.sum(vals * state["kernel"]))
                positions.append(start)
                outputs.append(out_val)

        state["positions"] = positions
        state["outputs"] = outputs
        state["current_idx"] = 0

        # Ažuriranje statusa i gumba
        if len(positions) == 0:
            _status_label.value = "Nema valjanih prozora. Podesi širinu jezgre / korak / popunu."
        else:
            _status_label.value = f"Broj izlaznih pozicija (prozora): {len(positions)}"

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

            if len(state["positions"]) == 0:
                print("Nema valjanih prozora. Pokušaj promijeniti parametre ili klikni 'Generiraj slučajne podatke'.")
                return

            idx = state["current_idx"]
            start = state["positions"][idx]
            kernel = state["kernel"]
            padded = state["padded"]
            dilation = _dilation.value

            K = len(kernel)
            window_idxs = start + dilation * np.arange(K)
            window_vals = padded[window_idxs]

            # HTML prikaz ulaza s istaknutim elementima
            active_idx_set = set(window_idxs.tolist())

            input_html_pieces = []
            for i, v in enumerate(padded):
                if i in active_idx_set:
                    # istaknut element
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
                input_html_pieces.append(cell)
            input_html = " ".join(input_html_pieces)

            # Prikaz jezgre
            kernel_html_pieces = []
            for v in kernel:
                cell = (
                    '<span style="display:inline-block; padding:2px 5px; margin:1px; '
                    'border-radius:4px; background-color:#e0f0ff; font-weight:bold;">'
                    f'{v}</span>'
                )
                kernel_html_pieces.append(cell)
            kernel_html = " ".join(kernel_html_pieces)

            # Izračun: suma(x_i * k_i)
            terms = [f"{x}×{k}" for x, k in zip(window_vals, kernel)]
            term_str = " + ".join(terms)
            out_val = state["outputs"][idx]

            # Naslov prozora
            header = f"<b>Prozor {idx+1} / {len(state['positions'])}</b> &nbsp; (indeks izlaza = y[{idx}])"

            html = f"""
            <div style="font-family: monospace; font-size: 13px;">
              {header}<br><br>
              <b>Ulaz (s popunom):</b><br>
              {input_html}<br><br>
              <b>Jezgra:</b><br>
              {kernel_html}<br><br>
              <b>Korišteni indeksi (u popunjenom ulazu):</b> {list(window_idxs)}<br>
              <b>Izračun:</b><br>
              y[{idx}] = {term_str} = <b>{out_val:.2f}</b>
            </div>
            """
            display(HTML(html))

    def _render_full_result():
        """Prikaz kompletnog izlaza konvolucije (uvijek vidljiv)."""
        with _out_result:
            _out_result.clear_output()
            if len(state["outputs"]) == 0:
                print("Nema izlaza konvolucije za prikaz.")
                return

            y = np.array(state["outputs"], dtype=float)

            # Lijepo formatirani HTML prikaz
            cells = [
                '<span style="display:inline-block; padding:2px 5px; margin:1px; '
                'border-radius:4px; background-color:#e8ffe8;">'
                f'{v:.2f}</span>'
                for v in y
            ]
            y_html = " ".join(cells)
            html = f"""
            <div style="font-family: monospace; font-size: 13px;">
              <b>Cijeli izlaz konvolucije y:</b><br>
              {y_html}
            </div>
            """
            display(HTML(html))

    # -------------------------
    # Callback funkcije
    # -------------------------
    def _on_generate_clicked(b):
        # Generiraj novi slučajni ulaz zadane duljine i ponovo izračunaj
        N = _array_len.value
        state["input"] = np.random.randint(1, 11, size=N)
        _compute_windows()
        _render_current_window()

    def _on_array_len_changed(change):
        # Promjena duljine niza -> generiraj novi ulaz i ponovo izračunaj
        if change["name"] == "value":
            _on_generate_clicked(None)

    def _on_param_changed(change):
        # Promjena širine jezgre / koraka / popune / dilatacije -> ponovni izračun s istim ulazom
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

    _array_len.observe(_on_array_len_changed, names="value")
    _kernel_size.observe(_on_param_changed, names="value")
    _stride.observe(_on_param_changed, names="value")
    _padding.observe(_on_param_changed, names="value")
    _dilation.observe(_on_param_changed, names="value")

    # -------------------------
    # Inicijalno postavljanje
    # -------------------------
    _on_generate_clicked(None)  # generiraj početne podatke i prikaži

    controls_top = HBox([_array_len, _kernel_size, _stride])
    controls_bot = HBox([_padding, _dilation, _btn_generate])
    nav_box = HBox(
        [_btn_prev, _btn_next, _status_label],
        layout=Layout(justify_content="flex-start", gap="10px")
    )

    ui = VBox([
        controls_top,
        controls_bot,
        _out_step,
        nav_box,
        _out_result
    ])

    return ui


# U Jupyter Lab-u:
# widget = create_conv1d_widget()
# display(widget)
