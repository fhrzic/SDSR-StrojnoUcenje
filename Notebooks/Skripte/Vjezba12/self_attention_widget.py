# self_attention_widget.py
# Spremi ovaj kod u datoteku (npr. self_attention_widget.py) i onda u Jupyteru radi:
#   from self_attention_widget import SelfAttentionDemoWidget
#   w = SelfAttentionDemoWidget()
#   w.display()

import math
import numpy as np
import torch
import torch.nn.functional as F
import ipywidgets as widgets
from IPython.display import display, clear_output


class SelfAttentionDemoWidget:
    """
    Widget za demonstraciju self-attentiona (3 vektora, dimenzija 3) na hrvatskom.

    Korištenje u Jupyter Notebooku/Labu:
        from self_attention_widget import SelfAttentionDemoWidget
        w = SelfAttentionDemoWidget()
        w.display()

    Što radi:
      - generira 3 slučajna vektora X oblika (T=3, D=3)
      - generira Mq, Mk, Mv oblika (3,3)
      - računa Q, K, V, scores = QK^T/sqrt(d), A = softmax(scores), Y = A@V
      - ispisuje sve međurezultate + matricu pažnje
    """

    def __init__(self):
        # UI elementi
        self.title = widgets.HTML("<h3>Self-attention demonstracija (3 vektora, dim=3)</h3>")

        self.btn_generate = widgets.Button(description="Generiraj slučajne vektore", button_style="primary")
        self.btn_compute = widgets.Button(description="Izračunaj self-attention", button_style="success")

        self.seed_toggle = widgets.Checkbox(value=True, description="Koristi seed (reproducibilno)")
        self.seed_input = widgets.IntText(value=0, description="Seed:", layout=widgets.Layout(width="220px"))

        self.low_slider = widgets.FloatSlider(value=0.0, min=-1.0, max=1.0, step=0.05, description="Min:", readout_format=".2f")
        self.high_slider = widgets.FloatSlider(value=1.0, min=-1.0, max=1.0, step=0.05, description="Max:", readout_format=".2f")

        self.decimals = widgets.IntSlider(value=4, min=2, max=8, step=1, description="Decimale:")

        self.out = widgets.Output()

        # stanje
        self._state = {"X": None, "Mq": None, "Mk": None, "Mv": None}

        # event handleri
        self.btn_generate.on_click(self._on_generate)
        self.btn_compute.on_click(self._on_compute)

        # layout
        self.controls_row1 = widgets.HBox([self.btn_generate, self.btn_compute])
        self.controls_row2 = widgets.HBox([self.seed_toggle, self.seed_input, self.decimals])
        self.controls_row3 = widgets.HBox([self.low_slider, self.high_slider])

        self.ui = widgets.VBox([self.title, self.controls_row1, self.controls_row2, self.controls_row3, self.out])

        # inicijalno generiraj
        self._on_generate(None)

    # -----------------------
    # Helper funkcije
    # -----------------------
    @staticmethod
    def _softmax_rows(scores: torch.Tensor) -> torch.Tensor:
        return F.softmax(scores, dim=-1)

    def _fmt_tensor(self, t: torch.Tensor) -> str:
        a = t.detach().cpu().numpy()
        with np.printoptions(precision=int(self.decimals.value), suppress=True):
            return str(a)

    @staticmethod
    def _make_random_vectors(low: float, high: float, seed: int | None) -> torch.Tensor:
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))
        X = (high - low) * torch.rand(3, 3) + low
        return X

    @staticmethod
    def _make_random_weight_matrix(seed: int | None) -> torch.Tensor:
        if seed is not None:
            torch.manual_seed(int(seed) + 12345)
        return 0.5 * torch.randn(3, 3)

    @staticmethod
    def _attention_demo(X: torch.Tensor, Mq: torch.Tensor, Mk: torch.Tensor, Mv: torch.Tensor):
        Q = X @ Mq
        K = X @ Mk
        V = X @ Mv

        d = X.shape[-1]
        scores = (Q @ K.T) / math.sqrt(d)
        A = F.softmax(scores, dim=-1)
        Y = A @ V

        return {"X": X, "Mq": Mq, "Mk": Mk, "Mv": Mv, "Q": Q, "K": K, "V": V, "scores": scores, "A": A, "Y": Y}

    # -----------------------
    # UI akcije
    # -----------------------
    def _warn(self, msg: str):
        print(f"Upozorenje: {msg}")

    def _on_generate(self, _):
        with self.out:
            clear_output()

            low = float(self.low_slider.value)
            high = float(self.high_slider.value)
            if high <= low:
                self._warn("Vrijednost 'Max' mora biti veća od 'Min'.")
                return

            seed = int(self.seed_input.value) if self.seed_toggle.value else None

            X = self._make_random_vectors(low, high, seed)
            Mq = self._make_random_weight_matrix(seed)
            Mk = self._make_random_weight_matrix(seed + 1 if seed is not None else None)
            Mv = self._make_random_weight_matrix(seed + 2 if seed is not None else None)

            self._state.update({"X": X, "Mq": Mq, "Mk": Mk, "Mv": Mv})

            print("Generirani ulazni vektori (embeddingi) X za 3 riječi (T=3, D=3):")
            print(self._fmt_tensor(X))

            print("\nGenerirane matrice težina (model ih inače uči):")
            print("Mq (Query):\n", self._fmt_tensor(Mq))
            print("Mk (Key):\n", self._fmt_tensor(Mk))
            print("Mv (Value):\n", self._fmt_tensor(Mv))

            print("\nKlikni 'Izračunaj self-attention' za sve međukorake i matricu pažnje.")

    def _on_compute(self, _):
        with self.out:
            clear_output()

            if self._state["X"] is None:
                self._warn("Prvo klikni 'Generiraj slučajne vektore'.")
                return

            X, Mq, Mk, Mv = self._state["X"], self._state["Mq"], self._state["Mk"], self._state["Mv"]
            res = self._attention_demo(X, Mq, Mk, Mv)

            d = X.shape[-1]

            print("1) Ulaz X (3 vektora, dim=3):")
            print(self._fmt_tensor(res["X"]))

            print("\n2) Projekcije (matrično množenje):")
            print("Q = X @ Mq  (Query):")
            print(self._fmt_tensor(res["Q"]))
            print("\nK = X @ Mk  (Key):")
            print(self._fmt_tensor(res["K"]))
            print("\nV = X @ Mv  (Value):")
            print(self._fmt_tensor(res["V"]))

            print("\n3) Similarity / alignment scores (skalirani dot produkt):")
            print(f"S = (Q @ K^T) / sqrt(d), gdje je d = {d}")
            print(self._fmt_tensor(res["scores"]))

            print("\n4) SoftMax po retcima (matrica pažnje):")
            print("A = SoftMax(S)  (svaki redak se zbraja na 1)")
            A = res["A"]
            print(self._fmt_tensor(A))
            print("\nProvjera (zbroj po retcima):")
            print(self._fmt_tensor(A.sum(dim=-1)))

            print("\n5) Završno ponderiranje (reweighing):")
            print("Y = A @ V  (kontekstualizirani embeddingi)")
            print(self._fmt_tensor(res["Y"]))

            print("\nSažetak:")
            print("- S (scores) govori koliko svaka riječ 'gleda' svaku drugu riječ.")
            print("- A (attention) su normalizirane težine (probabilistička distribucija po retku).")
            print("- Y su novi, kontekstualizirani vektori nakon primjene pažnje.")

    # -----------------------
    # Jupyter-friendly API
    # -----------------------
    def display(self):
        """Prikaži widget u Jupyter notebooku."""
        display(self.ui)

    def get_state(self):
        """Vrati trenutno stanje (X, Mq, Mk, Mv) kao torch tensore."""
        return dict(self._state)

    def compute_once(self):
        """
        Izračunaj i vrati sve međurezultate bez ispisa (korisno za daljnju analizu).
        """
        if self._state["X"] is None:
            raise RuntimeError("Nema generiranih vektora. Pozovi generate prvo (ili klikni gumb).")
        return self._attention_demo(self._state["X"], self._state["Mq"], self._state["Mk"], self._state["Mv"])
