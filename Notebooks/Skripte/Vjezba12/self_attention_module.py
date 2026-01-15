import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
from torchview import draw_graph
from IPython.display import display, SVG

def plot_graph(model: torch.nn.Module = None, 
               device:str = 'cpu',
               input_dim = None):
    """
    Funkcija koja crta graf modela.

    Argumenti:
        * model, pytorch model
        * device, str, gdje se nalaze model i podaci, zadano je: cpu
        * input_dim, tupple, ulazne dimenzije modela
    """
    model = model.to(device).eval()

    _graph = draw_graph(
        model,
        input_size=[input_dim],   # e.g. (1, 3, 224, 224)
        expand_nested=True,
        graph_name="Model",
        device=device,
    )

    # Option A: display directly from the Graphviz object
    _svg_bytes = _graph.visual_graph.pipe(format="svg")  # no temp file
    display(SVG(_svg_bytes))

def print_model_summary(model: torch.nn.Module = None,
                        device: str = 'cpu',
                        input_dim=None):
    """
    Funkcija koja ispisuje sažetak modela.

    Argumenti:
        * model, pytorch model.
        * device, str, gdje se nalaze model i podaci, zadano je: cpu.
        * input_dim, tupple, ulazne dimenzije modela.

    Izlaz:
        * vraća ispisivi sažetak modela
    """
    return torchinfo.summary(
        model=model,
        input_size=input_dim,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=16,
        device=device,
        row_settings=["var_names"]
    )

class SelfAttention(nn.Module):
    """
    Jednoglavi self-attention (Scaled Dot-Product Attention) u PyTorchu.

    Odgovara opisu s matricama Mq, Mk, Mv:
        Q = X @ Mq
        K = X @ Mk
        V = X @ Mv

        scores = Q @ K^T / sqrt(d_k)
        W = softmax(scores, dim=-1)
        Y = W @ V

    Ulaz:
        X: (B, T, D)  gdje je:
            B = batch size
            T = duljina sekvence (broj riječi / tokena)
            D = dimenzija embeddinga (k)

    Izlaz:
        Y: (B, T, D)
        attn: (B, T, T)  (opcionalno; matrica pažnje)
    """

    def __init__(self, embed_dim: int, bias: bool = True, return_attention: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.return_attention = return_attention

        # Mq, Mk, Mv (svaka je D x D)
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Parametri
        ---------
        x : torch.Tensor
            (B, T, D)
        mask : torch.Tensor | None
            Opcionalna maska pažnje.
            Očekivani oblici:
              - (B, T)      gdje je 1 za validne tokene, 0 za padding
              - (B, 1, T)   broadcastabilno na (B, T, T)
              - (B, T, T)   direktna maska po parovima
            Maska se primjenjuje tako da se zabranjenim pozicijama doda -inf prije SoftMax-a.

        Povrat
        ------
        y : torch.Tensor
            (B, T, D)
        attn (opcionalno) : torch.Tensor
            (B, T, T)
        """
        if x.dim() != 3:
            raise ValueError(f"Expected x with shape (B, T, D), got {tuple(x.shape)}")

        B, T, D = x.shape
        if D != self.embed_dim:
            raise ValueError(f"Expected embedding dim {self.embed_dim}, got {D}")

        # 1) Projekcije: Q, K, V
        Q = self.W_q(x)  # (B, T, D)
        K = self.W_k(x)  # (B, T, D)
        V = self.W_v(x)  # (B, T, D)

        # 2) Alignment scores: QK^T (scaled)
        # scores[b] = Q[b] @ K[b].T  -> (T, T)
        scores = torch.matmul(Q, K.transpose(-2, -1))  # (B, T, T)
        scores = scores / math.sqrt(D)

        # 3) Maska (opcionalno)
        if mask is not None:
            # Pretvori masku u broadcastabilni oblik za (B, T, T)
            if mask.dim() == 2:               # (B, T)
                mask_ = mask[:, None, :]      # (B, 1, T)
            else:
                mask_ = mask                  # (B, 1, T) ili (B, T, T)

            # Pretpostavka: mask == 1 znači "dozvoli", mask == 0 znači "zabrani"
            scores = scores.masked_fill(mask_ == 0, float("-inf"))

        # 4) SoftMax po zadnjoj dimenziji (po "keys" osi)
        attn = F.softmax(scores, dim=-1)  # (B, T, T)

        # 5) Reweighing: Y = attn @ V
        y = torch.matmul(attn, V)  # (B, T, D)

        if self.return_attention:
            return y, attn
        return y