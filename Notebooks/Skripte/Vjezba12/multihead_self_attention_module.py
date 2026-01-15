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

class MultiHeadSelfAttention(nn.Module):
    """
    Višeglavi self-attention (Multi-Head Scaled Dot-Product Attention) u PyTorchu.

    Ideja:
        Q = X @ Wq, K = X @ Wk, V = X @ Wv
        zatim se Q,K,V razdvoje po glavama (heads), računa se attention po svakoj glavi,
        a rezultati se konkateniraju i projiciraju natrag u embed_dim.

    Ulaz:
        X: (B, T, D)
          B = batch size
          T = duljina sekvence
          D = embed_dim

    Izlaz:
        Y: (B, T, D)
        attn: (B, H, T, T) (opcionalno)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        return_attention: bool = False,
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) mora biti djeljiv s num_heads ({num_heads})."
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.return_attention = return_attention

        # Projekcije za Q, K, V (svaka: D -> D)
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Završna projekcija nakon konkatenacije glava
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, T, D) -> (B, H, T, Dh)
        """
        B, T, D = x.shape
        x = x.view(B, T, self.num_heads, self.head_dim)
        return x.transpose(1, 2)  # (B, H, T, Dh)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, H, T, Dh) -> (B, T, D)
        """
        B, H, T, Dh = x.shape
        x = x.transpose(1, 2).contiguous()  # (B, T, H, Dh)
        return x.view(B, T, H * Dh)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Parametri
        ---------
        x : torch.Tensor
            (B, T, D)
        mask : torch.Tensor | None
            Opcionalna maska pažnje.
            Podržani oblici:
              - (B, T)         1=validno, 0=padding
              - (B, 1, T)      broadcast na (B, H, T, T)
              - (B, T, T)      broadcast na (B, H, T, T)
              - (B, H, T, T)   direktno po glavama
            Maska se primjenjuje kao -inf prije SoftMax-a.

        Povrat
        ------
        y : torch.Tensor
            (B, T, D)
        attn (opcionalno) : torch.Tensor
            (B, H, T, T)
        """
        if x.dim() != 3:
            raise ValueError(f"Expected x with shape (B, T, D), got {tuple(x.shape)}")

        B, T, D = x.shape
        if D != self.embed_dim:
            raise ValueError(f"Expected embedding dim {self.embed_dim}, got {D}")

        # 1) Projekcije Q, K, V
        Q = self.W_q(x)  # (B, T, D)
        K = self.W_k(x)  # (B, T, D)
        V = self.W_v(x)  # (B, T, D)

        # 2) Split po glavama
        Qh = self._split_heads(Q)  # (B, H, T, Dh)
        Kh = self._split_heads(K)  # (B, H, T, Dh)
        Vh = self._split_heads(V)  # (B, H, T, Dh)

        # 3) Scores: (Qh @ Kh^T) / sqrt(Dh)
        scores = torch.matmul(Qh, Kh.transpose(-2, -1))  # (B, H, T, T)
        scores = scores / math.sqrt(self.head_dim)

        # 4) Maska (opcionalno)
        if mask is not None:
            if mask.dim() == 2:                 # (B, T)
                mask_ = mask[:, None, None, :]  # (B, 1, 1, T)
            elif mask.dim() == 3:               # (B, 1, T) ili (B, T, T)
                # Ako je (B,1,T) -> (B,1,1,T); ako je (B,T,T) -> (B,1,T,T)
                mask_ = mask[:, None, :, :] if mask.shape[-2] == T else mask[:, None, None, :]
            else:
                mask_ = mask                    # (B, H, T, T) ili broadcastabilno

            scores = scores.masked_fill(mask_ == 0, float("-inf"))

        # 5) SoftMax -> attention weights
        attn = F.softmax(scores, dim=-1)         # (B, H, T, T)
        attn = self.attn_dropout(attn)

        # 6) Reweighing: Ah @ Vh
        out_heads = torch.matmul(attn, Vh)       # (B, H, T, Dh)

        # 7) Merge heads + output projection
        out = self._merge_heads(out_heads)       # (B, T, D)
        out = self.W_o(out)                      # (B, T, D)
        out = self.proj_dropout(out)

        if self.return_attention:
            return out, attn
        return out
