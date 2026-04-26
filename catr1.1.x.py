# target: Python 3.14+
import random
import threading
import time
import math
import re
import tkinter as tk
from tkinter import scrolledtext
from typing import List, Tuple, Generator

# ═══════════════════════════════════════════════════════════════════════════════
# ── UI Constants (CatR11.5 DeepSeek Theme) ────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
BG_MAIN    = "#1e1e2e"  # Deep space background
BG_SIDEBAR = "#181825"
BG_INPUT   = "#313244"
FG_MAIN    = "#cdd6f4"
FG_DIM     = "#a6adc8"
FG_THINK   = "#6c7086"  # DeepSeek R1 style reasoning text color
ACCENT     = "#89b4fa"  # DeepSeek blue
FONT_MAIN      = ("Helvetica", 11)
FONT_BOLD      = ("Helvetica", 11, "bold")
FONT_REASONING = ("Helvetica", 11, "italic")

# ═══════════════════════════════════════════════════════════════════════════════
# ── Matrix Utilities (The Core of BitNet) ─────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def zeros(r: int, c: int) -> List[List[float]]:
    return [[0.0] * c for _ in range(r)]

def transpose(A: List[List[float]]) -> List[List[float]]:
    if not A or not A[0]: return []
    return [list(col) for col in zip(*A)]

def matmul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Standard float matrix multiply for backward passes and non-quantized parts."""
    BT = transpose(B)
    return [[sum(a * b for a, b in zip(ra, cb)) for cb in BT] for ra in A]

def ternary_matmul(X: List[List[float]], W: List[List[int]]) -> List[List[float]]:
    """
    CatR11.5 Core: Matmul where W ∈ {-1,0,1} — ZERO MULTIPLICATIONS!
    This is what makes it a 'Real BitNet'.
    """
    WT = transpose(W)
    out = []
    for row in X:
        orow = []
        for col in WT:
            v = 0.0
            for x, w in zip(row, col):
                # Only additions and subtractions!
                if   w ==  1: v += x
                elif w == -1: v -= x
            orow.append(v)
        out.append(orow)
    return out

def softmax(x: List[float]) -> List[float]:
    m = max(x)
    e = [math.exp(v - m) for v in x]
    s = sum(e) + 1e-12
    return [v / s for v in e]

def clip_val(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

# ═══════════════════════════════════════════════════════════════════════════════
# ── BitNet 1.58b Layers (with full backprop) ─────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

class RMSNorm:
    def __init__(self, dim: int, eps: float = 1e-6):
        self.dim   = dim
        self.eps   = eps
        self.weight = [1.0] * dim
        self.gw     = [0.0] * dim

    def forward(self, x):
        self._x    = x
        self._irms = []
        out = []
        for row in x:
            ms   = sum(v * v for v in row) / len(row)
            irms = 1.0 / math.sqrt(ms + self.eps)
            self._irms.append(irms)
            out.append([v * irms * w for v, w in zip(row, self.weight)])
        return out

    def backward(self, go):
        gi = []
        n  = self.dim
        for row, gr, irms in zip(self._x, go, self._irms):
            for j in range(n):
                self.gw[j] += gr[j] * row[j] * irms
            dot = sum(gr[j] * self.weight[j] * row[j] for j in range(n))
            gi.append([
                gr[j] * self.weight[j] * irms - dot * (irms ** 3) * row[j] / n
                for j in range(n)
            ])
        return gi

    def step(self, lr: float):
        for j in range(self.dim):
            self.weight[j] -= lr * clip_val(self.gw[j], -5.0, 5.0)
            self.gw[j] = 0.0

class BitLinear:
    """
    BitNet 1.58b Linear — latent float weights quantised to {-1, 0, 1}
    Straight-Through Estimator for backward.
    """
    def __init__(self, inf: int, outf: int):
        self.inf  = inf
        self.outf = outf
        sc = math.sqrt(2.0 / inf)
        self.lw = [[random.gauss(0, sc) for _ in range(outf)] for _ in range(inf)]
        self.gw = None

    def _quantize(self):
        total = sum(abs(v) for row in self.lw for v in row)
        gamma = total / (self.inf * self.outf) + 1e-8
        # Strictly enforces -1, 0, 1
        return [[int(round(clip_val(v / gamma, -1, 1))) for v in row] for row in self.lw]

    def forward(self, x):
        self._x = x
        self._tw = self._quantize()
        return ternary_matmul(x, self._tw)

    def backward(self, go):
        self.gw = matmul(transpose(self._x), go)
        return matmul(go, transpose(self.lw))

    def step(self, lr: float):
        if self.gw:
            for i in range(self.inf):
                for j in range(self.outf):
                    self.lw[i][j] -= lr * clip_val(self.gw[i][j], -5.0, 5.0)

class CausalSelfAttention:
    def __init__(self, dim: int, heads: int):
        self.dim   = dim
        self.heads = heads
        self.hd    = dim // heads
        self.scale = 1.0 / math.sqrt(self.hd)
        self.qp = BitLinear(dim, dim)
        self.kp = BitLinear(dim, dim)
        self.vp = BitLinear(dim, dim)
        self.op = BitLinear(dim, dim)

    def forward(self, x):
        S = len(x)
        Q, K, V = self.qp.forward(x), self.kp.forward(x), self.vp.forward(x)
        self._Q, self._K, self._V = Q, K, V
        self._aw = []
        houts = []
        hd = self.hd
        for h in range(self.heads):
            s = h * hd
            Qh = [[Q[i][s + d] for d in range(hd)] for i in range(S)]
            Kh = [[K[i][s + d] for d in range(hd)] for i in range(S)]
            Vh = [[V[i][s + d] for d in range(hd)] for i in range(S)]
            sc = matmul(Qh, transpose(Kh))
            for i in range(S):
                for j in range(S):
                    sc[i][j] *= self.scale
                    if j > i: sc[i][j] = -1e9
            aw = [softmax(row) for row in sc]
            self._aw.append(aw)
            houts.append(matmul(aw, Vh))

        cat = [[v for h in range(self.heads) for v in houts[h][i]] for i in range(S)]
        return self.op.forward(cat)

    def backward(self, go):
        S, hd = len(go), self.hd
        gc  = self.op.backward(go)
        gQ, gK, gV = zeros(S, self.dim), zeros(S, self.dim), zeros(S, self.dim)

        for h in range(self.heads):
            s = h * hd
            gh = [[gc[i][s + d] for d in range(hd)] for i in range(S)]
            Qh = [[self._Q[i][s + d] for d in range(hd)] for i in range(S)]
            Kh = [[self._K[i][s + d] for d in range(hd)] for i in range(S)]
            Vh = [[self._V[i][s + d] for d in range(hd)] for i in range(S)]
            aw = self._aw[h]

            gaw = matmul(gh, transpose(Vh))
            gVh = matmul(transpose(aw), gh)

            gsc = []
            for i in range(S):
                dot = sum(aw[i][j] * gaw[i][j] for j in range(S))
                gsc.append([(gaw[i][j] - dot) * aw[i][j] * self.scale if j <= i else 0.0 for j in range(S)])

            gQh = matmul(gsc, Kh)
            gKh = matmul(transpose(gsc), Qh)

            for i in range(S):
                for d in range(hd):
                    gQ[i][s + d] += gQh[i][d]
                    gK[i][s + d] += gKh[i][d]
                    gV[i][s + d] += gVh[i][d]

        g1, g2, g3 = self.qp.backward(gQ), self.kp.backward(gK), self.vp.backward(gV)
        return [[g1[i][j] + g2[i][j] + g3[i][j] for j in range(self.dim)] for i in range(S)]

    def step(self, lr: float):
        for p in (self.qp, self.kp, self.vp, self.op): p.step(lr)

class FFN:
    def __init__(self, dim: int, hid: int):
        self.up   = BitLinear(dim, hid)
        self.down = BitLinear(hid, dim)

    def forward(self, x):
        h = self.up.forward(x)
        self._mask = [[1.0 if v > 0 else 0.0 for v in row] for row in h]
        h = [[max(0.0, v) for v in row] for row in h]
        return self.down.forward(h)

    def backward(self, go):
        g = self.down.backward(go)
        g = [[g[i][j] * self._mask[i][j] for j in range(len(g[0]))] for i in range(len(g))]
        return self.up.backward(g)

    def step(self, lr: float):
        self.up.step(lr)
        self.down.step(lr)

class Block:
    def __init__(self, dim: int, heads: int, ffn_dim: int):
        self.n1  = RMSNorm(dim)
        self.att = CausalSelfAttention(dim, heads)
        self.n2  = RMSNorm(dim)
        self.ffn = FFN(dim, ffn_dim)

    def forward(self, x):
        h = self.att.forward(self.n1.forward(x))
        x = [[a + b for a, b in zip(r, hr)] for r, hr in zip(x, h)]
        h = self.ffn.forward(self.n2.forward(x))
        return [[a + b for a, b in zip(r, hr)] for r, hr in zip(x, h)]

    def backward(self, g):
        gf = self.ffn.backward(self.n2.backward(g))
        g  = [[a + b for a, b in zip(ga, gb)] for ga, gb in zip(g, gf)]
        ga = self.att.backward(self.n1.backward(g))
        return [[a + b for a, b in zip(ga, gb)] for ga, gb in zip(g, ga)]

    def step(self, lr: float):
        self.n1.step(lr); self.att.step(lr)
        self.n2.step(lr); self.ffn.step(lr)

# ═══════════════════════════════════════════════════════════════════════════════
# ── CatR11.5 Model (R1 Style) ─────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

class CatR115Model:
    def __init__(self, vocab: int, dim=48, depth=1, heads=2, ffn_mult=2):
        self.V   = vocab
        self.dim = dim
        self.emb = [[random.gauss(0, 0.02) for _ in range(dim)] for _ in range(vocab)]
        self.layers = [Block(dim, heads, dim * ffn_mult) for _ in range(depth)]
        self.fnorm  = RMSNorm(dim)
        self.head   = BitLinear(dim, vocab)

    def forward(self, ids):
        self._ids = ids
        x = [list(self.emb[t]) for t in ids]
        for L in self.layers: x = L.forward(x)
        return self.head.forward(self.fnorm.forward(x))

    def loss_backward(self, logits, targets):
        S = len(logits)
        loss = 0.0
        gl = []
        for i in range(S):
            p = softmax(logits[i])
            loss -= math.log(max(p[targets[i]], 1e-10))
            g = list(p)
            g[targets[i]] -= 1.0
            gl.append([v / S for v in g])
        
        g = self.fnorm.backward(self.head.backward(gl))
        for L in reversed(self.layers): g = L.backward(g)

        self._eg = zeros(self.V, self.dim)
        for i, t in enumerate(self._ids):
            for j in range(self.dim): self._eg[t][j] += g[i][j]
        return loss / S

    def step(self, lr: float):
        for i in range(self.V):
            for j in range(self.dim):
                self._eg[i][j] = clip_val(self._eg[i][j], -5.0, 5.0)
                self.emb[i][j] -= lr * self._eg[i][j]
        for L in self.layers: L.step(lr)
        self.fnorm.step(lr); self.head.step(lr)

    def generate_token(self, ids, temperature=0.8, rep_penalty=1.4):
        logits = self.forward(ids)[-1]
        
        # Penalize recent tokens to prevent looping
        recent = set(ids[-10:])
        for tok in recent:
            if tok < len(logits):
                logits[tok] = logits[tok] / rep_penalty if logits[tok] > 0 else logits[tok] * rep_penalty
                
        if temperature > 0:
            pr = softmax([v / temperature for v in logits])
            indexed = sorted(enumerate(pr), key=lambda x: -x[1])[:15]
            total = sum(p for _, p in indexed)
            r = random.random() * total
            c = 0.0
            for idx, p in indexed:
                c += p
                if r < c: return idx
            return indexed[-1][0]
        return logits.index(max(logits))

# ═══════════════════════════════════════════════════════════════════════════════
# ── Word-Level Tokenizer (With Code & <think> Tag Support) ────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

class WordTokenizer:
    """Parses text including explicit <think> tags, words, numbers, and coding symbols."""
    # Updated Regex to handle Python code syntax, numbers, and newlines
    _PAT = re.compile(r"<think>|</think>|[a-z0-9]+|[.,!?#=():*\n]")

    def __init__(self, text: str):
        tokens = self._PAT.findall(text.lower())
        vocab  = sorted(set(tokens))
        self.w2i = {w: i for i, w in enumerate(vocab)}
        self.i2w = {i: w for i, w in enumerate(vocab)}
        self.vocab_size = len(vocab)
        
        self.THINK_START = self.w2i.get("<think>")
        self.THINK_END   = self.w2i.get("</think>")

    def encode(self, text: str) -> List[int]:
        return [self.w2i[t] for t in self._PAT.findall(text.lower()) if t in self.w2i]


# ═══════════════════════════════════════════════════════════════════════════════
# ── DeepSeek R1 Simulated Reasoning & Coding Corpus ───────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# Upgraded to teach English conversation + Python 3.14 Code Generation
CORPUS = (
    "what is a cat ? <think> a cat is a feline . it says meow . </think> a cat is a small furry animal . "
    "write some python code . <think> the user wants python code . i will use python 3.14 . </think> \n # target : python 3.14 \n import math \n def area ( r ) : \n return 3.14 * r * r \n "
    "hello who are you ? <think> i should introduce myself as cat r11.5 . </think> i am cat r11.5 , an ai model that can talk and code . "
    "import python 3.14 . <think> i will write the target comment and import a module . </think> \n # target : python 3.14 \n import random \n import time \n print ( ready ) \n "
    "can you code ? <think> yes , i can write simple python scripts . </think> yes , i can code . here is an example : \n def hello ( ) : \n print ( 1 ) \n "
    "what is the sun ? <think> the sun is a star . </think> the sun is a bright star in the sky . "
    "count to 3 . <think> i need to output numbers . </think> 1 , 2 , 3 . "
)

# ═══════════════════════════════════════════════════════════════════════════════
# ── CatR11.5 Engine ───────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

class CatR115Engine:
    MODEL_DIM   = 48
    MODEL_DEPTH = 1
    MODEL_HEADS = 2
    SEQ_LEN     = 24  # Increased to capture more coding context
    TRAIN_STEPS = 1500  # Increased steps for code structure learning
    LR          = 0.005

    def __init__(self):
        self.tok   = WordTokenizer(CORPUS)
        self.model = CatR115Model(
            self.tok.vocab_size,
            dim=self.MODEL_DIM, depth=self.MODEL_DEPTH,
            heads=self.MODEL_HEADS, ffn_mult=2,
        )
        self.data  = self.tok.encode(CORPUS)
        self.trained = False

    def train(self, callback=None):
        S, N = self.SEQ_LEN, len(self.data)
        for step in range(self.TRAIN_STEPS):
            i = random.randint(0, N - S - 1)
            inp = self.data[i : i + S]
            tgt = self.data[i + 1 : i + S + 1]
            logits = self.model.forward(inp)
            loss   = self.model.loss_backward(logits, tgt)
            self.model.step(self.LR)
            if callback and step % 5 == 0:
                callback(step + 1, self.TRAIN_STEPS, loss)
        self.trained = True

    def generate(self, prompt: str, max_tokens=70, temperature=0.7, stop_evt=None):
        """Generates tokens and yields (word, is_thinking_flag)."""
        ids = self.tok.encode(prompt.lower())
        
        # DeepSeek R1 behavior: encourage starting if empty
        if not ids: ids = self.tok.encode("hello")
        
        ctx_window = 32
        is_thinking = False

        for _ in range(max_tokens):
            if stop_evt and stop_evt.is_set(): break
            
            ctx = ids[-ctx_window:]
            nxt = self.model.generate_token(ctx, temperature)
            ids.append(nxt)
            
            word = self.tok.i2w.get(nxt, "?")
            
            # DeepSeek R1 state toggle based on tokens
            if word == "<think>":
                is_thinking = True
                yield ("\n\n[💭 Reasoning Trace]\n", True)
                continue
            elif word == "</think>":
                is_thinking = False
                yield ("\n[✅ Output]\n", False)
                continue
                
            # Better formatting for code generation and punctuation
            if word in (".", ",", "!", "?", ":", ")", "\n"):
                out_str = word
            else:
                out_str = " " + word
                
            yield (out_str, is_thinking)
            time.sleep(0.06)

# ═══════════════════════════════════════════════════════════════════════════════
# ── CatR11.5 UI (DeepSeek R1 Inspired) ────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

class CatR115App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CatR11.5 — Pure Python Ternary LLM")
        self.geometry("1050x750")
        self.minsize(820, 520)
        self.configure(bg=BG_MAIN)

        self._busy     = False
        self._stop_evt = threading.Event()
        self._engine   = CatR115Engine()

        self._build_ui()
        self._lock_input("Training from scratch…")
        self.after(300, self._start_training)

    def _build_ui(self):
        sb = tk.Frame(self, bg=BG_SIDEBAR, width=280)
        sb.pack(side="left", fill="y"); sb.pack_propagate(False)

        tk.Button(
            sb, text="+ New CatR11.5 Chat", bg=ACCENT, fg=BG_MAIN,
            activebackground="#749ce0", activeforeground=BG_MAIN,
            font=FONT_BOLD, relief="flat", bd=0, cursor="hand2", pady=12,
        ).pack(fill="x", padx=15, pady=20)

        tk.Label(sb, text="CatR11.5", bg=BG_SIDEBAR, fg=FG_MAIN,
                 font=("Helvetica", 12, "bold"), anchor="w").pack(fill="x", padx=15, pady=(10, 2))
        tk.Label(sb, text="Pure BitNet 1.58b Transformer", bg=BG_SIDEBAR,
                 fg=FG_DIM, font=("Helvetica", 9), anchor="w").pack(fill="x", padx=15)
        tk.Label(sb, text="DeepSeek R1 Reasoning Protocol", bg=BG_SIDEBAR,
                 fg=ACCENT, font=("Helvetica", 9, "italic"), anchor="w").pack(fill="x", padx=15, pady=2)

        ma = tk.Frame(self, bg=BG_MAIN)
        ma.pack(side="right", expand=True, fill="both")

        self.chat = scrolledtext.ScrolledText(
            ma, bg=BG_MAIN, fg=FG_MAIN, font=FONT_MAIN, wrap=tk.WORD,
            insertbackground=FG_MAIN, bd=0, highlightthickness=0, padx=40, pady=20,
        )
        self.chat.pack(expand=True, fill="both")
        self.chat.config(state=tk.DISABLED)
        
        # DeepSeek styled tags
        self.chat.tag_configure("user",      foreground=FG_MAIN, font=FONT_BOLD)
        self.chat.tag_configure("bot",       foreground=ACCENT,  font=FONT_BOLD)
        self.chat.tag_configure("reasoning", foreground=FG_THINK, font=FONT_REASONING)
        self.chat.tag_configure("normal",    foreground=FG_MAIN, font=FONT_MAIN)

        ic = tk.Frame(ma, bg=BG_MAIN)
        ic.pack(fill="x", side="bottom", pady=20, padx=40)

        ib = tk.Frame(ic, bg=BG_INPUT, highlightbackground="#45475a", highlightthickness=1)
        ib.pack(fill="x", ipady=8, ipadx=10)

        self.entry = tk.Entry(
            ib, bg=BG_INPUT, fg=FG_MAIN, font=FONT_MAIN,
            insertbackground=FG_MAIN, bd=0, highlightthickness=0,
        )
        self.entry.pack(side="left", expand=True, fill="x", padx=10, pady=5)
        self.entry.bind("<Return>", lambda e: self._on_send())

        self.btn = tk.Button(
            ib, text="⇧", bg=ACCENT, fg=BG_MAIN, font=("Helvetica", 14, "bold"),
            activebackground="#749ce0", activeforeground=BG_MAIN,
            relief="flat", cursor="hand2", command=self._on_send, width=3
        )
        self.btn.pack(side="right", padx=5)

        # progress bar
        self.prog_frame = tk.Frame(ma, bg=BG_MAIN)
        self.prog_frame.pack(fill="x", padx=80, pady=(0, 5), side="bottom")
        self.prog_canvas = tk.Canvas(self.prog_frame, height=12, bg=BG_INPUT, highlightthickness=0)
        self.prog_canvas.pack(fill="x")
        self.prog_label = tk.Label(self.prog_frame, text="", bg=BG_MAIN, fg=FG_DIM, font=("Helvetica", 9))
        self.prog_label.pack(pady=5)

    def _lock_input(self, placeholder):
        self.entry.config(state=tk.DISABLED)
        self.btn.config(state=tk.DISABLED)

    def _unlock_input(self):
        self.entry.config(state=tk.NORMAL)
        self.btn.config(state=tk.NORMAL)
        self.entry.focus_set()

    def _post(self, sender: str, text: str, tag: str):
        self.chat.config(state=tk.NORMAL)
        self.chat.insert(tk.END, f"{sender}\n", tag)
        self.chat.insert(tk.END, f"{text}\n\n", "normal")
        self.chat.config(state=tk.DISABLED)
        self.chat.see(tk.END)

    def _start_training(self):
        self._post("CatR11.5", "Initializing architecture. Training on DeepSeek style conversational and coding corpus...", "bot")
        threading.Thread(target=self._train_thread, daemon=True).start()

    def _draw_progress(self, step: int, total: int, loss: float):
        w = self.prog_canvas.winfo_width()
        if w < 10: w = 400
        frac = step / total
        self.prog_canvas.delete("all")
        self.prog_canvas.create_rectangle(0, 0, w, 12, fill=BG_INPUT, outline="")
        self.prog_canvas.create_rectangle(0, 0, int(w * frac), 12, fill=ACCENT, outline="")
        self.prog_label.config(text=f"Training: {step}/{total}  |  Loss: {loss:.4f}")

    def _train_thread(self):
        t0 = time.time()
        def cb(step, total, loss):
            self.after(0, self._draw_progress, step, total, loss)
        self._engine.train(callback=cb)
        elapsed = time.time() - t0

        def finish():
            self.prog_frame.pack_forget()
            info = (
                f"Training complete in {elapsed:.1f}s!\n"
                f"Model behaves like DeepSeek R1, generating a <think> trace before output.\n"
                f"CatR11.5 now understands conversational English and can write basic Python 3.14 code.\n"
                f"\nTry asking: 'write some python code' or 'hello who are you ?'"
            )
            self._post("System", info, "bot")
            self._unlock_input()
        self.after(0, finish)

    def _on_send(self):
        if self._busy or not self._engine.trained: return
        prompt = self.entry.get().strip()
        if not prompt: return
        self.entry.delete(0, tk.END)
        self._post("You", prompt, "user")
        self._busy = True
        self._stop_evt.clear()
        threading.Thread(target=self._gen_thread, args=(prompt,), daemon=True).start()

    def _gen_thread(self, prompt: str):
        self.chat.config(state=tk.NORMAL)
        self.chat.insert(tk.END, "CatR11.5\n", "bot")
        self.chat.see(tk.END)
        self.update_idletasks()

        # Generate tokens dynamically formatting reasoning vs output
        for text, is_thinking in self._engine.generate(prompt, max_tokens=70, temperature=0.7):
            tag = "reasoning" if is_thinking else "normal"
            self.chat.config(state=tk.NORMAL)
            self.chat.insert(tk.END, text, tag)
            self.chat.see(tk.END)
            self.update_idletasks()

        self.chat.config(state=tk.NORMAL)
        self.chat.insert(tk.END, "\n\n", "normal")
        self.chat.config(state=tk.DISABLED)
        self._busy = False

if __name__ == "__main__":
    app = CatR115App()
    app.mainloop()
