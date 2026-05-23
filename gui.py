#!/usr/bin/env python3

from logging import root
import tkinter as tk
from tkinter import ttk
import threading
import time
import math
import random
import numpy as np

# ── NumPy 2.x compatibility patch ────────────────────────────
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import gym
import gym_maze
from GreedyPolicies import EGreedyPolicy
from QLearningAgent import QLearningAgent
from SarsaAgent import SarsaAgent
from PolicyGradientAgent import PolicyGradientAgent
from AStarAgent import AStarAgent


# ═════════════════════════════════════════════════════════════
#  DESIGN TOKENS  (Retro-Futuristic Cyberpunk Dark Theme)
# ═════════════════════════════════════════════════════════════
BG       = "#080c10"        # near-black background
PANEL    = "#0d1219"        # slightly lighter panels
CARD     = "#111820"        # card surfaces
BORDER   = "#1e2d3d"        # subtle borders
GLOW     = "#00d4ff"        # cyan primary accent
GREEN    = "#00ff88"        # success / agent 2
ORANGE   = "#ff8c42"        # SARSA color
PURPLE   = "#c084fc"        # PG color
PINK     = "#ff4d6d"        # stop / danger
YELLOW   = "#ffd60a"        # goal star
TEXT     = "#e2e8f0"        # primary text
MUTED    = "#4a6080"        # secondary text
WALL_C   = "#1a2535"        # maze wall color
CELL_C   = "#0d1520"        # maze cell color
CELL_ALT = "#0a1018"        # alternating cell

# ── Algorithm → accent color ──────────────────────────────────
ALGO_COLOR = {
    "Q-Learning"                  : GLOW,
    "SARSA"                       : ORANGE,
    "Policy Gradient (REINFORCE)" : PURPLE,
}

# ── Team roster ───────────────────────────────────────────────
TEAM = [
    ("Ahmed Maged Motea",        GLOW),
    ("Hassan Mohammed",          GREEN),
    ("Ahmed Yasser",             ORANGE),
    ("Saher Ayman",              PURPLE),
    ("Abdulrahman Al-Husseini",  PINK),
]

# Maze wall bit flags  (SET bit = opening exists in that direction)
OPEN_W, OPEN_N, OPEN_E, OPEN_S = 0x1, 0x2, 0x4, 0x8


# ── Color utility (tkinter only accepts 6-digit hex) ──────────
def dim_color(hex_color: str, factor: float) -> str:
    """
    Blend hex_color toward BG by factor.
    factor=0.0 → original color, factor=1.0 → BG color.
    Returns a valid 6-digit #rrggbb string.
    """
    r1, g1, b1 = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    r2, g2, b2 = int(BG[1:3], 16),        int(BG[3:5], 16),        int(BG[5:7], 16)
    r = int(r1 * (1 - factor) + r2 * factor)
    g = int(g1 * (1 - factor) + g2 * factor)
    b = int(b1 * (1 - factor) + b2 * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


# Pre-computed dim variants for static use
GLOW_DIM    = dim_color(GLOW,   0.60)   # ~40% opacity on BG
GLOW_FAINT  = dim_color(GLOW,   0.85)   # ~15% opacity on BG
GREEN_FAINT = dim_color(GREEN,  0.78)
PURP_FAINT  = dim_color(PURPLE, 0.78)
ORNG_FAINT  = dim_color(ORANGE, 0.82)


# ═════════════════════════════════════════════════════════════
class SplashScreen:
    """
    Animated full-screen splash.
    Particle field + scanline effect + team credits.
    Auto-advances after 4 s or on click.
    """

    def __init__(self, root: tk.Tk, on_close):
        self.root     = root
        self.on_close = on_close
        self._alive   = True

        W, H = 920, 580
        self.W, self.H = W, H

        self.win = tk.Toplevel(root)
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        sx = (self.win.winfo_screenwidth()  - W) // 2
        sy = (self.win.winfo_screenheight() - H) // 2
        self.win.geometry(f"{W}x{H}+{sx}+{sy}")
        self.win.configure(bg=BG)

        self.c = tk.Canvas(self.win, width=W, height=H,
                           bg=BG, highlightthickness=0)
        self.c.pack()

        # Particles
        self._particles = [
            {
                "x" : random.uniform(0, W),
                "y" : random.uniform(0, H),
                "vx": random.uniform(-0.6, 0.6),
                "vy": random.uniform(-0.4, 0.4),
                "r" : random.uniform(1.0, 2.8),
                "color": random.choice([
                    GLOW_DIM, GREEN_FAINT,
                    PURP_FAINT, ORNG_FAINT,
                ]),
            }
            for _ in range(90)
        ]

        self._phase      = 0.0    # animation phase
        self._auto_timer = 0      # auto-launch after 4 s

        self._draw_static()
        self.win.bind("<Button-1>", self._launch)
        self.win.bind("<Key>", self._launch)
        self._loop()

    # ── Static background elements ───────────────────────────
    def _draw_static(self):
        c = self.canvas = self.c
        W, H = self.W, self.H

        # Subtle grid
        for x in range(0, W, 40):
            c.create_line(x, 0, x, H, fill="#0e1520", width=1)
        for y in range(0, H, 40):
            c.create_line(0, y, W, y, fill="#0e1520", width=1)

        # Corner brackets
        sz = 28
        for rx, ry, dx, dy in [
            (8, 8, 1, 1), (W-8, 8, -1, 1),
            (8, H-8, 1, -1), (W-8, H-8, -1, -1)
        ]:
            c.create_line(rx, ry, rx+dx*sz, ry,
                          fill=GLOW, width=2)
            c.create_line(rx, ry, rx, ry+dy*sz,
                          fill=GLOW, width=2)

        # ── Maze icon ────────────────────────────────────────
        self._draw_logo(c, W // 2, 88)

        # ── Title ────────────────────────────────────────────
        c.create_text(W//2, 158,
                      text="MAZE  SOLVER",
                      fill=TEXT, font=("Courier New", 32, "bold"),
                      tags="title")
        c.create_text(W//2, 192,
                      text="R E I N F O R C E M E N T   L E A R N I N G",
                      fill=GLOW, font=("Courier New", 11),
                      tags="subtitle")

        # Horizontal rule
        self._hline(c, 216, W)

        # ── "Our Team" heading ────────────────────────────────
        c.create_text(W//2, 236,
                      text="— O U R   T E A M —",
                      fill=MUTED, font=("Courier New", 9))

        # ── Team names  (3 left | 2 right) ───────────────────
        L, R = W//2 - 205, W//2 + 205
        left_team  = TEAM[:3]
        right_team = TEAM[3:]

        for i, (name, clr) in enumerate(left_team):
            c.create_text(L, 264 + i * 34,
                          text=f"▸  {name}",
                          fill=clr,
                          font=("Courier New", 11, "bold"),
                          anchor="center")

        for i, (name, clr) in enumerate(right_team):
            c.create_text(R, 264 + i * 34,
                          text=f"◂  {name}",
                          fill=clr,
                          font=("Courier New", 11, "bold"),
                          anchor="center")

        # Horizontal rule
        self._hline(c, 372, W)

        # ── Supervisor ────────────────────────────────────────
        c.create_text(W//2, 398,
                      text="Under Supervision of   Dr. Sara Khalil",
                      fill=TEXT, font=("Courier New", 12, "italic"))

        # ── Course tag ────────────────────────────────────────
        c.create_text(W//2, 430,
                      text="AI Course Project  •  2026",
                      fill=MUTED, font=("Courier New", 9))

        # Horizontal rule
        self._hline(c, 455, W)

        # ── Hint ─────────────────────────────────────────────
        c.create_text(W//2, H - 26,
                      text="[ CLICK  ANYWHERE  TO  LAUNCH ]",
                      fill=MUTED, font=("Courier New", 9),
                      tags="hint")

    def _hline(self, c, y, W):
        pad = 60
        c.create_line(pad, y, W - pad, y, fill=BORDER, width=1)

    def _draw_logo(self, canvas, cx, cy):
        """4×4 mini maze grid with agent and goal dots."""
        s, lw = 13, 2
        ox, oy = cx - 2*s, cy - 2*s

        # Outer box
        canvas.create_rectangle(ox, oy, ox+4*s, oy+4*s,
                                 outline=GLOW, width=lw)
        # Interior walls
        segs = [
            (1, 0, 1, 2), (1, 2, 3, 2),
            (3, 2, 3, 4), (2, 1, 2, 3),
        ]
        for x1, y1, x2, y2 in segs:
            canvas.create_line(
                ox + x1*s, oy + y1*s,
                ox + x2*s, oy + y2*s,
                fill=GLOW_DIM, width=lw)

        r = 5
        # Goal dot (green, top-left area)
        canvas.create_oval(ox + 0.5*s - r, oy + 0.5*s - r,
                           ox + 0.5*s + r, oy + 0.5*s + r,
                           fill=GREEN, outline="")
        # Agent dot (cyan, bottom-right)
        canvas.create_oval(ox + 3.5*s - r, oy + 3.5*s - r,
                           ox + 3.5*s + r, oy + 3.5*s + r,
                           fill=GLOW, outline="")

    # ── Animation loop ───────────────────────────────────────
    def _loop(self):
        if not self._alive:
            return

        self._phase      += 0.05
        self._auto_timer += 1
        c = self.c
        W, H = self.W, self.H

        # ── Particles ────────────────────────────────────────
        c.delete("fx")
        for p in self._particles:
            p["x"] = (p["x"] + p["vx"]) % W
            p["y"] = (p["y"] + p["vy"]) % H
            r = p["r"]
            c.create_oval(p["x"]-r, p["y"]-r,
                          p["x"]+r, p["y"]+r,
                          fill=GLOW_FAINT, outline="",
                          tags="fx")
        c.tag_lower("fx")

        # ── Pulsing outer glow border ─────────────────────────
        c.delete("border")
        pulse = 0.5 + 0.5 * math.sin(self._phase * 1.8)
        alpha = int(80 + 80 * pulse)
        color = f"#{alpha:02x}{min(alpha + 60, 255):02x}ff"
        c.create_rectangle(5, 5, W-5, H-5,
                           outline=color, width=2,
                           tags="border")

        # ── Scanline overlay (every 4px) ──────────────────────
        c.delete("scan")
        offset = int(self._phase * 8) % 4
        for yy in range(offset, H, 8):
            c.create_line(0, yy, W, yy,
                          fill="#0d0f12", width=1,
                          tags="scan")
        c.tag_raise("scan")

        # ── Hint blink ────────────────────────────────────────
        show = int(self._phase * 2) % 2 == 0
        c.itemconfig("hint",
                     fill=MUTED if show else BG)

        # Auto-launch after ~4 s (80 ticks × 50 ms)
        if self._auto_timer >= 80:
            self._launch()
            return

        self.win.after(50, self._loop)

    def _launch(self, _=None):
        if not self._alive:
            return
        self._alive = False
        self.win.destroy()
        self.on_close()


# ═════════════════════════════════════════════════════════════
class MazeApp:
    """Main application window."""

    CELL = 28   # تقليل حجم الخلية للشاشات الصغيرة

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Maze Solver — Reinforcement Learning")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)

        # ── Gym env ──────────────────────────────────────────
        self.env = gym.make("Maze-v0")
        self._detect_maze()

        # ── State vars ───────────────────────────────────────
        self.training        = False
        self.stop_flag       = False
        self.episode_rewards = []
        self.algo_var        = tk.StringVar(value="Q-Learning")
        self.episodes_var    = tk.IntVar(value=50)
        self.speed_var       = tk.DoubleVar(value=0.05)
        self.trained_agent   = None
        self.best_rl_steps   = None
        self.astar           = AStarAgent(self.env)

        # ── Build layout ──────────────────────────────────────
        self._build_header()
        self._build_body()
        self._build_footer()

        # ── Initial maze draw ─────────────────────────────────
        self.root.update()
        self._draw_maze()
        self._draw_agent(self.agent_start[0], self.agent_start[1])

    # ── Maze structure detection ──────────────────────────────
    def _detect_maze(self):
        """Read internal maze structure from gym env."""
        self.maze_cells  = None
        self.has_walls   = False
        self.MAZE_H      = 9
        self.MAZE_W      = 9
        self.goal        = [3, 3]
        self.agent_start = [4, 4]

        try:
            raw = self.env.unwrapped

            # 1. Grab maze_cells directly from the unwrapped environment
            if hasattr(raw, "maze_cells"):
                self.maze_cells = raw.maze_cells
                self.MAZE_H     = self.maze_cells.shape[0] - 2
                self.MAZE_W     = self.maze_cells.shape[1] - 2
                self.has_walls  = True

            for attr in ("goal", "goal_pos", "target", "objective_position"):
                if hasattr(raw, attr):
                    g = getattr(raw, attr)
                    self.goal = [int(g[0]), int(g[1])]
                    break

            # 2. Prioritize static start position
            for attr in ("player_marker_start_pos", "init_pos", "robot_pos", "state"):
                if hasattr(raw, attr):
                    s = getattr(raw, attr)
                    self.agent_start = [int(s[1]), int(s[0])]
                    break
        except Exception:
            pass

    # ── Header ────────────────────────────────────────────────
    def _build_header(self):
        hf = tk.Frame(self.root, bg=PANEL, pady=4)
        hf.pack(fill="x")

        tk.Label(
            hf,
            text="⬡  MAZE SOLVER — REINFORCEMENT LEARNING",
            bg=PANEL, fg=TEXT,
            font=("Courier New", 13, "bold")
        ).pack()

        # Team names row
        tf = tk.Frame(hf, bg=PANEL)
        tf.pack(pady=(2, 1))
        tk.Label(tf, text="Our Team : ",
                 bg=PANEL, fg=MUTED,
                 font=("Courier New", 7)).pack(side="left")
        for i, (name, clr) in enumerate(TEAM):
            tk.Label(tf, text=name,
                     bg=PANEL, fg=clr,
                     font=("Courier New", 7, "bold")
                     ).pack(side="left")
            if i < len(TEAM) - 1:
                tk.Label(tf, text="  |  ",
                         bg=PANEL, fg=BORDER,
                         font=("Courier New", 7)
                         ).pack(side="left")

        tk.Label(
            hf,
            text="Under Supervision of  Dr. Sara Khalil",
            bg=PANEL, fg=MUTED,
            font=("Courier New", 7, "italic")
        ).pack()

        # Accent line below header
        tk.Canvas(self.root, height=1, bg=GLOW,
                  highlightthickness=0).pack(fill="x")

    # ── Body ──────────────────────────────────────────────────
    def _build_body(self):
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=8, pady=6)
        self._build_maze_panel(body)
        self._build_right_panel(body)

    # ── Maze canvas ───────────────────────────────────────────
    def _build_maze_panel(self, parent):
        cw = self.MAZE_W * self.CELL
        ch = self.MAZE_H * self.CELL

        # Label above canvas
        lf = tk.Frame(parent, bg=BG)
        lf.pack(side="left", anchor="n", padx=(0, 8))

        tk.Label(lf, text="MAZE  ENVIRONMENT",
                 bg=BG, fg=MUTED,
                 font=("Courier New", 7, "bold")).pack(anchor="w", pady=(0, 2))

        # Canvas wrap with glow border
        wrap = tk.Frame(lf, bg=GLOW, padx=1, pady=1)
        wrap.pack()

        inner = tk.Frame(wrap, bg=CARD)
        inner.pack()

        self.canvas = tk.Canvas(inner, width=cw, height=ch,
                                bg=CELL_C, highlightthickness=0)
        self.canvas.pack()

    # ── Right control panel ───────────────────────────────────
    def _build_right_panel(self, parent):
        pf = tk.Frame(parent, bg=BG, width=250)
        pf.pack(side="left", fill="y")
        pf.pack_propagate(False)

        # ── Algorithm ──────────────────────────────────────
        self._sec(pf, "ALGORITHM")
        for algo, clr in ALGO_COLOR.items():
            row = tk.Frame(pf, bg=CARD, pady=2, padx=6,
                           highlightbackground=BORDER,
                           highlightthickness=1)
            row.pack(fill="x", pady=1)
            tk.Radiobutton(
                row, text=algo,
                variable=self.algo_var, value=algo,
                bg=CARD, fg=clr, selectcolor=BG,
                activebackground=CARD,
                font=("Courier New", 9, "bold"),
                relief="flat", indicatoron=1,
                highlightthickness=0
            ).pack(anchor="w")

        self._div(pf)

        # ── Parameters ─────────────────────────────────────
        self._sec(pf, "PARAMETERS")
        self._slider(pf, "Episodes",      self.episodes_var, 10, 200, 10)
        self._slider(pf, "Step delay(s)", self.speed_var,   0.0, 0.3, 0.01)

        self._div(pf)

        # ── Live stats ──────────────────────────────────────
        self._sec(pf, "LIVE STATISTICS")
        stat = tk.Frame(pf, bg=CARD, padx=6, pady=4,
                        highlightbackground=BORDER,
                        highlightthickness=1)
        stat.pack(fill="x")
        self.lbl_ep     = self._stat(stat, "Episode", "—")
        self.lbl_steps  = self._stat(stat, "Steps",   "—")
        self.lbl_reward = self._stat(stat, "Reward",  "—")
        self.lbl_status = self._stat(stat, "Status",  "Idle")

        self._div(pf)

        # ── Buttons ──────────────────────────────────────────
        self.btn_start = tk.Button(
            pf, text="▶ START",
            bg=GREEN, fg=BG,
            font=("Courier New", 10, "bold"),
            relief="flat", pady=5, cursor="hand2",
            activebackground="#00cc70",
            command=self._start_training)
        self.btn_start.pack(fill="x", pady=(0, 2))

        self.btn_stop = tk.Button(
            pf, text="■ STOP",
            bg=CARD, fg=PINK,
            font=("Courier New", 10, "bold"),
            relief="flat", pady=5, cursor="hand2",
            state="disabled",
            activebackground=CARD,
            command=self._stop_training)
        self.btn_stop.pack(fill="x")

        self._div(pf)

        # ── Reward chart ─────────────────────────────────────
        self._sec(pf, "REWARD CHART")
        self.chart = tk.Canvas(pf, bg=CARD, height=50,
                               highlightbackground=BORDER,
                               highlightthickness=1)
        self.chart.pack(fill="x")

        self._div(pf)

        # ── A* Comparison ────────────────────────────────────
        self._sec(pf, "A* vs RL")
        cmp_card = tk.Frame(pf, bg=CARD, padx=6, pady=4,
                            highlightbackground=BORDER,
                            highlightthickness=1)
        cmp_card.pack(fill="x")
        self.lbl_astar_steps = self._stat(cmp_card, "A* Steps",  "—")
        self.lbl_rl_best     = self._stat(cmp_card, "RL Best",   "—")
        self.lbl_efficiency  = self._stat(cmp_card, "RL / A*",   "—")

        btn_row = tk.Frame(pf, bg=BG)
        btn_row.pack(fill="x", pady=(4, 0))

        self.btn_astar = tk.Button(
            btn_row, text="⬡ A* PATH",
            bg=CARD, fg=GLOW,
            font=("Courier New", 8, "bold"),
            relief="flat", pady=4, cursor="hand2",
            highlightbackground=GLOW, highlightthickness=1,
            command=self._run_astar)
        self.btn_astar.pack(side="left", fill="x", expand=True, padx=(0, 2))

        self.btn_arrows = tk.Button(
            btn_row, text="⬡ POLICY",
            bg=CARD, fg=ORANGE,
            font=("Courier New", 8, "bold"),
            relief="flat", pady=4, cursor="hand2",
            highlightbackground=ORANGE, highlightthickness=1,
            state="disabled",
            command=self._draw_policy_arrows)
        self.btn_arrows.pack(side="left", fill="x", expand=True, padx=(2, 0))

    # ── Footer ────────────────────────────────────────────────
    def _build_footer(self):
        tk.Canvas(self.root, height=1, bg=GLOW,
                  highlightthickness=0).pack(fill="x")
        ff = tk.Frame(self.root, bg=PANEL, pady=3)
        ff.pack(fill="x")
        tk.Label(
            ff,
            text="RL Maze Solver • AI Course • 2026",
            bg=PANEL, fg=MUTED,
            font=("Courier New", 7)
        ).pack()

    # ── Widget helpers ────────────────────────────────────────
    def _sec(self, parent, title):
        tk.Label(parent, text=title,
                 bg=BG, fg=MUTED,
                 font=("Courier New", 6, "bold")
                 ).pack(anchor="w", pady=(4, 1))

    def _div(self, parent):
        tk.Canvas(parent, height=1, bg=BORDER,
                  highlightthickness=0).pack(fill="x", pady=2)

    def _slider(self, parent, label, var, lo, hi, res):
        f = tk.Frame(parent, bg=BG)
        f.pack(fill="x", pady=1)
        tk.Label(f, text=label, bg=BG, fg=TEXT,
                 font=("Courier New", 7), width=12,
                 anchor="w").pack(side="left")
        tk.Label(f, textvariable=var, bg=BG, fg=GLOW,
                 font=("Courier New", 7, "bold"), width=4
                 ).pack(side="right")
        ttk.Scale(f, from_=lo, to=hi, variable=var,
                  orient="horizontal"
                  ).pack(side="left", fill="x", expand=True, padx=2)

    def _stat(self, parent, label, init):
        f = tk.Frame(parent, bg=CARD)
        f.pack(fill="x", pady=0)
        tk.Label(f, text=f"{label}:", bg=CARD, fg=MUTED,
                 font=("Courier New", 7), width=8,
                 anchor="w").pack(side="left")
        lbl = tk.Label(f, text=init, bg=CARD, fg=TEXT,
                       font=("Courier New", 8, "bold"))
        lbl.pack(side="left")
        return lbl

    # ── Maze rendering ────────────────────────────────────────
    def _draw_maze(self):
        c    = self.canvas
        CELL = self.CELL
        H, W = self.MAZE_H, self.MAZE_W
        c.delete("maze")

        for row in range(H):
            for col in range(W):
                x0, y0 = col * CELL, row * CELL
                x1, y1 = x0 + CELL, y0 + CELL

                if [row + 1, col + 1] == self.goal:
                    fill = "#0d2010"
                else:
                    fill = CELL_C if (row + col) % 2 == 0 else CELL_ALT

                c.create_rectangle(x0, y0, x1, y1,
                                   fill=fill, outline="",
                                   tags="maze")

                if self.has_walls and self.maze_cells is not None:
                    v  = int(self.maze_cells[row + 1][col + 1])
                    ww = 2
                    if not (v & OPEN_N):
                        c.create_line(x0, y0, x1, y0,
                                      fill=WALL_C, width=ww, tags="maze")
                    if not (v & OPEN_S):
                        c.create_line(x0, y1, x1, y1,
                                      fill=WALL_C, width=ww, tags="maze")
                    if not (v & OPEN_W):
                        c.create_line(x0, y0, x0, y1,
                                      fill=WALL_C, width=ww, tags="maze")
                    if not (v & OPEN_E):
                        c.create_line(x1, y0, x1, y1,
                                      fill=WALL_C, width=ww, tags="maze")
                else:
                    c.create_rectangle(x0, y0, x1, y1,
                                       fill=fill,
                                       outline=WALL_C,
                                       tags="maze")

        # Outer border glow
        c.create_rectangle(0, 0, W*CELL - 1, H*CELL - 1,
                           outline=GLOW, width=1, tags="maze")

        # Goal marker
        gr, gc = self.goal
        gx = (gc - 1) * CELL + CELL // 2
        gy = (gr - 1) * CELL + CELL // 2
        r  = CELL // 3
        c.create_oval(gx-r, gy-r, gx+r, gy+r,
                      fill="#0d3018", outline=GREEN,
                      width=1, tags="maze")
        c.create_text(gx, gy, text="★",
                      fill=YELLOW,
                      font=("Courier New", int(CELL * 0.35), "bold"),
                      tags="maze")

    def _draw_agent(self, row, col):
        c    = self.canvas
        CELL = self.CELL
        c.delete("agent")
        cx = (col - 1) * CELL + CELL // 2
        cy = (row - 1) * CELL + CELL // 2
        r  = CELL // 3
        clr = ALGO_COLOR.get(self.algo_var.get(), GLOW)

        c.create_oval(cx-r-5, cy-r-5, cx+r+5, cy+r+5,
                      fill="", outline=dim_color(clr, 0.75),
                      width=3, tags="agent")
        c.create_oval(cx-r-1, cy-r-1, cx+r+1, cy+r+1,
                      fill="", outline=dim_color(clr, 0.45),
                      width=1, tags="agent")
        c.create_oval(cx-r, cy-r, cx+r, cy+r,
                      fill=clr, outline="white",
                      width=1, tags="agent")
        c.create_oval(cx-3, cy-3, cx+3, cy+3,
                      fill="white", outline="",
                      tags="agent")

    # ── Training logic ────────────────────────────────────────
    def _start_training(self):
        if self.training:
            return
        self.training        = True
        self.stop_flag       = False
        self.episode_rewards = []
        self.best_rl_steps   = None
        self.trained_agent   = None

        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.lbl_status.config(text="Training…", fg=GREEN)
        self._draw_maze()

        threading.Thread(target=self._train_thread,
                         daemon=True).start()

    def _stop_training(self):
        self.stop_flag = True

    def _train_thread(self):
        algo     = self.algo_var.get()
        episodes = int(self.episodes_var.get())
        delay    = float(self.speed_var.get())
        policy   = EGreedyPolicy(decay=True, epsilon=0.3)

        if algo == "Q-Learning":
            agent = QLearningAgent(env=self.env, policy=policy)
        elif algo == "SARSA":
            agent = SarsaAgent(env=self.env, policy=policy)
        else:
            agent = PolicyGradientAgent(env=self.env,
                                        alpha=0.01, gamma=0.99)

        self.env.initialize_env()

        # تحديث الـ maze بعد إعادة التوليد
        self._detect_maze()
        self.root.after(0, self._draw_maze)

        for ep in range(episodes):
            if self.stop_flag:
                break

            state, reward, done, _ = self.env.reset()
            total_reward = reward
            steps        = 0
            pg_buffer    = []

            if algo == "SARSA":
                action = agent.choose_action(state)

            while not done and not self.stop_flag:
                try:
                    pos = list(np.array(state).flatten())
                    row, col = int(pos[0]), int(pos[1])
                except Exception:
                    row, col = 0, 0

                self.root.after(0, self._draw_agent, row, col)

                if algo == "SARSA":
                    ns, r, done, _ = self.env.step(action)
                    na = agent.choose_action(ns)
                    agent.learn(state, action, r, ns, na, done)
                    action = na
                elif algo == "Q-Learning":
                    action = agent.choose_action(state)
                    ns, r, done, _ = self.env.step(action)
                    agent.learn(state, action, r, ns, done)
                else:
                    action = agent.choose_action(state)
                    ns, r, done, _ = self.env.step(action)
                    pg_buffer.append((state, action, r))

                state         = ns
                total_reward += r
                steps        += 1

                ep_txt = f"{ep + 1} / {episodes}"
                rw_txt = f"{total_reward:.2f}"
                st_txt = str(steps)
                self.root.after(
                    0, lambda t=ep_txt: self.lbl_ep.config(text=t))
                self.root.after(
                    0, lambda t=st_txt: self.lbl_steps.config(text=t))
                self.root.after(
                    0, lambda t=rw_txt: self.lbl_reward.config(text=t))

                time.sleep(delay)

            if algo == "Policy Gradient (REINFORCE)" and pg_buffer:
                agent.learn(pg_buffer)

            self.episode_rewards.append(total_reward)

            if done and (self.best_rl_steps is None or steps < self.best_rl_steps):
                self.best_rl_steps = steps

            self.root.after(0, self._update_chart)

        self.trained_agent = agent
        self.training = False
        self.root.after(0, self._training_done)

    def _training_done(self):
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        if self.stop_flag:
            self.lbl_status.config(text="Stopped", fg=ORANGE)
        else:
            self.lbl_status.config(text="Complete ✓", fg=GREEN)

        algo = self.algo_var.get()
        if algo in ("Q-Learning", "SARSA"):
            self.btn_arrows.config(state="normal")

        if self.best_rl_steps is not None:
            self.lbl_rl_best.config(text=str(self.best_rl_steps), fg=GREEN)
            astar_steps = self.astar.steps
            if astar_steps > 0:
                ratio = self.best_rl_steps / astar_steps
                color = GREEN if ratio <= 1.5 else (ORANGE if ratio <= 3 else PINK)
                self.lbl_efficiency.config(
                    text=f"{ratio:.2f}x", fg=color)

    # ── A* Pathfinding ────────────────────────────────────────
    def _run_astar(self):
        """Run A* and overlay its path on the maze canvas."""
        # ═══ التعديل المهم: تحديث A* بالـ maze الحالي ═══
        self.astar.env = self.env  # تحديث الـ environment
        self.astar._detect_maze()  # إعادة كشف الـ maze

        # تحديث الـ start position بالموقع الحالي للـ agent
        start = self.agent_start
        goal  = self.goal

        path = self.astar.find_path(start=start, goal=goal)

        if not path:
            self.lbl_astar_steps.config(text="No path!", fg=PINK)
            return

        self.lbl_astar_steps.config(text=str(self.astar.steps), fg=GLOW)

        if self.best_rl_steps is not None:
            ratio = self.best_rl_steps / max(self.astar.steps, 1)
            color = GREEN if ratio <= 1.5 else (ORANGE if ratio <= 3 else PINK)
            self.lbl_efficiency.config(text=f"{ratio:.2f}x", fg=color)

        self._draw_maze()
        self._draw_astar_path(path)

    def _draw_astar_path(self, path):
        """Draw A* path as a glowing trail on the canvas."""
        c    = self.canvas
        CELL = self.CELL
        c.delete("astar")

        if len(path) < 2:
            return

        for i, (row, col) in enumerate(path):
            cx = (col - 1) * CELL + CELL // 2
            cy = (row - 1) * CELL + CELL // 2

            if [row, col] == self.goal:
                continue

            x0, y0 = (col - 1) * CELL + 3, (row - 1) * CELL + 3
            x1, y1 = x0 + CELL - 6,  y0 + CELL - 6
            c.create_rectangle(x0, y0, x1, y1,
                               fill=dim_color(GLOW, 0.70),
                               outline=GLOW,
                               width=1, tags="astar")

            c.create_text(cx, cy,
                          text=str(i),
                          fill=GLOW,
                          font=("Courier New", int(CELL * 0.22), "bold"),
                          tags="astar")

        pts = []
        for row, col in path:
            pts.extend([(col - 1) * CELL + CELL // 2,
                        (row - 1) * CELL + CELL // 2])
        if len(pts) >= 4:
            c.create_line(*pts,
                          fill=GLOW, width=1,
                          dash=(4, 2), smooth=True,
                          tags="astar")

        c.create_text(4, 4,
                      text=f"A*: {self.astar.steps} steps",
                      fill=GLOW,
                      font=("Courier New", 7, "bold"),
                      anchor="nw", tags="astar")

    # ── Policy Arrows ─────────────────────────────────────────
    def _draw_policy_arrows(self):
        agent = self.trained_agent
        if agent is None or not hasattr(agent, "get_q_value"):
            return

        c    = self.canvas
        CELL = self.CELL
        c.delete("arrows")

        ACTION_MAP = {
            0: (-1,  0, "↑"),
            1: ( 1,  0, "↓"),
            2: ( 0, -1, "←"),
            3: ( 0,  1, "→"),
        }

        clr = ALGO_COLOR.get(self.algo_var.get(), GLOW)

        for row in range(self.MAZE_H):
            for col in range(self.MAZE_W):
                if self.maze_cells is not None:
                    if self.maze_cells[row + 1][col + 1] == 0:
                        continue

                if [row + 1, col + 1] == self.goal:
                    continue

                state = [row + 1, col + 1]
                actions = list(range(self.env.get_action_space().n))
                q_vals  = [agent.get_q_value(state, a) for a in actions]

                if max(abs(v) for v in q_vals) < 1e-6:
                    continue

                best_a  = actions[q_vals.index(max(q_vals))]
                symbol  = ACTION_MAP.get(best_a, ("?",))[2]

                cx = col * CELL + CELL // 2
                cy = row * CELL + CELL // 2

                c.create_text(cx, cy,
                              text=symbol,
                              fill=clr,
                              font=("Courier New",
                                    int(CELL * 0.32), "bold"),
                              tags="arrows")

        c.create_text(self.MAZE_W * CELL - 4, 4,
                      text="Policy ↑↓←→",
                      fill=clr,
                      font=("Courier New", 7, "bold"),
                      anchor="ne", tags="arrows")

    # ── Reward chart ──────────────────────────────────────────
    def _update_chart(self):
        c       = self.chart
        rewards = self.episode_rewards
        W       = c.winfo_width()
        H       = c.winfo_height()
        c.delete("all")

        if len(rewards) < 2 or W <= 1:
            return

        mn, mx = min(rewards), max(rewards)
        if mx == mn:
            mx = mn + 1.0
        PAD = 6

        def px(i):
            return PAD + (i / (len(rewards) - 1)) * (W - 2*PAD)

        def py(v):
            return H - PAD - ((v - mn) / (mx - mn)) * (H - 2*PAD)

        clr  = ALGO_COLOR.get(self.algo_var.get(), GLOW)
        pts  = [(px(i), py(v)) for i, v in enumerate(rewards)]

        poly = [(PAD, H-PAD)] + pts + [(W-PAD, H-PAD)]
        c.create_polygon(
            [coord for p in poly for coord in p],
            fill=dim_color(clr, 0.82), outline="")

        for i in range(len(pts) - 1):
            c.create_line(*pts[i], *pts[i+1],
                          fill=clr, width=1, smooth=True)

        c.create_oval(pts[-1][0]-3, pts[-1][1]-3,
                      pts[-1][0]+3, pts[-1][1]+3,
                      fill=clr, outline="white")

        c.create_text(W - 2, 4,
                      text=f"max {mx:.1f}",
                      fill=MUTED, font=("Courier New", 6),
                      anchor="ne")
        c.create_text(W - 2, H - 4,
                      text=f"min {mn:.1f}",
                      fill=MUTED, font=("Courier New", 6),
                      anchor="se")
        c.create_text(2, H // 2,
                      text=f"{rewards[-1]:.1f}",
                      fill=clr,
                      font=("Courier New", 8, "bold"),
                      anchor="w")


# ═════════════════════════════════════════════════════════════
def main():
    root = tk.Tk()
    root.withdraw()

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("Horizontal.TScale",
                    background=BG,
                    troughcolor=CARD,
                    sliderlength=12,
                    sliderwidth=12)

    def launch():
        root.deiconify()

        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()

        w = int(screen_w * 0.95)
        h = int(screen_h * 0.90)
        x = (screen_w - w) // 2
        y = (screen_h - h) // 2

        root.geometry(f"{w}x{h}+{x}+{y}")
        root.minsize(800, 600)

        MazeApp(root)
        root.lift()
        root.focus_force()

    SplashScreen(root, launch)
    root.mainloop()


if __name__ == "__main__":
    main()
