"""
DMD 扫描曝光仿真演示程序
运行: D:\installpath\Python311\python.exe main.py
"""
import sys
import math
import colorsys
import numpy as np
import matplotlib
import matplotlib.colors as mcolors
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size']        = 13
matplotlib.rcParams['axes.titlesize']   = 15
matplotlib.rcParams['axes.labelsize']   = 13
matplotlib.rcParams['xtick.labelsize']  = 12
matplotlib.rcParams['ytick.labelsize']  = 12
matplotlib.rcParams['legend.fontsize']  = 12
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QTabWidget, QLabel, QComboBox, QSpinBox,
    QPushButton, QGroupBox, QFormLayout, QSizePolicy, QFrame
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont


# ================================================================
# Config
# ================================================================
class Config:
    DMD_ROWS: int = 6
    DMD_COLS: int = 9
    N: int = 3
    L: float = 1.0
    PW: float = L / math.sqrt(N ** 2 + 1)
    DEFAULT_PATTERN: str = '横线'
    DEFAULT_LINE_WIDTH_PW: int = 3
    DEFAULT_M: int = 3
    BITMAP_SIZE: int = 60
    TIMER_MS: int = 120


# ================================================================
# DMDGeometry
# ================================================================
class DMDGeometry:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def pos(self, col: int, row: int):
        N = self.cfg.N
        return float(N * col - row), float(col + N * row)

    def all_positions(self) -> np.ndarray:
        rows, cols = self.cfg.DMD_ROWS, self.cfg.DMD_COLS
        arr = np.zeros((rows, cols, 2))
        for r in range(rows):
            for c in range(cols):
                arr[r, c] = self.pos(c, r)
        return arr

    def vertical_groups(self) -> dict:
        groups: dict = {}
        for r in range(self.cfg.DMD_ROWS):
            for c in range(self.cfg.DMD_COLS):
                x = int(self.cfg.N * c - r)
                groups.setdefault(x, []).append((c, r))
        return groups

    def mirror_corners(self, col: int, row: int) -> np.ndarray:
        N  = self.cfg.N
        cx, cy = self.pos(col, row)
        c   = np.array([cx, cy])
        hc  = np.array([ N / 2.0,  0.5   ])
        hr  = np.array([-0.5,      N / 2.0])
        return np.array([c + hc + hr, c + hc - hr, c - hc - hr, c - hc + hr])

    def x_range(self):
        pos = self.all_positions()
        return pos[:, :, 0].min(), pos[:, :, 0].max()

    def y_range(self):
        pos = self.all_positions()
        return pos[:, :, 1].min(), pos[:, :, 1].max()


# ================================================================
# PatternGen
# ================================================================
class PatternGen:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def generate(self, pattern: str, line_width_pw: int) -> np.ndarray:
        sz = self.cfg.BITMAP_SIZE
        bmp = np.zeros((sz, sz), dtype=np.float32)
        w = max(1, int(line_width_pw))
        period = w * 2

        if pattern == '横线':
            for y in range(sz):
                if (y % period) < w:
                    bmp[y, :] = 1.0
        elif pattern == '竖线':
            for x in range(sz):
                if (x % period) < w:
                    bmp[:, x] = 1.0
        elif pattern == '圆环':
            cx, cy = sz / 2.0, sz / 2.0
            yy, xx = np.mgrid[0:sz, 0:sz]
            r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            bmp = ((r % period) < w).astype(np.float32)

        return bmp


# ================================================================
# 全局样式 & 曝光颜色映射
# ================================================================
BG_DARK   = '#080c14'
BG_PANEL  = '#0d1525'
GRID_COL  = '#1e2d45'
TICK_COL  = '#64748b'

# 离散曝光次数颜色：0次→深色, 1次→蓝, 2次→琥珀, ≥3次→红
_CMAP_EXP  = mcolors.ListedColormap(['#0a0e1a', '#1e40af', '#d97706', '#b91c1c'])
_NORM_EXP  = mcolors.BoundaryNorm([0, 0.5, 1.5, 2.5, 1e9], _CMAP_EXP.N)

def _style_ax(ax, fig):
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_PANEL)
    ax.tick_params(colors=TICK_COL, labelsize=12)
    ax.xaxis.label.set_color(TICK_COL)
    ax.yaxis.label.set_color(TICK_COL)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e3a5f')
    ax.grid(True, color=GRID_COL, linewidth=0.6, linestyle='--', alpha=0.7)


def _pw_grid(ax, xl, xr, yb, yt, color='#2d4a70', lw=0.55, alpha=0.7):
    """在指定数据范围内绘制固定 1pw 像素网格（半整数位置），不随缩放改变。"""
    xs = np.arange(math.floor(xl) - 0.5, math.ceil(xr) + 1.0, 1.0)
    ys = np.arange(math.floor(yb) - 0.5, math.ceil(yt) + 1.0, 1.0)
    ax.set_xticks(xs, minor=True)
    ax.set_yticks(ys, minor=True)
    ax.tick_params(which='minor', length=0)
    ax.grid(which='minor', color=color, linewidth=lw, alpha=alpha)


# ================================================================
# Matplotlib Canvas
# ================================================================
class MplCanvas(FigureCanvas):
    def __init__(self, figsize=(5, 4), dpi=90):
        self.fig = Figure(figsize=figsize, dpi=dpi)
        self.fig.patch.set_facecolor(BG_DARK)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(100, 100)


class CanvasWithToolbar(QWidget):
    def __init__(self, figsize=(5, 4), dpi=90, parent=None):
        super().__init__(parent)
        self.canvas = MplCanvas(figsize=figsize, dpi=dpi)

        self._btn_home = QPushButton('⌂  复位视图')
        self._btn_home.setMaximumHeight(26)
        self._btn_home.setStyleSheet(
            'QPushButton{background:#0d1525;color:#60a5fa;border:none;'
            'font-size:12pt;padding:0 6px;border-radius:2px;}'
            'QPushButton:hover{background:#1e3a5f;}')
        self._btn_home.clicked.connect(self._reset_view)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self._btn_home)
        lay.addWidget(self.canvas)

        self._press_data = None
        self._press_lims = None
        self._home_lims  = {}

        self.canvas.mpl_connect('scroll_event',         self._on_scroll)
        self.canvas.mpl_connect('button_press_event',   self._on_press)
        self.canvas.mpl_connect('button_release_event', self._on_release)
        self.canvas.mpl_connect('motion_notify_event',  self._on_motion)

    @property
    def fig(self):
        return self.canvas.fig

    def draw(self):
        self.canvas.draw()
        for ax in self.canvas.fig.axes:
            if ax not in self._home_lims:
                self._home_lims[ax] = (ax.get_xlim(), ax.get_ylim())

    def _on_scroll(self, event):
        ax = event.inaxes
        if ax is None or event.xdata is None:
            return
        factor = 0.82 if event.button == 'up' else 1.0 / 0.82
        xd, yd = event.xdata, event.ydata
        xl, xr = ax.get_xlim();  yb, yt = ax.get_ylim()
        ax.set_xlim(xd + (xl - xd) * factor, xd + (xr - xd) * factor)
        ax.set_ylim(yd + (yb - yd) * factor, yd + (yt - yd) * factor)
        self.canvas.draw_idle()

    def _on_press(self, event):
        ax = event.inaxes
        if event.button != 1 or ax is None or event.xdata is None:
            return
        self._press_data = (event.xdata, event.ydata)
        self._press_lims = (list(ax.get_xlim()), list(ax.get_ylim()), ax)

    def _on_motion(self, event):
        if self._press_data is None or event.inaxes is None:
            return
        ax = self._press_lims[2]
        if event.xdata is None or event.ydata is None:
            return
        dx = event.xdata - self._press_data[0]
        dy = event.ydata - self._press_data[1]
        xl0, xr0 = self._press_lims[0];  yb0, yt0 = self._press_lims[1]
        ax.set_xlim(xl0 - dx, xr0 - dx)
        ax.set_ylim(yb0 - dy, yt0 - dy)
        self.canvas.draw_idle()

    def _on_release(self, event):
        if event.button == 1:
            self._press_data = None
            self._press_lims = None

    def _reset_view(self):
        for ax, (xl, yl) in self._home_lims.items():
            ax.set_xlim(xl);  ax.set_ylim(yl)
        self.canvas.draw_idle()


# ================================================================
# 界面 1：镜面位置关系
# ================================================================
class Tab1View(QWidget):
    def __init__(self, cfg: Config, geom: DMDGeometry):
        super().__init__()
        self.cfg = cfg
        self.geom = geom
        self._array_label_texts = []   # 供动态调字号

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.cv_tilt  = CanvasWithToolbar(figsize=(5, 5))
        self.cv_array = CanvasWithToolbar(figsize=(7, 5))
        layout.addWidget(self.cv_tilt, 2)
        layout.addWidget(self.cv_array, 3)
        self._draw_tilt()
        self._draw_array()

    # ---- 左图 ----
    def _draw_tilt(self):
        fig = self.cv_tilt.fig
        fig.clear()
        ax = fig.add_subplot(111)
        _style_ax(ax, fig)
        N = self.cfg.N

        SHOW_R, SHOW_C = 4, 4
        pts = {}
        for r in range(SHOW_R):
            for c in range(SHOW_C):
                x, y = self.geom.pos(c, r)
                pts[(c, r)] = (x, y)
                corners = self.geom.mirror_corners(c, r)
                poly = plt.Polygon(corners, closed=True,
                                   fc='#1e3a5f', ec='#4da6ff', lw=1.2, alpha=0.85, zorder=3)
                ax.add_patch(poly)
                ax.plot(x, y, 'o', color='#7dd3fc', ms=4, zorder=5)
                ax.text(x, y + self.cfg.PW * N * 0.7, f'({c},{r})',
                        fontsize=11, ha='center', color='#93c5fd', zorder=6)

        for c in range(SHOW_C - 1):
            x0, y0 = pts[(c, 0)];  x1, y1 = pts[(c + 1, 0)]
            ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle='->', color='#60a5fa', lw=1.8, mutation_scale=14))
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            ax.text(mx + 0.4, my, f'Δx={N}\nΔy=1  (pw)',
                    fontsize=11, ha='left', va='center', color='#93c5fd')

        for r in range(SHOW_R - 1):
            x0, y0 = pts[(0, r)];  x1, y1 = pts[(0, r + 1)]
            ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle='->', color='#f97316', lw=1.8, mutation_scale=14))
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            ax.text(mx + 0.4, my, f'Δx=-1\nΔy={N}  (pw)',
                    fontsize=11, ha='left', va='center', color='#fb923c')

        ax.set_title(f'4×4 微镜倾斜位置    N={N}    pitch={math.sqrt(N**2+1):.3f}pw=l',
                     fontsize=15, color='#e2e8f0', pad=8)
        ax.set_xlabel('x  (cross-scan, pw)', color='#94a3b8', fontsize=13)
        ax.set_ylabel('y  (scan 方向, pw)',   color='#94a3b8', fontsize=13)
        ax.set_aspect('equal')
        fig.tight_layout(pad=1.2)
        self.cv_tilt.draw()

    # ---- 右图：6×9 全阵列，数字自动缩放至微镜 2/3 大小 ----
    def _draw_array(self):
        fig = self.cv_array.fig
        fig.clear()
        ax = fig.add_subplot(111)
        _style_ax(ax, fig)

        groups = self.geom.vertical_groups()
        x_vals_sorted = sorted(groups.keys())
        phi = (math.sqrt(5) - 1) / 2.0
        color_of = {xv: colorsys.hsv_to_rgb((i * phi) % 1.0, 0.85, 0.95)
                    for i, xv in enumerate(x_vals_sorted)}

        # --- 先画轮廓和中心点，不画文字 ---
        for r in range(self.cfg.DMD_ROWS):
            for c in range(self.cfg.DMD_COLS):
                x, y = self.geom.pos(c, r)
                xi   = int(self.cfg.N * c - r)
                col  = color_of[xi]
                corners = self.geom.mirror_corners(c, r)
                poly = plt.Polygon(corners, closed=True,
                                   fc=(*col[:3], 0.22), ec=(*col[:3], 0.90),
                                   lw=1.3, zorder=3)
                ax.add_patch(poly)
                ax.plot(x, y, 'o', color=col, ms=3.5, zorder=5)

        for xv in x_vals_sorted:
            ax.axvline(xv, color=color_of[xv], alpha=0.18, lw=1.0, linestyle='--')

        ax.set_title('6×9 DMD 阵列    数字 = x (cross-scan, pw)    同色 = 同竖直列',
                     fontsize=15, color='#e2e8f0', pad=8)
        ax.set_xlabel('x  (cross-scan, pw)', color='#94a3b8', fontsize=13)
        ax.set_ylabel('y  (scan 方向, pw)',   color='#94a3b8', fontsize=13)
        ax.set_aspect('equal')

        # 设置明确的数据范围，供字号计算使用
        xmin, xmax = self.geom.x_range();  ymin, ymax = self.geom.y_range()
        margin = 2.5
        ax.set_xlim(xmin - margin, xmax + margin)
        ax.set_ylim(ymin - margin, ymax + margin)

        fig.tight_layout(pad=1.2)

        # --- 计算初始字号：基于 tight_layout 后的轴框比例 ---
        mirror_pw = math.sqrt(self.cfg.N ** 2 + 1)  # ≈ 3.16 pw

        def _compute_fs():
            try:
                ax_bbox = ax.get_window_extent()          # display pixels
                xl, xr_ = ax.get_xlim();  yb_, yt_ = ax.get_ylim()
                dw = xr_ - xl;  dh = yt_ - yb_
                if ax_bbox.width <= 0 or dw <= 0 or dh <= 0:
                    return 18.0
                # equal aspect: scale = smaller of the two ratios
                px_per_pw = min(ax_bbox.width / dw, ax_bbox.height / dh)
                return max(7.0, px_per_pw * (72.0 / fig.dpi) * mirror_pw * (2.0 / 3.0))
            except Exception:
                return 18.0

        # 初始字号估算（tight_layout 后，用轴框占比 × 图幅尺寸）
        ax_pos = ax.get_position()
        w_in   = fig.get_figwidth()  * ax_pos.width
        h_in   = fig.get_figheight() * ax_pos.height
        xl, xr_ = ax.get_xlim();  yb_, yt_ = ax.get_ylim()
        dw = xr_ - xl;  dh = yt_ - yb_
        if dw > 0 and dh > 0:
            in_per_pw = min(w_in / dw, h_in / dh)
            init_fs = max(7.0, in_per_pw * 72.0 * mirror_pw * (2.0 / 3.0))
        else:
            init_fs = 18.0

        # --- 添加文字标签 ---
        self._array_label_texts = []
        for r in range(self.cfg.DMD_ROWS):
            for c in range(self.cfg.DMD_COLS):
                x, y = self.geom.pos(c, r)
                xi   = int(self.cfg.N * c - r)
                t = ax.text(x, y, str(xi), ha='center', va='center',
                            fontsize=init_fs, color='white',
                            fontweight='bold', zorder=6)
                self._array_label_texts.append(t)

        # --- draw_event：每次渲染后更新字号（保证缩放/拉伸后仍正确） ---
        if hasattr(self, '_arr_draw_cid'):
            try:
                fig.canvas.mpl_disconnect(self._arr_draw_cid)
            except Exception:
                pass

        def _update_fs(event):
            fs = _compute_fs()
            for t in self._array_label_texts:
                if abs(t.get_fontsize() - fs) > 0.3:
                    t.set_fontsize(fs)
            # 不在此处调用 draw_idle，避免无限循环；
            # 下一次自然渲染（缩放/拉伸窗口等）会使用更新后的字号。

        self._arr_draw_cid = fig.canvas.mpl_connect('draw_event', _update_fs)
        self.cv_array.draw()


# ================================================================
# 界面 2：Bitmap 预览（右键拖拽可在 x/y 方向移动 DMD）
# ================================================================
class Tab2View(QWidget):
    def __init__(self, cfg: Config, pg: PatternGen, geom: DMDGeometry):
        super().__init__()
        self.cfg = cfg
        self.pg  = pg
        self.geom = geom

        self._dmd_x0: float = 0.0   # DMD 在 bitmap 中的 x 偏移（cross-scan）
        self._dmd_y0: float = 0.0   # DMD 在 bitmap 中的 y 偏移（scan）
        self._cur_pattern: str = cfg.DEFAULT_PATTERN
        self._cur_lw: int = cfg.DEFAULT_LINE_WIDTH_PW

        self._rdrag_start_x: float = None
        self._rdrag_start_y: float = None
        self._rdrag_dmd_x0:  float = None
        self._rdrag_dmd_y0:  float = None
        self._rdrag_axis:    str   = None  # 'x' 或 'y'，由拖拽方向决定
        self._on_dmd_moved = None          # 拖拽结束后的回调（由 MainWindow 设置）

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.canvas = CanvasWithToolbar(figsize=(6, 6))
        layout.addWidget(self.canvas)

        self.canvas.canvas.mpl_connect('button_press_event',   self._rclick_press)
        self.canvas.canvas.mpl_connect('motion_notify_event',  self._rclick_motion)
        self.canvas.canvas.mpl_connect('button_release_event', self._rclick_release)

    # ---- 右键拖拽 DMD ----
    # 规则：每次按下时自动判断主方向（水平→仅改x，竖直→仅改y），
    # 保证先单独确定x位置，再单独确定y位置，两者不相互干扰。
    def _rclick_press(self, event):
        if event.button == 3 and event.xdata is not None and event.ydata is not None:
            self._rdrag_start_x = event.xdata
            self._rdrag_start_y = event.ydata
            self._rdrag_dmd_x0  = self._dmd_x0
            self._rdrag_dmd_y0  = self._dmd_y0
            self._rdrag_axis    = None   # 移动超过阈值后再锁定方向

    def _rclick_motion(self, event):
        if self._rdrag_start_x is None or event.xdata is None or event.ydata is None:
            return
        dx = event.xdata - self._rdrag_start_x
        dy = event.ydata - self._rdrag_start_y

        # 移动超过 0.5 pw 后确定轴方向并锁定
        if self._rdrag_axis is None:
            if abs(dx) >= 0.5 or abs(dy) >= 0.5:
                self._rdrag_axis = 'x' if abs(dx) >= abs(dy) else 'y'
            else:
                return   # 移动量太小，尚未确定方向

        if self._rdrag_axis == 'x':
            self._dmd_x0 = float(round(self._rdrag_dmd_x0 + dx))   # 步进取整 → 镜面中心与像素中心重合
        else:
            self._dmd_y0 = float(round(self._rdrag_dmd_y0 + dy))
        self._redraw()

    def _rclick_release(self, event):
        if event.button == 3:
            moved = (self._rdrag_dmd_x0 != self._dmd_x0 or
                     self._rdrag_dmd_y0 != self._dmd_y0)
            self._rdrag_start_x = None
            self._rdrag_start_y = None
            self._rdrag_dmd_x0  = None
            self._rdrag_dmd_y0  = None
            self._rdrag_axis    = None
            if moved and self._on_dmd_moved is not None:
                self._on_dmd_moved()

    def get_dmd_x0(self) -> float:
        return self._dmd_x0

    def get_dmd_y0(self) -> float:
        return self._dmd_y0

    # ---- 刷新（外部：点击应用按钮） ----
    def refresh(self, pattern: str, lw: int):
        self._cur_pattern = pattern
        self._cur_lw      = lw
        self._dmd_x0      = 0.0
        self._dmd_y0      = 0.0
        self._redraw()

    def _redraw(self):
        pattern = self._cur_pattern
        lw      = self._cur_lw
        bmp = self.pg.generate(pattern, lw)
        sz  = self.cfg.BITMAP_SIZE
        dx  = self._dmd_x0
        dy  = self._dmd_y0

        fig = self.canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)
        _style_ax(ax, fig)
        # 关闭自适应主网格——只保留下方固定 1pw 次网格，
        # 确保网格精度不随缩放变化。
        ax.grid(False)

        # ── bitmap 显示 ──────────────────────────────────────────────
        # 像素 k 以整数坐标 k 为中心（占 [k-0.5, k+0.5]），
        # 与 DMD 镜面中心的整数 pw 坐标严格对应。
        ax.imshow(bmp, origin='lower', cmap='Blues', aspect='equal',
                  extent=[-0.5, sz - 0.5, -0.5, sz - 0.5],
                  vmin=0, vmax=1, alpha=0.9, interpolation='nearest')

        # 固定 1pw 像素网格（加强显示，不随缩放改变间距）
        _pw_grid(ax, -0.5, sz - 0.5, -0.5, sz - 0.5,
                 color='#3b6bc8', lw=0.75, alpha=0.85)

        # ── 叠加 DMD（整体偏移 dx, dy）────────────────────────────────
        for r in range(self.cfg.DMD_ROWS):
            for c in range(self.cfg.DMD_COLS):
                xr, yr = self.geom.pos(c, r)
                # 采样最近像素中心（round → 整数 → 像素索引）
                bx = int(round(xr + dx)) % sz
                by = int(round(yr + dy)) % sz
                on = bmp[by, bx] > 0.5
                corners = self.geom.mirror_corners(c, r)
                shifted = corners + np.array([dx, dy])
                poly = plt.Polygon(shifted, closed=True,
                                   fc='#ffe06644' if on else '#1e3a5f55',
                                   ec='#ffd700'   if on else '#4da6ff',
                                   lw=1.3, zorder=4)
                ax.add_patch(poly)
                # 实心点：镜面中心实际位置
                ax.plot(xr + dx, yr + dy, 'o',
                        color='#fbbf24' if on else '#60a5fa', ms=2.5, zorder=5)
                # 空心十字：采样的像素中心（round后坐标）
                sx, sy = round(xr + dx), round(yr + dy)
                ax.plot(sx, sy, '+',
                        color='#fbbf24' if on else '#60a5fa',
                        ms=4, markeredgewidth=0.8, zorder=6)

        from matplotlib.patches import Patch
        legend_els = [Patch(fc='#ffe06688', ec='#ffd700', label='ON  (曝光)'),
                      Patch(fc='#1e3a5f88', ec='#4da6ff', label='OFF (遮挡)')]
        ax.legend(handles=legend_els, loc='upper right', fontsize=12,
                  facecolor='#0f172a', edgecolor='#334155', labelcolor='#e2e8f0')

        ax.set_xlim(-0.5, sz - 0.5)
        ax.set_ylim(-0.5, sz - 0.5)

        # 标题提示操作方式
        axis_hint = '→水平拖拽调x  ↕竖直拖拽调y'
        ax.set_title(f'Bitmap · {pattern}  线宽={lw}pw  '
                     f'DMD偏移 x={dx:.1f} y={dy:.1f} pw    【右键{axis_hint}】',
                     fontsize=11, color='#e2e8f0', pad=8)
        ax.set_xlabel('x  (cross-scan, pw)', color='#94a3b8', fontsize=13)
        ax.set_ylabel('y  (scan 方向, pw)',   color='#94a3b8', fontsize=13)

        self.canvas._home_lims.clear()
        fig.tight_layout(pad=1.2)
        self.canvas.draw()


# ================================================================
# 界面 3：动态曝光仿真
# ================================================================
class Tab3View(QWidget):
    def __init__(self, cfg: Config, geom: DMDGeometry,
                 pg: PatternGen, get_params,
                 get_dmd_start_x=None, get_dmd_start_y=None):
        super().__init__()
        self.cfg  = cfg
        self.geom = geom
        self.pg   = pg
        self.get_params       = get_params
        self.get_dmd_start_x  = get_dmd_start_x   # callable → float
        self.get_dmd_start_y  = get_dmd_start_y   # callable → float
        self._tab4            = None               # 由 MainWindow 在构建后设置

        self.timer = QTimer()
        self.timer.timeout.connect(self._step)

        self.scan_pos: float  = 0.0
        self._start_x: float  = 0.0
        self._start_y: float  = 0.0

        sz = cfg.BITMAP_SIZE
        self.bmp        = np.zeros((sz, sz), dtype=np.float32)
        self.exposure   = np.zeros((sz, sz), dtype=np.int32)   # 曝光次数计数（bitmap坐标）
        self.substrate_exp: dict = {}                          # (实际x, 实际y) → 曝光次数
        self._last_states = np.zeros((cfg.DMD_ROWS, cfg.DMD_COLS))
        self._prev_frame_idx: int = -1
        self._history: list = []          # 用于后退一步
        self._mode: str = 'continuous'    # 'continuous' 或 'ttl'
        self._batch: bool = False         # 批量步进时抑制中间重绘
        self._initialized: bool = False   # 首次前进前先展示帧0采样

        # 预计算 DMD 空间范围（本地坐标）
        self._xmin, self._xmax = self.geom.x_range()
        self._ymin, self._ymax = self.geom.y_range()

        self._build_ui()
        self.reset()

    def _build_ui(self):
        vlay = QVBoxLayout(self)
        vlay.setContentsMargins(4, 4, 4, 4)

        hlay = QHBoxLayout()
        self.cv_dmd = CanvasWithToolbar(figsize=(5, 5))
        self.cv_bmp = CanvasWithToolbar(figsize=(5, 5))   # 右侧：bitmap 采样高亮
        hlay.addWidget(self.cv_dmd)
        hlay.addWidget(self.cv_bmp)
        vlay.addLayout(hlay, 1)

        btn_lay = QHBoxLayout()
        self.btn_run   = QPushButton('▶  连续曝光')
        self.btn_stop  = QPushButton('⏹  停止')
        self.btn_fwd   = QPushButton('▷  前进一步')
        self.btn_bwd   = QPushButton('◁  后退一步')
        self.btn_reset = QPushButton('↺  重置')
        self.lbl_status = QLabel('就绪')
        self.lbl_status.setAlignment(Qt.AlignCenter)

        for b in (self.btn_run, self.btn_stop, self.btn_fwd, self.btn_bwd, self.btn_reset):
            b.setMinimumHeight(34)
            btn_lay.addWidget(b)
        btn_lay.addWidget(self.lbl_status, 1)
        vlay.addLayout(btn_lay)

        self.btn_run.clicked.connect(self._start)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_fwd.clicked.connect(self._step_frame)
        self.btn_bwd.clicked.connect(self._step_frame_back)
        self.btn_reset.clicked.connect(self.reset)

    # ---- 仿真核心 ----
    def reset(self):
        self.timer.stop()
        self._start_x  = float(self.get_dmd_start_x()) if self.get_dmd_start_x else 0.0
        self._start_y  = float(self.get_dmd_start_y()) if self.get_dmd_start_y else 0.0
        self.scan_pos  = self._start_y
        self._prev_frame_idx = -1
        pattern, lw, _ = self.get_params()
        self.bmp = self.pg.generate(pattern, lw)
        sz = self.cfg.BITMAP_SIZE
        self.exposure     = np.zeros((sz, sz), dtype=np.int32)
        self.substrate_exp = {}
        self._last_states = np.zeros((self.cfg.DMD_ROWS, self.cfg.DMD_COLS))
        self._history.clear()
        self._initialized = False
        self._draw_dmd()
        self._draw_bitmap()
        if self._tab4 is not None:
            self._tab4.refresh(self.substrate_exp, self._start_x, self._start_y,
                               self.scan_pos, self._last_states,
                               self._start_y, self._mode)
        self._update_status(
            f'已重置  起始 x={self._start_x:.1f}  y={self._start_y:.1f} pw')

    def _start(self):
        self.timer.start(self.cfg.TIMER_MS)
        self._update_status('连续曝光中…')

    def _stop(self):
        self.timer.stop()
        self._update_status(f'暂停  scan_y={self.scan_pos:.1f}pw')

    def _step(self):
        _, lw, M = self.get_params()
        s  = self.scan_pos
        sz = self.cfg.BITMAP_SIZE

        if s >= self._start_y + sz:
            self._stop()
            self._update_status('曝光完成')
            return

        # 保存当前状态到历史（用于后退一步）
        self._history.append((
            self.exposure.copy(),
            dict(self.substrate_exp),
            self.scan_pos,
            self._prev_frame_idx,
            self._last_states.copy()
        ))
        if len(self._history) > 500:
            self._history.pop(0)

        frame_idx = int((s - self._start_y) / M)
        frame_y0  = self._start_y + frame_idx * M

        is_new_frame = (frame_idx != self._prev_frame_idx)
        if is_new_frame:
            states = np.zeros((self.cfg.DMD_ROWS, self.cfg.DMD_COLS), dtype=np.float32)
            for r in range(self.cfg.DMD_ROWS):
                for c in range(self.cfg.DMD_COLS):
                    xr, yr = self.geom.pos(c, r)
                    # 采样最近像素中心（与界面2一致）
                    bx = int(round(xr + self._start_x)) % sz
                    by = int(round(frame_y0 + yr)) % sz
                    states[r, c] = self.bmp[by, bx]
            self._last_states = states
            self._prev_frame_idx = frame_idx

        # 持续出光：每步都曝光；TTL：仅在换图瞬间曝光一次
        if self._mode == 'continuous' or is_new_frame:
            for r in range(self.cfg.DMD_ROWS):
                for c in range(self.cfg.DMD_COLS):
                    if self._last_states[r, c] > 0.5:
                        xr, yr = self.geom.pos(c, r)
                        wx = int(round(xr + self._start_x)) % sz
                        wy = int(round(s + yr)) % sz
                        self.exposure[wy, wx] += 1
                        # 实际基板坐标（不取模）用于界面4显示
                        ax = int(round(xr + self._start_x))
                        ay = int(round(s + yr))
                        self.substrate_exp[(ax, ay)] = self.substrate_exp.get((ax, ay), 0) + 1

        if not self._batch:
            self._draw_dmd()
            self._draw_bitmap()
            if self._tab4 is not None:
                self._tab4.refresh(self.substrate_exp, self._start_x, self._start_y,
                                   self.scan_pos, self._last_states,
                                   frame_y0, self._mode)

        self.scan_pos += 1.0
        if not self._batch:
            self._update_status(
                f'scan_y={self.scan_pos:.0f}pw  帧={frame_idx}  '
                f'ON镜={int(self._last_states.sum())}/{self.cfg.DMD_ROWS * self.cfg.DMD_COLS}')

    def _update_status(self, text: str):
        self.lbl_status.setText(text)
        if self._tab4 is not None:
            self._tab4.lbl_status.setText(text)

    def _step_back(self):
        """单 pw 后退（内部使用）。"""
        if not self._history:
            self._update_status('已是最初状态，无法后退')
            return
        self.timer.stop()
        exp_snap, sub_snap, sp, pfi, states_snap = self._history.pop()
        self.exposure        = exp_snap
        self.substrate_exp   = sub_snap
        self.scan_pos        = sp
        self._prev_frame_idx = pfi
        self._last_states    = states_snap
        self._draw_dmd()
        self._draw_bitmap()
        if self._tab4 is not None:
            _, _, M = self.get_params()
            M = max(M, 1)
            fi = int((self.scan_pos - self._start_y) / M)
            fy0 = self._start_y + fi * M
            self._tab4.refresh(self.substrate_exp, self._start_x, self._start_y,
                               self.scan_pos, self._last_states,
                               fy0, self._mode)
        self._update_status(f'后退  scan_y={self.scan_pos:.0f}pw')

    def _step_frame(self):
        """前进一帧（M 个 pw），批量执行不中间重绘。
        首次调用仅加载帧0采样点显示，不曝光不前进（scan_y=0）。"""
        _, _, M = self.get_params()
        M = max(M, 1)
        sz = self.cfg.BITMAP_SIZE

        # 首次点击：加载帧0采样，只显示，不曝光不前进
        if not self._initialized:
            self._initialized = True
            frame_y0 = self._start_y
            states = np.zeros((self.cfg.DMD_ROWS, self.cfg.DMD_COLS), dtype=np.float32)
            for r in range(self.cfg.DMD_ROWS):
                for c in range(self.cfg.DMD_COLS):
                    xr, yr = self.geom.pos(c, r)
                    bx = int(round(xr + self._start_x)) % sz
                    by = int(round(frame_y0 + yr)) % sz
                    states[r, c] = self.bmp[by, bx]
            self._last_states = states
            self._prev_frame_idx = 0
            self._draw_dmd()
            self._draw_bitmap()
            if self._tab4 is not None:
                self._tab4.refresh(self.substrate_exp, self._start_x, self._start_y,
                                   self.scan_pos, self._last_states,
                                   frame_y0, self._mode)
            self._update_status(
                f'帧0已加载  scan_y=0pw  '
                f'ON镜={int(states.sum())}/{self.cfg.DMD_ROWS * self.cfg.DMD_COLS}')
            return

        # 后续点击：批量步进 M 个 pw 并曝光
        frame_y0_before = self.scan_pos
        self._batch = True
        for _ in range(M):
            if self.scan_pos >= self._start_y + sz:
                break
            self._step()
        self._batch = False
        self._draw_dmd()
        self._draw_bitmap()
        if self._tab4 is not None:
            self._tab4.refresh(self.substrate_exp, self._start_x, self._start_y,
                               self.scan_pos, self._last_states,
                               frame_y0_before, self._mode)
        self._update_status(
            f'前进一帧  scan_y={self.scan_pos:.0f}pw  '
            f'ON镜={int(self._last_states.sum())}/{self.cfg.DMD_ROWS * self.cfg.DMD_COLS}')

    def _step_frame_back(self):
        """后退一帧（M 个 pw），弹出 M 条历史记录。"""
        if not self._history:
            self._update_status('已是最初状态，无法后退')
            return
        self.timer.stop()
        _, _, M = self.get_params()
        M = max(M, 1)
        snap = None
        for _ in range(M):
            if not self._history:
                break
            snap = self._history.pop()
        if snap is None:
            return
        exp_snap, sub_snap, sp, pfi, states_snap = snap
        self.exposure        = exp_snap
        self.substrate_exp   = sub_snap
        self.scan_pos        = sp
        self._prev_frame_idx = pfi
        self._last_states    = states_snap
        self._draw_dmd()
        self._draw_bitmap()
        if self._tab4 is not None:
            fi = int((self.scan_pos - self._start_y) / M) if M > 0 else 0
            fy0 = self._start_y + fi * M
            self._tab4.refresh(self.substrate_exp, self._start_x, self._start_y,
                               self.scan_pos, self._last_states,
                               fy0, self._mode)
        self._update_status(f'后退一帧  scan_y={self.scan_pos:.0f}pw')

    def toggle_mode(self) -> str:
        """切换曝光模式，返回新模式名称。"""
        self._mode = 'ttl' if self._mode == 'continuous' else 'continuous'
        return self._mode

    # ---- DMD 采样面板（仅显示 DMD 覆盖区） ----
    def _draw_dmd(self):
        _, lw, M = self.get_params()
        sz = self.cfg.BITMAP_SIZE
        s  = self.scan_pos
        frame_idx = int((s - self._start_y) / M) if M > 0 else 0
        frame_y0  = self._start_y + frame_idx * M

        xmin, xmax = self._xmin + self._start_x, self._xmax + self._start_x
        ymin, ymax = self._ymin, self._ymax
        margin = 2.0

        fig = self.cv_dmd.fig
        fig.clear()
        ax  = fig.add_subplot(111)
        ax.set_facecolor(BG_PANEL)

        # bitmap 局部背景
        x_lo = int(xmin) - int(margin) - 1
        x_hi = int(xmax) + int(margin) + 2
        y_lo = int(frame_y0 + ymin) - int(margin) - 1
        y_hi = int(frame_y0 + ymax) + int(margin) + 2
        bmp_crop = np.zeros((max(1, y_hi - y_lo), max(1, x_hi - x_lo)), dtype=np.float32)
        for iy in range(max(0, y_lo), min(sz, y_hi)):
            for ix in range(max(0, x_lo), min(sz, x_hi)):
                bmp_crop[iy - y_lo, ix - x_lo] = self.bmp[iy % sz, ix % sz]
        ax.imshow(bmp_crop, origin='lower', cmap='Blues', alpha=0.35,
                  extent=[x_lo - 0.5, x_hi - 0.5, y_lo - 0.5, y_hi - 0.5],
                  aspect='auto', vmin=0, vmax=1)

        ax.axhline(frame_y0,     color='#44ff88', lw=1, linestyle=':', alpha=0.7,
                   label=f'帧起始 y={frame_y0:.0f}')
        ax.axhline(frame_y0 + M, color='#44ff88', lw=1, linestyle=':', alpha=0.7)
        ax.axhline(s, color='#ff4455', lw=2, linestyle='--', label=f'扫描位 {s:.0f}pw')

        for r in range(self.cfg.DMD_ROWS):
            for c in range(self.cfg.DMD_COLS):
                xr, yr = self.geom.pos(c, r)
                on = self._last_states[r, c] > 0.5
                corners = self.geom.mirror_corners(c, r)
                shifted = corners + np.array([self._start_x, frame_y0])
                poly = plt.Polygon(shifted, closed=True,
                                   fc='#ffe06666' if on else '#1a2a4422',
                                   ec='#ffd700'   if on else '#3b82f6',
                                   lw=1.4, zorder=5)
                ax.add_patch(poly)
                ax.plot(xr + self._start_x, frame_y0 + yr, 'o',
                        color='#fbbf24' if on else '#60a5fa', ms=2.5, zorder=6)

        ax.set_xlim(xmin - margin, xmax + margin)
        ax.set_ylim(frame_y0 + ymin - margin, frame_y0 + ymax + margin)
        ax.set_aspect('equal')
        # 固定 1pw 像素网格（辅助显示采样位置）
        _pw_grid(ax, xmin - margin, xmax + margin,
                 frame_y0 + ymin - margin, frame_y0 + ymax + margin,
                 color='#2d5080', lw=0.55, alpha=0.7)
        ax.set_title(f'DMD 采样状态   帧={frame_idx}   bitmap y=[{frame_y0:.0f},{frame_y0+M:.0f})pw',
                     fontsize=13, color='white')
        ax.set_xlabel('x  (cross-scan, pw)',       color=TICK_COL, fontsize=13)
        ax.set_ylabel('bitmap y  (scan方向, pw)',   color=TICK_COL, fontsize=13)
        ax.tick_params(colors=TICK_COL, labelsize=12)
        ax.legend(fontsize=11, loc='upper right', facecolor='#222', labelcolor='white')
        self.cv_dmd._home_lims.clear()
        fig.tight_layout()
        self.cv_dmd.draw()

    # ---- Bitmap 采样面板（与左侧 DMD 状态对应，高亮当前被采样像素） ----
    def _draw_bitmap(self):
        _, lw, M = self.get_params()
        sz = self.cfg.BITMAP_SIZE
        s  = self.scan_pos
        frame_idx = int((s - self._start_y) / M) if M > 0 else 0
        frame_y0  = self._start_y + frame_idx * M

        xmin = self._xmin + self._start_x
        xmax = self._xmax + self._start_x
        ymin, ymax = self._ymin, self._ymax
        margin = 2.0

        fig = self.cv_bmp.fig
        fig.clear()
        ax  = fig.add_subplot(111)
        ax.set_facecolor(BG_PANEL)

        # 完整 bitmap（像素 k 中心在坐标 k，与左侧 DMD 面板坐标系一致）
        ax.imshow(self.bmp, origin='lower', cmap='Blues',
                  extent=[-0.5, sz - 0.5, -0.5, sz - 0.5],
                  aspect='auto', vmin=0, vmax=1, alpha=0.8,
                  interpolation='nearest')

        # 高亮每个镜面在 bitmap 中的采样像素
        for r in range(self.cfg.DMD_ROWS):
            for c in range(self.cfg.DMD_COLS):
                xr, yr = self.geom.pos(c, r)
                bx = int(round(xr + self._start_x)) % sz
                by = int(round(frame_y0 + yr)) % sz
                on = self._last_states[r, c] > 0.5
                # 高亮像素格（bx = xr + start_x，by = frame_y0 + yr，与左侧坐标一致）
                rect = mpatches.Rectangle(
                    (bx - 0.5, by - 0.5), 1.0, 1.0,
                    linewidth=1.5,
                    edgecolor='#ffd700' if on else '#4da6ff',
                    facecolor='#ffe06677' if on else '#1e3a5f44',
                    zorder=4)
                ax.add_patch(rect)
                ax.plot(bx, by, '+',
                        color='#fbbf24' if on else '#60a5fa',
                        ms=7, markeredgewidth=1.5, zorder=5)

        # 与左侧 DMD 面板保持相同 xlim/ylim，方便对照
        ax.set_xlim(xmin - margin, xmax + margin)
        ax.set_ylim(frame_y0 + ymin - margin, frame_y0 + ymax + margin)
        ax.set_aspect('equal')
        _pw_grid(ax, xmin - margin, xmax + margin,
                 frame_y0 + ymin - margin, frame_y0 + ymax + margin,
                 color='#2d5080', lw=0.55, alpha=0.7)

        from matplotlib.patches import Patch
        legend_els = [Patch(fc='#ffe06677', ec='#ffd700', label='ON  (采样=1)'),
                      Patch(fc='#1e3a5f44', ec='#4da6ff', label='OFF (采样=0)')]
        ax.legend(handles=legend_els, loc='upper right', fontsize=11,
                  facecolor='#0f172a', edgecolor='#334155', labelcolor='#e2e8f0')

        by_lo = int(round(frame_y0 + ymin))
        by_hi = int(round(frame_y0 + ymax))
        ax.set_title(f'Bitmap 采样点   帧={frame_idx}   '
                     f'bitmap y=[{by_lo}, {by_hi}]pw',
                     fontsize=13, color='white')
        ax.set_xlabel('x  (cross-scan, pw)', color=TICK_COL, fontsize=13)
        ax.set_ylabel('bitmap y  (pw)',      color=TICK_COL, fontsize=13)
        ax.tick_params(colors=TICK_COL, labelsize=12)
        self.cv_bmp._home_lims.clear()
        fig.tight_layout()
        self.cv_bmp.draw()


# ================================================================
# 界面 4：累积曝光结果（由 Tab3View 驱动刷新）
# ================================================================
class Tab4View(QWidget):
    def __init__(self, cfg: Config, geom: DMDGeometry):
        super().__init__()
        self.cfg  = cfg
        self.geom = geom
        self._xmin, self._xmax = geom.x_range()
        self._ymin, self._ymax = geom.y_range()
        self._tab3 = None          # 由 MainWindow 在构建后设置

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.canvas = CanvasWithToolbar(figsize=(6, 6))
        layout.addWidget(self.canvas, 1)

        btn_lay = QHBoxLayout()
        self.btn_run   = QPushButton('▶  连续曝光')
        self.btn_stop  = QPushButton('⏹  停止')
        self.btn_fwd   = QPushButton('▷  前进一步')
        self.btn_bwd   = QPushButton('◁  后退一步')
        self.btn_reset = QPushButton('↺  重置')
        self.btn_mode  = QPushButton('持续出光')
        self.lbl_status = QLabel('就绪')
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet('color:#94a3b8; font-size:12pt;')

        for b in (self.btn_run, self.btn_stop, self.btn_fwd, self.btn_bwd, self.btn_reset):
            b.setMinimumHeight(34)
            btn_lay.addWidget(b)
        self.btn_mode.setMinimumHeight(34)
        self._apply_mode_style('continuous')
        btn_lay.addWidget(self.btn_mode)
        btn_lay.addWidget(self.lbl_status, 1)
        layout.addLayout(btn_lay)

        self.btn_run.clicked.connect(lambda: self._tab3 and self._tab3._start())
        self.btn_stop.clicked.connect(lambda: self._tab3 and self._tab3._stop())
        self.btn_fwd.clicked.connect(lambda: self._tab3 and self._tab3._step_frame())
        self.btn_bwd.clicked.connect(lambda: self._tab3 and self._tab3._step_frame_back())
        self.btn_reset.clicked.connect(lambda: self._tab3 and self._tab3.reset())
        self.btn_mode.clicked.connect(self._on_mode_toggle)

        self._draw_empty()

    def _apply_mode_style(self, mode: str):
        if mode == 'continuous':
            self.btn_mode.setText('持续出光')
            self.btn_mode.setStyleSheet(
                'QPushButton{background:#164e36;color:#4ade80;border:1px solid #16a34a;'
                'border-radius:4px;padding:4px 10px;font-size:13pt;}'
                'QPushButton:hover{background:#15803d;}')
        else:
            self.btn_mode.setText('TTL 触发')
            self.btn_mode.setStyleSheet(
                'QPushButton{background:#431407;color:#fb923c;border:1px solid #c2410c;'
                'border-radius:4px;padding:4px 10px;font-size:13pt;}'
                'QPushButton:hover{background:#9a3412;}')

    def _on_mode_toggle(self):
        if self._tab3 is None:
            return
        new_mode = self._tab3.toggle_mode()
        self._apply_mode_style(new_mode)
        label = '持续出光' if new_mode == 'continuous' else 'TTL 触发'
        self.lbl_status.setText(f'模式已切换 → {label}')

    def _draw_empty(self):
        fig = self.canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_facecolor(BG_PANEL)
        fig.patch.set_facecolor(BG_DARK)
        ax.set_title('累积曝光结果  (请先在界面3运行仿真)',
                     fontsize=13, color='#64748b')
        fig.tight_layout()
        self.canvas.draw()

    def refresh(self, substrate_exp, start_x, start_y, scan_pos, last_states,
                frame_y0, mode):
        N    = self.cfg.N
        s    = scan_pos
        dy   = s - frame_y0          # 当前帧内已走步数（0 = 刚换图）
        xmin = self._xmin + start_x
        xmax = self._xmax + start_x
        ymin, ymax = self._ymin, self._ymax
        margin = 2.0

        fig = self.canvas.fig
        fig.clear()
        ax  = fig.add_subplot(111)
        ax.set_facecolor(BG_PANEL)

        # 角点偏移量（与 DMDGeometry.mirror_corners 相同）
        hc = np.array([N / 2.0,  0.5])
        hr = np.array([-0.5,     N / 2.0])
        corn_off = np.array([hc + hr, hc - hr, -hc - hr, -hc + hr])

        # 历史曝光光斑（统一颜色，不区分次数）
        for (wx, wy) in substrate_exp:
            corners = np.array([float(wx), float(wy)]) + corn_off
            poly = plt.Polygon(corners, closed=True,
                               fc='#1e40af', ec='#60a5fa',
                               lw=0.5, alpha=0.85, zorder=3)
            ax.add_patch(poly)

        # DMD 投影覆盖：
        #   持续出光 — 始终显示当前帧轮廓（固定在 frame_y0 位置）
        #   TTL 模式 — 仅在换图瞬间（dy==0）显示帧轮廓
        show_dmd = (mode == 'continuous') or (dy == 0)
        if show_dmd:
            for r in range(self.cfg.DMD_ROWS):
                for c in range(self.cfg.DMD_COLS):
                    xr, yr = self.geom.pos(c, r)
                    on = last_states[r, c] > 0.5
                    corners = self.geom.mirror_corners(c, r) + np.array([start_x, frame_y0])
                    poly = plt.Polygon(corners, closed=True,
                                       fc='#44ff8840' if on else 'none',
                                       ec='#44ff88'   if on else '#2d4a6a',
                                       lw=0.9, alpha=0.9, zorder=5)
                    ax.add_patch(poly)

        from matplotlib.patches import Patch
        mode_label = '持续出光' if mode == 'continuous' else 'TTL 触发'
        legend_els = [
            Patch(fc='#1e40af',   ec='#60a5fa', label='已曝光'),
            Patch(fc='#44ff8840', ec='#44ff88', label=f'当前帧 ON ({mode_label})'),
            Patch(fc='none',      ec='#2d4a6a', label='当前帧 OFF'),
        ]
        ax.legend(handles=legend_els, loc='upper right', fontsize=11,
                  facecolor='#0f172a', edgecolor='#334155', labelcolor='#e2e8f0',
                  title='图例', title_fontsize=11)

        y_lo = start_y + ymin - margin
        y_hi = max(s + ymax + margin, start_y + ymax + margin + 1)
        ax.set_xlim(xmin - margin, xmax + margin)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect('equal')
        ax.set_title(f'累积曝光结果   scan_y={s:.0f}pw  (绿框=当前DMD投影)',
                     fontsize=13, color='white')
        ax.set_xlabel('x  (cross-scan, pw)', color=TICK_COL, fontsize=13)
        ax.set_ylabel('y  (scan 方向, pw)',  color=TICK_COL, fontsize=13)
        ax.tick_params(colors=TICK_COL, labelsize=12)
        self.canvas._home_lims.clear()
        fig.tight_layout()
        self.canvas.draw()


# ================================================================
# 参数面板
# ================================================================
class ParamPanel(QWidget):
    def __init__(self, cfg: Config, on_apply):
        super().__init__()
        self.cfg = cfg
        self.on_apply = on_apply
        self.setMinimumWidth(170)
        self.setMaximumWidth(270)
        self._build_ui()

    def _build_ui(self):
        vlay = QVBoxLayout(self)
        vlay.setContentsMargins(8, 10, 8, 10)
        vlay.setSpacing(8)

        title = QLabel('参 数 设 置')
        title.setFont(QFont('Microsoft YaHei', 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet('color:#93c5fd; font-size:16pt;')
        vlay.addWidget(title)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet('color:#444')
        vlay.addWidget(sep)

        def lbl(s):
            l = QLabel(s); l.setStyleSheet('color:#ccc; font-size:13pt;'); return l

        grp_dmd = QGroupBox('DMD 规格（固定）')
        fl = QFormLayout(); fl.setSpacing(4)
        fl.addRow(lbl('行 × 列 :'),       lbl(f'{self.cfg.DMD_ROWS} × {self.cfg.DMD_COLS}'))
        fl.addRow(lbl('倾斜因子 N :'),     lbl(str(self.cfg.N)))
        fl.addRow(lbl('微镜边长 l :'),     lbl('1.0 (单位)'))
        fl.addRow(lbl('pw = l/√(N²+1) :'), lbl(f'{self.cfg.PW:.5f}'))
        grp_dmd.setLayout(fl)
        vlay.addWidget(grp_dmd)

        grp_pat = QGroupBox('图形设置')
        fl2 = QFormLayout(); fl2.setSpacing(4)
        self.combo = QComboBox()
        self.combo.addItems(['横线', '竖线', '圆环'])
        fl2.addRow(lbl('类型 :'), self.combo)
        self.spin_lw = QSpinBox()
        self.spin_lw.setRange(1, 20)
        self.spin_lw.setValue(self.cfg.DEFAULT_LINE_WIDTH_PW)
        self.spin_lw.setSuffix(' pw')
        fl2.addRow(lbl('线宽 :'), self.spin_lw)
        grp_pat.setLayout(fl2)
        vlay.addWidget(grp_pat)

        grp_scan = QGroupBox('扫描设置')
        fl3 = QFormLayout(); fl3.setSpacing(4)
        self.spin_M = QSpinBox()
        self.spin_M.setRange(1, 30)
        self.spin_M.setValue(self.cfg.DEFAULT_M)
        self.spin_M.setSuffix(' pw')
        fl3.addRow(lbl('换图距离 M :'), self.spin_M)
        grp_scan.setLayout(fl3)
        vlay.addWidget(grp_scan)

        btn = QPushButton('✔  应用 / 重置仿真')
        btn.setMinimumHeight(38)
        btn.setStyleSheet(
            'QPushButton{background:#1d4ed8;color:#dbeafe;border-radius:4px;'
            'font-weight:bold;font-size:13pt;border:1px solid #3b82f6;}'
            'QPushButton:hover{background:#2563eb;}')
        btn.clicked.connect(self.on_apply)
        vlay.addWidget(btn)
        vlay.addStretch()

    def get_params(self):
        return self.combo.currentText(), self.spin_lw.value(), self.spin_M.value()


# ================================================================
# 主窗口
# ================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cfg  = Config()
        self.geom = DMDGeometry(self.cfg)
        self.pg   = PatternGen(self.cfg)

        self.setWindowTitle('DMD 扫描曝光仿真演示  v0.1-demo')
        self.resize(1400, 860)
        self.setStyleSheet("""
            QMainWindow, QWidget   { background:#080c14; color:#cbd5e1; }
            QTabWidget::pane       { border:1px solid #1e3a5f; background:#080c14; }
            QTabBar::tab           { background:#0f1e33; color:#64748b;
                                     padding:8px 20px; border-radius:4px 4px 0 0;
                                     margin-right:2px; font-size:13pt; }
            QTabBar::tab:selected  { background:#1e3a5f; color:#93c5fd; font-weight:bold; }
            QTabBar::tab:hover     { background:#162d4a; color:#bfdbfe; }
            QGroupBox              { border:1px solid #1e3a5f; border-radius:5px;
                                     margin-top:8px; padding-top:6px;
                                     font-weight:bold; color:#60a5fa; font-size:13pt; }
            QGroupBox::title       { subcontrol-origin:margin; left:8px; }
            QLabel                 { color:#94a3b8; font-size:13pt; }
            QSpinBox, QComboBox    { background:#0f1e33; color:#cbd5e1;
                                     border:1px solid #1e3a5f; border-radius:3px;
                                     padding:2px 4px; font-size:13pt; }
            QSpinBox:focus, QComboBox:focus { border:1px solid #3b82f6; }
            QPushButton            { background:#1e3a5f; color:#93c5fd;
                                     border:1px solid #2563eb; border-radius:4px;
                                     padding:4px 10px; font-size:13pt; }
            QPushButton:hover      { background:#2563eb; color:#dbeafe; }
            QPushButton:pressed    { background:#1d4ed8; }
            QSplitter::handle      { background:#1e3a5f; width:3px; }
        """)

        self._build_ui()
        p, lw, _ = self.param_panel.get_params()
        self.tab2.refresh(p, lw)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)

        self.param_panel = ParamPanel(self.cfg, self._on_apply)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(
            'QTabWidget::pane{border:1px solid #333;}'
            'QTabBar::tab{background:#252540;color:#aaa;padding:7px 18px;font-size:13pt;}'
            'QTabBar::tab:selected{background:#3a3a6a;color:white;font-weight:bold;}')

        self.tab1 = Tab1View(self.cfg, self.geom)
        self.tab2 = Tab2View(self.cfg, self.pg, self.geom)
        self.tab3 = Tab3View(self.cfg, self.geom, self.pg,
                             self._get_params,
                             get_dmd_start_x=self.tab2.get_dmd_x0,
                             get_dmd_start_y=self.tab2.get_dmd_y0)
        self.tab4 = Tab4View(self.cfg, self.geom)
        self.tab3._tab4 = self.tab4   # Tab3 驱动 Tab4 刷新
        self.tab4._tab3 = self.tab3   # Tab4 按钮控制 Tab3 仿真
        self.tab2._on_dmd_moved = self.tab3.reset  # Tab2 拖拽结束→Tab3 重置起始位置

        self.tabs.addTab(self.tab1, '  界面1: 镜面位置  ')
        self.tabs.addTab(self.tab2, '  界面2: Bitmap  ')
        self.tabs.addTab(self.tab3, '  界面3: 采样仿真  ')
        self.tabs.addTab(self.tab4, '  界面4: 曝光结果  ')

        splitter.addWidget(self.tabs)
        splitter.addWidget(self.param_panel)
        splitter.setSizes([1160, 240])
        splitter.setHandleWidth(4)
        root.addWidget(splitter)

    def _get_params(self):
        return self.param_panel.get_params()

    def _on_apply(self):
        p, lw, _ = self.param_panel.get_params()
        self.tab2.refresh(p, lw)
        self.tab3.reset()


# ================================================================
# 入口
# ================================================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
