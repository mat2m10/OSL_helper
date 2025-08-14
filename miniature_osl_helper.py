"""
Miniature OSL Helper — Python module

Goals
-----
- Provide a starter Citadel paint library (sRGB hex values).
- Let you choose a light-source color, a surface/base color, and ambient light intensity,
  then predict the resulting perceived color via a simple Lambertian + ambient model.
- Suggest the closest single Citadel paint(s) to a target color (CIEDE2000 in Lab space).
- Suggest a simple two-paint mix (with 10% ratio steps) that best approximates a target color.

This is intentionally simple and fast so you can iterate while painting.
Extend/replace the CITADEL_COLORS dict with your own library as needed.

Usage (examples)
----------------
from miniature_osl_helper import (
    CITADEL_COLORS, hex_to_rgb, rgb_to_hex,
    simulate_osl, find_nearest_paints, suggest_two_paint_mix
)

# Choose colors by name
surface = CITADEL_COLORS['Mephiston Red']
light   = CITADEL_COLORS['Flash Gitz Yellow']

# Run a quick OSL simulation
result = simulate_osl(surface_rgb=surface,
                      light_rgb=light,
                      ambient_intensity=0.2,   # 0..1
                      light_intensity=1.0,     # arbitrary scale
                      ndotl=0.8,               # incidence (0..1)
                      distance=2.0,            # arbitrary units
                      falloff='inverse_square') # or 'none','inverse','linear'

print('Result color (hex):', rgb_to_hex(result))

# Suggest nearest single paints for the result color
nearest = find_nearest_paints(result, k=5)
for name, dE in nearest:
    print(f"{name}: ΔE2000 = {dE:.2f}")

# Suggest a simple two-paint mix for the result color
mix = suggest_two_paint_mix(result, step=10)  # ratios in 10% steps
print('Best mix:', mix)

"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List
import math
import colorsys
import plotly.graph_objects as go

# -----------------------------
# Paint library (starter set)
# sRGB values are 0..255 triplets.
# You can add/adjust as needed; values are approximations suitable for planning, not color-matched swatches.
# -----------------------------
CITADEL_COLORS: Dict[str, Tuple[int, int, int]] = {
    # Neutrals / Metals (approximate display values)
    'Abaddon Black': (0, 0, 0),
    'White Scar': (245, 245, 245),
    'Leadbelcher': (143, 148, 153),
    'Ironbreaker': (200, 205, 210),
    'Runefang Steel': (220, 225, 230),
    'Mechanicus Standard Grey': (89, 94, 98),
    'Eshin Grey': (67, 70, 74),
    'Dawnstone': (150, 155, 160),
    'Administratum Grey': (190, 195, 200),

    # Reds / Oranges / Yellows
    'Khorne Red': (92, 17, 20),
    'Mephiston Red': (150, 23, 28),
    'Evil Sunz Scarlet': (190, 40, 36),
    'Wild Rider Red': (220, 60, 40),
    'Troll Slayer Orange': (235, 90, 35),
    'Averland Sunset': (230, 170, 20),
    'Yriel Yellow': (255, 215, 0),
    'Flash Gitz Yellow': (255, 240, 0),

    # Browns / Bones / Flesh
    'Rhinox Hide': (60, 35, 25),
    'Mournfang Brown': (110, 65, 35),
    'XV-88': (135, 95, 45),
    'Zamesi Desert': (190, 150, 75),
    'Ushabti Bone': (220, 205, 160),
    'Rakarth Flesh': (200, 195, 180),
    'Bugman\'s Glow': (165, 95, 80),
    'Cadian Fleshtone': (210, 150, 120),

    # Greens
    'Caliban Green': (10, 50, 30),
    'Warpstone Glow': (30, 140, 70),
    'Moot Green': (120, 220, 80),
    'Castellan Green': (35, 60, 35),
    'Death Guard Green': (115, 125, 90),
    'Nurgling Green': (175, 190, 125),

    # Blues / Teals
    'Kantor Blue': (20, 35, 90),
    'Macragge Blue': (30, 60, 140),
    'Altdorf Guard Blue': (35, 85, 190),
    'Caledor Sky': (45, 120, 190),
    'Teclis Blue': (60, 160, 210),
    'Lothern Blue': (110, 200, 230),
    'Incubi Darkness': (10, 50, 60),
    'Stegadon Scale Green': (20, 80, 110),

    # Purples / Pinks
    'Naggaroth Night': (40, 20, 50),
    'Xereus Purple': (85, 40, 110),
    'Genestealer Purple': (150, 90, 170),
    'Screamer Pink': (150, 40, 90),
    'Emperor\'s Children': (235, 95, 175),
}

# -----------------------------
# Utility functions
# -----------------------------
import plotly.graph_objects as go
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    hex_str = hex_str.strip().lstrip('#')
    if len(hex_str) == 3:
        hex_str = ''.join([c*2 for c in hex_str])
    if len(hex_str) != 6:
        raise ValueError('hex must be #RGB or #RRGGBB')
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    return (r, g, b)


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"#{r:02X}{g:02X}{b:02X}"


def srgb_to_linear(c: float) -> float:
    # c in 0..1
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def linear_to_srgb(c: float) -> float:
    # c in 0..1
    return 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1/2.4)) - 0.055


def to_float(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    return tuple([v / 255.0 for v in rgb])  # type: ignore


def to_int(rgbf: Tuple[float, float, float]) -> Tuple[int, int, int]:
    return tuple([int(round(clamp01(v) * 255)) for v in rgbf])  # type: ignore


# XYZ/Lab conversion with D65/2°
# Matrices from sRGB spec
M_RGB_TO_XYZ = (
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
)
M_XYZ_TO_RGB = (
    ( 3.2404542, -1.5371385, -0.4985314),
    (-0.9692660,  1.8760108,  0.0415560),
    ( 0.0556434, -0.2040259,  1.0572252),
)

# Reference white (D65)
Xn, Yn, Zn = 0.95047, 1.00000, 1.08883


def srgb_to_xyz(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    r, g, b = to_float(rgb)
    rl, gl, bl = srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b)
    X = M_RGB_TO_XYZ[0][0]*rl + M_RGB_TO_XYZ[0][1]*gl + M_RGB_TO_XYZ[0][2]*bl
    Y = M_RGB_TO_XYZ[1][0]*rl + M_RGB_TO_XYZ[1][1]*gl + M_RGB_TO_XYZ[1][2]*bl
    Z = M_RGB_TO_XYZ[2][0]*rl + M_RGB_TO_XYZ[2][1]*gl + M_RGB_TO_XYZ[2][2]*bl
    return (X, Y, Z)


def xyz_to_srgb(XYZ: Tuple[float, float, float]) -> Tuple[int, int, int]:
    X, Y, Z = XYZ
    rl = M_XYZ_TO_RGB[0][0]*X + M_XYZ_TO_RGB[0][1]*Y + M_XYZ_TO_RGB[0][2]*Z
    gl = M_XYZ_TO_RGB[1][0]*X + M_XYZ_TO_RGB[1][1]*Y + M_XYZ_TO_RGB[1][2]*Z
    bl = M_XYZ_TO_RGB[2][0]*X + M_XYZ_TO_RGB[2][1]*Y + M_XYZ_TO_RGB[2][2]*Z
    r = linear_to_srgb(rl)
    g = linear_to_srgb(gl)
    b = linear_to_srgb(bl)
    return to_int((r, g, b))


def f_lab(t: float) -> float:
    delta = 6/29
    if t > delta**3:
        return t ** (1/3)
    return t / (3*delta**2) + 4/29


def srgb_to_lab(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    X, Y, Z = srgb_to_xyz(rgb)
    fx, fy, fz = f_lab(X/Xn), f_lab(Y/Yn), f_lab(Z/Zn)
    L = 116*fy - 16
    a = 500*(fx - fy)
    b = 200*(fy - fz)
    return (L, a, b)


# CIEDE2000 implementation
# Returns perceptual color difference (smaller is better)
# Reference: Sharma et al. (2005)

def ciede2000(Lab1: Tuple[float, float, float], Lab2: Tuple[float, float, float]) -> float:
    L1, a1, b1 = Lab1
    L2, a2, b2 = Lab2

    avg_L = (L1 + L2) / 2
    C1 = math.hypot(a1, b1)
    C2 = math.hypot(a2, b2)
    avg_C = (C1 + C2) / 2

    G = 0.5 * (1 - math.sqrt((avg_C**7) / (avg_C**7 + 25**7))) if avg_C != 0 else 0

    a1p = (1 + G) * a1
    a2p = (1 + G) * a2

    C1p = math.hypot(a1p, b1)
    C2p = math.hypot(a2p, b2)
    avg_Cp = (C1p + C2p) / 2

    h1p = math.degrees(math.atan2(b1, a1p)) % 360
    h2p = math.degrees(math.atan2(b2, a2p)) % 360

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = 0
    if C1p * C2p != 0:
        dh = h2p - h1p
        if dh > 180:
            dh -= 360
        elif dh < -180:
            dh += 360
        dhp = dh
    dHp = 2 * math.sqrt(C1p*C2p) * math.sin(math.radians(dhp/2))

    avg_hp = 0
    if C1p * C2p == 0:
        avg_hp = h1p + h2p
    else:
        hsum = h1p + h2p
        if abs(h1p - h2p) > 180:
            avg_hp = (hsum + 360) / 2 if hsum < 360 else (hsum - 360) / 2
        else:
            avg_hp = hsum / 2

    T = 1 - 0.17*math.cos(math.radians(avg_hp - 30)) + \
        0.24*math.cos(math.radians(2*avg_hp)) + \
        0.32*math.cos(math.radians(3*avg_hp + 6)) - \
        0.20*math.cos(math.radians(4*avg_hp - 63))

    d_ro = 30 * math.exp(-(((avg_hp - 275)/25)**2))
    RC = 2 * math.sqrt((avg_Cp**7) / (avg_Cp**7 + 25**7)) if avg_Cp != 0 else 0
    RT = -RC * math.sin(math.radians(2 * d_ro))

    SL = 1 + (0.015 * (avg_L - 50) ** 2) / math.sqrt(20 + (avg_L - 50) ** 2)
    SC = 1 + 0.045 * avg_Cp
    SH = 1 + 0.015 * avg_Cp * T

    dE = math.sqrt((dLp/SL)**2 + (dCp/SC)**2 + (dHp/SH)**2 + RT * (dCp/SC) * (dHp/SH))
    return dE


# -----------------------------
# Lighting model (simple OSL planner)
# -----------------------------
@dataclass
class OSLParams:
    ambient_intensity: float = 0.2   # 0..1
    light_intensity: float = 1.0     # arbitrary scale
    ndotl: float = 1.0               # 0..1 incidence factor
    distance: float = 1.0            # arbitrary units
    falloff: str = 'inverse_square'  # 'none', 'inverse', 'linear', 'inverse_square'


def _falloff_scale(distance: float, mode: str) -> float:
    d = max(1e-6, distance)
    m = mode.lower()
    if m == 'none':
        return 1.0
    if m == 'inverse':
        return 1.0 / d
    if m == 'linear':
        return max(0.0, 1.0 - d)
    # default inverse square
    return 1.0 / (d * d)


def simulate_osl(
    surface_rgb: Tuple[int, int, int],
    light_rgb: Tuple[int, int, int],
    ambient_intensity: float,
    light_intensity: float = 1.0,
    ndotl: float = 1.0,
    distance: float = 1.0,
    falloff: str = 'inverse_square'
) -> Tuple[int, int, int]:
    """
    Very simple Lambert + ambient model on sRGB colors.
    We linearize to work in linear RGB, blend, then return to sRGB.
    """
    # Clamp inputs
    ambient_intensity = clamp01(ambient_intensity)
    ndotl = clamp01(ndotl)
    light_intensity = max(0.0, light_intensity)

    # to linear RGB
    sr, sg, sb = to_float(surface_rgb)
    lr, lg, lb = to_float(light_rgb)

    srl, sgl, sbl = srgb_to_linear(sr), srgb_to_linear(sg), srgb_to_linear(sb)
    lrl, lgl, lbl = srgb_to_linear(lr), srgb_to_linear(lg), srgb_to_linear(lb)

    # Ambient term (scaled surface color)
    ambient = (
        srl * ambient_intensity,
        sgl * ambient_intensity,
        sbl * ambient_intensity,
    )

    # Direct light term (Lambertian: surface * light * ndotl * intensity * falloff)
    scale = ndotl * light_intensity * _falloff_scale(distance, falloff)
    direct = (
        srl * lrl * scale,
        sgl * lgl * scale,
        sbl * lbl * scale,
    )

    # Combine and convert back to sRGB
    out_lin = (
        ambient[0] + direct[0],
        ambient[1] + direct[1],
        ambient[2] + direct[2],
    )

    out_srgb = (
        linear_to_srgb(out_lin[0]),
        linear_to_srgb(out_lin[1]),
        linear_to_srgb(out_lin[2]),
    )
    return to_int(out_srgb)


# -----------------------------
# Matching utilities (nearest paint, two-paint mix)
# -----------------------------

def find_nearest_paints(target_rgb: Tuple[int, int, int], k: int = 5) -> List[Tuple[str, float]]:
    target_lab = srgb_to_lab(target_rgb)
    scored = []
    for name, rgb in CITADEL_COLORS.items():
        dE = ciede2000(target_lab, srgb_to_lab(rgb))
        scored.append((name, dE))
    scored.sort(key=lambda x: x[1])
    return scored[:k]


def mix_rgb(rgb1: Tuple[int, int, int], rgb2: Tuple[int, int, int], ratio_2: float) -> Tuple[int, int, int]:
    """Linear mix in linear RGB. ratio_2 = 0..1 is the fraction of color 2."""
    r1, g1, b1 = to_float(rgb1)
    r2, g2, b2 = to_float(rgb2)
    r1l, g1l, b1l = srgb_to_linear(r1), srgb_to_linear(g1), srgb_to_linear(b1)
    r2l, g2l, b2l = srgb_to_linear(r2), srgb_to_linear(g2), srgb_to_linear(b2)

    t = clamp01(ratio_2)
    rl = (1-t)*r1l + t*r2l
    gl = (1-t)*g1l + t*g2l
    bl = (1-t)*b1l + t*b2l

    return to_int((linear_to_srgb(rl), linear_to_srgb(gl), linear_to_srgb(bl)))


def suggest_two_paint_mix(target_rgb: Tuple[int, int, int], step: int = 10) -> Dict[str, object]:
    """
    Brute-force search over all unordered pairs of paints and ratio steps to approximate target.
    step: ratio step in percent (e.g., 10 -> 0%,10%,...,100%).
    Returns dict with best pair, ratio, dE, and the mixed color.
    """
    if step <= 0 or step > 50:
        raise ValueError('step should be in 1..50 (percent).')

    target_lab = srgb_to_lab(target_rgb)
    best = {
        'pair': None,
        'ratio_2_percent': None,
        'mixed_rgb': None,
        'deltaE2000': float('inf')
    }

    names = list(CITADEL_COLORS.keys())
    for i, name1 in enumerate(names):
        for j in range(i, len(names)):
            name2 = names[j]
            rgb1 = CITADEL_COLORS[name1]
            rgb2 = CITADEL_COLORS[name2]
            for r in range(0, 101, step):
                ratio_2 = r / 100.0
                mix = mix_rgb(rgb1, rgb2, ratio_2)
                dE = ciede2000(target_lab, srgb_to_lab(mix))
                if dE < best['deltaE2000']:
                    best = {
                        'pair': (name1, name2),
                        'ratio_2_percent': r,
                        'mixed_rgb': mix,
                        'deltaE2000': dE
                    }
    return best


# -----------------------------
# Convenience helpers for named paints
# -----------------------------

def get_paint(name: str) -> Tuple[int, int, int]:
    if name not in CITADEL_COLORS:
        raise KeyError(f'Paint "{name}" not found. Add it to CITADEL_COLORS.')
    return CITADEL_COLORS[name]


def simulate_osl_named(surface_paint: str, light_paint: str, params: OSLParams = OSLParams()) -> Tuple[int, int, int]:
    return simulate_osl(
        surface_rgb=get_paint(surface_paint),
        light_rgb=get_paint(light_paint),
        ambient_intensity=params.ambient_intensity,
        light_intensity=params.light_intensity,
        ndotl=params.ndotl,
        distance=params.distance,
        falloff=params.falloff,
    )

# -----------------------------
# Visualization helpers (distance gradient / color blocks)
# -----------------------------

def return_color_blocks(
    distances: List[float],
    surface_rgb: Tuple[int, int, int],
    light_rgb: Tuple[int, int, int],
    ambient_intensity: float,
    light_intensity: float = 1.0,
    ndotl: float = 1.0,
    falloff: str = 'inverse_square',
) -> List[Tuple[float, Tuple[int, int, int], str]]:
    """
    For a list of distances, compute OSL result colors.
    Returns a list of tuples: (distance, rgb, hex_string).
    """
    out: List[Tuple[float, Tuple[int,int,int], str]] = []
    for d in distances:
        rgb = simulate_osl(
            surface_rgb=surface_rgb,
            light_rgb=light_rgb,
            ambient_intensity=ambient_intensity,
            light_intensity=light_intensity,
            ndotl=ndotl,
            distance=d,
            falloff=falloff,
        )
        out.append((d, rgb, rgb_to_hex(rgb)))
    return out


def plot_distance_gradient(
    distances: List[float],
    surface_rgb: Tuple[int, int, int],
    light_rgb: Tuple[int, int, int],
    ambient_intensity: float,
    light_intensity: float = 1.0,
    ndotl: float = 1.0,
    falloff: str = 'inverse_square',
    block_size: int = 60,
    annotate_hex: bool = True,
):
    """
    Visualize a left-to-right strip of color blocks for the given distances.
    Lazy-imports matplotlib. Does not depend on it otherwise.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        raise ImportError('plot_distance_gradient requires matplotlib and numpy.')

    blocks = return_color_blocks(
        distances,
        surface_rgb,
        light_rgb,
        ambient_intensity,
        light_intensity,
        ndotl,
        falloff,
    )

    # Build an image array: height x (N*block_size) x 3
    N = len(blocks)
    height = block_size
    width = block_size * N
    img = np.zeros((height, width, 3), dtype=float)

    for i, (_, rgb, _) in enumerate(blocks):
        r, g, b = [v / 255.0 for v in rgb]
        img[:, i*block_size:(i+1)*block_size, 0] = r
        img[:, i*block_size:(i+1)*block_size, 1] = g
        img[:, i*block_size:(i+1)*block_size, 2] = b

    plt.figure(figsize=(max(3, N*1.2), 2))
    plt.imshow(img)
    plt.axis('off')

    if annotate_hex:
        # Add labels centered in each block
        for i, (d, _, hx) in enumerate(blocks):
            x = (i + 0.5) * block_size
            y = height * 0.5
            #plt.text(x, y, f"d={d}\\n{hx}", ha='center', va='center', fontsize=10, bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.6))
                     
    plt.tight_layout()
    plt.show()


def show_colors(color_dict):
    # Convert colors to hex strings
    names = list(color_dict.keys())
    hex_colors = [f'rgb{color_dict[name]}' for name in names]
    
    # Create positions for each color
    n_cols = 8
    x = [i % n_cols for i in range(len(names))]
    y = [-(i // n_cols) for i in range(len(names))]  # negative so top-down order
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(
            size=60,
            color=hex_colors,
            symbol='square'
        ),
        text=names,  # Hover text
        hoverinfo='text'
    ))
    
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        plot_bgcolor='white',
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    fig.show()


def _rgb_to_hsv(rgb):
    r, g, b = [v/255 for v in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)  # h in [0,1]
    return h*360, s, v

def _luminance(rgb):
    # Quick relative luminance for border contrast (sRGB approx)
    r, g, b = rgb
    return 0.299*r + 0.587*g + 0.114*b  # 0..255

def _as_rgb_str(rgb):
    return f"rgb{rgb}"

def _prepare_lists(color_dict):
    names = list(color_dict.keys())
    rgbs = [color_dict[n] for n in names]
    hsvs = [_rgb_to_hsv(rgb) for rgb in rgbs]
    colors = [_as_rgb_str(rgb) for rgb in rgbs]
    borders = ["white" if _luminance(rgb) < 128 else "black" for rgb in rgbs]
    return names, rgbs, hsvs, colors, borders

def show_colors_grid(color_dict, n_cols=8, sort='hue'):

    names, rgbs, hsvs, colors, borders = _prepare_lists(color_dict)

    if sort == 'hue':
        order = sorted(range(len(names)), key=lambda i: (hsvs[i][0], -hsvs[i][1], -hsvs[i][2]))
    elif sort == 'lightness':
        order = sorted(range(len(names)), key=lambda i: _luminance(rgbs[i]))
    else:
        order = list(range(len(names)))

    names = [names[i] for i in order]
    colors = [colors[i] for i in order]
    borders = [borders[i] for i in order]

    x = [i % n_cols for i in range(len(names))]
    y = [-(i // n_cols) for i in range(len(names))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='markers',
        marker=dict(size=60, color=colors, symbol='square',
                    line=dict(width=2, color=borders)),
        text=names, hoverinfo='text'
    ))
    fig.update_layout(
        showlegend=False, plot_bgcolor='white',
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        margin=dict(l=10, r=10, t=10, b=10),
        title=f"Citadel Colors — grid ({sort or 'original'} order)"
    )
    fig.show()

def show_colors_wheel(color_dict, radius_scale=0.9):
    """
    Hue wheel: angle = hue, radius ~ saturation (greys near center).
    """
    names, rgbs, hsvs, colors, borders = _prepare_lists(color_dict)

    # positions
    angles = [math.radians(h) for h, s, v in hsvs]
    radii  = [radius_scale * (0.15 + 0.85*s) for h, s, v in hsvs]  # keep a small inner radius so labels hover nicely

    x = [r*math.cos(a) for r, a in zip(radii, angles)]
    y = [r*math.sin(a) for r, a in zip(radii, angles)]

    # Sort trace drawing order by hue so the ring looks continuous
    order = sorted(range(len(names)), key=lambda i: hsvs[i][0])
    names = [names[i] for i in order]
    colors = [colors[i] for i in order]
    borders = [borders[i] for i in order]
    x = [x[i] for i in order]
    y = [y[i] for i in order]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='markers',
        marker=dict(size=60, color=colors, symbol='square',
                    line=dict(width=2, color=borders)),
        text=names, hoverinfo='text'
    ))

    # Add a faint guide circle
    theta = [math.radians(t) for t in range(0, 360)]
    gx = [radius_scale*math.cos(t) for t in theta]
    gy = [radius_scale*math.sin(t) for t in theta]
    fig.add_trace(go.Scatter(x=gx, y=gy, mode='lines',
                             line=dict(width=1, dash='dot'),
                             hoverinfo='skip', showlegend=False))

    fig.update_layout(
        showlegend=False, plot_bgcolor='white',
        xaxis=dict(showgrid=False, zeroline=False, visible=False, scaleanchor='y', scaleratio=1),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        margin=dict(l=10, r=10, t=10, b=10),
        title="Citadel Colors — hue wheel (r ∝ saturation)"
    )
    fig.show()



if __name__ == '__main__':
    # Quick demo
    surface = CITADEL_COLORS['Mephiston Red']
    light   = CITADEL_COLORS['Flash Gitz Yellow']
    out = simulate_osl(surface, light, ambient_intensity=0.25, ndotl=0.7, distance=1.5)
    print('Result sRGB:', out, 'hex:', rgb_to_hex(out))
    print('Nearest paints:')
    for name, dE in find_nearest_paints(out, k=5):
        print(f'  {name:24s} ΔE={dE:.2f}')
    print('Best two-paint mix:')
    bm = suggest_two_paint_mix(out, step=10)
    print(bm)
