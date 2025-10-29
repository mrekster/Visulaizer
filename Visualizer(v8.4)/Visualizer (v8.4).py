"""
==========================
VISUALIZER - Version 8.2
==========================
Major changes for v8:
- Demo mode
- WASD: removed
- [S] select all
- Grid control
----------
Changelog:
----------

v8.0:
- Exit screen no longer appears inside demo mode
- loading time shortened

v8.1:
- Fixed editing in demo mode
- fixed more leftover zoom parameters

v8.2:
- Added shift+ ./, for quicker Grid control
- Added Grid control limit size
- Added grid Control details in menu

v8.3:
- Menu reorganized

v8.4:
- Fixed copy/paste to: [Shift+D],  [Shift+W]

"""


import sys
import math
import copy
import pygame
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import random


# =======================
# Config
SCREEN_SIZE: Tuple[int, int] = (1920, 1080) #1920, 1080 standard
RES = "1920x1080"
FULLSCREEN: bool = True
# =======================

BG_COLOR = (18, 18, 22)
FG_COLOR = (230, 230, 235)
ACCENT = (120, 180, 255)
SELECTION = (255, 200, 90)
CENTER_MARK = (160, 210, 120)
GRID_COLOR = (35, 35, 40)
HANDLE_FILL = (255, 255, 255)
HANDLE_SIZE = 6
MOVE_STEP = 10
SCALE_STEP = 1.1
FONT_SIZE = 16
GRID_SPACING = 23
ROTATE_STEP_DEG = 5
CENTER_SIZE = 6
PASTE_OFFSET = 25
THICKNESS_HANDLE_RADIUS = 6
THICKNESS_HIT_RADIUS = 12
THICKNESS_MAX = 200

PALETTE = {
    pygame.K_1: (255, 255, 255),
    pygame.K_2: (255, 0, 0),
    pygame.K_3: (0, 200, 0),
    pygame.K_4: (0, 120, 255),
    pygame.K_5: (255, 200, 0),
    pygame.K_6: (255, 0, 200),
    pygame.K_7: (0, 255, 255),
    pygame.K_8: (180, 180, 180),
    pygame.K_9: (30, 220, 120),
}


pygame.init()
pygame.display.set_caption("Visualizer v11 — M for menu")
pygame.display.set_icon(pygame.image.load("Visualizer.png"))


def show_splash():

    temp = pygame.display.set_mode(SCREEN_SIZE, 0)
    bg = (18, 18, 22)


    logo = None
    for path in [r"C:\Users\marec\OneDrive\Obrázky\pictures\coding\Visualizer.png",
    ]:
        try:
            logo = pygame.image.load(path).convert_alpha()
            break
        except Exception:
            pass
    if logo is None:

        temp.fill(bg)
        f = pygame.font.SysFont("consolas", 42)
        t = f.render("Loading Visualizer…", True, (230, 230, 235))
        temp.blit(t, t.get_rect(center=(SCREEN_SIZE[0]//2, SCREEN_SIZE[1]//2)))
        pygame.display.flip()
        pygame.time.wait(800)
        return


    lw = min(SCREEN_SIZE[0], SCREEN_SIZE[1]) * 0.6
    scale = lw / max(logo.get_width(), logo.get_height())
    logo = pygame.transform.smoothscale(
        logo, (int(logo.get_width()*scale), int(logo.get_height()*scale))
    )
    rect = logo.get_rect(center=(SCREEN_SIZE[0]//2, SCREEN_SIZE[1]//2))

    clk = pygame.time.Clock()

    # --- Fade in ---
    alpha = 0
    while alpha < 255:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); sys.exit()
        temp.fill(bg)
        logo.set_alpha(alpha)
        temp.blit(logo, rect)
        pygame.display.flip()
        alpha += 10
        clk.tick(60)

    # --- Random hold ---
    hold_ms = int(random.uniform(1.0, 2.6) * 1000)
    elapsed = 0
    while elapsed < hold_ms:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); sys.exit()
        temp.fill(bg)
        logo.set_alpha(255)
        temp.blit(logo, rect)
        pygame.display.flip()
        dt = clk.tick(60)
        elapsed += dt


    jitter = lambda: random.randint(-4, 4)
    for frame in range(1):  # two quick flashes
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); sys.exit()
        temp.fill((230, 230, 230))
        jrect = rect.move(jitter(), jitter())
        pygame.display.flip()
        pygame.time.wait(7)
        temp.fill(bg)
        temp.blit(logo, rect)
        pygame.display.flip()
        pygame.time.wait(9)

    lw = min(SCREEN_SIZE[0], SCREEN_SIZE[1]) * 1
    scale = lw / max(logo.get_width(), logo.get_height())
    logo = pygame.transform.smoothscale(
        logo, (int(logo.get_width()*scale), int(logo.get_height()*scale))
    )
    rect = logo.get_rect(center=(SCREEN_SIZE[0]//2, SCREEN_SIZE[1]//2))

    clk = pygame.time.Clock()
    pygame.time.wait(18)

    # --- Fade out ---
    alpha = 255
    while alpha > 0:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); sys.exit()
        temp.fill(bg)
        logo.set_alpha(alpha)
        temp.blit(logo, rect)
        pygame.display.flip()
        alpha -= 12
        clk.tick(60)

    pygame.time.wait(120)

# run splash before creating your real screen
show_splash()

flags = pygame.FULLSCREEN if FULLSCREEN else 0
screen = pygame.display.set_mode(SCREEN_SIZE, flags)
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", FONT_SIZE)
grid_spacing = GRID_SPACING


def world_to_screen(pt, sw, sh):
    x, y = pt
    return (int(round(sw/2 + x)), int(round(sh/2 + y)))

def screen_to_world(px, sw, sh):
    mx, my = px
    return (mx - sw/2, my - sh/2)

def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])

def draw_grid_centered(surface, spacing_world):
    sw, sh = surface.get_size()
    spacing_px = spacing_world
    if spacing_px < 6:
        return
    left_world   = -(sw/2)
    right_world  =  (sw/2)
    top_world    = -(sh/2)
    bottom_world =  (sh/2)
    kx_start = math.floor(left_world  / spacing_world)
    kx_end   = math.ceil (right_world / spacing_world)
    ky_start = math.floor(top_world   / spacing_world)
    ky_end   = math.ceil (bottom_world/ spacing_world)
    for kx in range(kx_start, kx_end+1):
        x_world = kx * spacing_world
        x_screen, _ = world_to_screen((x_world, 0) , sw, sh)
        pygame.draw.line(surface, GRID_COLOR, (x_screen, 0), (x_screen, sh), 1)
    for ky in range(ky_start, ky_end+1):
        y_world = ky * spacing_world
        _, y_screen = world_to_screen((0, y_world) , sw, sh)
        pygame.draw.line(surface, GRID_COLOR, (0, y_screen), (sw, y_screen), 1)

def draw_axes_center(surface):
    sw, sh = surface.get_size()
    cx, cy = sw//2, sh//2
    pygame.draw.line(surface, (80,255,120), (cx, 0), (cx, sh), 2)
    pygame.draw.line(surface, (255,80,80),  (0, cy), (sw, cy), 2)

def clamp_int_tuple(pt):
    return (int(round(pt[0])), int(round(pt[1])))

def rnd_to_grid(v, grid=GRID_SPACING):
    return round(v / grid) * grid

def snap_point_to_centered_grid(x, y, grid=GRID_SPACING):
    sw, sh = screen.get_size()
    wx = x - sw/2
    wy = y - sh/2
    wx = rnd_to_grid(wx, grid)
    wy = rnd_to_grid(wy, grid)
    return (wx + sw/2, wy + sh/2)


def rotate_point(px, py, cx, cy, angle_rad):
    s, c = math.sin(angle_rad), math.cos(angle_rad)
    x, y = px - cx, py - cy
    xr = x * c - y * s
    yr = x * s + y * c
    return (cx + xr, cy + yr)

def draw_center_cross(surface, cx, cy, color=CENTER_MARK, size=CENTER_SIZE):
    cx_i, cy_i = int(round(cx)), int(round(cy))
    pygame.draw.line(surface, color, (cx_i - size, cy_i), (cx_i + size, cy_i), 1)
    pygame.draw.line(surface, color, (cx_i, cy_i - size), (cx_i, cy_i + size), 1)

class Shape:
    def move(self, dx, dy): ...
    def scale(self, factor): ...
    def draw(self, surface, selected: bool): ...
    def hit_test(self, pt) -> bool: ...
    def export_code(self) -> str: ...
    def toggle_fill(self): ...
    def set_color(self, color: Tuple[int, int, int]): ...
    def snap_to_grid(self): ...
    def rotate(self, deg: float): ...
    def center(self) -> Tuple[float, float]: ...
    def get_width(self) -> Optional[int]: ...
    def set_width(self, w: int): ...

@dataclass
class Line(Shape):
    a: Tuple[float, float]
    b: Tuple[float, float]
    color: Tuple[int, int, int] = FG_COLOR
    width: int = 2
    drag_mode: Optional[str] = None

    def move(self, dx, dy):
        self.a = (self.a[0]+dx, self.a[1]+dy)
        self.b = (self.b[0]+dx, self.b[1]+dy)

    def scale(self, factor):
        mx, my = self.center()
        ax = mx + (self.a[0]-mx)*factor
        ay = my + (self.a[1]-my)*factor
        bx = mx + (self.b[0]-mx)*factor
        by = my + (self.b[1]-my)*factor
        self.a, self.b = (ax, ay), (bx, by)

    def snap_to_grid(self):
        def snap_to_grid(self):
            self.a = snap_point_to_centered_grid(self.a[0], self.a[1], grid_spacing)
            self.b = snap_point_to_centered_grid(self.b[0], self.b[1], grid_spacing)

    def rotate(self, deg: float):
        mx, my = self.center()
        ang = math.radians(deg)
        self.a = rotate_point(self.a[0], self.a[1], mx, my, ang)
        self.b = rotate_point(self.b[0], self.b[1], mx, my, ang)

    def draw(self, surface, selected):
        pygame.draw.line(surface, self.color, self.a, self.b, self.width)
        if selected:
            for p in (self.a, self.b):
                pygame.draw.rect(surface, SELECTION,
                                 (*[int(p[0]-HANDLE_SIZE/2), int(p[1]-HANDLE_SIZE/2)], HANDLE_SIZE, HANDLE_SIZE), 1)
                pygame.draw.rect(surface, (255,255,255),
                                 (*[int(p[0]-2), int(p[1]-2)], 4, 4), 0)
        cx, cy = self.center()
        draw_center_cross(surface, cx, cy)

    def hit_test(self, pt) -> bool:
        ax, ay = self.a
        bx, by = self.b
        px, py = pt
        abx, aby = bx-ax, by-ay
        apx, apy = px-ax, py-ay
        ab2 = abx*abx + aby*aby
        if ab2 == 0:
            return dist(pt, self.a) <= 8
        t = max(0, min(1, (apx*abx + apy*aby)/ab2))
        proj = (ax + t*abx, ay + t*aby)
        return dist(pt, proj) <= 8

    def near_handle(self, pt) -> Optional[str]:
        if dist(pt, self.a) <= 10: return "a"
        if dist(pt, self.b) <= 10: return "b"
        return None

    def export_code(self) -> str:
        a = clamp_int_tuple(self.a)
        b = clamp_int_tuple(self.b)
        return f"pygame.draw.line(screen, {self.color}, {a}, {b}, {self.width})"

    def toggle_fill(self): pass
    def set_color(self, color): self.color = color

    def center(self) -> Tuple[float, float]:
        return ((self.a[0]+self.b[0])/2, (self.a[1]+self.b[1])/2)

    def get_width(self) -> Optional[int]: return self.width
    def set_width(self, w: int): self.width = max(1, min(THICKNESS_MAX, int(w)))

@dataclass
class Rect(Shape):
    x: float
    y: float
    w: float
    h: float
    color: Tuple[int, int, int] = FG_COLOR
    width: int = 2

    def move(self, dx, dy):
        self.x += dx; self.y += dy

    def scale(self, factor):
        cx, cy = self.center()
        self.w *= factor; self.h *= factor
        self.x = cx - self.w/2; self.y = cy - self.h/2

    def snap_to_grid(self):
        def snap_to_grid(self):
            cx, cy = self.center()
            sx, sy = snap_point_to_centered_grid(cx, cy, grid_spacing)
            self.x = sx - self.w / 2
            self.y = sy - self.h / 2

    def rotate(self, deg: float):
        cx, cy = self.center()
        pts = [
            (self.x, self.y),
            (self.x + self.w, self.y),
            (self.x + self.w, self.y + self.h),
            (self.x, self.y + self.h),
        ]
        ang = math.radians(deg)
        rpts = [rotate_point(px, py, cx, cy, ang) for (px, py) in pts]
        return Polygon(points=rpts, color=self.color, width=self.width)

    def draw(self, surface, selected):
        pygame.draw.rect(surface, self.color, pygame.Rect(self.x, self.y, self.w, self.h), self.width)
        if selected:
            r = pygame.Rect(self.x, self.y, self.w, self.h)
            pygame.draw.rect(surface, SELECTION, r, 1)
        cx, cy = self.center()
        draw_center_cross(surface, cx, cy)

    def hit_test(self, pt) -> bool:
        return pygame.Rect(self.x, self.y, self.w, self.h).inflate(8, 8).collidepoint(pt)

    def export_code(self) -> str:
        x, y, w, h = int(round(self.x)), int(round(self.y)), int(round(self.w)), int(round(self.h))
        return f"pygame.draw.rect(screen, {self.color}, pygame.Rect({x}, {y}, {w}, {h}), {self.width})"

    def toggle_fill(self): self.width = 0 if self.width != 0 else 2
    def set_color(self, color): self.color = color

    def center(self) -> Tuple[float, float]:
        return (self.x + self.w/2, self.y + self.h/2)

    def get_width(self) -> Optional[int]: return self.width
    def set_width(self, w: int): self.width = max(1, min(THICKNESS_MAX, int(w)))

@dataclass
class Circle(Shape):
    center_pt: Tuple[float, float]
    radius: float
    color: Tuple[int, int, int] = FG_COLOR
    width: int = 2

    def move(self, dx, dy):
        self.center_pt = (self.center_pt[0]+dx, self.center_pt[1]+dy)

    def scale(self, factor):
        self.radius = max(1, self.radius * factor)

    def snap_to_grid(self):
        self.center_pt = snap_point_to_centered_grid(self.center_pt[0], self.center_pt[1], grid_spacing)

    def rotate(self, deg: float): pass

    def draw(self, surface, selected):
        pygame.draw.circle(surface, self.color, clamp_int_tuple(self.center_pt), int(max(1, round(self.radius))), self.width)
        if selected:
            pygame.draw.circle(surface, SELECTION, clamp_int_tuple(self.center_pt), int(max(1, round(self.radius))), 1)
            h = (self.center_pt[0] + self.radius, self.center_pt[1])
            pygame.draw.rect(surface, HANDLE_FILL, (*[int(h[0]-2), int(h[1]-2)], 4, 4), 0)
            pygame.draw.rect(surface, SELECTION, (*[int(h[0]-HANDLE_SIZE/2), int(h[1]-HANDLE_SIZE/2)], HANDLE_SIZE, HANDLE_SIZE), 1)
        cx, cy = self.center()
        draw_center_cross(surface, cx, cy)

    def hit_test(self, pt) -> bool:
        d = dist(pt, self.center_pt)
        return d <= self.radius + 6

    def near_radius_handle(self, pt) -> bool:
        handle = (self.center_pt[0] + self.radius, self.center_pt[1])
        return dist(pt, handle) <= 10

    def export_code(self) -> str:
        c = clamp_int_tuple(self.center())
        r = int(round(self.radius))
        return f"pygame.draw.circle(screen, {self.color}, {c}, {r}, {self.width})"

    def toggle_fill(self): self.width = 0 if self.width != 0 else 2
    def set_color(self, color): self.color = color

    def center(self) -> Tuple[float, float]:
        return self.center_pt

    def get_width(self) -> Optional[int]: return self.width
    def set_width(self, w: int): self.width = max(1, min(THICKNESS_MAX, int(w)))

@dataclass
class Polygon(Shape):
    points: List[Tuple[float, float]] = field(default_factory=list)
    color: Tuple[int, int, int] = FG_COLOR
    width: int = 2

    def move(self, dx, dy):
        self.points = [(x+dx, y+dy) for (x, y) in self.points]

    def scale(self, factor):
        if not self.points: return
        cx, cy = self.center()
        self.points = [(cx + (x-cx)*factor, cy + (y-cy)*factor) for (x, y) in self.points]

    def snap_to_grid(self):
        self.points = [snap_point_to_centered_grid(x, y, grid_spacing) for (x, y) in self.points]

    def rotate(self, deg: float):
        if not self.points: return
        cx, cy = self.center()
        ang = math.radians(deg)
        self.points = [rotate_point(x, y, cx, cy, ang) for (x, y) in self.points]

    def draw(self, surface, selected):
        if len(self.points) >= 2:
            pygame.draw.polygon(surface, self.color, self.points, self.width)
        if selected:
            pygame.draw.polygon(surface, SELECTION, self.points, 1)
            for (x, y) in self.points:
                pygame.draw.rect(surface, HANDLE_FILL, (*[int(x-2), int(y-2)], 4, 4), 0)
                pygame.draw.rect(surface, SELECTION, (*[int(x-HANDLE_SIZE/2), int(y-HANDLE_SIZE/2)], HANDLE_SIZE, HANDLE_SIZE), 1)
        cx, cy = self.center()
        draw_center_cross(surface, cx, cy)

    def hit_test(self, pt) -> bool:
        if len(self.points) < 3:
            for i in range(len(self.points)-1):
                ax, ay = self.points[i]
                bx, by = self.points[i+1]
                abx, aby = bx-ax, by-ay
                apx, apy = pt[0]-ax, pt[1]-ay
                ab2 = abx*abx + aby*aby
                if ab2 == 0:
                    if dist(pt, (ax, ay)) <= 8:
                        return True
                else:
                    t = max(0, min(1, (apx*abx + apy*aby)/ab2))
                    proj = (ax + t*abx, ay + t*aby)
                    if dist(pt, proj) <= 8:
                        return True
            return False
        bb = pygame.Rect(min(p[0] for p in self.points),
                         min(p[1] for p in self.points),
                         max(p[0] for p in self.points)-min(p[0] for p in self.points),
                         max(p[1] for p in self.points)-min(p[1] for p in self.points)).inflate(8, 8)
        return bb.collidepoint(pt)

    def nearest_vertex(self, pt, threshold=10) -> Optional[int]:
        for i, p in enumerate(self.points):
            if dist(pt, p) <= threshold:
                return i
        return None

    def export_code(self) -> str:
        pts = [clamp_int_tuple(p) for p in self.points]
        return f"pygame.draw.polygon(screen, {self.color}, {pts}, {self.width})"

    def toggle_fill(self): self.width = 0 if self.width != 0 else 2
    def set_color(self, color): self.color = color

    def center(self) -> Tuple[float, float]:
        n = len(self.points)
        if n < 3:
            if n == 0: return (0.0, 0.0)
            sx = sum(p[0] for p in self.points); sy = sum(p[1] for p in self.points)
            return (sx/n, sy/n)
        area = 0.0
        cx = 0.0
        cy = 0.0
        for i in range(n):
            x0, y0 = self.points[i]
            x1, y1 = self.points[(i+1) % n]
            cross = x0*y1 - x1*y0
            area += cross
            cx += (x0 + x1) * cross
            cy += (y0 + y1) * cross
        area *= 0.5
        if abs(area) < 1e-6:
            sx = sum(p[0] for p in self.points); sy = sum(p[1] for p in self.points)
            return (sx/n, sy/n)
        cx /= (6.0 * area)
        cy /= (6.0 * area)
        return (cx, cy)

    def get_width(self) -> Optional[int]: return self.width
    def set_width(self, w: int): self.width = max(1, min(THICKNESS_MAX, int(w)))

shapes: List[Shape] = []
selected: List[Shape] = []
dragging = False
drag_offset = (0, 0)

creating_mode: Optional[str] = None
create_anchor: Optional[Tuple[float, float]] = None
temp_poly_points: List[Tuple[float, float]] = []
active_vertex_index: Optional[int] = None
circle_resizing = False

undo_stack: List[List[Shape]] = []

copied_shapes: List[Shape] = []

show_help = False
show_exit_confirm = False
demo_mode = False

thickness_mode = False
thickness_drag_shape: Optional[Shape] = None

def poly_area_abs(points: List[Tuple[float, float]]) -> float:
    n = len(points)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i+1) % n]
        area += x0*y1 - x1*y0
    return abs(area) * 0.5

def shape_pick_metric(shp: Shape) -> float:
    if isinstance(shp, Circle):
        return math.pi * shp.radius * shp.radius
    if isinstance(shp, Rect):
        return abs(shp.w * shp.h)
    if isinstance(shp, Polygon):
        area = poly_area_abs(shp.points)
        if area < 1e-6:
            if not shp.points:
                return float('inf')
            minx = min(p[0] for p in shp.points); maxx = max(p[0] for p in shp.points)
            miny = min(p[1] for p in shp.points); maxy = max(p[1] for p in shp.points)
            return abs((maxx-minx)*(maxy-miny))
        return area
    if isinstance(shp, Line):
        return dist(shp.a, shp.b)
    return float('inf')

def select_at(pt) -> Optional[Shape]:
    hits = [shp for shp in shapes if shp.hit_test(pt)]
    if not hits:
        return None
    best = None
    best_metric = None
    best_index = -1
    for shp in hits:
        m = shape_pick_metric(shp)
        original_z = shapes.index(shp)
        if best is None or m < best_metric - 1e-9 or (abs(m - best_metric) <= 1e-9 and original_z > best_index):
            best = shp
            best_metric = m
            best_index = original_z
    return best

def bring_to_top(shp: Shape):
    if shp in shapes:
        shapes.remove(shp)
        shapes.append(shp)

def push_undo():
    undo_stack.append(copy.deepcopy(shapes))
    if len(undo_stack) > 200:
        undo_stack.pop(0)

def undo():
    global shapes, selected
    if undo_stack:
        shapes[:] = undo_stack.pop()
        selected.clear()

def export_all_to_console():
    print("\n# ===== PYGAME DRAW EXPORT =====")
    print("# Paste these inside your draw loop (replace 'screen' if needed).")
    for shp in shapes:
        try:
            print(shp.export_code())
        except Exception as e:
            print(f"# Failed to export {shp}: {e}")
    print("# ===== END EXPORT =====\n")

def snap_one_vertices(shp: Shape):
    if hasattr(shp, "snap_to_grid"):
        shp.snap_to_grid()

def snap_one_center(shp: Shape):
    if hasattr(shp, "center"):
        cx, cy = shp.center()
        sx, sy = snap_point_to_centered_grid(cx, cy, GRID_SPACING)
        dx, dy = sx - cx, sy - cy
        shp.move(dx, dy)

def set_color_one(shp: Shape, color: Tuple[int, int, int]):
    if hasattr(shp, "set_color"):
        shp.set_color(color)

def rotate_one(shp: Shape, deg: float) -> Shape:
    if isinstance(shp, Rect):
        new_poly = shp.rotate(deg)
        if isinstance(new_poly, Polygon):
            return new_poly
        return shp
    else:
        if hasattr(shp, "rotate"):
            shp.rotate(deg)
        return shp

def replace_shape(old: Shape, new: Shape):
    if old is new: return
    try:
        idx = shapes.index(old)
        shapes[idx] = new
        if old in selected:
            si = selected.index(old)
            selected[si] = new
    except ValueError:
        pass

def eligible_for_thickness(shp: Shape) -> bool:
    w = shp.get_width() if hasattr(shp, "get_width") else None
    return (w is not None) and (w > 0)

def get_thickness_handle_pos(shp: Shape) -> Tuple[float, float]:
    cx, cy = shp.center()
    w = shp.get_width()
    d = max(1, int(w))
    return (cx + d, cy)

def draw_thickness_handles(surface):
    for shp in selected:
        if eligible_for_thickness(shp):
            cx, cy = shp.center()
            hx, hy = get_thickness_handle_pos(shp)
            pygame.draw.line(surface, ACCENT, (int(cx), int(cy)), (int(hx), int(hy)), 1)
            pygame.draw.circle(surface, ACCENT, (int(hx), int(hy)), THICKNESS_HANDLE_RADIUS, 0)

def thickness_pick_handle(pt) -> Optional[Shape]:
    best = None
    bestd = None
    for shp in selected:
        if eligible_for_thickness(shp):
            hx, hy = get_thickness_handle_pos(shp)
            d = dist(pt, (hx, hy))
            if d <= THICKNESS_HIT_RADIUS and (best is None or d < bestd):
                best = shp
                bestd = d
    return best

def thickness_update_from_mouse(shp: Shape, mouse: Tuple[int, int]):
    cx, cy = shp.center()
    d = dist((cx, cy), mouse)
    shp.set_width(max(1, min(THICKNESS_MAX, int(round(d/12)))))

def finish_creation():
    if demo_mode:
        return
    global creating_mode, create_anchor, temp_poly_points, selected
    if creating_mode == "rect" and create_anchor is not None:
        ax, ay = create_anchor
        mx, my = pygame.mouse.get_pos()
        w, h = mx - ax, my - ay
        rect = Rect(min(ax, mx), min(ay, my), abs(w), abs(h))
        push_undo()
        shapes.append(rect); selected = [rect]
    elif creating_mode == "circle" and create_anchor is not None:
        ax, ay = create_anchor
        mx, my = pygame.mouse.get_pos()
        radius = max(1, int(dist((ax, ay), (mx, my))))
        circ = Circle((ax, ay), radius)
        push_undo()
        shapes.append(circ); selected = [circ]
    elif creating_mode == "line" and create_anchor is not None:
        ax, ay = create_anchor
        mx, my = pygame.mouse.get_pos()
        push_undo()
        line = Line((ax, ay), (mx, my))
        shapes.append(line); selected = [line]
    elif creating_mode == "poly" and len(temp_poly_points) >= 3:
        if dist(temp_poly_points[-1], temp_poly_points[0]) <= 12:
            temp_poly_points[-1] = temp_poly_points[0]
        push_undo()
        poly = Polygon(list(temp_poly_points))
        shapes.append(poly); selected = [poly]
    reset_creation()

def reset_creation():
    global creating_mode, create_anchor, temp_poly_points, circle_resizing, active_vertex_index
    creating_mode = None
    create_anchor = None
    temp_poly_points = []
    circle_resizing = False
    active_vertex_index = None

running = True
while running:
    dt = clock.tick(120) / 1000.0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            show_exit_confirm = True

            # --- Demo toggle on Middle Mouse Button ---
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 2:
            demo_mode = not demo_mode
            if demo_mode:
                # entering demo: clear all interactive overlays/selection
                selected.clear()
                thickness_mode = False
                show_help = False
            continue  # don't process this event further

        if event.type == pygame.KEYDOWN:
            mods = pygame.key.get_mods()
            shift = bool(mods & pygame.KMOD_SHIFT)

            if show_exit_confirm:
                if event.key == pygame.K_ESCAPE:
                    show_exit_confirm = False
                elif event.key == pygame.K_RETURN:
                    running = False
                continue

            if demo_mode:
                if event.key == pygame.K_x and not shift:
                    show_exit_confirm = True


            elif event.key in (pygame.K_COMMA, pygame.K_PERIOD):
                step = 10 if shift else 1
                if event.key == pygame.K_COMMA:
                    grid_spacing = max(4, grid_spacing - step)  # clamp low
                else:
                    grid_spacing = min(2000, grid_spacing + step)  # clamp high
                print(f"Grid spacing: {grid_spacing}px")

            if event.key == pygame.K_ESCAPE:
                if creating_mode == "poly" and temp_poly_points:
                    reset_creation()

            elif event.key == pygame.K_RETURN:
                if creating_mode in ("rect", "circle", "line", "poly"):
                    finish_creation()

            elif event.key == pygame.K_c and not shift:
                reset_creation(); creating_mode = "circle"; create_anchor = None
            elif event.key == pygame.K_r and not shift:
                reset_creation(); creating_mode = "rect"; create_anchor = None
            elif event.key == pygame.K_l and not shift:
                reset_creation(); creating_mode = "line"; create_anchor = None
            elif event.key == pygame.K_p and not shift:
                reset_creation(); creating_mode = "poly"; temp_poly_points = []

            elif event.key in (pygame.K_COMMA, pygame.K_PERIOD):
                # Shift modifies by 10, otherwise by 1
                step = 10 if shift else 1
                if event.key == pygame.K_COMMA:
                    GRID_SPACING = max(4, GRID_SPACING - step)  # clamp to sane minimum
                else:  # PERIOD
                    GRID_SPACING = min(2000, GRID_SPACING + step)  # arbitrary high cap


            elif event.key == pygame.K_e and not shift:
                export_all_to_console()

            elif event.key == pygame.K_z and not shift:
                undo()

            elif event.key == pygame.K_TAB:
                if selected:
                    push_undo()
                    for shp in selected:
                        shp.toggle_fill()

            elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                push_undo()
                targets = selected if selected else shapes
                for shp in targets:
                    shp.scale(SCALE_STEP)
            elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                push_undo()
                targets = selected if selected else shapes
                for shp in targets:
                    shp.scale(1.0 / SCALE_STEP)

            elif event.key == pygame.K_g:
                push_undo()
                targets = selected if selected else shapes
                if shift:
                    for shp in targets:
                        snap_one_center(shp)
                else:
                    for shp in targets:
                        snap_one_vertices(shp)

            elif shift and event.key == pygame.K_c:
                targets = selected if selected else shapes
                _ = [s.center() for s in targets]
                print(f"Recalculated centers for {len(targets)} shape(s).")

            elif event.key == pygame.K_i:
                if selected:
                    push_undo()
                    for shp in list(selected):
                        new_shape = rotate_one(shp, ROTATE_STEP_DEG)
                        replace_shape(shp, new_shape)
            elif event.key == pygame.K_k:
                if selected:
                    push_undo()
                    for shp in list(selected):
                        new_shape = rotate_one(shp, -ROTATE_STEP_DEG)
                        replace_shape(shp, new_shape)

            elif event.key in PALETTE.keys() and not shift:
                if selected:
                    push_undo()
                    for shp in selected:
                        set_color_one(shp, PALETTE[event.key])

            elif event.key == pygame.K_m and not shift:
                show_help = not show_help

            elif event.key == pygame.K_a and not shift:
                selected = list(shapes)

            elif event.key == pygame.K_x and not shift:
                show_exit_confirm = True

            elif event.key == pygame.K_n and not shift:
                push_undo()
                if selected:
                    for shp in list(selected):
                        if shp in shapes:
                            shapes.remove(shp)
                    selected.clear()
                else:
                    shapes.clear()

            elif shift and event.key == pygame.K_d:
                if selected:
                    copied_shapes = copy.deepcopy(selected)
                    print(f"Taken {len(copied_shapes)} shape(s).")
            elif shift and event.key == pygame.K_w:
                if copied_shapes:
                    push_undo()
                    pasted = copy.deepcopy(copied_shapes)
                    for shp in pasted:
                        shp.move(PASTE_OFFSET, PASTE_OFFSET)
                        shapes.append(shp)
                    selected = pasted
                    print(f"Placed {len(pasted)} shape(s).")

            elif event.key == pygame.K_t and not shift:
                if any(eligible_for_thickness(s) for s in selected):
                    thickness_mode = not thickness_mode
                    thickness_drag_shape = None

        if event.type == pygame.MOUSEBUTTONDOWN:
            if demo_mode:
                continue
            mx, my = event.pos


            if show_exit_confirm:
                sw, sh = screen.get_size()
                panel_w, panel_h = 360, 160
                px = (sw - panel_w)//2; py = (sh - panel_h)//2
                btn_w, btn_h = 120, 36
                btn_gap = 20
                exit_rect = pygame.Rect(px + panel_w//2 - btn_gap//2 - btn_w, py + panel_h - btn_h - 16, btn_w, btn_h)
                cancel_rect = pygame.Rect(px + panel_w//2 + btn_gap//2, py + panel_h - btn_h - 16, btn_w, btn_h)
                if exit_rect.collidepoint((mx, my)) and event.button == 1:
                    running = False
                elif cancel_rect.collidepoint((mx, my)) and event.button == 1:
                    show_exit_confirm = False
                continue

            if event.button == 3:
                selected.clear()
                thickness_mode = False
                thickness_drag_shape = None
                continue

            if event.button == 1:
                if thickness_mode and selected:
                    shp = thickness_pick_handle((mx, my))
                    if shp is not None:
                        thickness_drag_shape = shp
                        continue
                if creating_mode in ("rect", "circle", "line"):
                    if create_anchor is None:
                        create_anchor = (mx, my)
                    else:
                        finish_creation()
                elif creating_mode == "poly":
                    if temp_poly_points and dist((mx, my), temp_poly_points[0]) <= 12 and len(temp_poly_points) >= 3:
                        temp_poly_points.append(temp_poly_points[0])
                        finish_creation()
                    else:
                        temp_poly_points.append((mx, my))
                else:
                    candidate = select_at((mx, my))
                    if selected:
                        if candidate and candidate not in selected:
                            bring_to_top(candidate)
                            selected.append(candidate)
                            dragging = True
                            drag_offset = (mx, my)
                        else:
                            if len(selected) == 1:
                                only = selected[0]
                                if isinstance(only, Polygon):
                                    idx = only.nearest_vertex((mx, my))
                                    if idx is not None:
                                        active_vertex_index = idx
                                        dragging = True
                                    else:
                                        dragging = True
                                        drag_offset = (mx, my)
                                elif isinstance(only, Line):
                                    handle = only.near_handle((mx, my))
                                    if handle:
                                        only.drag_mode = handle
                                        dragging = True
                                    else:
                                        only.drag_mode = "body"
                                        dragging = True
                                        drag_offset = (mx, my)
                                elif isinstance(only, Circle):
                                    if only.near_radius_handle((mx, my)):
                                        circle_resizing = True
                                        dragging = True
                                    else:
                                        circle_resizing = False
                                        dragging = True
                                        drag_offset = (mx, my)
                                else:
                                    dragging = True
                                    drag_offset = (mx, my)
                            else:
                                dragging = True
                                drag_offset = (mx, my)
                    else:
                        if candidate:
                            bring_to_top(candidate)
                            selected = [candidate]
                            if isinstance(candidate, Polygon):
                                idx = candidate.nearest_vertex((mx, my))
                                if idx is not None:
                                    active_vertex_index = idx
                                    dragging = True
                                else:
                                    dragging = True
                                    drag_offset = (mx, my)
                            elif isinstance(candidate, Line):
                                handle = candidate.near_handle((mx, my))
                                if handle:
                                    candidate.drag_mode = handle
                                    dragging = True
                                else:
                                    candidate.drag_mode = "body"
                                    dragging = True
                                    drag_offset = (mx, my)
                            elif isinstance(candidate, Circle):
                                if candidate.near_radius_handle((mx, my)):
                                    circle_resizing = True
                                    dragging = True
                                else:
                                    circle_resizing = False
                                    dragging = True
                                    drag_offset = (mx, my)
                            else:
                                dragging = True
                                drag_offset = (mx, my)
                        else:
                            dragging = False

        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                thickness_drag_shape = None
                dragging = False
                active_vertex_index = None
                circle_resizing = False
                if len(selected) == 1 and isinstance(selected[0], Line):
                    selected[0].drag_mode = None

        if event.type == pygame.MOUSEMOTION:
            mx, my = event.pos
            if thickness_drag_shape is not None:
                thickness_update_from_mouse(thickness_drag_shape, (mx, my))
            if dragging and selected:
                if len(selected) == 1:
                    only = selected[0]
                    if isinstance(only, Polygon):
                        if active_vertex_index is not None:
                            only.points[active_vertex_index] = (mx, my)
                        else:
                            dx = mx - drag_offset[0]
                            dy = my - drag_offset[1]
                            only.move(dx, dy)
                            drag_offset = (mx, my)
                    elif isinstance(only, Line):
                        if only.drag_mode == "a":
                            only.a = (mx, my)
                        elif only.drag_mode == "b":
                            only.b = (mx, my)
                        else:
                            dx = mx - drag_offset[0]
                            dy = my - drag_offset[1]
                            only.move(dx, dy)
                            drag_offset = (mx, my)
                    elif isinstance(only, Circle):
                        if circle_resizing:
                            only.radius = max(1, dist(only.center(), (mx, my)))
                        else:
                            dx = mx - drag_offset[0]
                            dy = my - drag_offset[1]
                            only.move(dx, dy)
                            drag_offset = (mx, my)
                    else:
                        dx = mx - drag_offset[0]
                        dy = my - drag_offset[1]
                        only.move(dx, dy)
                        drag_offset = (mx, my)
                else:
                    dx = mx - drag_offset[0]
                    dy = my - drag_offset[1]
                    for shp in selected:
                        shp.move(dx, dy)
                    drag_offset = (mx, my)


    screen.fill(BG_COLOR)

    if not demo_mode:
        draw_grid_centered(screen, grid_spacing)
        draw_axes_center(screen)

    for shp in shapes:
        shp.draw(screen, (shp in selected) if not demo_mode else False)

    if not demo_mode:
        if creating_mode == "rect" and create_anchor is not None:
            ...
        elif creating_mode == "circle" and create_anchor is not None:
            ...
        elif creating_mode == "line" and create_anchor is not None:
            ...
        elif creating_mode == "poly":
            ...

    if not demo_mode and thickness_mode and selected:
        draw_thickness_handles(screen)

    if not demo_mode and show_help:
        ...

    if creating_mode == "rect" and create_anchor is not None:
        ax, ay = create_anchor
        mx, my = pygame.mouse.get_pos()
        pygame.draw.rect(screen, ACCENT, pygame.Rect(min(ax, mx), min(ay, my), abs(mx-ax), abs(my-ay)), 1)
    elif creating_mode == "circle" and create_anchor is not None:
        ax, ay = create_anchor
        mx, my = pygame.mouse.get_pos()
        r = max(1, int(dist((ax, ay), (mx, my))))
        pygame.draw.circle(screen, ACCENT, (int(ax), int(ay)), r, 1)
    elif creating_mode == "line" and create_anchor is not None:
        ax, ay = create_anchor
        mx, my = pygame.mouse.get_pos()
        pygame.draw.line(screen, ACCENT, (ax, ay), (mx, my), 1)
    elif creating_mode == "poly":
        mouse_pt = pygame.mouse.get_pos()
        pts = temp_poly_points + [mouse_pt]
        if len(pts) >= 2:
            pygame.draw.lines(screen, ACCENT, False, pts, 1)
        for p in temp_poly_points:
            pygame.draw.rect(screen, ACCENT, (*[int(p[0]-HANDLE_SIZE/2), int(p[1]-HANDLE_SIZE/2)], HANDLE_SIZE, HANDLE_SIZE), 1)

    if thickness_mode and selected:
        draw_thickness_handles(screen)

    if show_help:
        lines = [
            "Visualizer Controls",
            "Create: [R]ectangle  [C]ircle  [L]ine  [P]olygon  |  [Enter] finish  [Esc] cancel",
            "Select: [Left-click] add  [Right-click] clear  [A] Select All",
            "Move/Transform: [Drag vertex] vertex edit  [drag side] move object ",
            "Operations: +/- scale  I/K rotate  T border | [G] vertex snap  [Shift+G] Weight snap   [MMB] Demo",
            "Style: [Tab] fill  [1-9] color  | [E] export  [Z] undo  [N] nullify sel/all  [X] quit",
            "Clipboard: [Shift+d] Duplicate  [Shift+w] Write (offset) ",
            f"Shapes: {len(shapes)}   Selected: {len(selected)}  Grid: {grid_spacing}px"
        ,
        ]
        padding = 10
        surf_lines = [font.render(t, True, FG_COLOR) for t in lines]
        width = max(s.get_width() for s in surf_lines) + padding * 2
        height = sum(s.get_height() for s in surf_lines) + padding * (len(surf_lines)+1)
        sw, sh = screen.get_size()
        x = (sw - width) // 2
        y = (sh - height) // 2
        panel = pygame.Surface((width, height), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 180))
        yy = padding
        for s in surf_lines:
            panel.blit(s, (padding, yy))
            yy += s.get_height() + padding
        screen.blit(panel, (x, y))

        res_text = RES

        res_surf = font.render(res_text, True, (230, 230, 235))
        pad = 6
        bg_rect = pygame.Rect(0, 0, res_surf.get_width() + pad * 2, res_surf.get_height() + pad * 2)
        bg_rect.bottomright = (sw - 8, sh - 8)
        hud_panel = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
        hud_panel.fill((0, 0, 0, 140))
        hud_panel.blit(res_surf, (pad, pad))
        screen.blit(hud_panel, bg_rect.topleft)

    if show_exit_confirm:
        if show_exit_confirm and not demo_mode:
            sw, sh = screen.get_size()
            panel_w, panel_h = 360, 160
            px = (sw - panel_w) // 2;
            py = (sh - panel_h) // 2
            panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
            panel.fill((0, 0, 0, 220))
            title = font.render("Exit Visualizer?", True, FG_COLOR)
            msg = font.render("Unsaved drawings will be lost.", True, (200, 200, 200))
            panel.blit(title, (20, 20))
            panel.blit(msg, (20, 50))
            btn_w, btn_h = 120, 36
            btn_gap = 20
            screen.blit(panel, (px, py))
            exit_rect = pygame.Rect(px + panel_w // 2 - btn_gap // 2 - btn_w, py + panel_h - btn_h - 16, btn_w, btn_h)
            cancel_rect = pygame.Rect(px + panel_w // 2 + btn_gap // 2, py + panel_h - btn_h - 16, btn_w, btn_h)
            pygame.draw.rect(screen, (180, 60, 60), exit_rect, border_radius=6)
            pygame.draw.rect(screen, (60, 160, 80), cancel_rect, border_radius=6)
            exit_txt = font.render("Exit", True, (0, 0, 0))
            cancel_txt = font.render("Cancel", True, (0, 0, 0))
            screen.blit(exit_txt, (exit_rect.x + (exit_rect.w - exit_txt.get_width()) // 2,
                                   exit_rect.y + (exit_rect.h - exit_txt.get_height()) // 2))
            screen.blit(cancel_txt, (cancel_rect.x + (cancel_rect.w - cancel_txt.get_width()) // 2,
                                     cancel_rect.y + (cancel_rect.h - cancel_txt.get_height()) // 2))



    pygame.display.flip()

pygame.quit()
sys.exit()
