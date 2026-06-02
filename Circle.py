import sys
import math
import subprocess


class Circle:

    epsilon: float = sys.float_info.epsilon * 10000

    def __init__(self, a: float, b: float, c: float, d: float, **attr) -> None:
        n = max(abs(a), abs(b), abs(c), abs(d), self.epsilon)
        self.a, self.b, self.c, self.d = a / n, b / n, c / n, d / n
        if abs(self.a) < self.epsilon:
            a = self.epsilon / 2 if self.a > 0 else -self.epsilon / 2
        rr = (b ** 2 + c ** 2 - 4 * a * d) / (4 * a * a)
        self.r = math.sqrt(rr) if rr >= 0 else -math.sqrt(-rr)
        self.x = -b / (2 * a)
        self.y = -c / (2 * a)
        self.attr = attr

    def center(self) -> tuple[float]:
        return self.x, self.y

    def inverse(self):
        return Circle(-self.a, -self.b, -self.c, -self.d)

    def is_bounded(self) -> bool:
        return self.a > self.epsilon

    def is_line(self) -> bool:
        return abs(self.a) <= self.epsilon

    def get_attr(self) -> dict:
        return self.attr
    
    def set_attr(self, **attr) -> None:
        self.attr.update(attr)

    # def contains_pt(self, x: float, y: float) -> bool:
    #     if self.a * x * (x + self.b) + self.a * y * (y + self.c) + self.d < self.epsilon:
    #         return True
    #     return False

    def contains_circ(self, c) -> bool:
        x, y = c.center()
        r = c.r
        if self.is_line():
            if c.is_line():
                u = self.b * c.c - self.c * c.b
                v = self.b * c.b + self.c * c.c
                if abs(u) > self.epsilon or v < 0:
                    return False
                if self.d / math.sqrt(self.b ** 2 + self.c **2) > c.d / math.sqrt(c.b ** 2 + c.c ** 2):
                    return False
                return True
            elif c.is_bounded():
                u = self.b * x + self.c * y + self.d
                if u >= 0:
                    return False
                dist = abs(u) / math.sqrt(self.b ** 2 + self.c ** 2)
                if dist + self.epsilon < r:
                    return False
                return True
            else:
                return False
        else:
            if self.r <= 0:
                return False if self.is_bounded() else True
            if c.is_line():
                if self.is_bounded():
                    return False
                u = c.b * self.x + c.c * self.y + c.d
                if u <= 0:
                    return False
                dist = abs(u) / math.sqrt(c.b ** 2 + c.c ** 2)
                if dist + self.epsilon < self.r:
                    return False
                return True
            if self.is_bounded() and c.is_bounded():
                if r - self.r > self.epsilon:
                    return False
                if (self.x - x) ** 2 + (self.y - y) ** 2 - (self.r - r) ** 2 < self.epsilon:
                    return True
            elif not self.is_bounded() and c.is_bounded():
                if (self.x - x) ** 2 + (self.y - y) ** 2 - (self.r + r) ** 2 > -self.epsilon:
                    return True
            elif not self.is_bounded() and not c.is_bounded():
                if self.r - r > self.epsilon:
                    return False
                if (self.x - x) ** 2 + (self.y - y) ** 2 - (self.r - r) ** 2 < self.epsilon:
                    return True
            return False

    def tikz_out(self, ox, oy, scale, r_min, frame):
        lines = []
        color = self.attr["color"] if "color" in self.attr else "white"
        if self.is_bounded():
            x = (self.x - ox) * scale
            y = (self.y - oy) * scale
            r = self.r * scale
            if r < r_min:
                return lines
            if "disc" in self.attr:
                if "bcolor" in self.attr:
                    bcolor = self.attr["bcolor"]
                    lines.append(f"  \\filldraw[fill={color},draw={bcolor}] ({x:.16f}pt, {y:.16f}pt) circle ({r:.16f}pt);\n")
                else:
                    lines.append(f"  \\fill[fill={color}] ({x:.16f}pt, {y:.16f}pt) circle ({r:.16f}pt);\n")
                #lines.append(f"  \\draw ({x}pt, {y}pt) node[scale = 0.1]" + "{*};\n")
            else:
                lines.append(f"  \\draw[draw={color}] ({x:.16f}pt, {y:.16f}pt) circle ({r:.16f}pt);\n")
        elif self.is_line():
            d = scale * (self.d + self.b * ox + self.c * oy)
            n = math.sqrt(self.b ** 2 + self.c ** 2)
            nx, ny = -self.b / n, -self.c / n
            x, y = -self.b * d / (n ** 2), -self.c * d / (n ** 2)
            w = max(math.sqrt(frame[0] ** 2 + frame[1] ** 2), math.sqrt(frame[2] ** 2 + frame[3] ** 2))
            if "disc" in self.attr:
                lines.append(f"  \\fill[fill={color}] ({x - ny * w:.16f}pt, {y + nx * w:.16f}pt)" + 
                             f" -- ({x + ny * w:.16f}pt, {y - nx * w:.16f}pt) -- ({x + (ny + 2 * nx) * w:.16f}pt, {y - (ny - 2 * ny) * w:.16f}pt)" +
                             f" -- ({x - (ny - 2 * nx) * w:.16f}pt, {y + (nx + 2 * ny) * w:.16f}pt) -- cycle;\n")
                if "bcolor" in self.attr:
                    bcolor = self.attr["bcolor"]
                    lines.append(f"  \\draw[draw={bcolor}] ({x - ny * w:.16f}pt, {y + nx * w:.16f}pt) -- ({x + ny * w:.16f}pt, {y - nx * w:.16f}pt);\n")
            else:
                lines.append(f"  \\draw[draw={color}] ({x - ny * w:.16f}pt, {y + nx * w:.16f}pt) -- ({x + ny * w:.16f}pt, {y - nx * w:.16f}pt);\n")
        else:
            x = (self.x - ox) * scale
            y = (self.y - oy) * scale
            r = self.r * scale
            if "disc" in self.attr:
                lines.append("  \\begin{scope}\n")
                lines.append(f"    \\clip ({frame[0]}pt,{frame[1]}pt) rectangle ({frame[2]}pt,{frame[3]}pt)\n    ({x:.16f}pt, {y:.16f}pt) circle ({r:.16f}pt);\n")
                lines.append(f"    \\fill[fill={color}] ({frame[0]}pt,{frame[1]}pt) rectangle ({frame[2]}pt,{frame[3]}pt);\n")
                lines.append("  \\end{scope}\n")
                if "bcolor" in self.attr:
                    bcolor = self.attr["bcolor"]
                    lines.append(f"  \\draw[draw={bcolor}] ({x:.16f}pt, {y:.16f}pt) circle ({r:.16f}pt);\n")
            else:
                lines.append(f"  \\draw[draw={color}] ({x:.16f}pt, {y:.16f}pt) circle ({r:.16f}pt);\n")
        return lines


class CircleArrangement:

    def reset(self) -> None:
        self.circles = []
        self.colors = {}
    
    def __init__(self) -> None:
        self.reset()

    def size(self) -> int:
        return len(self.circles)
        
    def add_circle(self, c: Circle, depth: float) -> bool:
        if any(a.contains_circ(c) for a, d in self.circles if d >= depth and 'disc' in a.attr):
            return False
        if 'disc' in c.attr:
            self.circles = [[a, d] for a, d in self.circles if d >= depth or not c.contains_circ(a)]
        self.circles.append([c, depth])
        return True

    def add_colors(self, entries):
        self.colors.update(entries)

    def find_frame(self):
        holes = [c for c, d in self.circles if not c.is_bounded() and not c.is_line() and 'disc' in c.attr]
        return holes[0] if holes else Circle(1, 0, 0, -1)

    def tikz_out(self, frame=None, width=300, height=300, r_min=0, background=None) -> list:
        self.circles.sort(key=lambda x: x[1])
        lines = ["\\begin{scope}\n"]
        for c in self.colors:
            lines.append("  \\definecolor{" + c + "}{rgb}{" + self.colors[c] + "}\n")
        if frame == None:
            frame = self.find_frame()
        w, h = width // 2, height // 2
        ox, oy = frame.center()
        scale = max(w, h) / frame.r
        #lines.append(f"  \\clip (0,0) circle ({width}pt);\n")
        lines.append(f"  \\clip (-{w}pt,-{h}pt) rectangle ({w}pt,{h}pt);\n")
        if background:
            lines.append(f"  \\fill[fill={background}] (-{w}pt,-{h}pt) rectangle ({w}pt,{h}pt);\n")
        for c, d in sorted(self.circles, key=lambda x: x[1]):
            lines.extend(c.tikz_out(ox, oy, scale, r_min, [-w, -h, w, h]))
        lines.append("\\end{scope}\n")
        return lines


class PDFPicture:

    def __init__(self) -> None:
        self.tikz_lines = []

    def append(self, lines: list) -> None:
        self.tikz_lines.extend(lines)

    def print(self, fname: str) -> None:
        lines = []
        lines.append("\\documentclass[tikz]{standalone}\n")
        lines.append("\\begin{document}\n")
        lines.append("\\begin{tikzpicture}\n")
        lines.extend(self.tikz_lines)
        lines.append("\\end{tikzpicture}\n")
        lines.append("\\end{document}\n")
        with open(fname, "w", encoding="ASCII") as file_out:
            file_out.writelines(lines)
        subprocess.run(["pdflatex", "\\nonstopmode\\input", fname], stdout=subprocess.DEVNULL)
