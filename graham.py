"""
Лабораторная работа: Алгоритм Грэхема.

Построение выпуклой оболочки множества точек на плоскости.
Сложность: O(n log n) (из-за сортировки).
"""

import math
import os
import textwrap


def cross(o, a, b):
    """
    Векторное произведение (OA x OB).
    > 0 — левый поворот, < 0 — правый, = 0 — точки коллинеарны.
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def graham_scan(points):
    """
    points — список точек (x, y).
    Возвращает вершины выпуклой оболочки в порядке обхода против часовой стрелки.
    """
    points = list(set(points))
    if len(points) < 3:
        return points

    # 1. Опорная точка — самая нижняя (при равенстве — самая левая)
    pivot = min(points, key=lambda p: (p[1], p[0]))

    # 2. Сортировка остальных точек по полярному углу относительно опорной,
    #    при равном угле — по расстоянию до опорной
    others = [p for p in points if p != pivot]
    others.sort(key=lambda p: (math.atan2(p[1] - pivot[1], p[0] - pivot[0]),
                               dist2(pivot, p)))

    # 3. Обход: удаляем точки, образующие правый поворот (или коллинеарные)
    hull = [pivot]
    for p in others:
        while len(hull) >= 2 and cross(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)

    return hull


def graham_steps(points):
    """
    Тот же алгоритм Грэхема, но с записью каждого шага
    (снимка состояния) для пошаговой анимации.
    """
    points = list(set(points))
    steps = []

    def snap(msg, stack, order=None, current=None, test=None, ok=None, removed=None,
             done=False):
        steps.append({"stack": list(stack), "order": order, "current": current, "test": test,
                      "ok": ok, "removed": removed, "done": done, "msg": msg})

    if len(points) < 3:
        snap("Меньше трёх различных точек — оболочка состоит из самих точек.", points,
             done=True)
        return steps

    pivot = min(points, key=lambda p: (p[1], p[0]))
    snap(f"Шаг 1. Опорная точка — самая нижняя (при равенстве — самая левая): {pivot}. "
         f"Она точно лежит на оболочке.", [pivot])

    others = [p for p in points if p != pivot]
    others.sort(key=lambda p: (math.atan2(p[1] - pivot[1], p[0] - pivot[0]),
                               dist2(pivot, p)))
    order = [pivot] + others
    snap("Шаг 2. Сортируем остальные точки по полярному углу относительно опорной "
         "(при равном угле — по расстоянию). Синие номера — порядок обхода.", [pivot], order)

    hull = [pivot]
    for p in others:
        snap(f"Рассматриваем точку {p}.", hull, order, current=p)
        while len(hull) >= 2 and cross(hull[-2], hull[-1], p) <= 0:
            kind = "правый поворот" if cross(hull[-2], hull[-1], p) < 0 else "точки на одной прямой"
            snap(f"{hull[-2]} → {hull[-1]} → {p}: {kind}. Точка {hull[-1]} не может быть "
                 f"вершиной оболочки — снимаем её со стека.", hull, order, current=p,
                 test=(hull[-2], hull[-1], p), ok=False, removed=hull[-1])
            hull.pop()
        if len(hull) >= 2:
            msg = f"{hull[-2]} → {hull[-1]} → {p}: левый поворот — кладём {p} в стек."
            test = (hull[-2], hull[-1], p)
        else:
            msg = f"В стеке одна точка — кладём {p} в стек."
            test = None
        hull.append(p)
        snap(msg, hull, order, current=p, test=test, ok=True)

    snap(f"Все точки обработаны. В стеке остались вершины выпуклой оболочки "
         f"({len(hull)} шт.) в порядке обхода против часовой стрелки.", hull, order, done=True)
    return steps


# --------------------------- Визуализация ----------------------------

def visualize(points, hull, save_path=None):
    """
    Рисует точки и выпуклую оболочку. Пунктиром показаны лучи от опорной
    точки, по углу которых сортируются точки; вершины оболочки
    пронумерованы в порядке обхода.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
    except ImportError:
        print("Для визуализации установите библиотеку: pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    pivot = hull[0]

    # лучи от опорной точки (порядок сортировки по полярному углу)
    for p in set(points):
        if p != pivot:
            ax.plot([pivot[0], p[0]], [pivot[1], p[1]], ls=":", color="lightgray", zorder=1)

    if len(hull) >= 3:
        ax.add_patch(Polygon(hull, closed=True, facecolor=(0.53, 0.81, 0.92, 0.35),
                             edgecolor="crimson", lw=2.5, zorder=2, label="выпуклая оболочка"))
    elif len(hull) == 2:
        ax.plot(*zip(*hull), color="crimson", lw=2.5, zorder=2, label="выпуклая оболочка")

    inner = [p for p in set(points) if p not in hull]
    if inner:
        ax.scatter(*zip(*inner), s=60, color="gray", zorder=3, label="остальные точки")
    ax.scatter(*zip(*hull), s=90, color="crimson", edgecolors="black", zorder=4,
               label="вершины оболочки")
    ax.scatter([pivot[0]], [pivot[1]], s=250, marker="*", color="gold", edgecolors="black",
               zorder=5, label="опорная точка")

    for i, p in enumerate(hull, 1):
        ax.annotate(f"{i}", p, textcoords="offset points", xytext=(8, 8),
                    fontsize=12, fontweight="bold", color="crimson")

    ax.set_title(f"Алгоритм Грэхема: вершин оболочки — {len(hull)} из {len(set(points))} точек",
                 fontsize=13)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.margins(0.12)
    ax.legend(loc="best")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
    else:
        plt.show()


# ------------------------ Пошаговая анимация -------------------------

def build_animation(points):
    """Возвращает (steps, draw_step) — список шагов и функцию отрисовки шага."""
    from matplotlib.patches import Polygon
    from matplotlib.lines import Line2D

    steps = graham_steps(points)
    unique = list(set(points))
    xs, ys = [p[0] for p in unique], [p[1] for p in unique]
    pad_x = (max(xs) - min(xs)) * 0.12 or 1
    pad_y = (max(ys) - min(ys)) * 0.12 or 1
    pivot = min(unique, key=lambda p: (p[1], p[0]))

    def draw_step(ax, s):
        stack, order = s["stack"], s["order"]

        if order:
            for p in order[1:]:
                ax.plot([pivot[0], p[0]], [pivot[1], p[1]], ls=":", color="lightgray", zorder=1)
            for i, p in enumerate(order[1:], 1):
                ax.annotate(str(i), p, textcoords="offset points", xytext=(-14, -14),
                            fontsize=9, color="royalblue")

        ax.scatter(*zip(*unique), s=50, color="gray", zorder=3)

        if s["done"] and len(stack) >= 3:
            ax.add_patch(Polygon(stack, closed=True, facecolor=(0.53, 0.81, 0.92, 0.35),
                                 edgecolor="crimson", lw=2.5, zorder=2))
        elif len(stack) >= 2:
            ax.plot(*zip(*stack), color="crimson", lw=2.5, zorder=2)
        ax.scatter(*zip(*stack), s=90, color="crimson", edgecolors="black", zorder=4)

        if s["test"]:
            a, b, c = s["test"]
            color = "limegreen" if s["ok"] else "darkorange"
            ax.plot([a[0], b[0], c[0]], [a[1], b[1], c[1]], color=color, lw=3, ls="--", zorder=5)
        if s["removed"]:
            ax.scatter(*s["removed"], s=300, marker="x", color="darkorange", linewidths=3,
                       zorder=6)
        if s["current"]:
            ax.scatter(*s["current"], s=260, facecolors="none", edgecolors="darkorange",
                       linewidths=2.5, zorder=6)
        ax.scatter(*pivot, s=280, marker="*", color="gold", edgecolors="black", zorder=7)

        ax.text(0.01, 0.01, "Стек: " + " ".join(f"({p[0]}, {p[1]})" for p in stack),
                transform=ax.transAxes, va="bottom", family="monospace", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9), wrap=True)
        ax.legend(handles=[
            Line2D([], [], marker="*", ls="", ms=15, mfc="gold", mec="black", label="опорная точка"),
            Line2D([], [], marker="o", ls="", ms=12, mfc="none", mec="darkorange", mew=2.5,
                   label="текущая точка"),
            Line2D([0], [0], color="crimson", lw=2.5, marker="o", label="стек (кандидаты)"),
            Line2D([0], [0], color="limegreen", lw=3, ls="--", label="левый поворот"),
            Line2D([0], [0], color="darkorange", lw=3, ls="--", label="правый поворот"),
            Line2D([], [], marker="x", ls="", ms=12, mec="darkorange", mew=3,
                   label="удаляемая точка"),
        ], loc="upper left", fontsize=9)

        ax.set_title("Алгоритм Грэхема\n" + textwrap.fill(s["msg"], 100), fontsize=12)
        ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
        ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)

    return steps, draw_step


def render_step(ax, steps, i, draw_step):
    ax.clear()
    draw_step(ax, steps[i])
    ax.text(0.99, 0.01, f"Шаг {i + 1} / {len(steps)}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=11, color="dimgray")


class StepPlayer:
    """
    Окно пошагового просмотра.
    Управление: кнопки внизу окна или клавиши ← / → (шаг назад / вперёд),
    пробел (автовоспроизведение), Home / End (в начало / в конец).
    """

    def __init__(self, steps, draw_step, interval=1200):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button

        # освобождаем стрелки и Home от стандартных действий matplotlib
        for name in ("keymap.back", "keymap.forward", "keymap.home"):
            plt.rcParams[name] = [k for k in plt.rcParams[name]
                                  if k not in ("left", "right", "home")]

        self.plt = plt
        self.steps = steps
        self.draw_step = draw_step
        self.i = 0
        self.playing = False

        self.fig = plt.figure(figsize=(12, 8.5))
        self.ax = self.fig.add_axes([0.04, 0.12, 0.92, 0.79])
        self.buttons = []
        for label, x, handler in [("<< Начало", 0.19, self.first), ("< Назад", 0.32, self.prev),
                                  ("Авто", 0.45, self.toggle), ("Вперёд >", 0.58, self.next),
                                  ("Конец >>", 0.71, self.last)]:
            button = Button(self.fig.add_axes([x, 0.02, 0.11, 0.055]), label)
            button.on_clicked(lambda event, h=handler: h())
            self.buttons.append(button)
        self.play_button = self.buttons[2]

        self.timer = self.fig.canvas.new_timer(interval=interval)
        self.timer.add_callback(self._tick)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def go(self, i):
        self.i = max(0, min(i, len(self.steps) - 1))
        render_step(self.ax, self.steps, self.i, self.draw_step)
        self.fig.canvas.draw_idle()

    def first(self):
        self.go(0)

    def last(self):
        self.go(len(self.steps) - 1)

    def next(self):
        self.go(self.i + 1)

    def prev(self):
        self.go(self.i - 1)

    def toggle(self):
        if self.playing:
            self._stop()
            return
        if self.i == len(self.steps) - 1:
            self.go(0)
        self.playing = True
        self.play_button.label.set_text("Пауза")
        self.timer.start()

    def _stop(self):
        self.playing = False
        self.play_button.label.set_text("Авто")
        self.timer.stop()
        self.fig.canvas.draw_idle()

    def _tick(self):
        if self.i >= len(self.steps) - 1:
            self._stop()
        else:
            self.next()

    def _on_key(self, event):
        actions = {"right": self.next, "left": self.prev, " ": self.toggle,
                   "home": self.first, "end": self.last}
        if event.key in actions:
            actions[event.key]()

    def show(self):
        self.go(0)
        self.plt.show()


def save_gif(steps, draw_step, path, fps=1):
    """Сохраняет анимацию в GIF (кадр на каждый шаг)."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_axes([0.04, 0.04, 0.92, 0.84])
    anim = FuncAnimation(fig, lambda i: render_step(ax, steps, i, draw_step),
                         frames=len(steps))
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def animate(points):
    StepPlayer(*build_animation(points)).show()


def save_animation(points, path="graham.gif"):
    steps, draw_step = build_animation(points)
    print(f"Сохраняю {len(steps)} кадров...")
    save_gif(steps, draw_step, path)
    print(f"Анимация сохранена: {os.path.abspath(path)}")


# ---------------------------- Ввод данных ----------------------------

def parse_number(s):
    """Преобразует строку в int, если возможно, иначе в float."""
    try:
        return int(s)
    except ValueError:
        return float(s)


def read_int(prompt, min_value=None):
    while True:
        try:
            value = int(input(prompt))
        except ValueError:
            print("  Ошибка: введите целое число.")
            continue
        if min_value is not None and value < min_value:
            print(f"  Ошибка: число должно быть не меньше {min_value}.")
            continue
        return value


def ask_yes_no(prompt):
    while True:
        answer = input(prompt).strip().lower()
        if answer in ("y", "yes", "д", "да"):
            return True
        if answer in ("n", "no", "н", "нет"):
            return False
        print("  Ошибка: введите y (да) или n (нет).")


def read_points():
    """Считывает точки с клавиатуры."""
    n = read_int("Количество точек: ", min_value=1)
    print("Введите точки в формате: <x> <y>  (например: 3 1.5)")
    points = []
    while len(points) < n:
        parts = input(f"  точка {len(points) + 1}: ").replace(",", " ").split()
        if len(parts) != 2:
            print("  Ошибка: нужно ровно 2 координаты.")
            continue
        try:
            points.append((parse_number(parts[0]), parse_number(parts[1])))
        except ValueError:
            print("  Ошибка: координаты должны быть числами.")
    return points


def run_menu(actions):
    """Меню визуализации. actions — словарь {пункт: функция}."""
    while True:
        print("\nВизуализация:\n"
              "  1 — итоговый результат\n"
              "  2 — пошаговая анимация\n"
              "  3 — сохранить анимацию в GIF\n"
              "  0 — выход")
        choice = input("Выберите пункт: ").strip()
        if choice == "0":
            break
        if choice not in actions:
            print("  Ошибка: введите 0, 1, 2 или 3.")
            continue
        try:
            actions[choice]()
        except ImportError:
            print("Для визуализации установите библиотеку: pip install -r requirements.txt")


def example_points():
    return [
        (0, 3), (1, 1), (2, 2), (4, 4), (0, 0),
        (1, 2), (3, 1), (3, 3), (2, 0), (4, 0),
    ]


if __name__ == "__main__":
    print("=== Алгоритм Грэхема (выпуклая оболочка) ===")
    if ask_yes_no("Использовать встроенный пример? (y/n): "):
        points = example_points()
    else:
        points = read_points()

    hull = graham_scan(points)

    print(f"\nИсходные точки: {points}")
    print(f"Выпуклая оболочка (вершин: {len(hull)}, обход против часовой стрелки):")
    for p in hull:
        print(f"  {p}")

    run_menu({
        "1": lambda: visualize(points, hull),
        "2": lambda: animate(points),
        "3": lambda: save_animation(points),
    })
