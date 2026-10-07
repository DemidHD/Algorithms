"""
Лабораторная работа: Алгоритм Тарьяна.

Поиск компонент сильной связности (КСС) в ориентированном графе
за один обход в глубину.
Сложность: O(V + E).
"""

import os
import sys
import textwrap

sys.setrecursionlimit(10 ** 6)


def tarjan(graph):
    """
    graph — словарь смежности: {вершина: [соседи, ...]}
    Возвращает список компонент сильной связности (каждая — список вершин).
    """
    index_counter = [0]
    index = {}      # порядковый номер вершины при обходе в глубину
    low = {}        # минимальный индекс, достижимый из поддерева вершины
    stack = []
    on_stack = set()
    components = []

    def strongconnect(v):
        index[v] = low[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        for w in graph[v]:
            if w not in index:
                # w ещё не посещена — идём в неё
                strongconnect(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                # w в стеке — значит, она в текущей КСС
                low[v] = min(low[v], index[w])

        # v — корень компоненты сильной связности
        if low[v] == index[v]:
            component = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                component.append(w)
                if w == v:
                    break
            components.append(component)

    for v in graph:
        if v not in index:
            strongconnect(v)

    return components


def tarjan_steps(graph):
    """
    Тот же алгоритм Тарьяна, но с записью каждого шага
    (снимка состояния) для пошаговой анимации.
    """
    counter = [0]
    index, low = {}, {}
    stack, on_stack = [], set()
    components = []
    path = []       # текущая цепочка рекурсивных вызовов
    steps = []

    def snap(msg, current=None, edge=None):
        steps.append({
            "index": dict(index), "low": dict(low), "stack": list(stack),
            "components": [list(c) for c in components], "path": list(path),
            "current": current, "edge": edge, "msg": msg,
        })

    def strongconnect(v):
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        path.append(v)
        snap(f"Заходим в {v}: index[{v}] = low[{v}] = {index[v]}, кладём {v} в стек.", v)

        for w in graph[v]:
            if w not in index:
                snap(f"Ребро {v}→{w}: {w} ещё не посещена — рекурсивно переходим в неё.",
                     v, (v, w))
                strongconnect(w)
                old = low[v]
                low[v] = min(low[v], low[w])
                snap(f"Возврат из {w} в {v}: low[{v}] = min({old}, low[{w}] = {low[w]}) "
                     f"= {low[v]}.", v, (v, w))
            elif w in on_stack:
                old = low[v]
                low[v] = min(low[v], index[w])
                snap(f"Ребро {v}→{w}: {w} в стеке (та же компонента) — "
                     f"low[{v}] = min({old}, index[{w}] = {index[w]}) = {low[v]}.", v, (v, w))
            else:
                snap(f"Ребро {v}→{w}: {w} уже в найденной КСС — ребро игнорируем.", v, (v, w))

        if low[v] == index[v]:
            component = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                component.append(w)
                if w == v:
                    break
            components.append(component)
            snap(f"low[{v}] = index[{v}] = {index[v]} — {v} корень КСС. Снимаем со стека "
                 f"вершины до {v}: {', '.join(map(str, component))}.", v)
        else:
            snap(f"low[{v}] = {low[v]} < index[{v}] = {index[v]} — {v} не корень, "
                 f"остаётся в стеке.", v)
        path.pop()

    snap("Начало: все вершины не посещены, стек пуст.")
    for v in graph:
        if v not in index:
            snap(f"Запускаем обход в глубину из непосещённой вершины {v}.", v)
            strongconnect(v)
    snap(f"Обход завершён. Найдено компонент сильной связности: {len(components)}.")
    return steps


# --------------------------- Визуализация ----------------------------

def visualize(graph, components, save_path=None):
    """
    Рисует граф: вершины одной компоненты сильной связности окрашены
    в один цвет, рёбра внутри компоненты — цветом компоненты,
    рёбра между компонентами — серым пунктиром.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        import networkx as nx
    except ImportError:
        print("Для визуализации установите библиотеки: pip install matplotlib networkx")
        return

    G = nx.DiGraph()
    G.add_nodes_from(graph)
    for u in graph:
        for v in graph[u]:
            G.add_edge(u, v)

    pos = nx.spring_layout(G, seed=42, k=1.5 / max(len(G), 1) ** 0.5)

    cmap = plt.get_cmap("tab10" if len(components) <= 10 else "tab20")
    comp_of = {}
    for i, comp in enumerate(components):
        for v in comp:
            comp_of[v] = i
    color_of = {v: cmap(comp_of[v] % cmap.N) for v in G.nodes()}

    inner = [(u, v) for u, v in G.edges() if comp_of[u] == comp_of[v]]
    outer = [(u, v) for u, v in G.edges() if comp_of[u] != comp_of[v]]

    fig, ax = plt.subplots(figsize=(10, 7))
    curved = {"connectionstyle": "arc3,rad=0.12"}

    nx.draw_networkx_nodes(G, pos, node_color=[color_of[v] for v in G.nodes()],
                           node_size=1000, edgecolors="black", ax=ax)
    nx.draw_networkx_labels(G, pos, font_weight="bold", font_color="white", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=inner, edge_color=[color_of[u] for u, _ in inner],
                           width=3, arrowsize=20, node_size=1000, ax=ax, **curved)
    nx.draw_networkx_edges(G, pos, edgelist=outer, edge_color="gray", style="dashed",
                           width=1.5, arrowsize=18, node_size=1000, ax=ax, **curved)

    ax.legend(handles=[
        Patch(facecolor=cmap(i % cmap.N), edgecolor="black",
              label=f"КСС {i + 1}: {', '.join(map(str, sorted(comp, key=str)))}")
        for i, comp in enumerate(components)
    ], loc="best")
    ax.set_title(f"Алгоритм Тарьяна: компонент сильной связности — {len(components)}",
                 fontsize=14)
    ax.axis("off")
    ax.margins(0.1)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
    else:
        plt.show()


# ------------------------ Пошаговая анимация -------------------------

def build_animation(graph):
    """Возвращает (steps, draw_step) — список шагов и функцию отрисовки шага."""
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    steps = tarjan_steps(graph)

    G = nx.DiGraph()
    G.add_nodes_from(graph)
    for u in graph:
        for v in graph[u]:
            G.add_edge(u, v)
    pos = nx.spring_layout(G, seed=42, k=1.5 / max(len(G), 1) ** 0.5)
    edges = list(G.edges())
    n_comp = len(steps[-1]["components"])
    cmap = plt.get_cmap("tab10" if n_comp <= 10 else "tab20")
    curved = {"connectionstyle": "arc3,rad=0.12"}

    def draw_step(ax, s):
        comp_of = {v: i for i, comp in enumerate(s["components"]) for v in comp}
        on_stack = set(s["stack"])
        path_edges = set(zip(s["path"], s["path"][1:]))

        def node_color(v):
            if v == s["current"]:
                return "orange"
            if v in comp_of:
                return cmap(comp_of[v] % cmap.N)
            if v in on_stack:
                return "khaki"
            return "white"

        edge_colors, widths = [], []
        for u, v in edges:
            if (u, v) == s["edge"]:
                edge_colors.append("black")
                widths.append(4.5)
            elif u in comp_of and v in comp_of and comp_of[u] == comp_of[v]:
                edge_colors.append(cmap(comp_of[u] % cmap.N))
                widths.append(3)
            elif (u, v) in path_edges:
                edge_colors.append("royalblue")
                widths.append(3)
            else:
                edge_colors.append("lightgray")
                widths.append(1.5)

        nx.draw_networkx_nodes(G, pos, node_color=[node_color(v) for v in G.nodes()],
                               node_size=1000, edgecolors="black", ax=ax)
        dark = [v for v in G.nodes() if v in comp_of and v != s["current"]]
        light = [v for v in G.nodes() if v not in dark]
        nx.draw_networkx_labels(G, pos, labels={v: v for v in dark}, font_weight="bold",
                                font_color="white", ax=ax)
        nx.draw_networkx_labels(G, pos, labels={v: v for v in light}, font_weight="bold", ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors, width=widths,
                               arrowsize=20, node_size=1000, ax=ax, **curved)
        for v, (x, y) in pos.items():
            if v in s["index"]:
                ax.annotate(f"{s['index'][v]} / {s['low'][v]}", (x, y),
                            textcoords="offset points", xytext=(0, -30), ha="center",
                            fontsize=10, fontweight="bold", color="darkblue",
                            annotation_clip=False)

        comps = "  ".join("{" + ", ".join(map(str, c)) + "}" for c in s["components"]) or "—"
        info = (f"Стек:         [{', '.join(map(str, s['stack']))}]\n"
                f"Путь обхода:  {' → '.join(map(str, s['path'])) or '—'}\n"
                f"Найденные КСС: {comps}")
        ax.text(0.01, 0.01, info, transform=ax.transAxes, va="bottom", family="monospace",
                fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
        ax.legend(handles=[
            Patch(facecolor="orange", edgecolor="black", label="текущая вершина"),
            Patch(facecolor="khaki", edgecolor="black", label="в стеке"),
            Patch(facecolor="white", edgecolor="black", label="не посещена"),
            Patch(facecolor=cmap(0), edgecolor="black", label="в найденной КСС"),
            Line2D([0], [0], color="black", lw=4, label="текущее ребро"),
            Line2D([0], [0], color="royalblue", lw=3, label="путь рекурсии"),
            Line2D([], [], ls="", label="под вершиной: index / low"),
        ], loc="upper right", fontsize=9)
        ax.set_title("Алгоритм Тарьяна\n" + textwrap.fill(s["msg"], 110), fontsize=12)
        ax.axis("off")
        ax.margins(0.12)

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
        self.ax = self.fig.add_axes([0.02, 0.10, 0.96, 0.82])
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
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.88])
    anim = FuncAnimation(fig, lambda i: render_step(ax, steps, i, draw_step),
                         frames=len(steps))
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def animate(graph):
    StepPlayer(*build_animation(graph)).show()


def save_animation(graph, path="tarjan.gif"):
    steps, draw_step = build_animation(graph)
    print(f"Сохраняю {len(steps)} кадров...")
    save_gif(steps, draw_step, path)
    print(f"Анимация сохранена: {os.path.abspath(path)}")


# ---------------------------- Ввод данных ----------------------------

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


def read_graph():
    """Считывает ориентированный граф с клавиатуры."""
    graph = {}
    vertices = input("Вершины через пробел (можно оставить пустым — "
                     "возьмутся из рёбер): ").split()
    for v in vertices:
        graph[v] = []

    m = read_int("Количество рёбер: ", min_value=0)
    print("Введите рёбра в формате: <откуда> <куда>  (например: 1 2)")
    i = 0
    while i < m:
        parts = input(f"  ребро {i + 1}: ").split()
        if len(parts) != 2:
            print("  Ошибка: нужно ровно 2 значения.")
            continue
        u, v = parts
        graph.setdefault(u, []).append(v)
        graph.setdefault(v, [])
        i += 1

    return graph


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
            print("Для визуализации установите библиотеки: pip install -r requirements.txt")


def example_graph():
    return {
        1: [2],
        2: [3],
        3: [1, 4],
        4: [5],
        5: [6],
        6: [4, 7],
        7: [8],
        8: [7],
    }


if __name__ == "__main__":
    print("=== Алгоритм Тарьяна (компоненты сильной связности) ===")
    if ask_yes_no("Использовать встроенный пример? (y/n): "):
        graph = example_graph()
    else:
        graph = read_graph()

    if not graph:
        print("Граф пуст.")
        sys.exit()

    components = tarjan(graph)

    print(f"\nНайдено компонент сильной связности: {len(components)}")
    for i, comp in enumerate(components, 1):
        print(f"  КСС {i}: {sorted(comp)}")

    run_menu({
        "1": lambda: visualize(graph, components),
        "2": lambda: animate(graph),
        "3": lambda: save_animation(graph),
    })
