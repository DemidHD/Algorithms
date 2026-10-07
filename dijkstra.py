"""
Лабораторная работа: Алгоритм Дейкстры.

Поиск кратчайших путей от одной вершины до всех остальных
во взвешенном графе с неотрицательными весами рёбер.
Сложность: O((V + E) log V) с использованием двоичной кучи.
"""

import heapq
import os
import textwrap

INF = float("inf")


def dijkstra(graph, start):
    """
    graph — словарь смежности: {вершина: [(сосед, вес), ...]}
    start — стартовая вершина.
    Возвращает (dist, prev):
        dist[v] — длина кратчайшего пути от start до v (inf, если недостижима),
        prev[v] — предыдущая вершина на кратчайшем пути.
    """
    dist = {v: float("inf") for v in graph}
    prev = {v: None for v in graph}
    dist[start] = 0

    heap = [(0, start)]
    visited = set()

    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)

        for v, w in graph[u]:
            if w < 0:
                raise ValueError("Алгоритм Дейкстры не работает с отрицательными весами")
            if d + w < dist[v]:
                dist[v] = d + w
                prev[v] = u
                heapq.heappush(heap, (dist[v], v))

    return dist, prev


def restore_path(prev, start, target):
    """Восстанавливает путь от start до target по массиву предков."""
    path = []
    v = target
    while v is not None:
        path.append(v)
        v = prev[v]
    path.reverse()
    return path if path[0] == start else []


def fmt(x):
    return "∞" if x == INF else str(x)


def dijkstra_steps(graph, start):
    """
    Тот же алгоритм Дейкстры, но с записью каждого шага
    (снимка состояния) для пошаговой анимации.
    """
    dist = {v: INF for v in graph}
    prev = {v: None for v in graph}
    dist[start] = 0
    heap = [(0, start)]
    visited = set()
    steps = []

    def snap(msg, current=None, edge=None, improved=None):
        steps.append({
            "dist": dict(dist), "prev": dict(prev), "visited": set(visited),
            "heap": sorted(heap), "current": current, "edge": edge,
            "improved": improved, "msg": msg,
        })

    snap(f"Инициализация: d[{start}] = 0, для остальных вершин d = ∞. "
         f"Кладём {start} в кучу.")

    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            snap(f"Из кучи извлечена запись ({d}, {u}), но {u} уже обработана — "
                 f"это устаревшая запись, пропускаем.", current=u)
            continue
        visited.add(u)
        snap(f"Извлекаем из кучи вершину {u} с минимальным d = {d}. "
             f"Это расстояние окончательное, {u} помечается обработанной.", current=u)

        for v, w in graph[u]:
            if v in visited:
                snap(f"Ребро {u}→{v} (вес {w}): {v} уже обработана — пропускаем.",
                     u, (u, v), False)
                continue
            new = d + w
            if new < dist[v]:
                old = dist[v]
                dist[v] = new
                prev[v] = u
                heapq.heappush(heap, (new, v))
                snap(f"Релаксация {u}→{v}: d[{u}] + {w} = {new} < {fmt(old)} — "
                     f"обновляем d[{v}] = {new}, prev[{v}] = {u}, кладём {v} в кучу.",
                     u, (u, v), True)
            else:
                snap(f"Релаксация {u}→{v}: d[{u}] + {w} = {new} ≥ {dist[v]} — "
                     f"путь не короче, ничего не меняем.", u, (u, v), False)

    snap("Куча пуста — алгоритм завершён. Красные рёбра образуют дерево кратчайших путей.")
    return steps


# --------------------------- Визуализация ----------------------------

def visualize(graph, dist, prev, start, directed=True, save_path=None):
    """
    Рисует граф: рёбра дерева кратчайших путей выделены красным,
    над каждой вершиной подписано расстояние от стартовой.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        import networkx as nx
    except ImportError:
        print("Для визуализации установите библиотеки: pip install matplotlib networkx")
        return

    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(graph)
    for u in graph:
        for v, w in graph[u]:
            G.add_edge(u, v, weight=w)

    pos = nx.spring_layout(G, seed=42)

    # рёбра дерева кратчайших путей
    tree = {(prev[v], v) for v in graph if prev[v] is not None}
    if not directed:
        tree |= {(v, u) for u, v in tree}

    edges = list(G.edges())
    edge_colors = ["crimson" if e in tree else "lightgray" for e in edges]
    widths = [3.5 if e in tree else 1.5 for e in edges]

    def node_color(v):
        if v == start:
            return "gold"
        if dist[v] == float("inf"):
            return "lightgray"
        return "skyblue"

    fig, ax = plt.subplots(figsize=(10, 7))
    curved = {"connectionstyle": "arc3,rad=0.12"} if directed else {}

    nx.draw_networkx_nodes(G, pos, node_color=[node_color(v) for v in G.nodes()],
                           node_size=1000, edgecolors="black", ax=ax)
    nx.draw_networkx_labels(G, pos, font_weight="bold", ax=ax)
    arrow_opts = {"arrowsize": 20, "node_size": 1000} if directed else {}
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors, width=widths,
                           arrows=directed, ax=ax, **arrow_opts, **curved)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, "weight"),
                                 font_size=10, ax=ax, **curved)

    # подписи расстояний над вершинами
    for v, (x, y) in pos.items():
        label = "d = ∞" if dist[v] == float("inf") else f"d = {dist[v]}"
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(0, 24), ha="center",
                    color="darkgreen", fontsize=10, fontweight="bold", annotation_clip=False)

    ax.legend(handles=[
        Patch(facecolor="gold", edgecolor="black", label="стартовая вершина"),
        Patch(facecolor="skyblue", edgecolor="black", label="достижимая вершина"),
        Patch(facecolor="lightgray", edgecolor="black", label="недостижимая вершина"),
        Line2D([0], [0], color="crimson", lw=3.5, label="ребро кратчайшего пути"),
    ], loc="best")
    ax.set_title(f"Алгоритм Дейкстры: кратчайшие пути из вершины {start}", fontsize=14)
    ax.axis("off")
    ax.margins(0.12)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
    else:
        plt.show()


# ------------------------ Пошаговая анимация -------------------------

def build_animation(graph, start, directed=True):
    """Возвращает (steps, draw_step) — список шагов и функцию отрисовки шага."""
    import networkx as nx
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    steps = dijkstra_steps(graph, start)

    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(graph)
    for u in graph:
        for v, w in graph[u]:
            G.add_edge(u, v, weight=w)
    pos = nx.spring_layout(G, seed=42)
    edges = list(G.edges())
    weights = nx.get_edge_attributes(G, "weight")
    curved = {"connectionstyle": "arc3,rad=0.12"} if directed else {}
    arrow_opts = {"arrowsize": 20, "node_size": 1000} if directed else {}

    def same(e, f):
        return e == f or (not directed and e == f[::-1])

    def draw_step(ax, s):
        def node_color(v):
            if v == s["current"]:
                return "orange"
            if v in s["visited"]:
                return "mediumseagreen"
            if s["dist"][v] < INF:
                return "skyblue"
            return "white"

        tree = [(p, v) for v, p in s["prev"].items() if p is not None]
        edge_colors, widths = [], []
        for e in edges:
            if s["edge"] is not None and same(e, s["edge"]):
                edge_colors.append("limegreen" if s["improved"] else "darkorange")
                widths.append(5)
            elif any(same(e, t) for t in tree):
                edge_colors.append("crimson")
                widths.append(3)
            else:
                edge_colors.append("lightgray")
                widths.append(1.5)

        nx.draw_networkx_nodes(G, pos, node_color=[node_color(v) for v in G.nodes()],
                               node_size=1000, edgecolors="black", ax=ax)
        nx.draw_networkx_labels(G, pos, font_weight="bold", ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors, width=widths,
                               arrows=directed, ax=ax, **arrow_opts, **curved)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=weights, font_size=10, ax=ax, **curved)
        for v, (x, y) in pos.items():
            ax.annotate(f"d = {fmt(s['dist'][v])}", (x, y), textcoords="offset points",
                        xytext=(0, 24), ha="center", color="darkgreen", fontsize=10,
                        fontweight="bold", annotation_clip=False)

        heap = ", ".join(f"({fmt(d)}, {v})" for d, v in s["heap"]) or "пусто"
        done = ", ".join(str(v) for v in graph if v in s["visited"]) or "—"
        ax.text(0.01, 0.01, f"Куча: {heap}\nОбработаны: {done}", transform=ax.transAxes,
                va="bottom", family="monospace", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
        ax.legend(handles=[
            Patch(facecolor="orange", edgecolor="black", label="текущая вершина"),
            Patch(facecolor="mediumseagreen", edgecolor="black", label="обработана"),
            Patch(facecolor="skyblue", edgecolor="black", label="в куче (d найдено)"),
            Patch(facecolor="white", edgecolor="black", label="не достигнута"),
            Line2D([0], [0], color="limegreen", lw=4, label="релаксация: улучшение"),
            Line2D([0], [0], color="darkorange", lw=4, label="релаксация: без улучшения"),
            Line2D([0], [0], color="crimson", lw=3, label="дерево кратчайших путей"),
        ], loc="upper right", fontsize=9)
        ax.set_title("Алгоритм Дейкстры\n" + textwrap.fill(s["msg"], 110), fontsize=12)
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


def animate(graph, start, directed=True):
    StepPlayer(*build_animation(graph, start, directed)).show()


def save_animation(graph, start, directed=True, path="dijkstra.gif"):
    steps, draw_step = build_animation(graph, start, directed)
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


def read_graph():
    """Считывает взвешенный граф с клавиатуры."""
    directed = ask_yes_no("Граф ориентированный? (y/n): ")
    m = read_int("Количество рёбер: ", min_value=1)

    graph = {}
    print("Введите рёбра в формате: <откуда> <куда> <вес>  (например: A B 5)")
    i = 0
    while i < m:
        parts = input(f"  ребро {i + 1}: ").split()
        if len(parts) != 3:
            print("  Ошибка: нужно ровно 3 значения.")
            continue
        u, v, w = parts
        try:
            w = parse_number(w)
        except ValueError:
            print("  Ошибка: вес должен быть числом.")
            continue
        if w < 0:
            print("  Ошибка: вес не может быть отрицательным.")
            continue

        graph.setdefault(u, []).append((v, w))
        graph.setdefault(v, [])
        if not directed:
            graph[v].append((u, w))
        i += 1

    return graph, directed


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
        "A": [("B", 4), ("C", 2)],
        "B": [("C", 5), ("D", 10)],
        "C": [("E", 3)],
        "D": [("F", 11)],
        "E": [("D", 4)],
        "F": [],
    }


if __name__ == "__main__":
    print("=== Алгоритм Дейкстры ===")
    if ask_yes_no("Использовать встроенный пример? (y/n): "):
        graph = example_graph()
        directed = True
        start = "A"
    else:
        graph, directed = read_graph()
        while True:
            start = input("Стартовая вершина: ").strip()
            if start in graph:
                break
            print(f"  Ошибка: такой вершины нет. Доступные: {', '.join(graph)}")

    dist, prev = dijkstra(graph, start)

    print(f"\nКратчайшие расстояния от вершины {start}:")
    for v in graph:
        path = restore_path(prev, start, v)
        path_str = " -> ".join(path) if path else "нет пути"
        print(f"  {v}: {dist[v]:<5}  путь: {path_str}")

    run_menu({
        "1": lambda: visualize(graph, dist, prev, start, directed),
        "2": lambda: animate(graph, start, directed),
        "3": lambda: save_animation(graph, start, directed),
    })
