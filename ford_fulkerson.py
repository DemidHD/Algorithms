"""
Лабораторная работа: Алгоритм Форда-Фалкерсона.

Поиск максимального потока в сети от истока s к стоку t.
Увеличивающие пути ищутся обходом в глубину по остаточной сети.
Сложность: O(E * F), где F — величина максимального потока.
"""

import os
import textwrap


def ford_fulkerson(capacity, source, sink):
    """
    capacity — матрица пропускных способностей n x n.
    Возвращает (max_flow, flow), где flow — матрица итогового потока.
    """
    n = len(capacity)
    # остаточная сеть
    residual = [row[:] for row in capacity]

    def dfs(u, visited, parent):
        """Ищет увеличивающий путь из u в sink, заполняя parent."""
        if u == sink:
            return True
        visited[u] = True
        for v in range(n):
            if not visited[v] and residual[u][v] > 0:
                parent[v] = u
                if dfs(v, visited, parent):
                    return True
        return False

    max_flow = 0
    while True:
        parent = [-1] * n
        if not dfs(source, [False] * n, parent):
            break

        # минимальная остаточная пропускная способность на найденном пути
        path_flow = float("inf")
        v = sink
        while v != source:
            u = parent[v]
            path_flow = min(path_flow, residual[u][v])
            v = u

        # обновляем остаточную сеть
        v = sink
        while v != source:
            u = parent[v]
            residual[u][v] -= path_flow
            residual[v][u] += path_flow
            v = u

        max_flow += path_flow

    flow = [[max(capacity[u][v] - residual[u][v], 0) for v in range(n)] for u in range(n)]
    return max_flow, flow


def min_cut(capacity, flow, source):
    """
    Множество S вершин, достижимых из истока в остаточной сети.
    Рёбра из S в остальные вершины образуют минимальный разрез.
    """
    n = len(capacity)
    seen = {source}
    stack = [source]
    while stack:
        u = stack.pop()
        for v in range(n):
            residual = capacity[u][v] - flow[u][v] + flow[v][u]
            if v not in seen and residual > 0:
                seen.add(v)
                stack.append(v)
    return seen


def ford_fulkerson_steps(capacity, source, sink):
    """
    Тот же алгоритм Форда-Фалкерсона, но с записью каждого шага
    (снимка состояния) для пошаговой анимации.
    """
    n = len(capacity)
    residual = [row[:] for row in capacity]
    total = 0
    steps = []

    def current_flow():
        return [[max(capacity[u][v] - residual[u][v], 0) for v in range(n)] for u in range(n)]

    def snap(msg, path_edges=(), path_nodes=(), cut=None):
        steps.append({"flow": current_flow(), "total": total, "path_edges": list(path_edges),
                      "path_nodes": list(path_nodes), "cut": cut, "msg": msg})

    def dfs(u, visited, parent):
        if u == sink:
            return True
        visited[u] = True
        for v in range(n):
            if not visited[v] and residual[u][v] > 0:
                parent[v] = u
                if dfs(v, visited, parent):
                    return True
        return False

    snap("Начало: поток по всем рёбрам равен 0. "
         "Ищем увеличивающий путь из истока в сток обходом в глубину по остаточной сети.")

    iteration = 0
    while True:
        parent = [-1] * n
        if not dfs(source, [False] * n, parent):
            break
        iteration += 1

        nodes = [sink]
        while nodes[-1] != source:
            nodes.append(parent[nodes[-1]])
        nodes.reverse()

        # каждый шаг пути — прямое ребро (есть свободная ёмкость)
        # или обратное (отмена ранее пущенного потока)
        flow = current_flow()
        path_edges = []
        for u, v in zip(nodes, nodes[1:]):
            if capacity[u][v] - flow[u][v] > 0:
                path_edges.append(((u, v), "forward"))
            else:
                path_edges.append(((v, u), "backward"))
        path_flow = min(residual[u][v] for u, v in zip(nodes, nodes[1:]))
        path_str = " → ".join(map(str, nodes))
        backward = [f"{a}→{b}" for (a, b), kind in path_edges if kind == "backward"]
        note = (f" Путь использует обратное ребро ({', '.join(backward)}): "
                f"часть потока по нему отменяется." if backward else "")

        snap(f"Итерация {iteration}: найден увеличивающий путь {path_str}. Узкое место "
             f"(минимальная остаточная ёмкость) = {path_flow}.{note}", path_edges, nodes)

        for u, v in zip(nodes, nodes[1:]):
            residual[u][v] -= path_flow
            residual[v][u] += path_flow
        total += path_flow

        snap(f"Пускаем {path_flow} ед. потока по пути {path_str}. "
             f"Общий поток = {total}.", path_edges, nodes)

    cut = min_cut(capacity, current_flow(), source)
    snap(f"Увеличивающих путей больше нет — поток максимален и равен {total}. "
         f"Красным пунктиром — рёбра минимального разреза.", cut=cut)
    return steps


# --------------------------- Визуализация ----------------------------

def visualize(capacity, flow, source, sink, max_flow, save_path=None):
    """
    Рисует сеть: на рёбрах подписано «поток / пропускная способность»,
    толщина ребра пропорциональна потоку. Вершины раскрашены по сторонам
    минимального разреза, рёбра разреза выделены.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        import networkx as nx
    except ImportError:
        print("Для визуализации установите библиотеки: pip install matplotlib networkx")
        return

    n = len(capacity)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for u in range(n):
        for v in range(n):
            if capacity[u][v] > 0:
                G.add_edge(u, v)

    # раскладка по слоям: слой = расстояние от истока (в рёбрах)
    layers = nx.single_source_shortest_path_length(G, source)
    last = max(layers.values(), default=0) + 1
    for v in G.nodes():
        G.nodes[v]["layer"] = layers.get(v, last)
    pos = nx.multipartite_layout(G, subset_key="layer")

    S = min_cut(capacity, flow, source)
    cut_edges = {(u, v) for u, v in G.edges() if u in S and v not in S}

    edges = list(G.edges())
    biggest = max((flow[u][v] for u, v in edges), default=0) or 1

    def edge_color(u, v):
        if (u, v) in cut_edges:
            return "crimson"
        if flow[u][v] == 0:
            return "lightgray"
        return "royalblue"

    def node_color(v):
        if v == source:
            return "limegreen"
        if v == sink:
            return "tomato"
        return "palegreen" if v in S else "lightsalmon"

    fig, ax = plt.subplots(figsize=(11, 7))
    curved = {"connectionstyle": "arc3,rad=0.15"}

    nx.draw_networkx_nodes(G, pos, node_color=[node_color(v) for v in G.nodes()],
                           node_size=1100, edgecolors="black", ax=ax)
    nx.draw_networkx_labels(G, pos, labels={v: ("s" if v == source else "t" if v == sink else v)
                                            for v in G.nodes()},
                            font_weight="bold", font_size=12, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=edges,
                           edge_color=[edge_color(u, v) for u, v in edges],
                           width=[1.5 + 4 * flow[u][v] / biggest for u, v in edges],
                           style=["dashed" if (u, v) in cut_edges else "solid" for u, v in edges],
                           arrowsize=20, node_size=1100, ax=ax, **curved)
    nx.draw_networkx_edge_labels(G, pos,
                                 edge_labels={(u, v): f"{flow[u][v]}/{capacity[u][v]}"
                                              for u, v in edges},
                                 font_size=10, label_pos=0.4, ax=ax, **curved)

    ax.legend(handles=[
        Patch(facecolor="palegreen", edgecolor="black", label="сторона истока (S)"),
        Patch(facecolor="lightsalmon", edgecolor="black", label="сторона стока (T)"),
        Line2D([0], [0], color="royalblue", lw=3, label="ребро с потоком"),
        Line2D([0], [0], color="crimson", lw=3, ls="--", label="ребро минимального разреза"),
        Line2D([0], [0], color="lightgray", lw=2, label="ребро без потока"),
    ], loc="best")
    ax.set_title(f"Алгоритм Форда-Фалкерсона: максимальный поток = {max_flow} "
                 f"(= пропускной способности минимального разреза)", fontsize=13)
    ax.axis("off")
    ax.margins(0.1)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
    else:
        plt.show()


# ------------------------ Пошаговая анимация -------------------------

def build_animation(capacity, source, sink):
    """Возвращает (steps, draw_step) — список шагов и функцию отрисовки шага."""
    import networkx as nx
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    steps = ford_fulkerson_steps(capacity, source, sink)

    n = len(capacity)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for u in range(n):
        for v in range(n):
            if capacity[u][v] > 0:
                G.add_edge(u, v)
    layers = nx.single_source_shortest_path_length(G, source)
    last = max(layers.values(), default=0) + 1
    for v in G.nodes():
        G.nodes[v]["layer"] = layers.get(v, last)
    pos = nx.multipartite_layout(G, subset_key="layer")
    edges = list(G.edges())
    biggest = max((capacity[u][v] for u, v in edges), default=1)
    curved = {"connectionstyle": "arc3,rad=0.15"}
    names = {v: ("s" if v == source else "t" if v == sink else v) for v in G.nodes()}

    def draw_step(ax, s):
        flow, cut = s["flow"], s["cut"]
        on_path = dict(s["path_edges"])

        def node_color(v):
            if v == source:
                return "limegreen"
            if v == sink:
                return "tomato"
            if cut is not None:
                return "palegreen" if v in cut else "lightsalmon"
            if v in s["path_nodes"]:
                return "gold"
            return "lightblue"

        edge_colors, widths, styles = [], [], []
        for u, v in edges:
            kind = on_path.get((u, v))
            if kind == "forward":
                edge_colors.append("darkorange")
                widths.append(5)
                styles.append("solid")
            elif kind == "backward":
                edge_colors.append("purple")
                widths.append(4)
                styles.append("dashed")
            elif cut is not None and u in cut and v not in cut:
                edge_colors.append("crimson")
                widths.append(3.5)
                styles.append("dashed")
            elif flow[u][v] > 0:
                edge_colors.append("royalblue")
                widths.append(1.5 + 3.5 * flow[u][v] / biggest)
                styles.append("solid")
            else:
                edge_colors.append("lightgray")
                widths.append(1.5)
                styles.append("solid")

        nx.draw_networkx_nodes(G, pos, node_color=[node_color(v) for v in G.nodes()],
                               node_size=1100, edgecolors="black", ax=ax)
        nx.draw_networkx_labels(G, pos, labels=names, font_weight="bold", font_size=12, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors, width=widths,
                               style=styles, arrowsize=20, node_size=1100, ax=ax, **curved)
        nx.draw_networkx_edge_labels(G, pos,
                                     edge_labels={(u, v): f"{flow[u][v]}/{capacity[u][v]}"
                                                  for u, v in edges},
                                     font_size=10, label_pos=0.4, ax=ax, **curved)

        ax.text(0.01, 0.01, f"Текущий поток: {s['total']}", transform=ax.transAxes,
                va="bottom", family="monospace", fontsize=12, fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
        handles = [
            Line2D([0], [0], color="darkorange", lw=4, label="увеличивающий путь"),
            Line2D([0], [0], color="purple", lw=4, ls="--", label="обратное ребро (отмена)"),
            Line2D([0], [0], color="royalblue", lw=3, label="ребро с потоком"),
            Line2D([0], [0], color="lightgray", lw=2, label="ребро без потока"),
        ]
        if cut is not None:
            handles += [
                Patch(facecolor="palegreen", edgecolor="black", label="сторона истока (S)"),
                Patch(facecolor="lightsalmon", edgecolor="black", label="сторона стока (T)"),
                Line2D([0], [0], color="crimson", lw=3, ls="--", label="минимальный разрез"),
            ]
        ax.legend(handles=handles, loc="upper right", fontsize=9)
        ax.set_title("Алгоритм Форда-Фалкерсона\n" + textwrap.fill(s["msg"], 110), fontsize=12)
        ax.axis("off")
        ax.margins(0.1)

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

    def __init__(self, steps, draw_step, interval=1500):
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


def save_gif(steps, draw_step, path, fps=0.7):
    """Сохраняет анимацию в GIF (кадр на каждый шаг)."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.88])
    anim = FuncAnimation(fig, lambda i: render_step(ax, steps, i, draw_step),
                         frames=len(steps))
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def animate(capacity, source, sink):
    StepPlayer(*build_animation(capacity, source, sink)).show()


def save_animation(capacity, source, sink, path="ford_fulkerson.gif"):
    steps, draw_step = build_animation(capacity, source, sink)
    print(f"Сохраняю {len(steps)} кадров...")
    save_gif(steps, draw_step, path)
    print(f"Анимация сохранена: {os.path.abspath(path)}")


# ---------------------------- Ввод данных ----------------------------

def read_int(prompt, min_value=None, max_value=None):
    while True:
        try:
            value = int(input(prompt))
        except ValueError:
            print("  Ошибка: введите целое число.")
            continue
        if min_value is not None and value < min_value:
            print(f"  Ошибка: число должно быть не меньше {min_value}.")
            continue
        if max_value is not None and value > max_value:
            print(f"  Ошибка: число должно быть не больше {max_value}.")
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


def read_network():
    """Считывает сеть с клавиатуры. Вершины нумеруются от 0 до n-1."""
    n = read_int("Количество вершин: ", min_value=2)
    m = read_int("Количество рёбер: ", min_value=1)

    capacity = [[0] * n for _ in range(n)]
    print(f"Введите рёбра в формате: <откуда> <куда> <пропускная способность>"
          f"  (вершины 0..{n - 1}, например: 0 1 16)")
    i = 0
    while i < m:
        parts = input(f"  ребро {i + 1}: ").split()
        if len(parts) != 3:
            print("  Ошибка: нужно ровно 3 значения.")
            continue
        try:
            u, v, c = map(int, parts)
        except ValueError:
            print("  Ошибка: все значения должны быть целыми числами.")
            continue
        if not (0 <= u < n and 0 <= v < n):
            print(f"  Ошибка: номера вершин должны быть от 0 до {n - 1}.")
            continue
        if u == v:
            print("  Ошибка: петли не допускаются.")
            continue
        if c < 0:
            print("  Ошибка: пропускная способность не может быть отрицательной.")
            continue
        capacity[u][v] += c  # параллельные рёбра складываются
        i += 1

    source = read_int("Исток: ", 0, n - 1)
    while True:
        sink = read_int("Сток: ", 0, n - 1)
        if sink != source:
            break
        print("  Ошибка: сток должен отличаться от истока.")

    return capacity, source, sink


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


def example_network():
    # Классический пример (CLRS): вершины 0..5, исток 0, сток 5
    capacity = [
        [0, 16, 13, 0, 0, 0],
        [0, 0, 10, 12, 0, 0],
        [0, 4, 0, 0, 14, 0],
        [0, 0, 9, 0, 0, 20],
        [0, 0, 0, 7, 0, 4],
        [0, 0, 0, 0, 0, 0],
    ]
    return capacity, 0, 5


if __name__ == "__main__":
    print("=== Алгоритм Форда-Фалкерсона ===")
    if ask_yes_no("Использовать встроенный пример? (y/n): "):
        capacity, source, sink = example_network()
    else:
        capacity, source, sink = read_network()

    max_flow, flow = ford_fulkerson(capacity, source, sink)

    print(f"\nМаксимальный поток из {source} в {sink}: {max_flow}")
    print("Поток по рёбрам (поток / пропускная способность):")
    for u in range(len(capacity)):
        for v in range(len(capacity)):
            if capacity[u][v] > 0:
                print(f"  {u} -> {v}: {flow[u][v]} / {capacity[u][v]}")

    S = min_cut(capacity, flow, source)
    cut = [(u, v) for u in S for v in range(len(capacity))
           if v not in S and capacity[u][v] > 0]
    print(f"Минимальный разрез: S = {sorted(S)}, рёбра разреза: "
          + ", ".join(f"{u}->{v}" for u, v in sorted(cut)))

    run_menu({
        "1": lambda: visualize(capacity, flow, source, sink, max_flow),
        "2": lambda: animate(capacity, source, sink),
        "3": lambda: save_animation(capacity, source, sink),
    })
