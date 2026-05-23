import heapq


class AStarAgent:
    """
    A* Pathfinding Algorithm for Maze Solving.

    Used as a comparison baseline against RL agents.
    A* finds the GUARANTEED optimal (shortest) path using:
        f(n) = g(n) + h(n)
        g(n) = cost from start to node n  (actual steps taken)
        h(n) = heuristic estimate to goal  (Manhattan distance)

    Unlike RL agents, A* requires full knowledge of the maze structure.
    RL agents learn WITHOUT this knowledge — trial and error only.
    """

    # Direction: (row_delta, col_delta) for N, S, E, W
    DIRS      = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    # Matching OPEN flags in maze_cells for each direction
    OPEN_FLAGS = [0x2, 0x8, 0x4, 0x1]   # N, S, E, W

    def __init__(self, env):
        self.env        = env
        self.path       = []        # list of (row, col) cells in order
        self.steps      = 0        # len(path) - 1
        self.maze_cells = None
        self.H          = 9
        self.W          = 9
        self.goal       = (0, 0)
        self.start      = (4, 4)
        self._detect_maze()

    # ──────────────────────────────────────────────────────────
    def _detect_maze(self):
        """Pull maze structure and goal/start from the gym env."""
        try:
            raw = self.env.unwrapped
            
            # 1. Grab maze_cells directly from the unwrapped environment
            if hasattr(raw, "maze_cells"):
                self.maze_cells    = raw.maze_cells
                self.H, self.W     = self.maze_cells.shape

            for attr in ("goal", "goal_pos", "target", "objective_position"):
                if hasattr(raw, attr):
                    g         = getattr(raw, attr)
                    self.goal = (int(g[0]), int(g[1]))
                    break

            # 2. Prioritize static start position over the constantly updating 'state'
            for attr in ("player_marker_start_pos", "init_pos", "robot_pos", "state"):
                if hasattr(raw, attr):
                    s          = getattr(raw, attr)
                    self.start = (int(s[1]), int(s[0]))
                    break
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────
    def _heuristic(self, pos, goal):
        """Manhattan distance — admissible heuristic for grid mazes."""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def _neighbors(self, row, col):
        """Return reachable neighbors of (row, col) using wall flags."""
        if self.maze_cells is None:
            return []
        cell_val  = int(self.maze_cells[row][col])
        neighbors = []
        for (dr, dc), flag in zip(self.DIRS, self.OPEN_FLAGS):
            if cell_val & flag:
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.H and 0 <= nc < self.W:
                    neighbors.append((nr, nc))
        return neighbors

    # ──────────────────────────────────────────────────────────
    def find_path(self, start=None, goal=None):
        """
        Run A* from start to goal.
        Returns list of (row, col) cells including start and goal.
        Sets self.path and self.steps as a side effect.
        """
        if start is None:
            start = self.start
        if goal is None:
            goal = self.goal

        start = tuple(start)
        goal  = tuple(goal)

        if self.maze_cells is None:
            self.path  = []
            self.steps = -1
            return []

        # Priority queue: (f_score, node)
        open_heap  = [(self._heuristic(start, goal), start)]
        came_from  = {}
        g_score    = {start: 0}

        visited = set()

        while open_heap:
            _, current = heapq.heappop(open_heap)

            if current in visited:
                continue
            visited.add(current)

            if current == goal:
                # Reconstruct path
                path = []
                node = current
                while node in came_from:
                    path.append(node)
                    node = came_from[node]
                path.append(start)
                path.reverse()
                self.path  = path
                self.steps = len(path) - 1
                return path

            for neighbor in self._neighbors(*current):
                if neighbor in visited:
                    continue
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor]  = current
                    g_score[neighbor]    = tentative_g
                    f                    = tentative_g + self._heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (f, neighbor))

        # No path found
        self.path  = []
        self.steps = -1
        return []
