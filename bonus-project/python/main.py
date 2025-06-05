import argparse
import copy
import heapq
import os
import random
import time
from typing import List, Tuple, Optional, Set as PySet

from rich.console import Console
from rich.panel import Panel
from rich.prompt import IntPrompt, Prompt
from rich.table import Table

# $ pip install rich psutil

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False


class PuzzleState:
    def __init__(self, board: List[List[int]], size: int, moves: int = 0, parent: Optional['PuzzleState'] = None):
        self.board = board
        self.size = size
        self.moves = moves
        self.parent = parent
        self.blank_pos = self._find_blank()
        self.manhattan = self._calculate_manhattan()
        self.f_cost_value: float = 0.0

    def _find_blank(self) -> Tuple[int, int]:
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    return (i, j)
        raise ValueError("No blank tile found in the puzzle board.")

    def _calculate_manhattan(self) -> int:
        total_distance = 0
        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i][j]
                if value != 0:
                    target_row = (value - 1) // self.size
                    target_col = (value - 1) % self.size
                    total_distance += abs(i - target_row) + abs(j - target_col)
        return total_distance

    def f_cost(self) -> int:
        """Calculates the f-cost (g + h) for A*."""
        return self.moves + self.manhattan

    def __lt__(self, other: 'PuzzleState') -> bool:
        """Comparison for priority queue in A*."""
        return self.f_cost() < other.f_cost()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PuzzleState):
            return NotImplemented
        return self.board == other.board

    def __hash__(self) -> int:
        """Hashing for storage in sets or as dictionary keys."""
        return hash(tuple(map(tuple, self.board)))


def get_neighbors(state: PuzzleState) -> List[PuzzleState]:
    """Generate neighboring states by moving the blank tile."""
    neighbors = []
    directions = [(-1, 0, "Up"), (1, 0, "Down"), (0, -1, "Left"), (0, 1, "Right")]
    row, col = state.blank_pos

    for dr, dc, _ in directions:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < state.size and 0 <= new_col < state.size:
            new_board = copy.deepcopy(state.board)
            new_board[row][col], new_board[new_row][new_col] = new_board[new_row][new_col], new_board[row][col]
            neighbors.append(PuzzleState(new_board, state.size, state.moves + 1, state))
    return neighbors


def get_goal_board(size: int) -> List[List[int]]:
    """Generates the goal state for an N x N puzzle."""
    board = [[(i * size + j + 1) for j in range(size)] for i in range(size)]
    board[size - 1][size - 1] = 0
    return board


def is_solvable(board: List[List[int]], size: int) -> bool:
    """Check if the N-puzzle is solvable."""
    flat_board = [num for row in board for num in row if num != 0]
    inversions = 0
    for i in range(len(flat_board)):
        for j in range(i + 1, len(flat_board)):
            if flat_board[i] > flat_board[j]:
                inversions += 1

    blank_row = -1
    for i in range(size):
        for j in range(size):
            if board[i][j] == 0:
                blank_row = i
                break
        if blank_row != -1:
            break
    if blank_row == -1:
        raise ValueError("No black tile found")
    if size % 2 == 1:
        return inversions % 2 == 0
    else:
        blank_row_from_bottom = size - blank_row
        return (inversions + blank_row_from_bottom) % 2 == 1


def generate_solvable_puzzle(n: int, scramble_moves: Optional[int] = None) -> List[List[int]]:
    """Generate a solvable N x N puzzle."""
    if n < 2:
        raise ValueError("Puzzle size must be at least 2x2.")

    if scramble_moves is None:
        scramble_moves = n ** 3

    max_attempts = 100
    attempt = 0

    while attempt < max_attempts:
        board = get_goal_board(n)
        blank_r, blank_c = n - 1, n - 1
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for _ in range(scramble_moves):
            valid_moves = []
            for dr, dc in moves:
                nr, nc = blank_r + dr, blank_c + dc
                if 0 <= nr < n and 0 <= nc < n:
                    valid_moves.append((nr, nc))
            if not valid_moves:
                continue
            new_r, new_c = random.choice(valid_moves)
            board[blank_r][blank_c], board[new_r][new_c] = board[new_r][new_c], board[blank_r][blank_c]
            blank_r, blank_c = new_r, new_c

        if is_solvable(board, n):
            return board

        attempt += 1
        scramble_moves = max(1, scramble_moves + random.randint(-5, 5))

    raise ValueError(f"Could not generate a solvable puzzle after {max_attempts} attempts.")


def a_star_search(initial_board: List[List[int]], size: int) -> Tuple[Optional[PuzzleState], float, float, int]:
    """Run A* search on the puzzle."""
    start_time = time.time()
    nodes_expanded = 0
    memory_used = -1.0
    current_process = None
    if PSUTIL_AVAILABLE and psutil and os:
        current_process = psutil.Process(os.getpid())

    initial_state = PuzzleState(initial_board, size)
    goal_state_board = get_goal_board(size)
    goal_key = PuzzleState(goal_state_board, size).__hash__()

    open_list: List[PuzzleState] = [initial_state]
    heapq.heapify(open_list)
    closed_set: PySet[int] = set()

    while open_list:
        current_state = heapq.heappop(open_list)
        current_key = current_state.__hash__()

        if current_key == goal_key:
            if PSUTIL_AVAILABLE and current_process:
                memory_used = current_process.memory_info().rss / (1024 * 1024)  # MB
            return current_state, time.time() - start_time, memory_used, nodes_expanded

        if current_key in closed_set:
            continue
        closed_set.add(current_key)

        nodes_expanded += 1
        for neighbor in get_neighbors(current_state):
            if neighbor.__hash__() not in closed_set:
                heapq.heappush(open_list, neighbor)

    if PSUTIL_AVAILABLE and current_process:
        memory_used = current_process.memory_info().rss / (1024 * 1024)  # MB
    return None, time.time() - start_time, memory_used, nodes_expanded


def _get_board_tuple_key(board: List[List[int]]) -> Tuple[Tuple[int, ...], ...]:
    """Converts a board (list of lists) to a hashable tuple of tuples."""
    return tuple(map(tuple, board))


def _rbfs_recursive(
        state: PuzzleState,
        goal_key: Tuple[Tuple[int, ...], ...],
        f_limit: float,
        nodes_expanded_counter: List[int]
) -> Tuple[Optional[PuzzleState], float]:
    current_state_key = _get_board_tuple_key(state.board)
    if current_state_key == goal_key:
        return state, state.f_cost_value

    nodes_expanded_counter[0] += 1

    successors = get_neighbors(state)
    if not successors:
        return None, float('inf')

    for s_node in successors:
        s_node.f_cost_value = max(s_node.moves + s_node.manhattan, state.f_cost_value)

    while True:
        successors.sort(key=lambda x: x.f_cost_value)

        best_successor = successors[0]

        if best_successor.f_cost_value > f_limit:
            return None, best_successor.f_cost_value

        alternative_f_value = successors[1].f_cost_value if len(successors) > 1 else float('inf')

        result_state, best_f_updated = _rbfs_recursive(
            best_successor,
            goal_key,
            min(f_limit, alternative_f_value),
            nodes_expanded_counter
        )
        best_successor.f_cost_value = best_f_updated

        if result_state is not None:
            return result_state, best_f_updated


def solve_rbfs(initial_board: List[List[int]], size: int) -> Tuple[Optional[PuzzleState], float, float, int]:
    start_time = time.time()
    nodes_expanded_counter = [0]
    current_process = None

    if PSUTIL_AVAILABLE and psutil and os:
        current_process = psutil.Process(os.getpid())

    if not is_solvable(initial_board, size):
        print("RBFS: Initial puzzle is not solvable.")
        elapsed_time = time.time() - start_time
        if PSUTIL_AVAILABLE and current_process:
            memory_used = current_process.memory_info().rss / (1024 * 1024)
        return None, elapsed_time, memory_used, 0

    initial_state = PuzzleState(board=initial_board, size=size, moves=0, parent=None)
    initial_state.f_cost_value = initial_state.f_cost()

    goal_b = get_goal_board(size)
    goal_key_tuple = _get_board_tuple_key(goal_b)

    result_state, final_f_val = _rbfs_recursive(
        initial_state,
        goal_key_tuple,
        float('inf'),
        nodes_expanded_counter
    )

    elapsed_time = time.time() - start_time
    if PSUTIL_AVAILABLE and current_process and hasattr(current_process, 'memory_info'):
        memory_used = current_process.memory_info().rss / (1024 * 1024)  # MB

    num_expanded = nodes_expanded_counter[0]
    return result_state, elapsed_time, memory_used, num_expanded


def print_puzzle(state_data: List[List[int]] | List[int], n_size: int, console_obj: Console, title: str = "Puzzle"):
    """Display the puzzle state as a table within a panel."""
    if state_data and isinstance(state_data[0], list):
        flat_state = [item for sublist in state_data for item in sublist]
    else:
        flat_state = state_data

    table = Table(show_header=False, title=title,
                  title_style="bold black on white")
    for _ in range(n_size):
        table.add_column(justify="center", width=3, style="bold bright_white")

    for i in range(0, n_size * n_size, n_size):
        row_items = [str(flat_state[i + j]) if flat_state[i + j] != 0 else " " for j in range(n_size)]
        table.add_row(*row_items)
    console_obj.print(table)


def print_solution_rich(result: Optional[PuzzleState], algorithm_name: str, time_taken: float,
                        memory_used: float, nodes_expanded: int, puzzle_size: int,
                        show_steps_flag: bool, console_obj: Console):
    """Display the solution and metrics using a rich Table."""
    if result is None:
        console_obj.print(f"[bold red]No solution found by {algorithm_name}.[/bold red]")
        return None

    metrics_table = Table(title=f"{algorithm_name} Search Results", style="bold blue", border_style="blue")
    metrics_table.add_column("Metric", style="cyan", justify="right")
    metrics_table.add_column("Value", style="magenta", justify="left")

    metrics_table.add_row("Solution Found", "[bold green]Yes[/bold green]")
    metrics_table.add_row("Time Taken (s)", f"{time_taken:.4f}")
    metrics_table.add_row("Memory Used (MB)", f"{memory_used:.2f}" if memory_used >= 0 else "[yellow]N/A[/yellow]")
    metrics_table.add_row("Nodes Expanded", f"{nodes_expanded:,}")

    # console_obj.print(metrics_table)

    solution_path: List[List[List[int]]] = []
    current_node = result
    while current_node:
        solution_path.append(current_node.board)
        current_node = current_node.parent
    solution_path.reverse()

    if solution_path:
        # console_obj.print(f"[bold green]Solution path length: {len(solution_path) - 1} moves[/bold green]")
        if show_steps_flag:
            console_obj.print(Panel("[bold yellow]Solution Steps:[/bold yellow]", expand=False, border_style="yellow"))
            for i, board_state in enumerate(solution_path):
                print_puzzle(board_state, puzzle_size, console_obj, title=f"Step {i}")
    else:
        console_obj.print(f"[yellow]No valid solution path could be reconstructed for {algorithm_name}.[/yellow]")

    return {"time": time_taken, "memory": memory_used, "nodes": nodes_expanded, "steps": len(solution_path) - 1}


def print_comparison_table(a_star_metrics_dict: Optional[dict], rbfs_metrics_dict: Optional[dict],
                           console_obj: Console):
    """Display a comparison table for A* and RBFS metrics."""
    comp_table = Table(title="[bold]A* vs RBFS Comparison[/bold]", border_style="green")
    comp_table.add_column("Metric", style="cyan", justify="right")
    comp_table.add_column("A*", style="blue", justify="left")
    comp_table.add_column("RBFS", style="magenta", justify="left")

    a_star_found = "[bold green]Yes[/bold green]" if a_star_metrics_dict else "[bold red]No[/bold red]"
    rbfs_found = "[bold green]Yes[/bold green]" if rbfs_metrics_dict else "[bold red]No[/bold red]"
    comp_table.add_row("Solution Found", a_star_found, rbfs_found)

    a_star_time = f"{a_star_metrics_dict['time']:.4f} s" if a_star_metrics_dict else "[yellow]N/A[/yellow]"
    rbfs_time = f"{rbfs_metrics_dict['time']:.4f} s" if rbfs_metrics_dict else "[yellow]N/A[/yellow]"
    comp_table.add_row("Time Taken", a_star_time, rbfs_time)

    a_star_mem = (f"{a_star_metrics_dict['memory']:.2f} MB"
                  if a_star_metrics_dict and a_star_metrics_dict['memory'] >= 0 else "[yellow]N/A[/yellow]")
    rbfs_mem = (f"{rbfs_metrics_dict['memory']:.2f} MB"
                if rbfs_metrics_dict and rbfs_metrics_dict['memory'] >= 0 else "[yellow]N/A[/yellow]")
    comp_table.add_row("Memory Used", a_star_mem, rbfs_mem)

    a_star_nodes = f"{a_star_metrics_dict['nodes']:,}" if a_star_metrics_dict else "[yellow]N/A[/yellow]"
    rbfs_nodes = f"{rbfs_metrics_dict['nodes']:,}" if rbfs_metrics_dict else "[yellow]N/A[/yellow]"
    comp_table.add_row("Nodes Expanded", a_star_nodes, rbfs_nodes)

    a_star_steps = f"{a_star_metrics_dict['steps']:,}" if a_star_metrics_dict else "[yellow]N/A[/yellow]"
    rbfs_steps = f"{rbfs_metrics_dict['steps']:,}" if rbfs_metrics_dict else "[yellow]N/A[/yellow]"
    comp_table.add_row("Path length", a_star_steps, rbfs_steps)

    console_obj.print(comp_table)


def main():
    parser = argparse.ArgumentParser(description="N-Puzzle Solver using A* and RBFS algorithms.")
    parser.add_argument("--size", type=int, default=3, help="Puzzle size 'n' (e.g., 3 for 3x3, 4 for 4x4, min 2)")
    parser.add_argument("--scramble", type=int, help="Number of random scramble moves (default: n^3)")
    parser.add_argument("--show-steps", action="store_true", help="Show solution steps")
    args = parser.parse_args()
    console = Console()
    console.print(
        Panel.fit("[bold blue]N-Puzzle Solver[/bold blue]", style="bold white on blue", border_style="blue"))
    console.print("Welcome to the N-Puzzle Solver! Solves N-Puzzles using A* and RBFS algorithms.", style="italic dim")
    console.print("-" * 50)
    use_args = args.size is not None
    if use_args:
        current_size = args.size
        num_scramble_moves = args.scramble if args.scramble is not None else current_size ** 3
        show_steps = args.show_steps
    else:
        while True:
            current_size = IntPrompt.ask(
                "Enter puzzle size 'n' (e.g., 3 for 3x3, 4 for 4x4, min 2)",
                default=3,
                console=console
            )
            if current_size < 2:
                console.print("[bold red]Puzzle size must be at least 2.[/bold red]")
            else:
                break

        console.print("-" * 30)
        default_scramble = current_size ** 3
        while True:
            scramble_input = Prompt.ask(
                f"Enter number of random scramble moves (default: {default_scramble})",
                default=str(default_scramble),
                console=console
            )
            try:
                num_scramble_moves = int(scramble_input)
                if num_scramble_moves < 0:
                    console.print("[bold red]Scramble moves cannot be negative.[/bold red]")
                    continue
                break
            except ValueError:
                console.print("[bold red]Invalid input. Please enter an integer or press Enter for default.[/bold red]")

        console.print("-" * 30)
        show_steps_input = Prompt.ask(
            "Show solution steps? ([bold green]y[/bold green]/[bold red]n[/bold red])",
            choices=["y", "n"],
            default="n",
            console=console
        ).lower()
        show_steps = (show_steps_input == "y")

    if current_size < 2:
        console.print("[bold red]Puzzle size must be at least 2.[/bold red]")
        return

    if num_scramble_moves < 0:
        console.print("[bold red]Scramble moves cannot be negative.[/bold red]")
        return

    console.print("-" * 50)

    console.print(
        f"\n[bold yellow]Generating a solvable {current_size}x{current_size} puzzle with {num_scramble_moves} moves...[/bold yellow]\n")
    initial_board = generate_solvable_puzzle(current_size, scramble_moves=num_scramble_moves)

    if not is_solvable(initial_board, current_size):
        console.print("[bold red]Error: The generated puzzle is not solvable. Attempting to regenerate...[/bold red]")
        initial_board = generate_solvable_puzzle(current_size,
                                                 scramble_moves=num_scramble_moves + 1)
        if not is_solvable(initial_board, current_size):
            console.print("[bold red]Fatal: Could not generate a solvable puzzle. Exiting.[/bold red]")
            exit(1)
        console.print("[bold green]Successfully regenerated a solvable puzzle.[/bold green]")

    print_puzzle(initial_board, current_size, console, title="Initial Puzzle State")
    console.print("-" * 50)

    # --- Run A* Search ---
    console.print("[bold cyan]Running A* Search Algorithm...[/bold cyan]")
    a_star_initial_board_copy = copy.deepcopy(initial_board)
    a_star_result, a_star_time, a_star_memory, a_star_nodes = a_star_search(a_star_initial_board_copy, current_size)
    a_star_metrics = print_solution_rich(a_star_result, "A*", a_star_time, a_star_memory, a_star_nodes,
                                         current_size, show_steps, console)
    # console.print("-" * 50)

    # --- Run RBFS Search ---
    console.print("\n[bold magenta]Running RBFS Algorithm...[/bold magenta]")
    rbfs_initial_board_copy = copy.deepcopy(initial_board)
    rbfs_result, rbfs_time, rbfs_memory, rbfs_nodes = solve_rbfs(rbfs_initial_board_copy, current_size)
    rbfs_metrics = print_solution_rich(rbfs_result, "RBFS", rbfs_time, rbfs_memory, rbfs_nodes,
                                       current_size, show_steps, console)
    console.print("-" * 50)

    # --- Comparison Table ---
    console.print("\n[bold magenta]Comparison of Search Algorithms[/bold magenta]")
    print_comparison_table(a_star_metrics, rbfs_metrics, console)
    console.print("-" * 50)
    console.print("[bold yellow]Solver finished.[/bold yellow]")


if __name__ == "__main__":
    main()
