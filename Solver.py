from image_processing import gray_image, detect_grid_size, process_starbattle_map

# Star Battle Map
gray = gray_image('Star Battle Puzzle 3.png')
grid = detect_grid_size(gray)
df = process_starbattle_map(gray, grid)

import numpy as np
from collections import defaultdict

class StarBattleSolver:
    def __init__(self, df):
        self.df = df.copy()
        self.rows, self.cols = df.shape

        # State: 0 = blank, 1 = star, -1 = dot
        self.state = np.zeros((self.rows, self.cols), dtype=int)

        # Track groups and their positions
        self.groups = self._analyze_groups()

    def _analyze_groups(self):
        """Analyze the groups and their positions"""
        groups = defaultdict(list)
        for i in range(self.rows):
            for j in range(self.cols):
                group_id = self.df.iloc[i, j]
                groups[group_id].append((i, j))
        return dict(groups)

    def _get_available_positions(self, group_id):
        """Get available positions (blank spaces) for a group"""
        positions = self.groups[group_id]
        return [(r, c) for r, c in positions if self.state[r, c] == 0]

    def _place_star(self, row, col):
        """Place a star and mark surrounding areas with dots"""
        if self.state[row, col] != 0:
            return False

        self.state[row, col] = 1

        # Mark entire row and column with dots (except the star itself)
        for c in range(self.cols):
            if c != col and self.state[row, c] == 0:
                self.state[row, c] = -1
        for r in range(self.rows):
            if r != row and self.state[r, col] == 0:
                self.state[r, col] = -1

        # Mark adjacent cells with dots
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if self.state[nr, nc] == 0:
                        self.state[nr, nc] = -1

        return True

    def _place_dot(self, row, col):
        """Place a dot"""
        if self.state[row, col] == 0:
            self.state[row, col] = -1
            return True
        return False

    def _rule_single_position(self):
        """Rule 1: If a group has only one available position, place a star there"""
        changed = False
        for group_id, positions in self.groups.items():
            available = self._get_available_positions(group_id)
            if len(available) == 1:
                row, col = available[0]
                if self._place_star(row, col):
                    print(f"Rule 1: Placed star at ({row}, {col}) in group {group_id} (only position)")
                    changed = True
        return changed

    def _rule_group_on_single_line(self):
        """Rule 2: If a group occupies only one row/column, mark other groups' positions in that line"""
        changed = False

        for group_id, positions in self.groups.items():
            # Check if group is on a single row
            rows = set(pos[0] for pos in positions)
            if len(rows) == 1:
                row = list(rows)[0]
                # Mark all other groups' positions in this row with dots
                for c in range(self.cols):
                    if self.df.iloc[row, c] != group_id and self.state[row, c] == 0:
                        self._place_dot(row, c)
                        changed = True

            # Check if group is on a single column
            cols = set(pos[1] for pos in positions)
            if len(cols) == 1:
                col = list(cols)[0]
                # Mark all other groups' positions in this column with dots
                for r in range(self.rows):
                    if self.df.iloc[r, col] != group_id and self.state[r, col] == 0:
                        self._place_dot(r, col)
                        changed = True

        return changed

    def _rule_counting_constraint(self):
        """Rule 3: Use counting constraints to deduce forced placements"""
        changed = False

        # Count groups per row and column
        for row in range(self.rows):
            groups_in_row = set(self.df.iloc[row, :])
            available_positions = sum(1 for c in range(self.cols) if self.state[row, c] == 0)
            stars_in_row = sum(1 for c in range(self.cols) if self.state[row, c] == 1)

            # If we need exactly one more star and have limited positions
            if stars_in_row == 0 and available_positions == len(groups_in_row):
                # Each group in this row must place its star here
                for group_id in groups_in_row:
                    group_positions_in_row = [(row, c) for c in range(self.cols)
                                              if self.df.iloc[row, c] == group_id and self.state[row, c] == 0]
                    if len(group_positions_in_row) == 1:
                        r, c = group_positions_in_row[0]
                        if self._place_star(r, c):
                            print(f"Rule 3: Placed star at ({r}, {c}) by counting constraint")
                            changed = True

        for col in range(self.cols):
            groups_in_col = set(self.df.iloc[:, col])
            available_positions = sum(1 for r in range(self.rows) if self.state[r, col] == 0)
            stars_in_col = sum(1 for r in range(self.rows) if self.state[r, col] == 1)

            # If we need exactly one more star and have limited positions
            if stars_in_col == 0 and available_positions == len(groups_in_col):
                # Each group in this column must place its star here
                for group_id in groups_in_col:
                    group_positions_in_col = [(r, col) for r in range(self.rows)
                                              if self.df.iloc[r, col] == group_id and self.state[r, col] == 0]
                    if len(group_positions_in_col) == 1:
                        r, c = group_positions_in_col[0]
                        if self._place_star(r, c):
                            print(f"Rule 3: Placed star at ({r}, {c}) by counting constraint")
                            changed = True

        return changed

    def _is_solved(self):
        """Check if the puzzle is solved"""
        # Check if each group has exactly one star
        for group_id, positions in self.groups.items():
            stars_in_group = sum(1 for r, c in positions if self.state[r, c] == 1)
            if stars_in_group != 1:
                return False

        # Check if each row and column has exactly one star
        for row in range(self.rows):
            stars_in_row = sum(1 for c in range(self.cols) if self.state[row, c] == 1)
            if stars_in_row != 1:
                return False

        for col in range(self.cols):
            stars_in_col = sum(1 for r in range(self.rows) if self.state[r, col] == 1)
            if stars_in_col != 1:
                return False

        return True

    def _backtrack_search(self):
        """Use backtracking search for remaining positions"""
        if self._is_solved():
            return True

        # Find the group with the fewest available positions
        min_positions = float('inf')
        best_group = None

        for group_id, positions in self.groups.items():
            available = self._get_available_positions(group_id)
            stars_in_group = sum(1 for r, c in positions if self.state[r, c] == 1)

            if stars_in_group == 0 and 0 < len(available) < min_positions:
                min_positions = len(available)
                best_group = group_id

        if best_group is None:
            return False

        # Try placing a star in each available position
        available_positions = self._get_available_positions(best_group)
        for row, col in available_positions:
            # Save current state
            old_state = self.state.copy()

            # Try placing star
            if self._place_star(row, col):
                if self._backtrack_search():
                    return True

            # Restore state
            self.state = old_state

        return False

    def solve(self):
        """Main solving method"""
        print("Starting to solve Star Battle puzzle...")
        print(f"Puzzle size: {self.rows}x{self.cols}")
        print(f"Number of groups: {len(self.groups)}")

        iteration = 0
        while iteration < 100:  # Prevent infinite loops
            iteration += 1
            changed = False

            print(f"\n--- Iteration {iteration} ---")

            # Apply logical rules
            if self._rule_single_position():
                changed = True
            if self._rule_group_on_single_line():
                changed = True
            if self._rule_counting_constraint():
                changed = True

            if not changed:
                print("No more logical deductions possible, trying backtracking...")
                break

            if self._is_solved():
                print("Puzzle solved using logical deduction!")
                return True

        # If logical deduction isn't enough, use backtracking
        print("Applying backtracking search...")
        if self._backtrack_search():
            print("Puzzle solved using backtracking!")
            return True
        else:
            print("No solution found!")
            return False

    def display_solution(self):
        """Display the current state of the puzzle"""
        print("\nCurrent state:")
        print("Groups:")
        print(self.df)
        print("\nSolution (1=star, -1=dot, 0=blank):")
        print(self.state)

        # Create a visual representation
        visual = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                if self.state[i, j] == 1:
                    row.append('★')
                elif self.state[i, j] == -1:
                    row.append('·')
                else:
                    row.append(' ')
            visual.append(row)

        print("\nVisual representation:")
        for row in visual:
            print(' '.join(f"{cell:2}" for cell in row))


# Example usage with your data
if __name__ == "__main__":
    # Your dataframe

    solver = StarBattleSolver(df)

    if solver.solve():
        solver.display_solution()
    else:
        print("Could not solve the puzzle")
        solver.display_solution()