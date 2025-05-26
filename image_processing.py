import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt

def gray_image(image_path):

    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return gray


def detect_grid_size(gray_image):
    """
    Detect the grid size by counting vertical lines (both thick and thin).
    For an NxN grid, there should be N+1 vertical lines.
    Uses Hough transform for line detection.
    """
    height, width = gray_image.shape

    # Use edge detection
    edges = cv2.Canny(gray_image, 50, 150)

    # Detect vertical lines using HoughLines
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=int(height * 0.3))

    vertical_lines = []
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            # Check if line is approximately vertical (theta close to 0 or π)
            if abs(theta) < 0.1 or abs(theta - np.pi) < 0.1:
                # Calculate x-coordinate of the line
                x = rho / np.cos(theta) if abs(np.cos(theta)) > 0.1 else rho / np.sin(theta)
                if 0 <= x <= width:
                    vertical_lines.append(x)

    # Remove duplicate lines (lines that are very close to each other)
    min_distance = width // 20  # Minimum distance between lines
    vertical_lines = sorted(vertical_lines)
    filtered_lines = []
    for line in vertical_lines:
        if not filtered_lines or abs(line - filtered_lines[-1]) > min_distance:
            filtered_lines.append(line)

    num_vertical_lines = len(filtered_lines)
    print(f"Hough transform detected {num_vertical_lines} vertical lines")

    # Grid size is number of lines minus 1
    grid_size = num_vertical_lines - 1

    print(f"Detected grid size: {grid_size}x{grid_size}")

    # Ensure grid size is reasonable (between 5 and 12 for typical puzzles)
    if grid_size < 5:
        print(f"Warning: Detected grid size {grid_size} seems too small, using 8x8 as default")
        grid_size = 8
    elif grid_size > 12:
        print(f"Warning: Detected grid size {grid_size} seems too large, using 10x10 as default")
        grid_size = 10

    return grid_size


def process_starbattle_map(gray, grid_size):
    """
    Process a starbattle map image and create a dataframe with group numbers.
    Groups are defined by thick black lines, while thin lines are grid boundaries.
    """

    # Get image dimensions
    height, width = gray.shape

    # Create binary image (black lines = 0, everything else = 255)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Detect thick lines by using morphological operations
    # Create kernels for detecting horizontal and vertical thick lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))

    # Invert binary image so lines are white (255) and background is black (0)
    inverted = cv2.bitwise_not(binary)

    # Detect thick horizontal lines
    horizontal_lines = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, horizontal_kernel)

    # Detect thick vertical lines
    vertical_lines = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, vertical_kernel)

    # Combine thick lines
    thick_lines = cv2.bitwise_or(horizontal_lines, vertical_lines)

    # Dilate thick lines to ensure they form complete boundaries
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thick_lines = cv2.dilate(thick_lines, kernel_dilate, iterations=2)

    # Create mask for region growing (invert so background is 255, lines are 0)
    region_mask = cv2.bitwise_not(thick_lines)

    # Estimate grid size by analyzing the image dimensions
    # Assuming roughly 8x8 grid based on typical starbattle puzzles
    #grid_size = 8
    cell_height = height // grid_size
    cell_width = width // grid_size

    # Create result dataframe
    result_df = pd.DataFrame(index=range(grid_size), columns=range(grid_size))

    # Use flood fill to identify connected regions
    visited = np.zeros_like(region_mask, dtype=bool)
    group_id = 0

    # Dictionary to store group assignments for each cell
    cell_groups = {}

    for row in range(grid_size):
        for col in range(grid_size):
            # Calculate center point of current cell
            center_y = int((row + 0.5) * cell_height)
            center_x = int((col + 0.5) * cell_width)

            if not visited[center_y, center_x] and region_mask[center_y, center_x] > 0:
                # Perform flood fill from this point
                mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
                cv2.floodFill(region_mask.copy(), mask, (center_x, center_y), 128)

                # Extract the filled region
                filled_region = (mask[1:-1, 1:-1] == 1).astype(np.uint8) * 255

                # Find all cells that belong to this region
                current_group_cells = []
                for r in range(grid_size):
                    for c in range(grid_size):
                        cell_center_y = int((r + 0.5) * cell_height)
                        cell_center_x = int((c + 0.5) * cell_width)

                        if filled_region[cell_center_y, cell_center_x] > 0:
                            current_group_cells.append((r, c))
                            visited[cell_center_y, cell_center_x] = True

                # Assign group ID to all cells in this region
                for r, c in current_group_cells:
                    cell_groups[(r, c)] = group_id

                group_id += 1

    # Alternative approach: Analyze each cell individually
    # This is more reliable for irregular grids
    for row in range(grid_size):
        for col in range(grid_size):
            if (row, col) not in cell_groups:
                # Calculate cell boundaries
                top = row * cell_height
                bottom = (row + 1) * cell_height
                left = col * cell_width
                right = (col + 1) * cell_width

                # Extract cell region
                cell_region = region_mask[top:bottom, left:right]

                # Find connected components in this cell
                labeled, num_features = ndimage.label(cell_region)

                # Assign a unique group ID if not already assigned
                if (row, col) not in cell_groups:
                    cell_groups[(row, col)] = group_id
                    group_id += 1

    # Fill the dataframe
    for row in range(grid_size):
        for col in range(grid_size):
            if (row, col) in cell_groups:
                result_df.iloc[row, col] = cell_groups[(row, col)]
            else:
                result_df.iloc[row, col] = 0

    # Normalize group numbers to start from 0 and be consecutive
    unique_groups = sorted(set(cell_groups.values()))
    group_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_groups)}

    for row in range(grid_size):
        for col in range(grid_size):
            if result_df.iloc[row, col] is not None:
                result_df.iloc[row, col] = group_mapping.get(result_df.iloc[row, col], 0)

    return result_df.astype(int)


# Example usage
def main():
    """
    Main function to demonstrate usage
    """
    try:
        # Try automatic processing first
        # Replace 'your_image_path.png' with the actual path to your image
        gray = gray_image('Star Battle Puzzle 3.png')
        grid = detect_grid_size(gray)
        df = process_starbattle_map(gray, grid)
        print("Automatic processing successful:")
        print(df)

    except Exception as e:
        print(f"Automatic processing failed: {e}")

    # Visualize the result
    plt.figure(figsize=(8, 8))
    plt.imshow(df.values, cmap='tab10', interpolation='nearest')
    plt.title('Starbattle Map Groups')
    plt.colorbar(label='Group ID')

    # Add grid lines
    for i in range(len(df.columns) + 1):
        plt.axvline(i - 0.5, color='black', linewidth=1)
    for i in range(len(df.index) + 1):
        plt.axhline(i - 0.5, color='black', linewidth=1)

    # Add text annotations
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            plt.text(j, i, str(df.iloc[i, j]), ha='center', va='center',
                     fontsize=12, fontweight='bold')

    plt.xticks(range(len(df.columns)))
    plt.yticks(range(len(df.index)))
    plt.show()

    return df


if __name__ == "__main__":
    result_df = main()