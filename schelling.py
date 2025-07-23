import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from IPython.display import Image
import random
import os

class SchellingModel:
    def __init__(self, width=20, height=20, density=0.9, similarity_threshold=0.5):
        self.width = width
        self.height = height
        self.density = density
        self.similarity_threshold = similarity_threshold

        # 0 = empty, 1 = group A (blue), 2 = group B (orange)
        self.EMPTY = 0
        self.GROUP_A = 1
        self.GROUP_B = 2

        self.grid = np.zeros((self.height, self.width), dtype=int)
        self.init_agents()

    def init_agents(self):
        """
        Initialise the agents
        """
        for y in range(self.height):
            for x in range(self.width):
                if random.random() < self.density:
                    self.grid[y, x] = random.choice([self.GROUP_A, self.GROUP_B])
                else:
                    self.grid[y, x] = self.EMPTY

    def is_happy(self, x, y):
        """
        Checks if the agent is happy or not.

        Parameters
        ----------

        return
        ------
        """
        agent = self.grid[y, x]
        if agent == self.EMPTY:
            return True  # empty cells are trivially happy

        neighbors = self.get_neighbors(x, y)
        if not neighbors:
            return True  # no neighbors to compare to

        similar = sum(1 for n in neighbors if n == agent)
        return (similar / len(neighbors)) >= self.similarity_threshold

    def get_neighbors(self, x, y):
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    n = self.grid[ny, nx]
                    if n != self.EMPTY:
                        neighbors.append(n)
        return neighbors

    def step(self):
        unhappy_agents = [(x, y) for y in range(self.height)
                          for x in range(self.width)
                          if self.grid[y, x] != self.EMPTY and not self.is_happy(x, y)]

        random.shuffle(unhappy_agents)
        empty_spaces = [(x, y) for y in range(self.height)
                        for x in range(self.width)
                        if self.grid[y, x] == self.EMPTY]
        random.shuffle(empty_spaces)

        for (x, y) in unhappy_agents:
            if not empty_spaces:
                break
            new_x, new_y = empty_spaces.pop()
            self.grid[new_y, new_x] = self.grid[y, x]
            self.grid[y, x] = self.EMPTY

    def percent_unhappy(self):
        agents = [(x, y) for y in range(self.height)
                  for x in range(self.width)
                  if self.grid[y, x] != self.EMPTY]
        if not agents:
            return 0
        unhappy = [1 for x, y in agents if not self.is_happy(x, y)]
        return 100 * sum(unhappy) / len(agents)

    def percent_similar(self):
        agents = [(x, y) for y in range(self.height)
                  for x in range(self.width)
                  if self.grid[y, x] != self.EMPTY]
        if not agents:
            return 0
        total_similar = 0
        total_neighbors = 0
        for x, y in agents:
            neighbors = self.get_neighbors(x, y)
            similar = sum(1 for n in neighbors if n == self.grid[y, x])
            total_similar += similar
            total_neighbors += len(neighbors)
        if total_neighbors == 0:
            return 100
        return 100 * total_similar / total_neighbors
    
def run_schelling_model(
        steps:int=30,
        width:int=30,
        height:int=30,
        density:float=0.9,
        similarity_threshold:float=0.5
):
    rf"""
    Runs the Schelling Agent Based Model (ABM) and saves the animation png as `\schelling_ani.png`

    Parameters
    ----------
    `steps`: int, default=30
        the number of increments the simulation runs
    `width`: int, default=30
        the width of the square grid spatial context
    `height`: int, default=30
        the height of the square grid spatial context
    `density`: float, default=0.9
        the ratio of the occupied to the unoccupied square grid spatial units
    `similarity_threshold`: float, default=0.5
        the threshold ratio of surrounding similar to different agents around an agent "a" (similar and different are relative to "a")

    Returns
    -------
    A png image displayable in a Jupyter notebook.
    """

    # Find python directory
    python_directory = os.getcwd()

    # Create model
    model = SchellingModel(width=width, height=height, density=density, similarity_threshold=similarity_threshold)

    fig, ax = plt.subplots()

    # Define colors
    cmap = mcolors.ListedColormap(['white', 'blue', 'orange'])
    im = ax.imshow(model.grid, cmap=cmap, vmin=0, vmax=2)

    # Run the model and save the animation
    def update(frame):
        model.step()
        im.set_array(model.grid)
        ax.set_title(f"Step {frame+1}\n% Similar: {model.percent_similar():.2f} | % Unhappy: {model.percent_unhappy():.2f}")
        # im = ax.imshow(model.grid, cmap=cmap, vmin=0, vmax=2)
        fig.canvas.draw_idle()
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=steps, interval=800, repeat=False, blit=False)
    ani.save(rf"{python_directory}/schelling_ani.gif")
    plt.close()

    # Required to show animation in notebook
    return Image(filename=rf"{python_directory}/schelling_ani.gif")