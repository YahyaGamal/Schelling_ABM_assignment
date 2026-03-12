import numpy as np
import random
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from IPython.display import Image
python_directory = os.getcwd()


class WolfSheepModel:

    def __init__(
        self,
        width=50,
        height=50,
        initial_sheep=100,
        initial_wolves=50,
        sheep_reproduce=4,
        wolf_reproduce=5,
        sheep_gain_from_food=4,
        wolf_gain_from_food=20,
        grass_regrowth_time=30
    ):

        self.width = width
        self.height = height

        self.sheep_reproduce = sheep_reproduce
        self.wolf_reproduce = wolf_reproduce
        self.sheep_gain_from_food = sheep_gain_from_food
        self.wolf_gain_from_food = wolf_gain_from_food
        self.grass_regrowth_time = grass_regrowth_time

        self.grid_grass = np.ones((height, width))
        self.countdown = np.zeros((height, width))

        self.sheep = []
        self.wolves = []

        ## Initialise the agents 
        ## (no class created for the sheep and wolves to keep the model simple as both have very similar behaviours)
        for _ in range(initial_sheep):
            self.sheep.append({
                "x": random.randrange(width),
                "y": random.randrange(height),
                "energy": random.randrange(2 * sheep_gain_from_food)
            })

        for _ in range(initial_wolves):
            self.wolves.append({
                "x": random.randrange(width),
                "y": random.randrange(height),
                "energy": random.randrange(2 * wolf_gain_from_food)
            })

    ## Randomly move the agents in space
    def move(self, agent):

        dx = random.choice([-1, 0, 1])
        dy = random.choice([-1, 0, 1])

        agent["x"] = (agent["x"] + dx) % self.width
        agent["y"] = (agent["y"] + dy) % self.height

    ## Increment the model by 1 time step (behaviours defined in this function for sheep and wolves)
    def step(self):

        new_sheep = []
        new_wolves = []

        random.shuffle(self.sheep)

        ## Manage the sheep
        for sheep in self.sheep:
            ## Move randomly in space
            self.move(sheep)
            ## Decrease energy by 1
            sheep["energy"] -= 1
            ## Extract the location of the sheep
            x, y = sheep["x"], sheep["y"]
            ## If the sheep finds grass, eat and increase energy
            if self.grid_grass[y, x] == 1:
                self.grid_grass[y, x] = 0
                self.countdown[y, x] = self.grass_regrowth_time
                sheep["energy"] += self.sheep_gain_from_food
            ## If the sheep survives, add it to the list of the sheep in the next time steps
            if sheep["energy"] >= 0:
                new_sheep.append(sheep)
            ## If the sheep will reproduce, generate a new sheep and add it to the list of sheeps in the next time step
            if random.random() * 100 < self.sheep_reproduce:
                sheep["energy"] /= 2
                new_sheep.append({
                    "x": sheep["x"],
                    "y": sheep["y"],
                    "energy": sheep["energy"]
                })
        ## Allocate the list of sheep in the next time step to the model
        self.sheep = new_sheep

        random.shuffle(self.wolves)
        ## Manage the wolves
        for wolf in self.wolves:
            ## Move randomly in space
            self.move(wolf)
            ## Decrease energy
            wolf["energy"] -= 1
            ## Iterate through all the sheep and check if any is in the current location of the predator
            prey = None
            for sheep in self.sheep:
                if sheep["x"] == wolf["x"] and sheep["y"] == wolf["y"]:
                    prey = sheep
                    break
            ## If a prey sheep is found, consume it
            if prey:
                self.sheep.remove(prey)
                wolf["energy"] += self.wolf_gain_from_food
            ## If the wolf is surviving, add it to the list of wolves in the next time step
            if wolf["energy"] >= 0:
                new_wolves.append(wolf)
            ## If the wolf will reproduce, generate a clone of it with half the energy
            if random.random() * 100 < self.wolf_reproduce:
                wolf["energy"] /= 2
                new_wolves.append({
                    "x": wolf["x"],
                    "y": wolf["y"],
                    "energy": wolf["energy"]
                })
        ## Assign the list of the wolves in the next time step to the model
        self.wolves = new_wolves
        ## Manage regrowing grass
        self.countdown[self.grid_grass == 0] -= 1
        regrow = self.countdown <= 0
        self.grid_grass[regrow] = 1

    def get_grid(self):

        grid = np.zeros((self.height, self.width))

        for y in range(self.height):
            for x in range(self.width):
                if self.grid_grass[y, x] == 1:
                    grid[y, x] = 1

        for sheep in self.sheep:
            grid[sheep["y"], sheep["x"]] = 2

        for wolf in self.wolves:
            grid[wolf["y"], wolf["x"]] = 3

        return grid

    def sheep_count(self):
        return len(self.sheep)

    def wolf_count(self):
        return len(self.wolves)


def run_predator_prey_model(
    steps=200,
    width=50,
    height=50,
    initial_sheep=100,
    initial_wolves=50,
    sheep_reproduce=4,
    wolf_reproduce=5,
    sheep_gain_from_food=4,
    wolf_gain_from_food=20,
    grass_regrowth_time=30
):

    model = WolfSheepModel(
        width=width,
        height=height,
        initial_sheep=initial_sheep,
        initial_wolves=initial_wolves,
        sheep_reproduce=sheep_reproduce,
        wolf_reproduce=wolf_reproduce,
        sheep_gain_from_food=sheep_gain_from_food,
        wolf_gain_from_food=wolf_gain_from_food,
        grass_regrowth_time=grass_regrowth_time,
    )

    fig, (ax_grid, ax_plot) = plt.subplots(1, 2, figsize=(12, 5))

    cmap = mcolors.ListedColormap(["brown", "green", "white", "black"])

    im = ax_grid.imshow(model.get_grid(), cmap=cmap, vmin=0, vmax=3)

    ax_grid.set_title("Wolf–Sheep Environment")
    ax_grid.axis("off")

    sheep_vals = []
    wolf_vals = []
    x_vals = []

    line_sheep, = ax_plot.plot([], [], label="Sheep")
    line_wolves, = ax_plot.plot([], [], label="Wolves")

    ax_plot.set_xlim(0, steps)
    ax_plot.set_ylim(0, max(initial_sheep * 3, initial_wolves * 3))

    ax_plot.set_xlabel("Step")
    ax_plot.set_ylabel("Population")
    ax_plot.set_title("Population Dynamics")

    ax_plot.legend()

    def update(frame):

        model.step()

        im.set_array(model.get_grid())

        x_vals.append(frame)
        sheep_vals.append(model.sheep_count())
        wolf_vals.append(model.wolf_count())

        line_sheep.set_data(x_vals, sheep_vals)
        line_wolves.set_data(x_vals, wolf_vals)

        return [im, line_sheep, line_wolves]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=steps,
        interval=50,
        repeat=False,
        blit=False
    )

    save_path = os.path.join(python_directory, "wolf_sheep_ani.gif")

    ani.save(save_path, writer="pillow")

    plt.close()

    return Image(filename=save_path)