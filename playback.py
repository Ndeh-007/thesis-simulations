
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

# modify the plotter
matplotlib.rcParams["font.family"] = ["serif"]

# import the colors
from helpers import *


class PlayBack:
    def __init__(
        self, fp: list[tuple[str, str]], cns: list[str] | None, grid: list[int]
    ):
        self.folder_pairs = fp
        self.folder_plot_data: dict[str, dict] = curate_plot_data(self.folder_pairs)
        self.paused = False

        self.grid = grid
        self.case_names = (
            np.array(["" for _ in range(int(grid[0] * grid[1]))])
            .reshape((grid[0], grid[1]))
            .tolist()
        )
        self.folder_pair_gird = []
        for i in range(0, len(fp), 2):
            l = fp[i]
            r = fp[i + 1]
            self.folder_pair_gird.append([l, r])
        self.n_images = int(len(self.folder_plot_data))
        self.fig, self.axes = plt.subplots(grid[0], grid[1], figsize=(10, 10))
        # self.fig.subplots_adjust(hspace=20, left=20, right=40)

        if self.n_images == 1:
            self.axes = [self.axes]  # Ensure axes is a list for consistent indexing

        # force a two d grid
        if self.grid[1] == 1:
            self.axes = [[_ax] for _ax in self.axes]

        self.contours = []
        self.images = []

        z = 0
        for i in range(self.grid[0]):
            ctr = []
            imgs = []
            for j in range(self.grid[1]): 
                ax = self.axes[i][j]
                l = self.folder_pair_gird[i]
                set_dir = self.folder_pair_gird[i][j][0]

                opts = self.folder_plot_data[set_dir]

                ann_img = ax.imshow(
                    opts["annulus_images"][0],
                    aspect="equal",
                )
                ann_contour = ax.contour(
                    opts["annulus_streamlines_x"],
                    opts["annulus_streamlines_y"],
                    opts["annulus_streamlines"][0],
                    levels=10,
                    linewidths=(0.5,),
                    linestyles=("solid",),
                    colors="white",
                )

                ax.set_title(self.case_names[i][j])

                ax.set_xticks(
                    opts["ticks"]["annulus"]["x_ticks"],
                    labels=opts["ticks"]["annulus"]["x_ticks_labels"],
                )
                ax.set_yticks(
                    opts["ticks"]["annulus"]["y_ticks"],
                    labels=opts["ticks"]["annulus"]["y_ticks_labels"],
                )
                ctr.append(ann_contour)
                imgs.append(ann_img)


                legend_elements = [
                    Line2D([], [], color='white', label=f'Image {self.letters[z]}'),
                ]
                ax.legend(handles=legend_elements, loc="lower left", prop={'weight':'bold'})
                z += 1

            self.contours.append(ctr)
            self.images.append(imgs)

        # obtain the max number of frames
        self.n_frames = 10 # set a base number of frames
        for i in range(self.grid[0]):
            for j in range(self.grid[1]):
                l = len(
                    self.folder_plot_data[self.folder_pair_gird[i][j][0]][
                        "annulus_images"
                    ]
                )
                if self.n_frames < l:
                    self.n_frames = l

        # Create the animation
        # n_frames = len(folder_plot_data[folder_pairs[0][0]]['annulus_images'])  # Assuming equal frame count across all
        self.anim = FuncAnimation(
            self.fig, self.update, frames=self.n_frames, interval=50, blit=False
        )

        self.fig.canvas.mpl_connect("button_press_event", self.toggle_pause)

        legend_patches = [
            mpatches.Patch(color=color, label=name)
            for color, name in zip(COLORS, COLOR_NAMES)
        ]
        self.fig.legend(
            handles=legend_patches,
            loc="upper center",
            ncol=5,
            bbox_to_anchor=(0.5, 1.0),
        )

    # Define the update function for animation
    def update(self, frame):

        if frame + 1 == self.n_frames:
            self.toggle_pause()

        for i in range(self.grid[0]):
            for j in range(self.grid[1]):
                img = self.images[i][j]
                contour = self.contours[i][j]

                opts = self.folder_plot_data[self.folder_pair_gird[i][j][0]]

                if frame not in range(len(opts["annulus_images"])):
                    # skip areas where the frames are not consistent
                    continue

                # Update the image data
                img.set_data(opts["annulus_images"][frame])

                # Clear the old contours
                for c in contour.collections:
                    c.remove()

                # Add new contours
                new_contour = self.axes[i][j].contour(
                    opts["annulus_streamlines_x"],
                    opts["annulus_streamlines_y"],
                    opts["annulus_streamlines"][frame],
                    levels=10,
                    linewidths=(0.5,),
                    linestyles=("solid",),
                    colors="white",
                )
                self.contours[i][j] = new_contour  # Update the contour reference

        # return self.images + [
        #     c for contour in self.contours for c in contour.collections
        # ]

    def toggle_pause(self, *args, **kwargs):
        if self.paused:
            self.anim.pause()
        else:
            self.anim.resume()

        self.paused = not self.paused

