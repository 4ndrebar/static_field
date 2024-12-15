from typing import Tuple
import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, generate_binary_structure
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import copy
import ipywidgets as widgets
from IPython.display import display, clear_output



# TODO
# [ ] check for conductor intersections
# [ ] add insulators
# [ ] add von neumann BC for insulators


class Conductor:
    def __init__(self, V: float, object):
        self.V = V
        if not isinstance(object, Shape):
            raise TypeError("The shape must be a Shape object")
        self.shape = object


class Insulator:
    pass


class Shape(ABC):
    def __init__(self, center=np.zeros(3)) -> None:
        self.center = np.array(center)
        self.bbox = None


class Cube(Shape):
    def __init__(self, side: int = 1):
        center = (side / 2, side / 2, side / 2)
        super().__init__(center)
        self.side = side


class Brick(Shape):
    def __init__(self, dimensions: Tuple[float, float, float]):
        center = np.arrray(dimensions) / 2
        super().__init__(center)
        if not (isinstance(dimensions, tuple) and len(dimensions) == 3):
            raise TypeError("dimensions must be a tuple of three integers.")
        self.dimensions = dimensions


class Sphere(Shape):
    def __init__(self, radius: float):
        center = radius * np.ones(3)
        super().__init__(center)
        self.radius = radius


class Slab(Shape):
    def __init__(self, orientation_normal: int, dimensions: Tuple[float, float]):
        if orientation_normal not in [1, 2, 3]:
            raise ValueError("The normal direction should be a value in 1,2,3")
        self.normal = orientation_normal
        center_ = np.array(dimensions) / 2
        center = np.insert(center_, self.normal - 1, 1)
        super().__init__(center)
        if not (isinstance(dimensions, tuple) and len(dimensions) == 2):
            raise TypeError("tuple_arg must be a tuple of two integers.")
        self.dimensions = dimensions


class StaticSolver:
    """
    StaticSolver class for simulating and visualizing electrostatic potentials and fields.

    Attributes:
        size (np.ndarray): Physical dimensions of the simulation grid (x, y, z) in units.
        resolution (float): Spatial resolution of the simulation grid.
        V (np.ndarray): Potential grid initialized with zeros.
        conductor_pixels (np.ndarray): Boolean array indicating conductor locations.
        conductors (list): List of Conductor objects in the simulation.
        kern3d (np.ndarray): 3D convolution kernel for simulating potential evolution.
        kern2d (np.ndarray): 2D convolution kernel for simulating slice potential evolution.
        V_evolved (np.ndarray): Evolved potential grid after simulation.
        err (list): Error convergence history from the simulation.

    Methods:
        __init__(simulation_size, resolution=0.1):
            Initializes the simulation grid, resolution, and convolution kernels.

        evolve(iterations=2000, sigma=3, etol=1e-5):
            Simulates potential evolution over the 3D grid.

        evolve_slice(z_slice, iterations=2000, sigma=3, etol=1e-5):
            Simulates potential evolution for a 2D slice at a given z-coordinate.

        get_bbox(obj):
            Computes the bounding box for a given Shape object.

        _validate_tuple(obj, length, dtype, name):
            Validates the structure of a tuple parameter.

        _insert_shape(shape, position, array, value=None):
            Inserts a Shape object into a grid array with optional voltage value.

        add_conductor(voltage, shape, position):
            Adds a Conductor object with specified voltage and position.

        viz_slice(z_slice=None, levels=40, plot_err=True, quiver_scale=(0.1, 1.0)):
            Visualizes a 2D slice of the evolved potential and its gradient.
    """

    def __init__(self, simulation_size: Tuple[int, int, int], resolution=0.1):
        if not (
            isinstance(simulation_size, tuple)
            and len(simulation_size) == 3
            and all(isinstance(i, int) for i in simulation_size)
        ):
            raise TypeError("tuple_arg must be a tuple of three integers.")
        for size in simulation_size:
            if size < resolution:
                raise ValueError(
                    f"The size is smaller than the resolution ({resolution})"
                )
        else:
            self.size = np.array(simulation_size)
            self.resolution = resolution
            self.V = np.zeros(
                shape=np.round(self.size / self.resolution).astype(int),
                dtype=np.float32,
            )
            self.conductor_pixels = np.zeros(
                shape=np.round(self.size / self.resolution).astype(int), dtype=np.bool_
            )
            self.conductors = []
            kern = generate_binary_structure(3, 1).astype(float) / 6
            kern[1, 1, 1] = 0
            self.kern3d = kern
            kern2d = np.array(
                [[0.0, 1 / 4, 0.0], [1 / 4, 0.0, 1 / 4], [0.0, 1 / 4, 0.0]]
            )
            self.kern2d = kern2d

    def evolve(self, iterations=2000, sigma=None, etol=1e-5, rtol=1e-5):
        """
        Evolve the entire 3D grid over a specified number of iterations.

        Parameters:
        iterations: int, maximum number of iterations to run.
        sigma: float, standard deviation for Gaussian smoothing.
        etol: float, tolerance for early stopping based on error convergence.
        rtol: float, relative tolerance for early stopping.
        """
        # Optionally smooth the initial grid
        if sigma is not None:
            self.V = gaussian_filter(self.V, sigma=sigma, mode="constant")

        # Preallocate error history
        self.err = np.zeros(iterations)
        self.rerr = np.zeros(iterations)

        # Preallocate temporary array
        V_ = np.zeros_like(self.V, dtype=self.V.dtype)

        grid_size = self.V.size  # For efficient error calculations

        for i in tqdm(range(iterations)):
            # Perform convolution
            convolve(self.V, self.kern3d, output=V_, mode="constant")

            # Enforce boundary conditions
            V_[self.conductor_pixels] = self.V[self.conductor_pixels]

            # Compute errors
            diff_sum = np.sum((self.V - V_) ** 2)
            grid_sum = np.sum(self.V**2)
            abs_error = diff_sum / grid_size
            rel_error = abs_error / (grid_sum / grid_size)

            # Record errors
            self.err[i] = abs_error
            self.rerr[i] = rel_error

            # Check convergence criteria
            if abs_error < etol or rel_error < rtol:
                self.err = self.err[: i + 1]
                self.rerr = self.rerr[: i + 1]
                break

            # Swap V and V_ without deep copying
            self.V, V_ = V_, self.V

        # Save the final evolved grid
        self.V_evolved = self.V

    def evolve_slice(
        self, z_slice: float, iterations=2000, sigma=None, etol=1e-5, rtol=1e-5
    ):
        z_index = round(z_slice / self.resolution)

        # Extract slice without unnecessary copies
        grid = self.V[:, :, z_index].copy()
        conductor_pixels_slice = self.conductor_pixels[:, :, z_index]

        # Optionally smooth the initial slice
        if sigma is not None:
            grid = gaussian_filter(grid, sigma=sigma, mode="constant")

        # Preallocate error history
        self.err = np.zeros(iterations)
        self.rerr = np.zeros(iterations)

        # Preallocate temporary arrays
        V_ = np.zeros_like(grid, dtype=grid.dtype)

        grid_size = grid.size  # For efficiency in error calculation

        for i in tqdm(range(iterations)):
            # Perform convolution
            convolve(grid, self.kern2d, output=V_, mode="constant")

            # Enforce boundary conditions
            V_[conductor_pixels_slice] = grid[conductor_pixels_slice]

            # Compute errors
            diff_sum = np.sum((grid - V_) ** 2)
            grid_sum = np.sum(grid**2)
            abs_error = diff_sum / grid_size
            rel_error = abs_error / (grid_sum / grid_size)

            # Record errors
            self.err[i] = abs_error
            self.rerr[i] = rel_error

            # Check convergence criteria
            if abs_error < etol or rel_error < rtol:
                self.err = self.err[: i + 1]
                self.rerr = self.rerr[: i + 1]
                break

            # Swap grid and V_ without deep copying
            grid, V_ = V_, grid

        # Save the final evolved slice
        self.V_evolved = grid

    def get_bbox(self, obj: Shape):
        shape_type = type(obj).__name__

        if isinstance(obj, Cube):
            dimensions = (obj.side, obj.side, obj.side)
        elif isinstance(obj, Brick):
            dimensions = obj.dimensions
        elif isinstance(obj, Sphere):
            dimensions = (2 * obj.radius,) * 3
        elif isinstance(obj, Slab):
            dimensions_2d = obj.dimensions
            dimensions = np.insert(dimensions_2d, obj.normal - 1, 1)
        else:
            raise TypeError(f"Unsupported shape type: {shape_type}")

        scaled_dimensions = tuple(
            int(max(dim / self.resolution, 1)) for dim in dimensions
        )
        bbox = np.zeros(scaled_dimensions, dtype=bool)

        if isinstance(obj, Sphere):
            center = scaled_dimensions[0] // 2
            radius_voxel = obj.radius / self.resolution
            x, y, z = np.ogrid[
                : scaled_dimensions[0], : scaled_dimensions[1], : scaled_dimensions[2]
            ]
            distance = np.sqrt(
                (x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2
            )
            bbox[distance <= radius_voxel] = True
        else:
            bbox.fill(True)

        return bbox

    def _validate_tuple(self, obj, length, dtype, name):
        if not (
            isinstance(obj, tuple)
            and len(obj) == length
            and all(isinstance(i, dtype) for i in obj)
        ):
            raise TypeError(f"{name} must be a tuple of {length} {dtype.__name__}s.")

    def _insert_shape(
        self, shape: Shape, position: Tuple[float, float, float], array, value=None
    ):
        self._validate_tuple(position, 3, (int, float), "position")

        bbox = self.get_bbox(shape)
        bbox_center = np.round(np.array(shape.center) / self.resolution).astype(int)
        position_index = np.round(np.array(position) / self.resolution).astype(int)

        # Compute slices
        start_index = position_index - bbox_center
        end_index = start_index + np.array(bbox.shape)
        large_slice = tuple(slice(start_index[i], end_index[i]) for i in range(3))

        if value is not None:
            array[large_slice] = np.where(bbox, value, array[large_slice])
        else:
            array[large_slice] = np.logical_or(array[large_slice], bbox)

    def add_conductor(
        self, voltage: float, shape: Shape, position: Tuple[float, float, float]
    ):
        self.conductors.append(Conductor(voltage, shape))
        self._insert_shape(shape, position, self.V, value=voltage)
        self._insert_shape(shape, position, self.conductor_pixels)

    def viz_slice(
        self, z_slice=None, levels=40, plot_err=True, quiver_scale=(0.1, 1.0)
    ):
        """
        Visualize a 2D slice of the evolved potential and its gradient.

        Parameters:
          z_slice: float, the z-coordinate of the slice to plot (optional for 3D systems).
          levels: int, the number of contour levels for the potential.
          plot_err: bool, whether to plot the error log.
          quiver_scale: tuple, (min_scale, max_scale) to control quiver arrow lengths.
        """
        # Handle 2D vs 3D arrays
        if len(np.shape(self.V_evolved)) == 2:
            array = self.V_evolved[:, :, np.newaxis]  # Convert 2D to pseudo-3D
            z_index = 0  # Only one slice available
        else:
            array = self.V_evolved
            z_index = round(z_slice / self.resolution)

        # Create physical coordinates grid
        x, y, _ = [np.arange(0, size, self.resolution) for size in self.size]
        xx, yy = np.meshgrid(x, y)  # Grid for plotting

        # Extract and transpose the slice for correct orientation
        array_slice = array[:, :, z_index].T

        # --- First Plot: Potential (Imshow + Contours) ---
        _, ax1 = plt.subplots(figsize=(8, 6))
        im = ax1.imshow(
            array_slice,
            extent=[x[0], x[-1], y[0], y[-1]],
            cmap="plasma",  # Vibrant colormap
            origin="lower",
            aspect="equal",
        )
        CS = ax1.contour(
            xx,
            yy,
            array_slice,
            levels=levels,
            colors="black",
            linewidths=0.7,
            alpha=0.8,
        )
        ax1.clabel(CS, CS.levels, inline=True, fontsize=6, fmt="%.1f")
        ax1.set_title(f"Electric Potential Slice at z = {z_slice}", fontsize=12)
        ax1.set_xlabel("$x$")
        ax1.set_ylabel("$y$")
        plt.colorbar(im, ax=ax1, label="Potential (V)", shrink=0.8)
        plt.tight_layout()

        # --- Second Plot: Gradient Map with Quiver ---
        _, ax2 = plt.subplots(figsize=(8, 6))

        # Compute gradient (negative for electric field)
        E = np.gradient(-array_slice, self.resolution, self.resolution)
        Ey, Ex = E
        # Compute the magnitude of the gradient
        E_magnitude = np.sqrt(Ex**2 + Ey**2)

        # Normalize and scale the gradient vectors
        max_magnitude = E_magnitude.max()
        scale_factor = (quiver_scale[1] - quiver_scale[0]) / max_magnitude
        Ey_norm, Ex_norm = [Ei * scale_factor for Ei in E]

        # Background potential map
        im2 = ax2.imshow(
            array_slice,
            extent=[x[0], x[-1], y[0], y[-1]],
            cmap="coolwarm",
            origin="lower",
            aspect="equal",
            alpha=0.6,
        )
        plt.colorbar(im2, ax=ax2, label="Potential (V)", shrink=0.8)

        # Quiver plot with scaled arrow lengths
        ax2.quiver(
            xx, yy, Ex_norm, Ey_norm, color="black", width=0.002, headwidth=3, alpha=0.8
        )
        ax2.set_title(f"Electric Field Gradient Slice at z = {z_slice}", fontsize=12)
        ax2.set_xlabel("$x$")
        ax2.set_ylabel("$y$")
        ax2.set_aspect("equal")

        # --- Third Plot : Plot of electric field magnitude ---
        _, ax3 = plt.subplots(figsize=(8, 6))
        im3 = ax3.imshow(
            E_magnitude,
            extent=[x[0], x[-1], y[0], y[-1]],
            cmap="plasma",  # Vibrant colormap
            origin="lower",
            aspect="equal",
        )
        plt.colorbar(im3, ax=ax3, label="Electric field (V/m)", shrink=0.8)

        ax3.set_title(f"Electric Field Magnitude Slice at z = {z_slice}", fontsize=12)
        ax3.set_xlabel("$x$")
        ax3.set_ylabel("$y$")
        ax3.set_aspect("equal")

        # --- Fourth Plot (Optional): Error Log ---
        if plot_err:
            _, ax4 = plt.subplots(figsize=(8, 6))
            ax4.semilogy(np.arange(len(self.err)), self.err, label="absolute error")
            ax4.semilogy(np.arange(len(self.rerr)), self.rerr, label="relative error")
            ax4.set_title("Error Log")
            ax4.set_xlabel("Epoch")
            ax4.set_ylabel("Error")
            ax4.legend()
            plt.tight_layout()

        # Show all plots
        plt.show()

    def interactive_viz(self, z_slice=None, levels=40):
        """
        Method to visualize a 2D slice of the evolved potential and its gradient interactively.
        
        Parameters:
        z_slice: float, the z-coordinate of the slice to plot (optional for 3D systems).
        levels: int, the number of contour levels for the potential.
        """
        def interactive_plot(z_slice):
            # Clear the previous plot before drawing the new one
            clear_output(wait=True)

            # Visualize the slice with the updated z_slice value
            self.viz_slice(z_slice=z_slice, levels=levels, plot_err=False)
            plt.show()

        # Create the slider for z_slice
        slider = widgets.interactive(interactive_plot, z_slice=(0, self.size[2]-1, 1))
        display(slider)


if __name__ == "__main__":
    resolution = 0.1
    solver = StaticSolver(simulation_size=(30, 30, 30), resolution=resolution)
    s = Sphere(2)
    c = Cube(3)
    sl = Slab(1, (10, 10))
    ss = Sphere(1)
    solver.add_conductor(5, s, (20, 15, 15))
    solver.add_conductor(13, sl, (15, 23, 15))
    solver.add_conductor(15, s, (10, 10, 15))
    solver.add_conductor(-10, c, (5, 5, 15))
    sl = Slab(2, (10, 10))
    solver.add_conductor(-12, sl, (20, 10, 15))
    z_slice = 15
    # solver.evolve_slice(z_slice=z_slice, iterations=1000)
    solver.evolve()
    solver.interactive_viz(z_slice=z_slice)
    plt.show()
