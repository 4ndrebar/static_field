from typing import Tuple
import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

#TODO
#[ ] check for conductor intersections

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
        center = np.insert(center_, self.normal-1, 1)
        super().__init__(center)
        if not (isinstance(dimensions, tuple) and len(dimensions) == 2):
            raise TypeError("tuple_arg must be a tuple of two integers.")
        self.dimensions = dimensions


class StaticSolver:
    """
    A solver for the 3D static electric field
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
            self.V = np.zeros(shape=np.round(self.size / self.resolution).astype(int), dtype=float)
            self.conductor_pixels = np.zeros(
                shape=np.round(self.size / self.resolution).astype(int), dtype=bool
            )
            self.conductors = []

    def get_bbox(self, obj: Shape):
        if isinstance(obj, Cube):
            sidelen = int(max(obj.side / self.resolution, 1))
            bbox = np.full((sidelen, sidelen, sidelen), True, dtype=bool)

        elif isinstance(obj, Brick):
            dimensions = tuple(
                int(max(dim / self.resolution, 1)) for dim in obj.dimensions
            )
            bbox = np.full(dimensions, True, dtype=bool)

        elif isinstance(obj, Sphere):
            sidelen = int(max(2 * obj.radius / self.resolution, 1))
            bbox = np.zeros(
                (sidelen, sidelen, sidelen), dtype=bool
            )  # Initialize with False

            # Create a grid of coordinates
            center = sidelen // 2
            radius_voxel = obj.radius / self.resolution
            x, y, z = np.ogrid[:sidelen, :sidelen, :sidelen]
            distance = np.sqrt(
                (x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2
            )

            # Fill `True` for points inside the sphere
            bbox[distance <= radius_voxel] = True

        elif isinstance(obj, Slab):
            dimensions = tuple(
                int(max(dim / self.resolution, 1)) for dim in obj.dimensions
            )
            bbox2d = np.full(dimensions, True, dtype=bool)
            bbox = np.expand_dims(bbox2d, axis=obj.normal - 1)

        else:
            raise TypeError(f"Unsupported shape type: {type(obj).__name__}")

        return bbox

    def add_conductor(self, voltage:float , object:Shape, position: Tuple[float, float, float]):
        cond = Conductor(voltage, object)

        if not (isinstance(position, tuple) and len(position) == 3):
            raise TypeError("dimensions must be a tuple of three integers.")
        self.conductors.append(cond)
        self.add_object(cond.shape, position)
    
        bbox = self.get_bbox(object)
        center_scaled = object.center / self.resolution
        bbox_center = np.round(center_scaled).astype(int)

        position = np.array(position)
        position_scaled = position / self.resolution
        position_index = np.round(position_scaled).astype(int)
        # Compute slices
        start_index = [position_index[i] - bbox_center[i] for i in range(3)]
        end_index = [start_index[i] + bbox.shape[i] for i in range(3)]

        # Generate slices for both arrays
        large_slice = tuple(slice(start_index[i], end_index[i]) for i in range(3))
        small_slice = tuple(slice(None) for _ in range(3))  # Full range of small array
        self.V[large_slice] = np.where(
            bbox[small_slice], voltage, self.V[large_slice]
        )


    def add_object(self, object: Shape, position: Tuple[float, float, float]):
        """
        Adds an Insulator or a Conductor to the simulation
        """
        if not (isinstance(position, tuple) and len(position) == 3):
            raise TypeError("dimensions must be a tuple of three integers.")
        bbox = self.get_bbox(object)
        center_scaled = object.center / self.resolution
        bbox_center = np.round(center_scaled).astype(int)

        position = np.array(position)
        position_scaled = position / self.resolution
        position_index = np.round(position_scaled).astype(int)
        # Compute slices
        start_index = [position_index[i] - bbox_center[i] for i in range(3)]
        end_index = [start_index[i] + bbox.shape[i] for i in range(3)]

        # Generate slices for both arrays
        large_slice = tuple(slice(start_index[i], end_index[i]) for i in range(3))
        small_slice = tuple(slice(None) for _ in range(3))  # Full range of small array
        self.conductor_pixels[large_slice] = np.where(
            bbox[small_slice], 1, self.conductor_pixels[large_slice]
        )


# Function to plot slices along different axes
def plot_slices(array, slice_indices=None):
    fig, ax = plt.subplots()
    ax.imshow(array[:, :, slice_indices], cmap="viridis", vmin = 0, vmax = 10)
    ax.set_title(f"Slice {i}")
    ax.axis("off")


if __name__ == "__main__":
    solver = StaticSolver(simulation_size=(30, 30, 30))
    s = Sphere(2)
    c = Cube(3)
    sl = Slab(2, (10,10))
    for i in range(5):
        solver.add_conductor(3+i,s,(10+3*i,2+i**2,3+2*i))
    solver.get_bbox(s)
    solver.add_object(s, (10, 10, 10))
    solver.add_object(s, (11, 10, 23))
    solver.add_object(s, (11, 10, 27))
    solver.add_object(s, (18, 10, 13))
    solver.add_object(s, (21, 11, 10))
    solver.add_object(c, (13, 14, 15))
    solver.add_object(sl, (13,16, 15))
    
    for i in range(300):
        plot_slices(solver.V, i)
        plt.show()
