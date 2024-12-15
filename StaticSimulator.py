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

        scaled_dimensions = tuple(int(max(dim / self.resolution, 1)) for dim in dimensions)
        bbox = np.zeros(scaled_dimensions, dtype=bool)

        if isinstance(obj, Sphere):
            center = scaled_dimensions[0] // 2
            radius_voxel = obj.radius / self.resolution
            x, y, z = np.ogrid[:scaled_dimensions[0], :scaled_dimensions[1], :scaled_dimensions[2]]
            distance = np.sqrt((x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2)
            bbox[distance <= radius_voxel] = True
        else:
            bbox.fill(True)

        return bbox





    def _validate_tuple(obj, length, dtype, name):
        if not (isinstance(obj, tuple) and len(obj) == length and all(isinstance(i, dtype) for i in obj)):
            raise TypeError(f"{name} must be a tuple of {length} {dtype.__name__}s.")



    def _insert_shape(self, shape: Shape, position: Tuple[float, float, float], array, value=None):
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


    def add_object(self, shape: Shape, position: Tuple[float, float, float]):
        self._insert_shape(shape, position, self.conductor_pixels)

    def add_conductor(self, voltage: float, shape: Shape, position: Tuple[float, float, float]):
        self.conductors.append(Conductor(voltage, shape))
        self._insert_shape(shape, position, self.V, value=voltage)



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
