from typing import Tuple
import numpy as np
from abc import ABC, abstractmethod


class Conductor:
    def __init__(self, V: float, shape):
        self.V = V
        if not isinstance(shape, Shape):
            raise TypeError("The shape must be a Shape object")
        self.shape = shape


class Insulator:
    pass


class Shape(ABC):
    def __init__(self) -> None:
        self.center = np.zeros(2)
        self.bbox = None


class Cube(Shape):
    def __init__(self, side: int = 1):
        super().__init__()
        self.side = side


class Brick(Shape):
    def __init__(self, dimensions: Tuple[float, float, float]):
        super().__init__()
        if not (isinstance(dimensions, tuple) and len(dimensions) == 3):
            raise TypeError("tuple_arg must be a tuple of three integers.")
        self.dimensions = dimensions


class Sphere(Shape):
    def __init__(self, radius: float):
        super().__init__()
        self.radius = radius


class Slab(Shape):
    def __init__(self, orientation_normal: int, dimensions: Tuple[float, float]):
        super().__init__()
        if not (isinstance(dimensions, tuple) and len(dimensions) == 2):
            raise TypeError("tuple_arg must be a tuple of two integers.")
        self.dimensions = dimensions
        if orientation_normal not in [1,2,3]:
            raise ValueError("The normal direction should be a value in 1,2,3")
        self.normal = orientation_normal


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
        else:
            self.size = simulation_size
            self.resolution = resolution
            self.V = np.zeros(shape=simulation_size)

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
            bbox = np.expand_dims(bbox2d, axis=obj.normal-1)

        else:
            raise TypeError(f"Unsupported shape type: {type(obj).__name__}")

        return bbox

    def add_object(self, object: Shape):
        """
        Adds an Insulator or a Conductor to the simulation
        """


if __name__ == "__main__":
    solver = StaticSolver(simulation_size=(30, 30, 30))
    s = Slab(2,(20,10))
    solver.get_bbox(s)
