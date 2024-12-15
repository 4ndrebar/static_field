# Static Field Simulator

## Overview
The Static Field Simulator is a Python-based simulation tool designed to model and evolve a static electric field in a 2D grid. The system applies numerical methods such as convolution and Gaussian filtering to simulate the evolution of electric potentials and fields within a given space. It is particularly useful for exploring how static fields evolve under boundary conditions and other constraints.

## Features
- Simulate the evolution of a static field over multiple iterations.
- Convolution and Gaussian filtering for smoothing and processing.
- Support for boundary conditions (e.g., conductors) in the simulation.
- Early stopping criteria based on error convergence.
- Efficient handling of 2D slices of 3D data for faster processing.

## Installation

### Prerequisites
- Python 3.x
- NumPy
- SciPy
- Matplotlib (for visualization, optional)

You can install the required dependencies using `pip`:

```
pip install numpy scipy matplotlib
```

### Clone the repository

```
git clone https://github.com/4ndrebar/static_field
cd static-field-simulator
```

## Usage

### Basic Setup

1. Create an instance of the `StaticFieldSolver` class with the necessary parameters such as grid size and resolution.
2. Add conductors in the simulation region and assign them a voltage.
3a. Call the `evolve_slice()` method to simulate the evolution of the static field for a given slice.
3b. Call the `evolve()` method to simulate the evolution of the static field for the entire simulation region.

### Example Usage

```python
from StaticSimulator import *
# Visualize or analyze the evolved field (optional)
import matplotlib.pyplot as plt
plt.imshow(solver.V_evolved)
plt.colorbar()
plt.show()

# Initialize the solver with grid size, resolution, and boundary conditions
solver = StaticSolver(simulation_size=(30, 30, 30), resolution=resolution)
s = Sphere(2)
c = Cube(3)
sl = Slab(1, (10, 10))
ss = Sphere(1)
solver.add_conductor(5, s, (20, 15, 15))
solver.add_conductor(15, s, (10, 10, 15))
solver.add_conductor(13, sl, (15, 23, 15))
solver.add_conductor(-10, c, (5, 5, 15))
z_slice = 15
# Evolve the entire grid and visuzlize the field at a specific z-coordinate
solver.evolve()
solver.interactive_viz(z_slice=z_slice)
plt.show()
```

### Parameters for `evolve_slice()`:
- **z_slice**: The z-coordinate of the slice to evolve (in grid units).
- **iterations**: Maximum number of iterations to run.
- **sigma**: Standard deviation for Gaussian smoothing (optional).
- **etol**: Absolute error tolerance for early stopping.
- **rtol**: Relative error tolerance for early stopping.

## Methods

### `StaticFieldSolver.evolve_slice()`
Evolve a single slice of the static field along the z-axis for a specified number of iterations. 
This method uses convolution for smoothing and applies boundary conditions for conductor pixels.

### `StaticFieldSolver.evolve()`
Evolve the entire 3D field for a specified number of iterations.

## Performance Considerations
The performance of the simulation depends on the size of the grid and the number of iterations. 
The `evolve_slice()` method allows you to simulate a single slice, which is useful for debugging or analyzing specific layers of the field.
For 3D simulations, evolving multiple slices can be slow, so consider optimizations or parallelizing the process for larger grids.

## Contributing

If you'd like to contribute to the project, feel free to fork the repository, make your changes, and create a pull request. Ensure that your code adheres to the existing coding standards and includes appropriate tests for any new functionality.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
