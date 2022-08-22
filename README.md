# pygeo3d

## About pygeo3d

pygeo3d is a vasrt mathematical module providing (real) 3-dimensional objects, i.e. Points, Lines and Planes as Python objects through classes.

*Date of creation:* `April 22, 2021` \
*Date of first release on [PyPI](https://pypi.org/):* `May 06, 2021`

Using this module, 3-dimensional geometry can be easily executed in Python. Along with providing calulative tools, it also allows a user to visualize Points, Lines and Planes on a 3D plot, and can be used as a learning tool to enhance a student's 3D visualization capabilities. The module is enriched with help-text and examples.

## Included classes and functions

To access the help-text of a class to know more about it, run:

```python
help(pygeo3d.classname)
```

### About the class `Point`

`Point(x, y, z)` represents the position vector of a Point `(x, y, z)` in 3-dimensional Cartesian coordinate space. The difference of two Points gives a Vector, while the addition of a Vector to a Point gives another Point.

### About the class `Line`

`Line(a, b)` represents the vector equation of a Line `r = a + Kb` in 3-dimensional Cartesian coordinate space. A Line can be indexed/sliced by real numbers to get the Point(s) that lie on the line for the given values of parameter k. Membership testing on a Line allowes a user to check if a Point lies on it.

### About the class `Plane`

`Plane(r, n)` represents the vector equation of a Plane `r • n = d` in 3-dimensional Cartesian coordinate space. Membership testing on a Plane allows a user to check if a Point/Line lies on it.

All classes have various other methods, and even alternate constructors. Users are advised to look at the documentation for more efficient usage.

### Some functions

A variety of functions have been provided inn the module, which include functions to calculate/check:
- Euclidean distance between Points, Lines, and Planes
- Angle between Lines and Planes
- Intersections of Lines and Planes
- Images and Projections of Points and Lines in/on Lines/Planes
- If a list of Points are collinear/coplanar
- If some Lines or Planes are parallel or perpendicular

And so on. Users are advised to run the following command in Python to get to know about all the available functions:

```python
dir(pygeo3d)
```

## Update History

### Updates (0.0.5)

- Minor bug fixes
- Alternate constructor `Point.FromSequence(seq)` for `Point` can now accept generators/generator-expressions as argument
- More colors available for plotting 

### Updates (0.0.6)

`Line` objects now support slicing, which returns a `tuple` of `Points` on the Line at the indices given in the slice

### Updates (0.0.7)

Minor bug fixes

### Updates (0.0.8)

- Minor bug fixes: Fixed more issues of floating point precision
- Removed the function `plot_random_objects()`
- Better type hinting

## Footnotes

This project is dependent upon the following of my own modules, available through *PyPI*:
- `pyvectors`
- `linear_equations`

## Run

To use, execute:

```
pip install pygeo3d
```

Import this file in your project, whenever needed, using:

```python
import pygeo3d
```
