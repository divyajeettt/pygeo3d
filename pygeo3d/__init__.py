"""pygeo3d - 3-dimensional Geometry
Defines classes Point, Line and Plane in ℝ³ Cartesian Co-ordinate system
and provides many functions for manipulation of 3D - objects, i.e.
Points, Lines and Planes"""


from pyvectors import Vector
from types import GeneratorType
import linear_equations as eqns
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import numpy as np
import math


################################################################################


class Point:
    """Point(x, y, z) -> (x)î + (y)ĵ + (z)k̂
    represents the Position Vector of a Point in 3D space

    see help(__init__) for help on Point creation
    Truth Value of Points:
        bool(Origin) == False"""


    def __init__(self, x: float, y: float, z: float, /) -> None:
        """initializes Point instance
        x, y, z are Real Numbers which represent co-ordinates of required Point
        attrs of Point Object:
            • self.x = x
            • self.y = y
            • self.z = z
            • self.posn_vector = Vector(x, y, z)"""

        point = [x, y, z]

        for i in range(3):
            if not isinstance(point[i], int|float):
                raise TypeError(
                    "invalid component type for 'Point': must be 'int' / 'float'"
                )
            if str(abs(point[i])) in {"inf", "nan"}:
                raise ValueError(f"invalid component for 'Point': '{point[i]}'")

            if float(point[i]).is_integer():
                point[i] = int(point[i])

        self.tuple = self.x, self.y, self.z = tuple(point)
        self.posn_vector = Vector.FromSequence(self.tuple)


    def __str__(self, /) -> str:
        """defines the str() method for Point"""

        return str(self.posn_vector)


    def __repr__(self, /) -> str:
        """defines the repr() method for Point"""

        return f"Point({self.x}, {self.y}, {self.z})"


    def __bool__(self, /) -> bool:
        """defines the truth value of a Point
        bool(Point(0, 0, 0)) == False"""

        return set(self.tuple) != {0}


    def __hash__(self, /) -> int:
        """defines the hash value of a Point"""

        return hash(repr(self))


    def __getitem__(self, axis: str) -> float:
        """defines indexing property for Point
        indices defined here:
          'x' = X-component
          'y' = Y-component
          'z' = Z-component"""

        if not isinstance(axis, str):
            raise TypeError("'Point' indices must be 'str'")

        if axis not in {"x", "y", "z"}:
            raise IndexError(" ".join((
                "invalid index for 'Point':",
                "see help(__getitem__) for help on valid indices"
            )))
        else:
            return {"x": self.x, "y": self.y, "z": self.z}[axis]


    def __round__(self, digits: int|None = 0) -> 'Point':
        """defines the round() method for Point
        round each component of the Point to given number of digits"""

        result = [round(self.tuple[i], digits) for i in range(3)]
        return Point.FromSequence(result)


    def __neg__(self, /) -> 'Point':
        """defines the negative of Point using unary '-' operator
        -Point(x, y, z) == Point(-x, -y, -z)"""

        return Point(-self.x, -self.y, -self.z)


    def __abs__(self, /) -> float:
        """defines the abs() method for Point
        returns the distance of the Point from the Origin (0, 0, 0)"""

        return self.posn_vector.mod()


    def __eq__(self, other: 'Point', /) -> bool:
        """defines the equality of Point Objects using '==' operator"""

        if type(self) is not type(other):
            return False
        else:
            return self.tuple == other.tuple


    def __add__(self, other: Vector, /) -> 'Point':
        """sum of a Point Object with a Vector Object using '+' operator
        returns a Point (self + other)"""

        if not isinstance(other, Vector):
            raise TypeError("'+' for Point requires Vector as right operand")
        else:
            return Point.FromSequence(
                [self.tuple[i] + other.tuple[i] for i in range(3)]
            )


    def __sub__(self, other: 'Point', /) -> Vector:
        """difference of two Point Objects using '-' operator
        returns a Vector (self - other)"""

        return Vector.FromSequence(
            [self.tuple[i] - other.tuple[i] for i in range(3)]
        )


    @classmethod
    def FromSequence(
            cls, sequence: dict|list|tuple|GeneratorType, /
        ) -> 'Point':
        """creates a Point from the given sequence
        the sequence must have len(sequence) == 3 / the generator must yield 3
        int / float numbers
        valid sequences defined here:
            • (x, y, z)
            • [x, y, z]
            • {'x': x, 'y': y, 'z': z}"""

        if not isinstance(sequence, dict|list|tuple|GeneratorType):
            raise TypeError(" ".join((
                "invalid sequence type for 'Point':",
                "see help(FromSequence) for help on valid sequences"
            )))
        if isinstance(sequence, GeneratorType):
            sequence = tuple(sequence)

        if len(sequence) != 3:
            raise ValueError(" ".join((
                "len of sequence must be 3:",
                "see help(FromSequence) for help on valid sequences"
            )))
        if isinstance(sequence, dict):
            if set(sequence.keys()) != {"x", "y", "z"}:
                raise KeyError(" ".join((
                    "invalid keys for 'Point':",
                    "see help(FromSequence) for help on valid keys"
                )))
            else:
                sequence = sequence["x"], sequence["y"], sequence["z"]
        return cls(*sequence)


    @classmethod
    def FromVector(cls, vector: Vector, /) -> 'Point':
        """creates the unique Point associated with the given position Vector
        FromVector(Vector(x, y, z)) == Point(x, y, z)"""

        return cls.FromSequence(Vector.components(vector))


    @staticmethod
    def plot_points(*points: 'Point', show_legend: bool|None = True) -> None:
        """plots all given Points in the same plot
        show_legend is a boolean which displays legend on the plot if True"""

        for point in points:
            if not isinstance(point, Point):
                raise TypeError(
                    "'plot_points' expects all arguments as 'Point'"
                )
        plot_objects(*points, show_legend=show_legend)


    def plot(self, /) -> None:
        """plots the Point on a 3D plot and displays it"""

        plot_objects(self)


    def components(self, /) -> tuple[float]:
        """returns a 'tuple' containing rectangular components of the Point
        components(Point(x, y, z)) == (x, y, z)"""

        return self.tuple


    def octant(self, /) -> int:
        """returns the octant in which the given Point lies as 'int'
        returns NotImplemented if any component of the Point is zero
        i.e., Point(x, y, z), where x, y, z ≠ 0
        octants are numbered as per convention:
            1. (+, +, +)      5.  (+, +, -)
            2. (-, +, +)      6.  (-, +, -)
            3. (-, -, +)      7.  (-, -, -)
            4. (+, -, +)      8.  (+, -, -)"""

        if self.tuple.count(0):
            return NotImplemented

        if self.z > 0:
            if self.y > 0:
                if self.x > 0:
                    return 1
                return 2

            if self.x < 0:
                return 3
            return 4

        if self.y > 0:
            if self.x > 0:
                return 5
            return 6

        if self.x < 0:
            return 7
        return 8


    def axis(self, /) -> int:
        """returns the axis on which the given Point lies as 'int'
        returns NotImplemented if two components of the Point are not zero
        i.e., Point(x, 0, 0), Point(0, y, 0), Point(0, 0, z), where x, y, z ≠ 0
        axes are numbered as per convention:
            1, 2, 3 = 'X', 'Y', 'Z'"""

        if self.tuple.count(0) != 2:
            return NotImplemented

        if not self.z:
            if not self.y:
                return 1
            return 2
        return 3


    def position_vector(self, /) -> Vector:
        """returns the unique position Vector associated with the given Point
        position_vector(Point(x, y, z)) == Vector(x, y, z)"""

        return self.posn_vector


################################################################################


class Line:
    """Line(a, b) -> r = a + 𝝀b
    represents the Vector Equation of a Line in 3D space

    see help(__init__) for help on Line creation
    Truth Value of Lines:
        bool(Line parallel to X, Y or Z-axis) == False"""


    def __init__(self, a: Point, b: Vector, /) -> None:
        """initialize Line instance
        a is a Point Object that lies on the required Line
        b is a Vector Object parallel to the required Line (b ≠ Vector.null)
        attrs of Line Object:
            • self.a = a
            • self.b = b"""

        if not isinstance(a, Point):
            raise TypeError("invalid operand type for 'a': must be 'Point'")

        if not isinstance(b, Vector):
            raise TypeError("invalid operand type for 'b': must be 'Vector'")

        if not b:
            raise ValueError("invalid Vector for 'b': cannot be Vector.null")

        nums = list(b.tuple)
        smallest = int(abs(min({num for num in nums if num}, key=abs)))

        for divisor in {*nums, *range(-smallest, smallest+1)}:
            if not divisor:
                continue
            for component in nums:
                if not (component / divisor).is_integer():
                    break
            else:
                nums = [num / divisor for num in nums]

        if not nums[0]:
            req = nums[2] if not nums[1] else nums[1]
            nums = [num * (-1 if req < 0 else +1) for num in nums]
        else:
            nums = [num * (-1 if nums[0] < 0 else +1) for num in nums]

        self.a, self.b = a, Vector.FromSequence(nums)


    def __str__(self, /) -> str:
        """defines the str() method for Line"""

        return f"r = ({self.a}) + 𝝀({self.b})"


    def __repr__(self, /) -> str:
        """defines the repr() method for Line"""

        return f"Line({self.a !r}, {self.b !r})"


    def __bool__(self, /) -> bool:
        """defines the truth value of a Line
        Lines parallel to X, Y or Z-axis evaluate to False"""

        return not any([
            self.b.isparallel(Vector.i), self.b.isparallel(Vector.j),
            self.b.isparallel(Vector.k)
        ])


    def __hash__(self, /) -> int:
        """defines the hash value of a Line"""

        return hash(self.b.unit())


    def __getitem__(self, k: int|float|slice) -> Point|tuple[Point]:
        """defines indexing property for Line
        indices for a Line must be int / float / slice
        returns the Point on the Line at index u -> a + ub
        slicing the Line returns a tuple of Points on the Line at indices given
        in the slice"""

        if isinstance(k, slice):
            if k.start is None or k.stop is None:
                raise ValueError("slicing a Line requires 'start' and 'stop'")
            else:
                k1, k2 = k.start, k.stop
                step = 1 if k.step is None else k.step

            for num in k1, k2, step:
                if not isinstance(num, int|float):
                    raise TypeError("slice indices must be 'int' / 'float'")
            if not step:
                raise ValueError("slice step cannot be zero")

            points, index = [], k1
            if step > 0:
                if k1 > k2:
                    return ()
                else:
                    while index < k2:
                        points.append(self[index])
                        index += step
            else:
                if k1 < k2:
                    return ()
                else:
                    while index > k2:
                        points.append(self[index])
                        index += step
            return tuple(points)

        else:
            return self.a + self.b.scale(k)


    def __contains__(self, item: Point, /) -> bool:
        """defines membership property for Line
        returns True if given item (Point) lies on the Line, False otherwise"""

        if not isinstance(item, Point):
            raise TypeError(" ".join((
                "'in <Line>' requires left operand as 'Point'",
                f"not {item.__class__.__name__ !r}"
            )))
        try:
            return (item - self.a).isscaledof(self.b)
        except ValueError:
            return True


    def __eq__(self, other: 'Line', /) -> bool:
        """defines the equality of Line Objects using '==' operator"""

        if type(self) is not type(other):
            return False
        else:
            return self.b.isscaledof(other.b) and (other.a in self)


    @classmethod
    def FromAngles(
            cls, alpha: float, beta: float, gamma: float, /, *,
            point: Point|None = Point(0, 0, 0)
        ) -> 'Line':
        """creates a Line along the given angles passing through given Point
        (angles are in degrees taken counter-clockwise)
        𝜶, 𝜷, 𝜸 = angles subtended by Line at the axes 'X', 'Y', 'Z'
        direction cosines: cos²𝜶 + cos²𝜷 + cos²𝜸 == 1"""

        if not isinstance(point, Point):
            raise TypeError(
                "invalid operand type for 'point': must be 'Point'"
            )
        return cls(point, Vector.FromAngles(alpha, beta, gamma))


    @classmethod
    def From2Points(cls, point1: Point, point2: Point, /) -> 'Line':
        """creates a Line passing through the given Points"""

        for point in point1, point2:
            if not isinstance(point, Point):
                raise TypeError(
                    "'From2Points' expects all arguments as 'Point'"
                )
        if point1 == point2:
            raise ValueError("Line must be described by two unique Points")
        else:
            return cls(point1, (point2 - point1))


    @staticmethod
    def plot_lines(*lines: 'Line', show_legend: bool|None = True) -> None:
        """plots all given Lines in the same plot
        show_legend is a boolean which displays legend on the plot if True"""

        for line in lines:
            if not isinstance(line, Line):
                raise TypeError("'plot_lines' expects all arguments as 'Line'")
        plot_objects(*lines, show_legend=show_legend)


    def plot(self, /) -> None:
        """plots the Line on a 3D plot and displays it"""

        plot_objects(self)


    def directioncos(self, /) -> tuple[float]:
        """returns a tuple containing direction cosines of the Line
        directioncos(FromAngles(𝜶, 𝜷, 𝜸, point=p)) == (cos𝜶, cos𝜷, cos𝜸)
        see help(FromAngles) for help on angles & direction cosines"""

        return self.b.unit().tuple


    def directionratios(self, /, *, k: float|None = 1) -> tuple[float]:
        """returns a tuple containing direction ratios of the Line
        k is a real number by which the DRs are scaled"""

        if isinstance(k, int|float) and not k:
            raise ValueError("invalid value for 'k': cannot be zero")
        else:
            return self.b.scale(k).tuple


    def directing_vector(self, /) -> Vector:
        """returns the directing Vector of the Line
        i.e. Vector parallel to the Line"""

        return self.b


    def get_points(
            self, k1: float, k2: float, /, *, step: float|None = 1
        ) -> tuple[Point]:
        """returns a tuple of Points on the Line from index k1 to index k2
        (including k1 and excluding k2) separated by step
        analogous to: self[k1:k2:step]
        see help(__getitem__) for help on valid indices"""

        for num in k1, k2, step:
            if not isinstance(num, int|float):
                raise TypeError(
                    "'get_points' expects all arguments as 'int' / 'float'"
                )
        if not step:
            raise ValueError("invalid value for 'step': cannot be zero")
        if step > 0:
            if k1 > k2:
                raise ValueError("'k1' must be < 'k2' for +ve 'step' value")
        else:
            if k1 < k2:
                raise ValueError("'k1' must be > 'k2' for -ve 'step' value")

        return self[k1:k2:step]


################################################################################


class Plane:
    """Plane(n, d) -> r • n = d
    represents the Vector Equation of a Plane in 3D space

    see help(__init__) for help on Plane creation
    Truth Value of Planes:
        bool(Plane parallel to XY, YZ or ZX-Plane) == False"""


    def __init__(self, n: Vector, d: float, /) -> None:
        """initialize Plane instance
        n is a Vector Object normal to the required Plane (n ≠ Vector.null)
        d is the constant for the Plane in the equation (r • n = d)
        attrs of Plane Object:
            • self.n = n
            • self.d = d
        Plane is formed in its lowest possible reduced form
        ex: Plane(Vector(6, 8, 2), 10) -> Plane(Vector(3, 4, 1), 5)"""

        if not isinstance(n, Vector):
            raise TypeError("invalid operand type for 'n': must be 'Vector'")

        if not n:
            raise ValueError("invalid Vector for 'n': cannot be Vector.null")

        if not isinstance(d, int|float):
            raise TypeError(
                "invalid operand type for 'd': must be 'int' / 'float'"
            )

        nums = [*n.tuple, d]
        smallest = int(abs(min({num for num in nums if num}, key=abs)))

        for divisor in {*nums, *range(-smallest, smallest+1)}:
            if not divisor:
                continue
            for component in nums:
                if not (component / divisor).is_integer():
                    break
            else:
                nums = [num / divisor for num in nums]

        if not nums[0]:
            req = nums[2] if not nums[1] else nums[1]
            nums = [num * (-1 if req < 0 else +1) for num in nums]
        else:
            nums = [num * (-1 if nums[0] < 0 else +1) for num in nums]

        self.n, self.d = Vector.FromSequence(nums[:-1]), nums[-1]
        if float(self.d).is_integer():
            self.d = int(self.d)


    def __str__(self, /) -> str:
        """defines the str() method for Plane"""

        return f"r • ({self.n}) = {self.d}"


    def __repr__(self, /) -> str:
        """defines the repr() method for Plane"""

        return f"Plane({self.n !r}, {self.d})"


    def __bool__(self, /) -> bool:
        """defines the truth value of a Plane
        Planes parallel to XY, YZ or ZX-Plane evaluate to False"""

        return not any([
            self.n.isparallel(Vector.i), self.n.isparallel(Vector.j),
            self.n.isparallel(Vector.k)
        ])


    def __hash__(self, /) -> int:
        """defines the hash value of a Plane"""

        return hash(repr(self.unit_normal_repr()))


    def __contains__(self, item: Point|Line, /) -> bool:
        """defines membership property for Plane
        returns True if given Point / Line lies on / in the Plane,
        False otherwise"""

        if not isinstance(item, Point|Line):
            raise TypeError(" ".join((
                "'in <Plane>' requires left operand as 'Point' / 'Line'",
                f"not {item.__class__.__name__ !r}"
            )))
        if isinstance(item, Point):
            return item.posn_vector.dot(self.n) == self.d
        else:
            return item.b.isperpendicular(self.n) and (item.a in self)


    def __eq__(self, other: 'Plane', /) -> bool:
        """defines the equality of Plane Objects using '==' operator"""

        if (type(self) is not type(other)) or not self.n.isscaledof(other.n):
            return False
        elif self.d == other.d == 0:
            return True
        else:
            if not all({self.d, other.d}):
                return False
            else:
                return other.d / self.d == self.n.scalefactor(other.n)


    @classmethod
    def FromPointNormal(
            cls, n: Vector, /, *, point: Point|None = Point(0, 0, 0)
        ) -> 'Plane':
        """creates a Plane with normal Vector n (n ≠ Vector.null) and
        passing through the given Point. returns Plane:
            (r - a) • n = 0    i.e.,    r • n = a • n
            where a is the Point on the required Plane"""

        if not isinstance(point, Point):
            raise TypeError("invalid operand type for 'point': must be 'Point'")
        else:
            return cls(n, point.posn_vector.dot(n))


    @classmethod
    def From3Points(
            cls, point1: Point, point2: Point, point3: Point, /
        ) -> 'Plane':
        """creates a Plane passing through the given Points"""

        for point in point1, point2, point3:
            if not isinstance(point, Point):
                raise TypeError(
                    "'From3Points' expects all arguments as 'Point'"
                )
        if len({point1, point2, point3}) != 3:
            raise ValueError("Plane must be described by three unique Points")

        if collinear(point1, point2, point3):
            raise ValueError(
                "Plane must be described by three unique non - collinear Points"
            )
        return cls.FromPointNormal(
           (point2 - point1).cross(point3 - point1), point=point1
        )

    @classmethod
    def From2Lines(cls, line1: Line, line2: Line, /) -> 'Line':
        """creates a Plane passing through the the given Lines"""

        for line in line1, line2:
            if not isinstance(line, Line):
                raise TypeError("'From2Lines' expects all arguments as 'Line'")

        if type_lines(line1, line2) in {0, 3}:
            raise ValueError(
                "Plane must be described by two unique non - skew Lines"
            )
        if line1.b.isparallel(line2.b):
            return cls.From3Points(line1.a, line2.a, line1[1])
        else:
            return cls.FromPointNormal(line1.b.cross(line2.b), point=line1.a)


    @classmethod
    def From2Planes(
            cls, plane1: 'Plane', plane2: 'Plane', /, *,
            point: Point|None = Point(0, 0, 0)
        ) -> 'Plane':
        """creates a Plane through the Line of intersection of Planes 1 and 2
        passing through the given Point"""

        for plane in plane1, plane2:
            if not isinstance(plane, cls):
                raise TypeError(
                    "'From2Planes' expects positional arguments as 'Plane'"
                )
        if not isinstance(point, Point):
            raise TypeError("invalid operand type for 'point': must be 'Point'")

        if (line := intersection(plane1, plane2)) is NotImplemented:
            raise ValueError("given Planes do not intersect")

        if point in line:
            raise ValueError(" ".join((
                "given Point should not lie on the Line of",
                "intersection of the two Planes"
            )))

        try:
            eqn = eqns.LinearEquation1D(
                point.posn_vector.dot(plane2.n) - plane2.d,
                point.posn_vector.dot(plane1.n) - plane1.d
            )
        except ValueError:
            return plane1 if point in plane1 else plane2
        else:
            k = eqns.solve1D(eqn)
            n = (plane1.n + plane2.n.scale(k)).scale(eqn.a)
            d = (plane1.d + k*plane2.d) * eqn.a

        return cls(n, d)


    @classmethod
    def FromLinePoint(cls, *, line: Line, point: Point) -> 'Plane':
        """creates a Plane containing the given Line passing through the
        given Point"""

        if not isinstance(line, Line):
            raise TypeError("invalid operand type for 'line': must be 'Line'")

        if not isinstance(point, Point):
            raise TypeError("invalid operand type for 'point': must be 'Point'")

        if point in line:
            raise ValueError("given Point must not lie on the given Line")

        normal = (line.a - point).cross(line.b)
        return cls.FromPointNormal(normal, point=point)


    @staticmethod
    def plot_planes(*planes: 'Plane', show_legend: bool|None = True) -> None:
        """plots all given Planes in the same plot
        show_legend is a boolean which displays legend on the plot if True"""

        for plane in plane:
            if not isinstance(plane, Plane):
                raise TypeError(
                    "'plot_planes' expects all arguments as 'Plane'"
                )
        plot_objects(*planes, show_legend=show_legend)


    def plot(self, /) -> None:
        """plots the Plane on a 3D plot and displays it"""

        plot_objects(self)


    def cartesian_repr(self, /) -> str:
        """returns a str of representation of the Plane in Cartesian form
        the str returned is of the form (ax + by + cz = d)"""

        return " ".join((
            str(self.n).replace("î", "x").replace("ĵ", "y").replace("k̂", "z"),
            "=", str(self.d)
        ))


    def unit_normal_repr(self, /) -> str:
        """returns a str of representation of the Plane in the form of its
        unit normal Vector and distance from the Origin (0, 0, 0)
        the str returned is of the form (r • n̂ = D)"""

        n, d = self.n.unit(), (self.d / self.n.mod())
        return f"r • ({n}) = {int(d) if float(d).is_integer() else d}"


    def directing_vector(self, /) -> Vector:
        """returns the directing Vector of the Plane
        i.e. Vector normal to the Plane"""

        return self.n


################################################################################


def plot_linesegment(point1: Point, point2: Point, /) -> None:
    """plots a Line Segment joining the two unique Points and displays it"""

    for point in point1, point2:
        if not isinstance(point, Point):
            raise TypeError(
                "'plot_linesegment' expects all arguments as 'Point'"
            )
    plot_objects((point1, point2))


def plot_objects(
        *objects: Vector|Point|Line|Plane|tuple[Point], show_legend: bool|None = True,
        plot_intersections: bool|None = False
    ) -> None:
    """plots all given 3D - objects in the same plot and displays it
    to refer to a Line Segment, pass a tuple of two unique Points
    ex: plot_objects(xy_plane, (Point(2, 3, 4), Point(-1, 3, -5)))
    show_legend is a boolean which displays legend on the plot if True
    plot_intersections is a boolean which if True, plots the objects along with
    their Points / Lines of intersection (if any)"""

    if not objects:
        return

    points, lines, planes, vectors, linesegs = set(), set(), set(), set(), set()
    for obj in objects:
        if isinstance(obj, Point):
            points.add(obj)
        elif isinstance(obj, Line):
            lines.add(obj)
        elif isinstance(obj, Plane):
            planes.add(obj)
        elif isinstance(obj, Vector):
            vectors.add(obj)
        elif isinstance(obj, tuple):
            if len(obj) != 2:
                raise ValueError(" ".join((
                    "to refer to Line Segment, pass a tuple of two Points.",
                    "see help(plot_objects) for further help"
                )))
            for point in obj:
                if not isinstance(point, Point):
                    raise TypeError(" ".join((
                        "to refer to Line Segment, pass a tuple of two Points.",
                        "see help(plot_objects) for further help"
                    )))
            if len(set(obj)) < 2:
                raise ValueError(
                    "Line Segment must be described by two unique Points"
                )
            else:
                linesegs.add(obj) if (obj[::-1] not in linesegs) else None
        else:
            raise TypeError(" ".join((
                f"cannot plot object type '{obj.__class__.__name__}':",
                "must be 'Point', 'Line', 'Plane', 'Vector' or Line Segment"
            )))

    if plot_intersections:
        intersections = set()
        for obj1 in objects:
            for obj2 in objects:
                try:
                    intersections.add(intersection(obj1, obj2))
                except TypeError:
                    pass
                for obj3 in objects:
                    try:
                        intersections.add(intersection(obj1, obj2, obj3))
                    except TypeError:
                        pass
        for obj4 in intersections.copy():
            for obj5 in intersections.copy():
                try:
                    intersections.add(intersection(obj4, obj5))
                except TypeError:
                    pass
            for obj6 in objects:
                try:
                    intersections.add(intersection(obj4, obj6))
                except TypeError:
                    pass
        for obj in intersections:
            if isinstance(obj, Point):
                points.add(obj)
            elif isinstance(obj, Line):
                lines.add(obj)

    ax = plt.axes(projection="3d")
    patches = []
    colors = [
        "red", "orange", "green", "blue", "purple",
        "cyan", "magenta", "yellow", "brown", "gray",
    ] * len(objects)

    if planes:
        arange1, arange2 = np.arange(-50, 50), np.arange(-75, 75)

    for plane in planes:
        x = y = z = arange1 if len(plane.n) == 3 else arange2
        if plane.n.z:
            X, Y = np.meshgrid(x, y)
            Z = (plane.d - (plane.n.x*X + plane.n.y*Y)) / plane.n.z
        elif plane.n.x:
            Y, Z = np.meshgrid(y, z)
            X = (plane.d - (plane.n.y*Y + plane.n.z*Z)) / plane.n.x
        else:
            Z, X = np.meshgrid(z, x)
            Y = (plane.d - (plane.n.z*Z + plane.n.x*X)) / plane.n.y
        ax.plot_wireframe(X, Y, Z, color=colors[0], rstride=5, cstride=5)
        patches.append(mp.Patch(color=colors[0], label=plane))
        colors.pop(0)

    ax.plot3D([-100, 100], [0, 0], [0, 0], color="black", linewidth=3)
    ax.plot3D([0, 0], [-100, 100], [0, 0], color="black", linewidth=3)
    ax.plot3D([0, 0], [0, 0], [-100, 100], color="black", linewidth=3)

    for line in lines:
        p1 = p2 = Point(0, 0, 0)
        for i in range(1, 101):
            if abs(p1) > 100 or abs(p2) > 100:
                break
            if abs(line[i]) > abs(p1):
                p1 = line[i]
            if abs(line[-i]) > abs(p2):
                p2 = line[-i]
        ax.plot3D(
            [p1.x, p2.x], [p1.y, p2.y], [p1.z, p2.z],
            linewidth=3, color=colors[0]
        )
        patches.append(mp.Patch(
            color=colors[0], label=str(line).replace("𝝀", "λ")
        ))
        colors.pop(0)

    ax.plot3D(0, 0, 0, color="black", marker="o")

    for lineseg in linesegs:
        point1, point2 = lineseg
        ax.plot3D(
            [point1.x, point2.x], [point1.y, point2.y], [point1.z, point2.z],
            linewidth=3, color=colors[0]
        )
        patches.append(mp.Patch(
            color=colors[0], label=f"({point2}) - ({point1})"
        ))
        ax.plot3D(*point1.tuple, color=colors[0], marker="o")
        ax.plot3D(*point2.tuple, color=colors[0], marker="o")
        colors.pop(0)

    for vector in vectors:
        ax.quiver(
            0, 0, 0, *vector.tuple, linewidth=3,
            color=(colors[0] if vector else "black")
        )
        patches.append(mp.Patch(
            color=(colors[0] if vector else "black"), label=vector
        ))
        colors.pop(0)

    for point in points:
        ax.plot3D(
            point.x, point.y, point.z, marker="o",
            color=(colors[0] if point else "black")
        )
        patches.append(mp.Patch(
            color=(colors[0] if point else "black"), label=point
        ))
        colors.pop(0)

    ax.set_xlabel("î")
    ax.set_ylabel("ĵ")
    ax.set_zlabel("k̂")

    if show_legend:
        ax.legend(handles=patches)

    plt.show()


def distance(obj1: Point|Line|Plane, obj2: Point|Line|Plane, /) -> float:
    """returns the shortest Euclidean distance between the two 3D - objects
    objects can be of type Point, Line and / or Plane
    i.e., distance() can return the shortest distance between:
        • two Points
        • two Lines
        • two Planes
        • Point and Line (and vice - versa)
        • Line and Plane (and vice - versa)
        • Plane and Point (and vice - versa)"""

    if isinstance(obj1, Point):
        if isinstance(obj2, Point):
            return math.dist(obj1.tuple, obj2.tuple)

        if isinstance(obj2, Line):
            return abs(obj2.b.cross(obj1 - obj2.a)) / obj2.b.mod()

        if isinstance(obj2, Plane):
            return abs(obj1.posn_vector.dot(obj2.n) - obj2.d) / obj2.n.mod()

    if isinstance(obj1, Line):
        if isinstance(obj2, Point):
            return distance(obj2, obj1)

        if isinstance(obj2, Line):
            if obj1.b.isscaledof(obj2.b):
                return abs((obj2.a - obj1.a).cross(obj1.b)) / obj1.b.mod()
            else:
                vec = obj1.b.cross(obj2.b)
                return abs((obj2.a - obj1.a).dot(vec)) / vec.mod()

        if isinstance(obj2, Plane):
            if parallel(obj1, obj2):
                return distance(obj1.a, obj2)
            else:
                return 0.0

    if isinstance(obj1, Plane):
        if isinstance(obj2, Point|Line):
            return distance(obj2, obj1)

        if isinstance(obj2, Plane):
            if obj1.n.isscaledof(obj2.n):
                obj2 = Plane(obj1.n, obj2.d*obj2.n.scalefactor(obj1.n))
                return abs(obj2.d - obj1.d) / obj1.n.mod()
            else:
                return 0.0

    raise TypeError(" ".join((
        "invalid argument type(s) for 'distance':",
        "must be 'Point', 'Line' and / or 'Plane'.",
        "see help(distance) for help on valid arguments."
    )))


def angle(obj1: Line|Plane, obj2: Line|Plane, /) -> float:
    """returns the acute angle (in degrees) between the two 3D - objects
    objects can be of type Line and / or Plane
    i.e., angle() can return the angle between:
        • two Lines
        • two Planes
        • Line and Plane (and vice - versa)"""

    if isinstance(obj1, Line):
        if isinstance(obj2, Line):
            ang1, ang2 = obj1.b.angle(obj2.b), obj1.b.angle(-obj2.b)
            return ang1 if ang1 <= 90 else ang2

        if isinstance(obj2, Plane):
            return abs(90 - obj1.b.angle(obj2.n))

    if isinstance(obj1, Plane):
        if isinstance(obj2, Line):
            return angle(obj2, obj1)

        if isinstance(obj2, Plane):
            ang1, ang2 = obj1.n.angle(obj2.n), obj1.n.angle(-obj2.n)
            return ang1 if ang1 <= 90 else ang2

    raise TypeError(" ".join((
        "invalid argument type(s) for 'angle':",
        "must be 'Line' and / or 'Plane'.",
        "see help(angle) for help on valid arguments."
    )))


def parallel(*objects: Line|Plane) -> bool:
    """returns True if all the given 3D - objects are parallel, False otherwise
    objects can be of type Line and / or Plane
    i.e., parallel() can check if the following are parallel:
        • Lines
        • Planes
        • Lines and Planes (and vice - versa)
    note: at least two arguments are required for parallel()"""

    for obj in objects:
        if not isinstance(obj, Line|Plane):
            raise TypeError(" ".join((
                "invalid argument type(s) for 'parallel':",
                "must be 'Line' and / or 'Plane'.",
                "see help(parallel) for help on valid arguments."
            )))

    if len(objects) < 2:
        raise ValueError("at least two arguments required for 'parallel'")

    for i in range(len(objects)):
        obj1, obj2 = objects[i-1], objects[i]
        if math.isclose(angle(obj1, obj2), 0, abs_tol=1e-09):
            return False
    else:
        return True


def perpendicular(obj1: Line|Plane, obj2: Line|Plane, /) -> bool:
    """returns True if the two 3D - objects are perpendicular, False otherwise
    objects can be of type Line and / or Plane
    i.e., perpendicular() can check if the following are perpendicular:
        • two Lines
        • two Planes
        • Line and Plane (and vice - versa)"""

    for obj in obj1, obj2:
        if not isinstance(obj, Line|Plane):
            raise TypeError(" ".join((
                "invalid argument type(s) for 'perpendicular':",
                "must be 'Line' and / or 'Plane'.",
                "see help(perpendicular) for help on valid arguments."
            )))
    return math.isclose(angle(obj1, obj2), 90, abs_tol=1e-09)


def isclose(
        point1: Point, point2: Point, /, *,
        rel_tol: float|None = 1e-09, abs_tol: float|None = 0.0
    ) -> bool:
    """returns True if given Points are close to each other, False otherwise
    for the Points to be considered "close", the distance between them must be
    smaller than at-least one of the tolerances
    rel_tol:
        maximum difference for being considered "close", relative to the
        magnitude of the input values
    abs_tol:
        maximum difference for being considered "close", regardless of the
        magnitude of the input values"""

    for point in point1, point2:
        if not isinstance(point, Point):
            raise TypeError("'isclose' expects all arguments as 'Point'")

    length = distance(point1, point2)
    return math.isclose(length, 0, rel_tol=rel_tol, abs_tol=abs_tol)


def section(
        point1: Point, point2: Point, /, m: float, n: float,
        *, external: bool|None = False
    ) -> Point:
    """returns the Point that sections / divides the Line Segment joining the
    given Points in the ratio m : n internally (default)
    for external division, pass keyword argument external as True
    m and n must be positive Real Numbers
    returns NotImplemented if no such Point exists (only in case of external
    division, i.e. when m = n)"""

    for point in point1, point2:
        if not isinstance(point, Point):
            raise TypeError(
                "'section' expects all positional only arguments as 'Point'"
            )
    if point1 == point2:
        raise ValueError(
            "'section' expects two unique Points as positional only arguments"
        )

    divide = Vector.section_external if external else Vector.section_internal

    if (m == n) and external:
        return NotImplemented
    else:
        return Point.FromVector(
            divide(point1.posn_vector, point2.posn_vector, m=m, n=n)
        )


def midpoint(point1: Point, point2: Point, /) -> Point:
    """returns the midpoint of the Line Segment joining the two unique Points"""

    for point in point1, point2:
        if not isinstance(point, Point):
            raise TypeError("'midpoint' expects all arguments as 'Point'")

    if point1 == point2:
        raise ValueError("'midpoint' expects two unique Points as arguments")
    else:
        return section(point1, point2, m=1, n=1)


def area(point1: Point, point2: Point, point3: Point, /) -> float:
    """returns the area of the triangle formed by the three unique Points"""

    for point in point1, point2, point3:
        if not isinstance(point, Point):
            raise TypeError("'area' expects all arguments as 'Point'")

    if len({point1, point2, point3}) < 3:
        raise ValueError("'area' expects three unique Points as arguments")

    a = distance(point1, point2)
    b = distance(point2, point3)
    c = distance(point3, point1)
    s = (a + b + c) / 2

    return (s * (s-a) * (s-b) * (s-c)) ** 0.5


def centroid(point1: Point, point2: Point, point3: Point, /) -> Point:
    """returns the centroid of the triangle formed by the three unique Points"""

    for point in point1, point2, point3:
        if not isinstance(point, Point):
            raise TypeError("'centroid' expects all arguments as 'Point'")

    if len({point1, point2, point3}) < 3:
        raise ValueError("'centroid' expects three unique Points as arguments")

    if not area(point1, point2, point3):
        raise ValueError("given Points do not form a triangle")

    return Point.FromSequence(
        [(point1[i] + point2[i] + point3[i]) / 3 for i in ("x", "y", "z")]
    )


def type_lines(line1: Line, line2: Line, /) -> int:
    """checks and returns an int to refer if two Lines in 3D space are:
        • 0 -> coincident
        • 1 -> parallel
        • 2 -> intersecting
        • 3 -> skew"""

    for line in line1, line2:
        if not isinstance(line, Line):
            raise TypeError("'type_lines' expects all arguments as 'Line'")

    not_det = math.isclose(
        (line2.a - line1.a).scalar_triple(line1.b, line2.b), 0, abs_tol=1e-09
    )

    if line1 == line2:
        return 0
    elif parallel(line1, line2):
        return 1
    elif not_det:
        return 2
    else:
        return 3


def collinear(*points: Point) -> bool:
    """returns True if all given Points are collinear, False otherwise
    note: at least three unique arguments are required for collinear()"""

    unique = []
    for point in points:
        if not isinstance(point, Point):
            raise TypeError(
                "invalid argument type(s) for 'collinear': must be 'Point'"
            )
        unique.append(point) if point not in unique else None

    if len(unique) < 3:
        raise ValueError(
            "at least three unique arguments are required for 'collinear'"
        )

    line = Line.From2Points(unique[0], unique[1])
    for point in unique:
        if point not in line:
            return False
    return True


def coplanar(*points: Point) -> bool:
    """returns True if all given Points are coplanar, False otherwise
    note: at least four unique arguments are required for coplanar()"""

    unique = []
    for point in points:
        if not isinstance(point, Point):
            raise TypeError(
                "invalid argument type(s) for 'coplanar': must be 'Point'"
            )
        unique.append(point) if point not in unique else None

    if len(unique) < 4:
        raise ValueError(
            "at least four unique arguments are required for 'coplanar'"
        )

    if collinear(*unique):
        return False

    point1, point2 = unique[0], unique[1]
    line = Line.From2Points(point1, point2)

    for point3 in unique[2:]:
        if point3 not in line:
            plane = Plane.From3Points(point1, point2, point3)
            break

    for point in points:
        if point not in plane:
            return False
    return True


def intersection(
        obj1: Line|Plane, obj2: Line|Plane, obj3: Plane|None = None, /
    ) -> Point|Line:
    """returns the 3D object of intersection of the given 3D objects
    i.e., intersection() can return:
        • Point of intersection of two Lines
        • Point of intersection of Line and Plane (and vice - versa)
        • Point of intersection of three Planes
        • Line of intersection of two Planes
    returns NotImplemented if:
        • the given objects do not intersect each other
        • any of the given objects are coincident
        • given Line in contained in the Plane (in Point of intersection
        of Line and Plane)
    note: argument 3 will be ignored in all cases except in case of Point
    of intersection of three Planes"""

    if isinstance(obj1, Line):
        if isinstance(obj2, Line):
            if type_lines(obj1, obj2) != 2:
                return NotImplemented
            else:
                A, B, C = (obj2.b - obj1.b).tuple

            try:
                if A:
                    x = (obj1.a.x*obj2.b.x - obj2.a.x*obj1.b.x) / A
                    k = (x - obj1.a.x) / obj1.b.x
                elif B:
                    y = (obj1.a.y*obj2.b.y - obj2.a.y*obj1.b.y) / B
                    k = (y - obj1.a.y) / obj1.b.y
                else:
                    z = (obj1.a.z*obj2.b.z - obj2.a.z*obj1.b.z) / C
                    k = (z - obj1.a.z) / obj1.b.z
            except ZeroDivisionError:
                return intersection(obj2, obj1)

            if obj1[k] in obj2:
                return obj1[k]

            system = [
                eqns.LinearEquation2D(obj1.b[i], -obj2.b[i], (obj1.a-obj2.a)[i])
                for i in ("x", "y", "z")
            ]

            eqn1 = system[0]
            eqn2 = system[1] if system[1] != eqn1 else system[2]
            eqn3 = system[2] if system[2] != eqn2 else system[1]

            k1, k2 = eqns.solve2D(eqn1, eqn2)
            close = isclose(obj1[k1], obj2[k2], abs_tol=1e-09)
            if eqns.satisfies(eqn3, k1, k2) or close:
                return obj1[k1]

        if isinstance(obj2, Plane):
            if parallel(obj1, obj2) or obj1 in obj2:
                return NotImplemented

            eqn = eqns.LinearEquation1D(
                obj1.b.dot(obj2.n), obj1.a.posn_vector.dot(obj2.n) - obj2.d
            )
            return obj1[eqns.solve1D(eqn)]

    if isinstance(obj1, Plane):
        if isinstance(obj2, Line):
            return intersection(obj2, obj1)

        if isinstance(obj2, Plane):
            if obj3 is None:
                if parallel(obj1, obj2):
                    return NotImplemented
                else:
                    parallel_vec = obj1.n.cross(obj2.n)
                    X = Y = Z = 0

                if not (obj1 and obj2):
                    if parallel(obj1, xy_plane):
                        Z = intersection(obj1, z_axis).z
                    elif parallel(obj1, yz_plane):
                        X = intersection(obj1, x_axis).x
                    else:
                        Y = intersection(obj1, y_axis).y

                    if parallel(obj2, xy_plane):
                        Z = intersection(obj2, z_axis).z
                    elif parallel(obj2, yz_plane):
                        X = intersection(obj2, x_axis).x
                    else:
                        Y = intersection(obj2, y_axis).y

                    return Line(Point(X, Y, Z), parallel_vec)

                if not parallel_vec.isperpendicular(xy_plane.n):
                    z = 0
                    eqn1 = eqns.LinearEquation2D(obj1.n.x, obj1.n.y, -obj1.d)
                    eqn2 = eqns.LinearEquation2D(obj2.n.x, obj2.n.y, -obj2.d)
                    x, y = eqns.solve2D(eqn1, eqn2)

                elif not parallel_vec.isperpendicular(yz_plane.n):
                    x = 0
                    eqn1 = eqns.LinearEquation2D(obj1.n.y, obj1.n.z, -obj1.d)
                    eqn2 = eqns.LinearEquation2D(obj2.n.y, obj2.n.z, -obj2.d)
                    y, z = eqns.solve2D(eqn1, eqn2)

                else:
                    y = 0
                    eqn1 = eqns.LinearEquation2D(obj1.n.z, obj1.n.x, -obj1.d)
                    eqn2 = eqns.LinearEquation2D(obj2.n.z, obj2.n.x, -obj2.d)
                    z, x = eqns.solve2D(eqn1, eqn2)

                return Line(Point(x, y, z), parallel_vec)

            if isinstance(obj3, Plane):
                if any({
                    parallel(obj1, obj2), parallel(obj2, obj3),
                    parallel(obj3, obj1)
                }):
                    return NotImplemented
                else:
                    return intersection(intersection(obj1, obj2), obj3)

            raise TypeError("argument 3 if given, must be of type 'Plane'")

    raise TypeError(" ".join((
        "invalid argument type(s) for 'intersection':",
        "must be 'Line' and / or 'Plane'.",
        "see help(intersection) for help on valid arguments."
    )))


def image(obj1: Point|Line|Plane, obj2: Line|Plane, /) -> Point|Line|Plane:
    """returns the image of 3D - object1 taking object2 as a mirror
    i.e., image() can return the image of:
        • Point in a Line
        • Point in a Plane
        • Line in a Line
        • Line in a Plane
        • Plane in a Plane
    returns obj1 if:
        • obj1 is contained in obj2
        • the given objects are coincident"""

    if isinstance(obj1, Point):
        if isinstance(obj2, Line):
            plane = Plane(obj2.b, obj1.posn_vector.dot(obj2.b))
            point = [2*obj2.a[i] - obj1[i] for i in ("x", "y", "z")]
            line = Line(Point.FromSequence(point), obj2.b.scale(2))
            return intersection(line, plane)

        if isinstance(obj2, Plane):
            line = Line(obj1, obj2.n)
            index = obj2.n.dot(obj1.posn_vector) - obj2.d
            index *= (-2 / obj2.n.mod()**2)
            return line[index]

    if isinstance(obj1, Line):
        if isinstance(obj2, Line):
            return Line.From2Points(image(obj1[-1], obj2), image(obj1[1], obj2))

        if isinstance(obj2, Plane):
            return Line.From2Points(image(obj1[-1], obj2), image(obj1[1], obj2))

    if isinstance(obj1, Plane):
        if isinstance(obj2, Plane):
            p1 = image(projection(Point(1, 0, 0), obj1), obj2)
            p2 = image(projection(Point(0, 1, 0), obj1), obj2)
            p3 = image(projection(Point(0, 0, 1), obj1), obj2)
            return Plane.From3Points(p1, p2, p3)

    raise TypeError(" ".join((
        "invalid argument type(s) for 'image':",
        "must be 'Point', 'Line' and / or 'Plane'.",
        "see help(image) for help on valid arguments."
    )))


def projection(obj1: Point|Line, obj2: Line|Plane, /) -> Point|Line:
    """returns the projection of 3D - object1 on object2
    object1 can be of type Point or Line
    object2 can be of type Line or Plane
    i.e., projection() can return the projection of:
        • Point on a Line
        • Point on a Plane
        • Line on a Plane
    returns obj1 if obj1 is contained in obj2"""

    if isinstance(obj1, Point):
        if isinstance(obj2, Line):
            return midpoint(obj1, image(obj1, obj2))

        if isinstance(obj2, Plane):
            line = Line(obj1, obj2.n)
            index = obj2.n.dot(obj1.posn_vector) - obj2.d
            index *= (-1 / obj2.n.mod()**2)
            return line[index]

    if isinstance(obj1, Line):
        if isinstance(obj2, Plane):
            point1 = projection(obj1[1], obj2)
            point2 = projection(obj1[-1], obj2)
            return Line.From2Points(point1, point2)

    raise TypeError(" ".join((
        "invalid argument type(s) for 'projection':",
        "must be 'Point' and / or 'Line'.",
        "see help(projection) for help on valid arguments."
    )))


################################################################################

# CONSTANTS
ORIGIN = origin = Point(0, 0, 0)

X_AXIS = x_axis = Line(origin, Vector.i)
Y_AXIS = y_axis = Line(origin, Vector.j)
Z_AXIS = z_axis = Line(origin, Vector.k)

XY_PLANE = xy_plane = Plane(Vector.k, 0)
YZ_PLANE = yz_plane = Plane(Vector.i, 0)
ZX_PLANE = zx_plane = Plane(Vector.j, 0)
