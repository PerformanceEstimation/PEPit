from PEPit.point import Point
from PEPit.expression import Expression


def proximal_step(x0, f, step):
	"""
	This routine performs a proximal step of step size 'step', starting from 'x0', and on function 'f'.
	That is, it performs :
	prox(y) = argmin_x { gamma * f(x) + 1/2 * ||x-y||^2 }
	<=>
	0 \in \gamma \partial f(y) + x-y
	<=>
	y=x-gamma * \partial f(y)

	Args:
		x0 (Point): starting point x0
		f (Function): function on which the (sub)gradient will be evaluated
		step (float): step size of the proximal step

	Returns:
		- x (Point).
		- gx (Point) the (sub)gradient of f at x.
		- fx (Expression) the function f evaluated at x.

	"""

	gx = Point()
	fx = Expression()
	x = x0 - step * gx
	f.add_point((x, gx, fx))

	return x, gx, fx
