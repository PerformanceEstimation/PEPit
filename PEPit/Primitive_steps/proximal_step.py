from PEPit.point import Point
from PEPit.expression import Expression


def proximal_step(x0, f, step):
	"""
	This routine performs a proximal step of step size 'step', starting from 'x0', and on function 'f'.
	That is, it performs :
	prox(y) = argmin_x { f(x) + 1/2/gamma * ||x-y||^2 }
	<=>
	0 \in \partial f(y) + 1/gamma * (x-y)
	<=>
	y=x-gamma*\partial f(y)

	:param x0 (Point): starting point x0
	:param f (function): function on which the (sub)radient will be evaluated
	:param step (float): step size of the proximal step

	:return:
		- x (Point).
		- gx (Point) the (sub)gradient of f at x.
		- fx (Expression) the function f evaluated at x.
	"""

	gx = Point()
	fx = Expression()
	x = x0 - step * gx
	f.add_point((x, gx, fx))

	return x, gx, fx