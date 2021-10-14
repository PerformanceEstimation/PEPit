from PEPit.point import Point
from PEPit.expression import Expression


def inexact_proximal_step(x0, f, step, opt):
	"""
	This routine performs a proximal step of step size 'step', starting from 'x0', and on function 'f'.
	That is, it performs :
		y = x0 - step * (\partial f(y) + e),
	where \partial f(y) is a (sub)gradient of the function f at y, and e is some computation error
	whose characteristic are provided in the settings structure.

	prox(y) = argmin_x { f(x) + 1/2/gamma * ||x-y||^2 }
	<=>
	0 \in \partial f(y) + 1/gamma * (x-y)
	<=>
	y=x-gamma*\partial f(y)

	:param x0 (Point): starting point
	:param f (function): function on which the (sub)gradient will be evaluated
	:param step (float): step size gamma of the proximal step

	:return: y, where \partial f(y) is a eps-subgradient of the funciton f at y
			gy a subgradient of y
			fy function f evaluated at y
			w a primal point (possibly = y) such tht y is a subgradient of the function f at w
			v is subgradient of f at w
			fw function f evaluated at w
			epsVar : requiredaccuracy, which the user can bound
	"""

	gx = Point()
	fx = Expression()
	x = x0 - step * gx
	f.add_point((x, gx, fx))

	return x, gx, fx