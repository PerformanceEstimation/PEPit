from PEPit.point import Point
from PEPit.expression import Expression


def proximal_step(x0, f, step):
	"""
	TODO: CANEVAS DESCRIPTION DE CE QU'on FAIT
	Output the proximal step...

	prox(y) = argmin_x { f(x) + 1/2/gamma * ||x-y||^2 }
	<=>
	0 \in \partial f(y) + 1/gamma * (x-y)
	<=>
	y=x-gamma*\partial f(y)

	:param x0:
	:param f:
	:param step:
	:return:
	"""

	gx = Point()
	fx = Expression()
	x = x0 - step * gx
	f.add_point((x, gx, fx))

	return x, gx, fx
