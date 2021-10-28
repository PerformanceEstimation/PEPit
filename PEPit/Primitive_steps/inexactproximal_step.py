from PEPit.point import Point
from PEPit.expression import Expression


def inexact_proximal_step(x0, f, step, opt='PD_gapII'):
	"""
	This routine performs an inexact proximal step with step size gamma,
	starting from x0, and on function f, that is :
		x = x0 - gamma*(v+e), where v is a (epsilon-sub)gradient of f at
							  x, and e is some computation error whose
							  characteristics are provided in the algorithm.

	Four optimization criterion are available :
	- 'PD_gapI' :
		PD gap(x,v;x0) <= epsVar for the proximal subproblem.
	- 'PD_gapII' :
		PD gap(x,v;x0) <= epsVar for the proximal subproblem,
								with v a subgradient of f at x.
	- 'PD_gapIII' :
		PD gap(x,v;x0) <= epsVar for the proximal subproblem,
								with v = (x_0 - x)/step
	- 'Orip-style' (see [1] below) :
	Approximate proximal operator outputs x such that
		<v, e> + epsilon/step <= espVar for the proximal subproblem,
									with x = x_0 - step * (v - e),
									with v an epsilon-subgradient of f at x.
		PD gap(x,v;x0) <= epsVar for the proximal subproblem.

	ORIP: optimized relatively inexact proximal point algorithm (see [1])
	[1] M. Barre, A. Taylor, F. Bach. Principled analyses and design of
	     first-order methods with inexact proximal operators


	:param x0: starting point x0.
	:param f: function on which the (epsilon-sub)gradient will be evaluated.
	:param step: step size of the proximal step.
	:param opt: inaccuracy parameters in the "opt" structure.

	:return: x, gx, fx, w, v , fw, epsVar
	NOTE : v is an epxilon-subgradient of f at x, and is the subgradient of f evaluated at w.
	epsVar is te required accuracy (a variable), which the user can (should) bound.
	"""
	if opt == 'PD_gapI':
		"""
		Approximate the proximal operator outputs x such that
			PD_gap(x,v:x0) <=epsVar
		with v some dual variable, and 
			PD_gap(x,v;x0) = 1/2*(x - x0 + step*(fx + f^*(v) - <v;x>)
		in which we use : f^*(v) = <v, w> - f(w) for some w such that v is a subgradient of f at w.
		"""
		v = Point()
		w = Point()
		fw = Expression()
		f.add_point((w, v, fw))

		x = Point()
		gx = Point()
		fx = Expression()
		f.add_point((x, gx, fx))

		epsVar = Expression()
		e = x - x0 + step*v
		eps_sub = fx - fw - v*(x-w)
		f.add_constraint(e**2/2 + step*eps_sub <= epsVar)

	if opt == 'PD_gapII':
		"""
		Approximate the proximal operator outputs x such that
			||e||**2 <= epsVar
		withx = x0 - step * (gx - e), and gx a subgradient of f at x.
		"""
		e = Point()
		gx = Point()
		x = x0 - step * (gx - e)
		fx = Expression()
		f.add_point((x, gx, fx))
		epsVar = Expression()
		f.add_constraint(e**2 <= epsVar)
		w, v, fw = x, gx, fx

	if opt == 'PD_gapIII':
		"""
		Approximate the proximal operator outputs x such that
			step * (fx - fw - v*(x - w) <= epsVar
		with v = (x0 - x)/step and w a point such tat v is a subgradient of f at w.
		"""
		x, gx, w = Point(), Point(), Point()
		v = (x0 - x)/step
		fw, fx = Expression(), Expression()
		f.add_point((x, gx, fx))
		f.add_point((w, v, fw))
		epsVar = Expression()
		eps_sub = fx - fw - v * (x - w)
		f.add_constraint(step*eps_sub <= epsVar)

	if opt == 'Orip-style':
		"""
		Approximate proximal operator outputs x such that :
			<v, e> + epsilon/step <= espVar
		with x = x0 - step * (v - e), wit v an epsilon subgradient of f at x.
		"""
		v = Point()
		w = Point()
		fw = Expression()
		f.add_point((w, v, fw))

		e = Point()
		gx = Point()
		fx = Expression()
		x = x0 - step * (v - e)
		f.add_point((x, gx, fx))

		epsVar = Expression()
		eps_sub = fx - fw - v*(x-w)
		f.add_constraint(e*v + eps_sub/step <= epsVar)

	return x, gx, fx, w, v, fw, epsVar