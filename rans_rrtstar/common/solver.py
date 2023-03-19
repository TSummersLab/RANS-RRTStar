
from casadi import SX, nlpsol


def dummy_problem_ipopt():
    # This is just called to make the annoying IPOPT print banner show up before doing other stuff
    x = SX.sym('x')
    nlp = {'x': x, 'f': x**2}
    settings = {'verbose': 0,
                'ipopt.print_level': 0,
                'print_time': 0}
    sol = nlpsol('sol', 'ipopt', nlp, settings)
    return sol()
