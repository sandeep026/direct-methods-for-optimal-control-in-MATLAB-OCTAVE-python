import matplotlib.pyplot as plt
import casadi as cs
from typing import Union

# number of states and control
Nx = 2
Nu = 1

#symbolic states and control to define Lagrange term and dynamics
x = cs.MX.sym("x", Nx, 1)
u = cs.MX.sym("u", Nu, 1)

# lagrange
L = u**2
L = cs.Function('L', [x, u], [L])

# dynamics
f = cs.vcat([x[1], -x[1]+u])
f = cs.Function('L', [x, u], [f])

# bc parameters
x_t0 = cs.DM.zeros(2, 1)
a, b = (2.694528, 1.155356)

#discretized time grid
t0 = 0
tf = 2
N = 100
t = cs.linspace(t0, tf, N+1)
dt = (tf-t0)/N

# create a optimization object using Opti class
nlp = cs.Opti()
X = nlp.variable(Nx, N+1)
U = nlp.variable(Nu, N)

# variable can be of the following tyoe
cs_type = Union[cs.MX, cs.SX, cs.DM]

# function to integrate the Lagrange term numerically
def obj_int(dt: cs_type, xv: cs_type, uv: cs_type, L: cs.Function, op: str):

    if op == "rie":
        return dt*cs.sum2(L(xv[:, 0:N], uv))
    elif op == "trap":
        return 0.5*dt*cs.sum2(L(xv[:, 0:N], uv)+L(xv[:, 1:N+1], uv))
    else:
        raise ValueError("wrong option")

# function to discretize the dynamics
def dyn_int(dt: cs_type, xv: cs_type, uv: cs_type, f: cs.Function, op: str):

    if op == "ef":
        xl = xv[:, 0:N]
        xr = xv[:, 1:N+1]
        return cs.vec(xr-xl-dt*(f(xl, uv)))
    elif op == "trap":
        xl = xv[:, 0:N]
        xr = xv[:, 1:N+1]
        return cs.vec(xr-xl-dt*(f(xl, uv)+f(xr, uv)))
    elif op == "her":
        xl = xv[:, 0:N]
        xr = xv[:, 1:N+1]
        xc = 0.5*(xl+xr)+1/8*(f(xl, uv)-f(xr, uv))
        return cs.vec(xr-xl-dt/6*(f(xl, uv)+4*f(xc, uv)+f(xr, uv)))
    elif op == "rk4":
        xl = xv[:, 0:N]
        xr = xv[:, 1:N+1]
        k1 = f(xl, uv)
        k2 = f(xl+k1*dt/2, uv)
        k3 = f(xl+k2*dt/2, uv)
        k4 = f(xl+k3*dt, uv)
        return cs.vec(xr-xl-1/6*dt*(k1+2*k2+2*k3+k4))
    else:
        raise ValueError("wrong option")

# choice of numerical methods
oop = 'trap'
dop = 'rk4'

# construct the objective function and discretize the dynamics
obj_L = obj_int(dt, X, U, L, oop)
obj_E = 0
res = dyn_int(dt, X, U, f, dop)

# construct and solve NLP
nlp.minimize(obj_L+obj_E)
nlp.subject_to(X[:, 0]-x_t0 == 0)
nlp.subject_to(res == 0)
nlp.subject_to(X[0, N]-a*X[1, N]+b == 0)
nlp.solver('ipopt')
sol=nlp.solve()

# plot the solution
plt.plot(t,sol.value(X[0,:]))
plt.show()


