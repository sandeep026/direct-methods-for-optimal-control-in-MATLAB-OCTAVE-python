# %% [markdown]
# ## Import all packages and module

# %%
import casadi as cs
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from lglpsmethods import LGL

# %% [markdown]
# ## use symbolic casadi variables to create state and control vectors

# %%
Nx = 2
Nu = 1
x = cs.MX.sym("x", Nx, 1)
u = cs.MX.sym("u", Nu, 1)

# %% [markdown]
# ## create functions of the optimal control problem (Lagrange term, dynamics)

# %%
L = u**2
L = cs.Function('L', [x, u], [L])

f = cs.vcat([x[1], -x[1]+u])
f = cs.Function('L', [x, u], [f])

x_t0 = cs.DM.zeros(2, 1)
a, b = (2.694528, 1.155356)

# %% [markdown]
# ## create pseudospectral grid and compute the differentiation matrices and quadratures

# %%
t0 = 0
tf = 2
N = 30
dt = (tf-t0)/N
T=tf-t0
ps=LGL(N+1)
D=ps.D
tau=ps.tau
wi=ps.wi
t=(tf-t0)/2*tau+(tf+t0)/2

# %%
## create NLP using Opti() class

# %%
nlp = cs.Opti()
X = nlp.variable(Nx, N+1)
U = nlp.variable(Nu, N+1)

# %% [markdown]
# ## transcribe the optimal control problem

# %%
cs_type = Union[cs.MX, cs.SX, cs.DM]
def obj_int(wi,T: cs_type, xv: cs_type, uv: cs_type, L: cs.Function):

    return T/2*cs.dot(L(xv, uv),wi.T)
 
def dyn_int(D,T: cs_type, xv: cs_type, uv: cs_type, f: cs.Function):

    return cs.vec(2/T*(D@X.T).T-f(xv, uv))

obj_L = obj_int(wi,T,X,U,L)
obj_E = 0 # mayer term
res = dyn_int(D,T, X, U, f)

# %% [markdown]
# ## solve the optimization problem using IPOPT

# %%
nlp.minimize(obj_L+obj_E)
nlp.subject_to(X[:, 0]-x_t0 == 0)
nlp.subject_to(res == 0)
nlp.subject_to(X[0, N]-a*X[1, N]+b == 0)
nlp.solver('ipopt')
sol=nlp.solve()

# %% [markdown]
# ## analytical solution

# %%
t=(t.flatten())
us=np.exp(t)/4-1/2
z1s=-3/8*np.exp(-t)+np.exp(t)/8-1/2*t+1/4
z2s=3/8*np.exp(-t)+1/8*np.exp(t)-1/2
Js=1/32*(8*2-8*np.exp(2)+np.exp(2*2))-1/32*(8*0-8*np.exp(0)+np.exp(2*0))
print('Optimal objective error (|Analytical-Numerical|)--',np.abs(Js-sol.value(nlp.f)))

# %% [markdown]
# ## plots

# %%
plt.plot(sol.value(X[0,:]),sol.value(X[1,:]),label='LGL')
plt.scatter(z1s,z2s,label='analytical',c='r')
plt.xlabel('z1')
plt.ylabel('z2')
plt.legend()
plt.grid()
plt.show()
plt.figure()
plt.plot(t,sol.value(U),label='LGL')
plt.plot(t,us,label='analytical',ls='-.')
plt.xlabel('time [s]')
plt.ylabel('f')
plt.legend()
plt.grid()
plt.show()


