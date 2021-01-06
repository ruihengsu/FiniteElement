from dolfin import *
from mshr import *

beta = 0.25
gamma = 0.5

t = 0.0
dt = 0.001
T = 10.0

mesh = RectangleMesh(Point(-1, -1), Point(1, 1), 64, 64)

order = 2  # order of basis functions
V = FunctionSpace(mesh, "CG", order)

initial_displacment = Constant(0.0)
initial_velocity = Constant(0.0)

u0 = interpolate(initial_displacment, V)
v0 = interpolate(initial_velocity, V)

u = TrialFunction(V)
v = TestFunction(V)
u_new = Function(V)                   # displacement (solution)
v_new = Function(V)                   # velocity
a_new = Function(V)                   # acceleration


def boundary1(x, on_boundary):
    cond1 = on_boundary and near(x[0], -1)
    cond2 = x[1] >= -1.0/3.0 and x[1] <= 1.0/3.0
    return cond1 and cond2


def boundary2(x, on_boundary):
    cond1 = on_boundary
    cond2 = not (x[1] >= -1.0/3.0 and x[1] <= 1.0/3.0 and near(x[0], -1))
    return cond1 and cond2


u_D1 = Constant(0.0)
u_D2 = Constant(0.0)

bc1 = DirichletBC(V, u_D1, boundary1)
bc2 = DirichletBC(V, u_D2, boundary2)
bc = [bc1, bc2]

# neumann = Expression("(x[0] + 1 <= tol && x[1] > -1.0/3.0 && x[1] < 1.0/3.0) ? 4*pi*cos(4*pi*t) : 0",
#                      t = 0,
#                      tol = DOLFIN_EPS,
#                      degree = order)


k = inner(grad(u), grad(v))*dx  # stiffness matrix integrand
m = u*v*dx  # mass matrix integrand
# f = v*neumann*ds # neumann boundary condition

K = assemble(k)  # assemble stiffness matrix
M = assemble(m)  # assemble mass matrix
# F = assemble(f)

a0 = Function(V)
solve(m == -k*u0, a0, bc)  # this gives us a0 to start our algorithm

vtkFile_a = File("results/p{}_acc.pvd".format(order))
vtkFile_a << a0

vtkFile_u = File("results/p{}_displacement.pvd".format(order))
vtkFile_v = File("results/p{}_vel.pvd".format(order))

step = 0

while t <= T:

    if step % 10 == 0:
        vtkFile_u << (u0, t)
        vtkFile_v << (v0, t)

    # predictor
    u_p = u0.vector() + dt*v0.vector() + pow(dt, 2)*(0.5 - beta)*a0.vector()
    v_p = v0.vector() + dt*(1-gamma)*a0.vector()

    b = -1*K*u_p
    A = M + pow(dt, 2)*beta*K

    for boundary_condition in bc:
        boundary_condition.apply(A, b)

    solve(A, a_new.vector(), b)

    u_new.vector()[:] = u_p + pow(dt, 2)*beta*a_new.vector()
    v_new.vector()[:] = v_p + dt*gamma*a_new.vector()

    u0.assign(u_new)
    v0.assign(v_new)
    a0.assign(a_new)

    t += dt
    # neumann.t = t

    if t <= 0.5:
        u_D1.assign(sin(4*pi*t))
    else:
        u_D1.assign(0)
    step += 1
