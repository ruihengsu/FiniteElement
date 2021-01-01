from dolfin import *
from mshr import *

plane_w = 2.0/3
plane_h = 8.0/3

wall_w = 0.005/3
wall_h = 0.05/3
slit_h = 0.5/3

plane = Rectangle(Point(plane_w*4, plane_h), Point(-plane_w, -plane_h))
wall1 = Rectangle(Point(wall_w, slit_h), Point(-wall_w, -slit_h))
wall2 = Rectangle(Point(wall_w, plane_h), Point(-wall_w, slit_h+wall_h))
wall3 = Rectangle(Point(wall_w, -plane_h), Point(-wall_w, -slit_h-wall_h))
wall4 = Rectangle(Point(-plane_w, plane_h), Point(-wall_w, slit_h*2))
wall5 = Rectangle(Point(-plane_w, -plane_h), Point(-wall_w, -slit_h*2))

domain = plane - wall1 - wall2 - wall3 - wall4 - wall5
mesh = generate_mesh(domain, 128)

V = FunctionSpace(mesh, "CG", 1)

p = Function(V, name="p")
phi = Function(V, name="phi")

u = TrialFunction(V)
v = TestFunction(V)

# write the initial conditions:
vtkFile = File("solution/wave.pvd")
# this is zero everywhere
vtkFile << phi


def source_boundary(x, on_boundary):
    return on_boundary and near(x[0], -plane_w, DOLFIN_EPS)


bcval = Constant(0.0)
bc = DirichletBC(V, bcval, source_boundary)

t = 0
dt = 0.001
T = 10.0
step = 0
while t <= T:
    step += 1
    # u_source.t = t # updates the time, this changes our boundary conditions
    bcval.assign(sin(10*pi*t))

    phi.assign(phi - dt / 2 * p)

    solve(u*v*dx == v*p*dx + dt*inner(grad(v), grad(phi))*dx,
          p,
          bc)

    phi.assign(phi - dt / 2 * p)

    t += dt
    if step % 10 == 0:
        vtkFile << (phi, t)
