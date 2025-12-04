"""
SteadyFlowCylinder_final_with_LBFGS.py
Full PINN (mixed variables psi,p,s11,s22,s12) + Adam + L-BFGS optimization
+ high-resolution postprocessing (500x500)
Python 3.10+, TensorFlow 2.x, SciPy
"""

import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pyDOE2 import lhs
from scipy.optimize import minimize

# -------------------------
# Configuration (change if needed)
# -------------------------
tf.keras.backend.set_floatx('float32')
np.random.seed(1234)
tf.random.set_seed(1234)

# Domain and cylinder geometry (same as earlier)
LB = np.array([0.0, 0.0], dtype=np.float32)
UB = np.array([1.1, 0.41], dtype=np.float32)
CYL_X, CYL_Y, CYL_R = 0.2, 0.2, 0.05

# Network architecture
LAYERS = [2] + 8*[40] + [5]   # output: [psi, p, s11, s22, s12]

# Training hyperparameters (Paper-level)
ADAM_EPOCHS = 1000     # you chose 'C' paper-level
ADAM_LR = 5e-4
LBFGS_MAXITER = 1000

# Postproc grid resolution (you requested option C)
NX = 500
NY = 500

# Collocation sampling size
N_COLLO = 40000

# Device: prefer GPU if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("Using GPU:", gpus[0])
    except Exception:
        pass
else:
    print("No GPU found — running on CPU.")

# -------------------------
# Utilities
# -------------------------
def DelCylPT(XY_c, xc=CYL_X, yc=CYL_Y, r=CYL_R):
    dst = np.sqrt((XY_c[:,0]-xc)**2 + (XY_c[:,1]-yc)**2)
    return XY_c[dst >= r]

# chunked evaluation utility to avoid memory spikes
def chunked_predict(fn, X_all, chunk=20000):
    N = X_all.shape[0]
    out_list = []
    for i in range(0, N, chunk):
        Xi = X_all[i:i+chunk]
        out_list.append(fn(Xi))
    return np.vstack(out_list)

# -------------------------
# PINN model (mixed formulation)
# -------------------------
class PINN_mixed(tf.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.weights = []
        self.biases = []
        # Initialize weights and biases as tf.Variable float32
        for i in range(len(layers)-1):
            w = tf.Variable(tf.random.truncated_normal([layers[i], layers[i+1]],
                                                       stddev=np.sqrt(2/(layers[i]+layers[i+1]))),
                            dtype=tf.float32)
            b = tf.Variable(tf.zeros([1, layers[i+1]], dtype=tf.float32), dtype=tf.float32)
            self.weights.append(w)
            self.biases.append(b)

    def neural_net(self, X):
        H = X
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            H = tf.tanh(tf.matmul(H, W) + b)
        Y = tf.matmul(H, self.weights[-1]) + self.biases[-1]
        return Y  # (N,5): psi,p,s11,s22,s12

    def net_mixed(self, X):
        """Return u,v,p,s11,s22,s12 computed from psi and network outputs"""
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        # compute inside gradient tape so psi is connected to X
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(X)
            out = self.neural_net(X)
            psi = out[:,0:1]
            p   = out[:,1:2]
            s11 = out[:,2:3]
            s22 = out[:,3:4]
            s12 = out[:,4:5]
            grads = tape.gradient(psi, X)   # (N,2): [dpsi/dx, dpsi/dy]
        u = grads[:,1:2]   # dpsi/dy
        v = -grads[:,0:1]  # -dpsi/dx
        # return TensorFlow tensors (float32)
        return u, v, p, s11, s22, s12

    def residuals(self, X):
        """Compute residuals f_u,f_v,f_s11,f_s22,f_s12,f_p at points X"""
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        rho = tf.constant(1.0, dtype=tf.float32)
        mu  = tf.constant(0.02, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as t2:
            t2.watch(X)
            u, v, p, s11, s22, s12 = self.net_mixed(X)

            # first derivatives
            u_x = t2.gradient(u, X)[:,0:1]
            u_y = t2.gradient(u, X)[:,1:2]
            v_x = t2.gradient(v, X)[:,0:1]
            v_y = t2.gradient(v, X)[:,1:2]

            s11_x = t2.gradient(s11, X)[:,0:1]
            s12_x = t2.gradient(s12, X)[:,0:1]
            s12_y = t2.gradient(s12, X)[:,1:2]
            s22_y = t2.gradient(s22, X)[:,1:2]

        # cast to float32 to avoid dtype conflicts
        u = tf.cast(u, tf.float32); v = tf.cast(v, tf.float32); p = tf.cast(p, tf.float32)
        s11 = tf.cast(s11, tf.float32); s22 = tf.cast(s22, tf.float32); s12 = tf.cast(s12, tf.float32)
        u_x = tf.cast(u_x, tf.float32); u_y = tf.cast(u_y, tf.float32)
        v_x = tf.cast(v_x, tf.float32); v_y = tf.cast(v_y, tf.float32)
        s11_x = tf.cast(s11_x, tf.float32); s12_x = tf.cast(s12_x, tf.float32)
        s12_y = tf.cast(s12_y, tf.float32); s22_y = tf.cast(s22_y, tf.float32)

        f_u = rho*(u * u_x + v * u_y) - s11_x - s12_y
        f_v = rho*(u * v_x + v * v_y) - s12_x - s22_y

        f_s11 = -p + 2.0 * mu * u_x - s11
        f_s22 = -p + 2.0 * mu * v_y - s22
        f_s12 = mu * (u_y + v_x) - s12

        f_p = p + 0.5*(s11 + s22)

        return f_u, f_v, f_s11, f_s22, f_s12, f_p

# -------------------------
# Build datasets (collocation + boundaries)
# -------------------------
# Collocation: Latin hypercube sampling, delete cylinder interior
XY_c = LB + (UB - LB) * lhs(2, N_COLLO)
XY_c = DelCylPT(XY_c, xc=CYL_X, yc=CYL_Y, r=CYL_R).astype(np.float32)

# Cylinder surface (for plotting)
theta = np.linspace(0, 2*np.pi, 300)
CYLD = np.vstack([CYL_X + CYL_R*np.cos(theta), CYL_Y + CYL_R*np.sin(theta)]).T.astype(np.float32)

# Channel walls (top/bottom)
xwall = np.linspace(LB[0], UB[0], 600)[:,None].astype(np.float32)
WALL_bottom = np.hstack([xwall, np.full_like(xwall, LB[1])])
WALL_top    = np.hstack([xwall, np.full_like(xwall, UB[1])])
WALL = np.vstack([WALL_bottom, WALL_top]).astype(np.float32)

# INLET (x=LB[0]) Poiseuille
ny_in = 600
y_in = np.linspace(LB[1], UB[1], ny_in)[:,None].astype(np.float32)
INLET_xy = np.hstack([np.zeros_like(y_in), y_in])
Umax = 1.0
u_INLET = (4.0 * Umax * y_in * (UB[1] - y_in) / (UB[1]**2)).astype(np.float32)
v_INLET = np.zeros_like(u_INLET, dtype=np.float32)

# OUTLET (x=UB[0]), enforce p ~ 0
OUTLET_xy = np.hstack([np.full_like(y_in, UB[0]), y_in]).astype(np.float32)

# -------------------------
# Instantiate model
# -------------------------
pinn = PINN_mixed(LAYERS)
# Create Adam optimizer
adam_optimizer = tf.keras.optimizers.Adam(learning_rate=ADAM_LR)

# -------------------------
# Loss wrappers and train step
# -------------------------
@tf.function
def compute_total_loss(Xc, Xin, uin, vin, Xwall, Xout):
    # PDE residuals
    f_u, f_v, f_s11, f_s22, f_s12, f_p = pinn.residuals(Xc)
    loss_f = tf.reduce_mean(tf.square(f_u)) + tf.reduce_mean(tf.square(f_v)) \
             + tf.reduce_mean(tf.square(f_s11)) + tf.reduce_mean(tf.square(f_s22)) \
             + tf.reduce_mean(tf.square(f_s12)) + tf.reduce_mean(tf.square(f_p))

    # Wall BC (no-slip)
    u_w, v_w, _, _, _, _ = pinn.net_mixed(Xwall)
    loss_wall = tf.reduce_mean(tf.square(u_w)) + tf.reduce_mean(tf.square(v_w))

    # Inlet BC: velocity match
    u_in_pred, v_in_pred, _, _, _, _ = pinn.net_mixed(Xin)
    loss_in = tf.reduce_mean(tf.square(u_in_pred - uin)) + tf.reduce_mean(tf.square(v_in_pred - vin))

    # Outlet BC: pressure ~ 0
    _, _, p_out, _, _, _ = pinn.net_mixed(Xout)
    loss_out = tf.reduce_mean(tf.square(p_out))

    total = loss_f + 2.0*(loss_wall + loss_in + loss_out)
    return total

@tf.function
def adam_train_step(Xc, Xin, uin, vin, Xwall, Xout):
    with tf.GradientTape() as tape:
        loss_val = compute_total_loss(Xc, Xin, uin, vin, Xwall, Xout)
    grads = tape.gradient(loss_val, pinn.weights + pinn.biases)
    adam_optimizer.apply_gradients(zip(grads, pinn.weights + pinn.biases))
    return loss_val

# -------------------------
# Run Adam training
# -------------------------
print("Starting Adam training: epochs =", ADAM_EPOCHS)
t0 = time.time()
# Prepare numpy arrays for feeding
Xc_train = XY_c.astype(np.float32)
Xin_train = INLET_xy.astype(np.float32)
uin_train = u_INLET.astype(np.float32)
vin_train = v_INLET.astype(np.float32)
Xwall_train = WALL.astype(np.float32)
Xout_train = OUTLET_xy.astype(np.float32)

# Run epochs (status printed every 100)
for ep in range(ADAM_EPOCHS):
    lval = adam_train_step(Xc_train, Xin_train, uin_train, vin_train, Xwall_train, Xout_train)
    if (ep % 100) == 0:
        print(f"Epoch {ep:6d}, Loss = {lval.numpy():.6e}")
print("Adam training completed in {:.1f} s".format(time.time()-t0))

# -------------------------
# L-BFGS-B refinement using SciPy
# -------------------------
# Flatten model variables to a single 1D vector and provide set/get utilities
def get_variables_vector():
    vecs = []
    for W in pinn.weights:
        vecs.append(W.numpy().ravel())
    for b in pinn.biases:
        vecs.append(b.numpy().ravel())
    return np.concatenate(vecs).astype(np.float64)

def set_variables_vector(x):
    # assign values back to weights and biases from 1D numpy array x (double)
    x = x.astype(np.float32)  # convert to float32 for tf.assign
    idx = 0
    for W in pinn.weights:
        shape = W.shape
        size = tf.size(W).numpy()
        newW = x[idx:idx+size].reshape(shape)
        W.assign(newW)
        idx += size
    for b in pinn.biases:
        shape = b.shape
        size = tf.size(b).numpy()
        newb = x[idx:idx+size].reshape(shape)
        b.assign(newb)
        idx += size

# Objective function for SciPy (value + gradient)
def scipy_loss_and_grad(x_numpy):
    # set variables
    set_variables_vector(x_numpy)
    # compute loss and gradients using TF
    with tf.GradientTape() as tape:
        # make trainable variables list
        vars_tf = pinn.weights + pinn.biases
        loss_tf = compute_total_loss(Xc_train, Xin_train, uin_train, vin_train, Xwall_train, Xout_train)
    grads_tf = tape.gradient(loss_tf, vars_tf)
    # flatten gradients to 1D numpy in same order as get_variables_vector
    grads_list = []
    for g in grads_tf:
        grads_list.append(tf.reshape(g, [-1]).numpy().astype(np.float64))
    grad_numpy = np.concatenate(grads_list)
    return loss_tf.numpy().astype(np.float64), grad_numpy

# SciPy minimize wrapper (L-BFGS-B)
print("Starting L-BFGS-B optimization (may take time)...")
x0 = get_variables_vector()
result = minimize(fun=lambda x: scipy_loss_and_grad(x)[0],
                  x0=x0,
                  jac=lambda x: scipy_loss_and_grad(x)[1],
                  method='L-BFGS-B',
                  options={'maxiter': LBFGS_MAXITER, 'maxcor':50, 'ftol':1e-12, 'gtol':1e-8, 'disp': True})
# assign final weights
set_variables_vector(result.x)
print("L-BFGS-B finished, message:", result.message)

# -------------------------
# Postprocessing (high-resolution 500x500 grid)
# -------------------------
print("Starting post-processing: building high-resolution grid", NX, "x", NY)
xg = np.linspace(LB[0], UB[0], NX)
yg = np.linspace(LB[1], UB[1], NY)
Xg, Yg = np.meshgrid(xg, yg)
points = np.vstack([Xg.ravel(), Yg.ravel()]).T.astype(np.float32)

# mask out points inside cylinder
dist = np.sqrt((points[:,0]-CYL_X)**2 + (points[:,1]-CYL_Y)**2)
mask = dist >= CYL_R
points_out = points[mask]

# predict in chunks to avoid OOM
def predict_mixed_numpy(Xpts):
    # returns u,v,p,s11,s22,s12 as numpy arrays
    Xtf = tf.convert_to_tensor(Xpts.astype(np.float32))
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(Xtf)
        out = pinn.neural_net(Xtf)
        psi = out[:,0:1]
        p   = out[:,1:2]
        s11 = out[:,2:3]
        s22 = out[:,3:4]
        s12 = out[:,4:5]
        grads = tape.gradient(psi, Xtf)
    u = grads[:,1:2].numpy()
    v = (-grads[:,0:1]).numpy()
    return u, v, p.numpy(), s11.numpy(), s22.numpy(), s12.numpy()

u_all, v_all, p_all, s11_all, s22_all, s12_all = [],[],[],[],[],[]
chunk = 20000
for i in range(0, points_out.shape[0], chunk):
    Xi = points_out[i:i+chunk]
    u_ch, v_ch, p_ch, s11_ch, s22_ch, s12_ch = predict_mixed_numpy(Xi)
    u_all.append(u_ch); v_all.append(v_ch); p_all.append(p_ch)
    s11_all.append(s11_ch); s22_all.append(s22_ch); s12_all.append(s12_ch)
u_all = np.vstack(u_all).flatten()
v_all = np.vstack(v_all).flatten()
p_all = np.vstack(p_all).flatten()
s11_all = np.vstack(s11_all).flatten()
s22_all = np.vstack(s22_all).flatten()
s12_all = np.vstack(s12_all).flatten()

# fill full grids (with nan inside cylinder)
Ugrid = np.full(points.shape[0], np.nan, dtype=np.float32)
Vgrid = np.full(points.shape[0], np.nan, dtype=np.float32)
Pgrid = np.full(points.shape[0], np.nan, dtype=np.float32)
S11grid = np.full(points.shape[0], np.nan, dtype=np.float32)
S22grid = np.full(points.shape[0], np.nan, dtype=np.float32)
S12grid = np.full(points.shape[0], np.nan, dtype=np.float32)

Ugrid[mask] = u_all
Vgrid[mask] = v_all
Pgrid[mask] = p_all
S11grid[mask] = s11_all
S22grid[mask] = s22_all
S12grid[mask] = s12_all

Umat = Ugrid.reshape(Xg.shape)
Vmat = Vgrid.reshape(Xg.shape)
Pmat = Pgrid.reshape(Xg.shape)
S11mat = S11grid.reshape(Xg.shape)
S22mat = S22grid.reshape(Xg.shape)
S12mat = S12grid.reshape(Xg.shape)

# -------------------------
# Plot and save figures
# -------------------------
print("Saving figures to disk...")

# Collocation and boundary points
plt.figure(figsize=(8,4))
plt.scatter(XY_c[:,0], XY_c[:,1], s=0.7, alpha=0.15, label='collocation')
plt.scatter(WALL[:,0], WALL[:,1], s=4, color='green', alpha=0.6, label='walls')
plt.scatter(CYLD[:,0], CYLD[:,1], s=6, color='red', label='cylinder')
plt.scatter(INLET_xy[:,0], INLET_xy[:,1], s=2, color='orange', label='inlet')
plt.scatter(OUTLET_xy[:,0], OUTLET_xy[:,1], s=2, color='purple', label='outlet')
plt.legend(markerscale=3, fontsize=8)
plt.title('Collocation and boundary points')
plt.xlabel('x'); plt.ylabel('y')
plt.axis('equal')
plt.tight_layout()
plt.savefig('collocation_final.png', dpi=200)
plt.show()

# u (PINN) contour
plt.figure(figsize=(8,4))
plt.contourf(Xg, Yg, Umat, 50, cmap='viridis')
plt.colorbar(label='u (m/s)')
circle = plt.Circle((CYL_X, CYL_Y), CYL_R, color='k', fill=False, linewidth=1.0)
plt.gca().add_patch(circle)
plt.title('u velocity (PINN)')
plt.xlabel('x'); plt.ylabel('y'); plt.axis('equal')
plt.tight_layout()
plt.savefig('u_pinn.png', dpi=200)
plt.show()

# v (PINN) contour
plt.figure(figsize=(8,4))
plt.contourf(Xg, Yg, Vmat, 50, cmap='viridis')
plt.colorbar(label='v (m/s)')
plt.gca().add_patch(circle)
plt.title('v velocity (PINN)')
plt.xlabel('x'); plt.ylabel('y'); plt.axis('equal')
plt.tight_layout()
plt.savefig('v_pinn.png', dpi=200)
plt.show()

# p (PINN) contour
plt.figure(figsize=(8,4))
plt.contourf(Xg, Yg, Pmat, 50, cmap='rainbow')
plt.colorbar(label='p (Pa)')
plt.gca().add_patch(circle)
plt.title('pressure (PINN)')
plt.xlabel('x'); plt.ylabel('y'); plt.axis('equal')
plt.tight_layout()
plt.savefig('p_pinn.png', dpi=200)
plt.show()

# Streamfunction (psi) contour: evaluate psi on grid (masked)
def predict_psi(Xpts):
    Xtf = tf.convert_to_tensor(Xpts.astype(np.float32))
    out = pinn.neural_net(Xtf).numpy()
    psi_vals = out[:,0]
    return psi_vals

psi_vals_all = []
chunk = 20000
for i in range(0, points_out.shape[0], chunk):
    psi_vals_all.append(predict_psi(points_out[i:i+chunk]))
psi_vals_all = np.concatenate(psi_vals_all)
psi_grid = np.full(Xg.size, np.nan, dtype=np.float32)
psi_grid[mask] = psi_vals_all
Psimat = psi_grid.reshape(Xg.shape)

plt.figure(figsize=(8,4))
cs = plt.contourf(Xg, Yg, Psimat, 60, cmap='magma')
plt.gca().add_patch(circle)
plt.colorbar(label='streamfunction ψ')
plt.title('Streamfunction (ψ) - PINN')
plt.xlabel('x'); plt.ylabel('y'); plt.axis('equal')
plt.tight_layout()
plt.savefig('psi_contour_final.png', dpi=200)
plt.show()

# Stress component examples
plt.figure(figsize=(8,4))
plt.contourf(Xg, Yg, S11mat, 40, cmap='coolwarm')
plt.colorbar(label='s11')
plt.gca().add_patch(circle)
plt.title('Stress s11 (PINN)')
plt.axis('equal')
plt.tight_layout()
plt.savefig('s11_pinn.png', dpi=200)
plt.show()

print("All postprocessing images saved to current folder:")
print("collocation_final.png, u_pinn.png, v_pinn.png, p_pinn.png, psi_contour_final.png, s11_pinn.png")
print("Done.")
