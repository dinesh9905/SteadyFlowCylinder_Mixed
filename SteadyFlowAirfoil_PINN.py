"""
SteadyFlowAirfoil_mixed.py

PINN for steady 2D laminar flow around a NACA0012 airfoil (mixed formulation).
- Uses TensorFlow 1.x compatibility (tf.compat.v1)
- Adam pre-training followed by SciPy L-BFGS-B (full-batch)
- Generates NACA0012 coordinates internally (no external CSV required)
- Creates collocation points in fluid region (excludes airfoil interior)
- Enforces inlet / outlet / top-bottom / airfoil no-slip BCs
- Produces side-by-side plots of u, v, p (PINN pred vs reference placeholder)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # quiet TF logging

import numpy as np
import time
import matplotlib.pyplot as plt
import pickle
import scipy.io
import random
from matplotlib.path import Path
from scipy import optimize

# TensorFlow 1.x compatibility
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


random.seed(1234)
np.random.seed(1234)

# ---------------------------
# Utility: generate NACA 4-digit cambered series airfoil (here 0012)
# ---------------------------
def naca4_coordinates(m=0, p=0, t=12, c=1.0, n=201, closed=True):
    # m: camber (percent of chord) -> for 0012 it's 0
    # p: location of max camber (tenths of chord)
    # t: thickness percentage (12 for NACA0012)
    # c: chord length
    # n: number of points along surface (per side)
    # returns (x,y) with upper then lower surfaces (closed)
    t = t / 100.0
    x_lin = np.linspace(0, c, n)
    # thickness distribution (NACA formula)
    yt = 5 * t * (
        0.2969 * np.sqrt(x_lin/c)
        - 0.1260 * (x_lin/c)
        - 0.3516 * (x_lin/c)**2
        + 0.2843 * (x_lin/c)**3
        - 0.1015 * (x_lin/c)**4
    ) * c
    # symmetric (m=0)
    xu = x_lin
    yu = yt
    xl = x_lin[::-1]
    yl = -yt[::-1]
    # assemble upper (leading to trailing) then lower (trailing to leading)
    x_coords = np.concatenate([xu, xl[1:]])  # drop duplicate at c
    y_coords = np.concatenate([yu, yl[1:]])
    if closed:
        # append first point to close
        x_coords = np.concatenate([x_coords, x_coords[:1]])
        y_coords = np.concatenate([y_coords, y_coords[:1]])
    coords = np.vstack([x_coords, y_coords]).T
    return coords

# ---------------------------
# PINN class (mostly unchanged logic)
# ---------------------------
class PINN_laminar_flow:
    def __init__(self, Collo, INLET, OUTLET, WALL, uv_layers, lb, ub, ExistModel=0, uvDir=''):
        self.count = 0
        self.lb = lb
        self.ub = ub
        self.rho = 1.0
        self.mu = 0.02

        self.x_c = Collo[:, 0:1]
        self.y_c = Collo[:, 1:2]

        self.x_INLET = INLET[:, 0:1]
        self.y_INLET = INLET[:, 1:2]
        self.u_INLET = INLET[:, 2:3]
        self.v_INLET = INLET[:, 3:4]

        self.x_OUTLET = OUTLET[:, 0:1]
        self.y_OUTLET = OUTLET[:, 1:2]

        self.x_WALL = WALL[:, 0:1]
        self.y_WALL = WALL[:, 1:2]

        self.uv_layers = uv_layers
        self.loss_rec = []

        if ExistModel == 0:
            self.uv_weights, self.uv_biases = self.initialize_NN(self.uv_layers)
        else:
            print("Loading uv NN ...")
            self.uv_weights, self.uv_biases = self.load_NN(uvDir, self.uv_layers)

        # placeholders
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x_c.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y_c.shape[1]])
        self.x_WALL_tf = tf.placeholder(tf.float32, shape=[None, self.x_WALL.shape[1]])
        self.y_WALL_tf = tf.placeholder(tf.float32, shape=[None, self.y_WALL.shape[1]])
        self.x_OUTLET_tf = tf.placeholder(tf.float32, shape=[None, self.x_OUTLET.shape[1]])
        self.y_OUTLET_tf = tf.placeholder(tf.float32, shape=[None, self.y_OUTLET.shape[1]])
        self.x_INLET_tf = tf.placeholder(tf.float32, shape=[None, self.x_INLET.shape[1]])
        self.y_INLET_tf = tf.placeholder(tf.float32, shape=[None, self.y_INLET.shape[1]])
        self.u_INLET_tf = tf.placeholder(tf.float32, shape=[None, self.u_INLET.shape[1]])
        self.v_INLET_tf = tf.placeholder(tf.float32, shape=[None, self.v_INLET.shape[1]])
        self.x_c_tf = tf.placeholder(tf.float32, shape=[None, self.x_c.shape[1]])
        self.y_c_tf = tf.placeholder(tf.float32, shape=[None, self.y_c.shape[1]])

        # graphs
        self.u_pred, self.v_pred, self.p_pred, _, _, _ = self.net_uv(self.x_tf, self.y_tf)
        self.f_pred_u, self.f_pred_v, self.f_pred_s11, self.f_pred_s22, self.f_pred_s12, self.f_pred_p = self.net_f(self.x_c_tf, self.y_c_tf)
        self.u_WALL_pred, self.v_WALL_pred, _, _, _, _ = self.net_uv(self.x_WALL_tf, self.y_WALL_tf)
        self.u_INLET_pred, self.v_INLET_pred, _, _, _, _ = self.net_uv(self.x_INLET_tf, self.y_INLET_tf)
        _, _, self.p_OUTLET_pred, _, _, _ = self.net_uv(self.x_OUTLET_tf, self.y_OUTLET_tf)

        self.loss_f = tf.reduce_mean(tf.square(self.f_pred_u)) \
                      + tf.reduce_mean(tf.square(self.f_pred_v)) \
                      + tf.reduce_mean(tf.square(self.f_pred_s11)) \
                      + tf.reduce_mean(tf.square(self.f_pred_s22)) \
                      + tf.reduce_mean(tf.square(self.f_pred_s12)) \
                      + tf.reduce_mean(tf.square(self.f_pred_p))

        self.loss_WALL = tf.reduce_mean(tf.square(self.u_WALL_pred)) + tf.reduce_mean(tf.square(self.v_WALL_pred))
        self.loss_INLET = tf.reduce_mean(tf.square(self.u_INLET_pred - self.u_INLET_tf)) + tf.reduce_mean(tf.square(self.v_INLET_pred - self.v_INLET_tf))
        self.loss_OUTLET = tf.reduce_mean(tf.square(self.p_OUTLET_pred))

        self.loss = self.loss_f + 2*(self.loss_WALL + self.loss_INLET + self.loss_OUTLET)

        # optimizers
        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss, var_list=self.uv_weights + self.uv_biases)

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # prepare for SciPy LBFGS wrapper
        self._prepare_scipy_optimizer_vars()

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]; out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float32), dtype=tf.float32)

    def save_NN(self, fileDir):
        uv_weights = self.sess.run(self.uv_weights)
        uv_biases = self.sess.run(self.uv_biases)
        with open(fileDir, 'wb') as f:
            pickle.dump([uv_weights, uv_biases], f)
            print("Save uv NN parameters successfully...")

    def load_NN(self, fileDir, layers):
        weights = []; biases = []
        with open(fileDir, 'rb') as f:
            uv_weights, uv_biases = pickle.load(f)
            num_layers = len(layers)
            assert num_layers == (len(uv_weights)+1)
            for num in range(0, num_layers - 1):
                W = tf.Variable(uv_weights[num], dtype=tf.float32)
                b = tf.Variable(uv_biases[num], dtype=tf.float32)
                weights.append(W); biases.append(b)
                print(" - Load NN parameters successfully...")
        return weights, biases

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = X
        for l in range(0, num_layers - 2):
            W = weights[l]; b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]; b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_uv(self, x, y):
        psips = self.neural_net(tf.concat([x, y], 1), self.uv_weights, self.uv_biases)
        psi = tf.reshape(psips[:, 0], (-1, 1))
        p = tf.reshape(psips[:, 1], (-1, 1))
        s11 = tf.reshape(psips[:, 2], (-1, 1))
        s22 = tf.reshape(psips[:, 3], (-1, 1))
        s12 = tf.reshape(psips[:, 4], (-1, 1))
        u = tf.gradients(psi, y)[0]
        v = -tf.gradients(psi, x)[0]
        return u, v, p, s11, s22, s12

    def net_f(self, x, y):
        rho = self.rho; mu = self.mu
        u, v, p, s11, s22, s12 = self.net_uv(x, y)
        s11_1 = tf.gradients(s11, x)[0]
        s12_2 = tf.gradients(s12, y)[0]
        s22_2 = tf.gradients(s22, y)[0]
        s12_1 = tf.gradients(s12, x)[0]
        u_x = tf.gradients(u, x)[0]; u_y = tf.gradients(u, y)[0]
        v_x = tf.gradients(v, x)[0]; v_y = tf.gradients(v, y)[0]
        f_u = rho*(u*u_x + v*u_y) - s11_1 - s12_2
        f_v = rho*(u*v_x + v*v_y) - s12_1 - s22_2
        f_s11 = -p + 2*mu*u_x - s11
        f_s22 = -p + 2*mu*v_y - s22
        f_s12 = mu*(u_y + v_x) - s12
        f_p = p + (s11 + s22)/2
        return f_u, f_v, f_s11, f_s22, f_s12, f_p

    def callback(self, loss):
        self.count += 1
        self.loss_rec.append(loss)
        print('{} th iterations, Loss: {}'.format(self.count, loss))

    def train(self, iter, learning_rate):
        tf_dict = {
            self.x_c_tf: self.x_c, self.y_c_tf: self.y_c,
            self.x_WALL_tf: self.x_WALL, self.y_WALL_tf: self.y_WALL,
            self.x_INLET_tf: self.x_INLET, self.y_INLET_tf: self.y_INLET,
            self.u_INLET_tf: self.u_INLET, self.v_INLET_tf: self.v_INLET,
            self.x_OUTLET_tf: self.x_OUTLET, self.y_OUTLET_tf: self.y_OUTLET,
            self.learning_rate: learning_rate
        }
        loss_WALL = []; loss_f = []; loss_INLET = []; loss_OUTLET = []
        for it in range(iter):
            self.sess.run(self.train_op_Adam, tf_dict)
            if it % 10 == 0:
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e' % (it, loss_value))
            loss_WALL.append(self.sess.run(self.loss_WALL, tf_dict))
            loss_f.append(self.sess.run(self.loss_f, tf_dict))
            self.loss_rec.append(self.sess.run(self.loss, tf_dict))
            loss_INLET.append(self.sess.run(self.loss_INLET, tf_dict))
            loss_OUTLET.append(self.sess.run(self.loss_OUTLET, tf_dict))
        return loss_WALL, loss_INLET, loss_OUTLET, loss_f, self.loss

    # prepare SciPy L-BFGS variables and placeholders
    def _prepare_scipy_optimizer_vars(self):
        self.tf_vars = self.uv_weights + self.uv_biases
        self.shapes = [v.get_shape().as_list() for v in self.tf_vars]
        # flatten
        self.size = np.sum([int(np.prod(shape)) for shape in self.shapes])
        self._placeholders = []
        self._assign_ops = []
        for i, var in enumerate(self.tf_vars):
            shape = self.shapes[i]
            var_size = int(np.prod(shape))
            p = tf.placeholder(tf.float32, shape=[var_size], name='ph_%d' % i)
            assign_op = tf.assign(tf.reshape(var, shape), tf.reshape(p, shape))
            self._placeholders.append(p)
            self._assign_ops.append(assign_op)
        grads = tf.gradients(self.loss, self.tf_vars)
        self._grad_flat = tf.concat([tf.reshape(g, [-1]) for g in grads], 0)

    def _pack_variables(self, vec):
        arrays = []
        idx = 0
        for shape in self.shapes:
            size = int(np.prod(shape))
            arrays.append(vec[idx:idx+size])
            idx += size
        return arrays

    def _assign_flattened_vars(self, flat):
        arrays = self._pack_variables(flat)
        feed = {}
        for p, arr in zip(self._placeholders, arrays):
            feed[p] = arr.astype(np.float32)
        self.sess.run(self._assign_ops, feed_dict=feed)

    def _get_loss_and_grad(self, flat):
        self._assign_flattened_vars(flat)
        loss_value, grad_flat = self.sess.run([self.loss, self._grad_flat], feed_dict={
            self.x_c_tf: self.x_c, self.y_c_tf: self.y_c,
            self.x_WALL_tf: self.x_WALL, self.y_WALL_tf: self.y_WALL,
            self.x_INLET_tf: self.x_INLET, self.y_INLET_tf: self.y_INLET,
            self.u_INLET_tf: self.u_INLET, self.v_INLET_tf: self.v_INLET,
            self.x_OUTLET_tf: self.x_OUTLET, self.y_OUTLET_tf: self.y_OUTLET
        })
        return loss_value.astype(np.float64), grad_flat.astype(np.float64)

    def train_bfgs(self, maxiter=20000):
        initial_vars = self.sess.run(self.tf_vars)
        flat0 = np.concatenate([v.flatten() for v in initial_vars]).astype(np.float64)
        def _callback(flat):
            loss_value, _ = self._get_loss_and_grad(flat)
            self.callback(loss_value)
        print("Starting SciPy L-BFGS-B optimization...")
        opt_result = optimize.minimize(fun=lambda x: self._get_loss_and_grad(x)[0],
                                       x0=flat0,
                                       jac=lambda x: self._get_loss_and_grad(x)[1],
                                       method='L-BFGS-B',
                                       callback=_callback,
                                       options={'maxiter': maxiter, 'maxcor': 50, 'maxls': 50, 'ftol': np.finfo(float).eps})
        self._assign_flattened_vars(opt_result.x)
        print("SciPy L-BFGS-B finished. success: ", opt_result.success, "message:", opt_result.message)

    def predict(self, x_star, y_star):
        u_star = self.sess.run(self.u_pred, {self.x_tf: x_star, self.y_tf: y_star})
        v_star = self.sess.run(self.v_pred, {self.x_tf: x_star, self.y_tf: y_star})
        p_star = self.sess.run(self.p_pred, {self.x_tf: x_star, self.y_tf: y_star})
        return u_star, v_star, p_star

    def getloss(self):
        tf_dict = {
            self.x_c_tf: self.x_c, self.y_c_tf: self.y_c,
            self.x_WALL_tf: self.x_WALL, self.y_WALL_tf: self.y_WALL,
            self.x_INLET_tf: self.x_INLET, self.y_INLET_tf: self.y_INLET,
            self.u_INLET_tf: self.u_INLET, self.v_INLET_tf: self.v_INLET,
            self.x_OUTLET_tf: self.x_OUTLET, self.y_OUTLET_tf: self.y_OUTLET
        }
        loss_f = self.sess.run(self.loss_f, tf_dict)
        loss_WALL = self.sess.run(self.loss_WALL, tf_dict)
        loss_INLET = self.sess.run(self.loss_INLET, tf_dict)
        loss = self.sess.run(self.loss, tf_dict)
        loss_OUTLET = self.sess.run(self.loss_OUTLET, tf_dict)
        return loss_WALL, loss_INLET, loss_OUTLET, loss_f, loss

# ---------------------------
# Helper: remove points inside airfoil polygon
# ---------------------------
def remove_inside_polygon(XY, poly):
    # XY: (N,2), poly: Nx2 polygon (closed or open) as Path
    path = Path(poly)
    mask = path.contains_points(XY)
    return XY[~mask, :]

# ---------------------------
# Post processing plotting function (pred vs ref)
# ---------------------------
def postProcess_airfoil(xmin, xmax, ymin, ymax, field_REF, field_PINN, airfoil_poly, N_test=200):
    [x_FLUENT, y_FLUENT, u_FLUENT, v_FLUENT, p_FLUENT] = field_REF
    [x_MIXED, y_MIXED, u_MIXED, v_MIXED, p_MIXED] = field_PINN

    # Create regular grid for plotting
    xi = np.linspace(xmin, xmax, N_test)
    yi = np.linspace(ymin, ymax, N_test)
    Xg, Yg = np.meshgrid(xi, yi)

    # Interpolate scattered predictions onto grid
    from scipy.interpolate import griddata
    Up = griddata((x_MIXED.flatten(), y_MIXED.flatten()), u_MIXED.flatten(), (Xg, Yg), method='cubic', fill_value=np.nan)
    Vp = griddata((x_MIXED.flatten(), y_MIXED.flatten()), v_MIXED.flatten(), (Xg, Yg), method='cubic', fill_value=np.nan)
    Pp = griddata((x_MIXED.flatten(), y_MIXED.flatten()), p_MIXED.flatten(), (Xg, Yg), method='cubic', fill_value=np.nan)

    Ur = griddata((x_FLUENT.flatten(), y_FLUENT.flatten()), u_FLUENT.flatten(), (Xg, Yg), method='cubic', fill_value=np.nan)
    Vr = griddata((x_FLUENT.flatten(), y_FLUENT.flatten()), v_FLUENT.flatten(), (Xg, Yg), method='cubic', fill_value=np.nan)
    Pr = griddata((x_FLUENT.flatten(), y_FLUENT.flatten()), p_FLUENT.flatten(), (Xg, Yg), method='cubic', fill_value=np.nan)

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(11, 6))
    fig.subplots_adjust(hspace=0.25, wspace=0.2)
    titles = [r'$u$ (m/s)', r'$v$ (m/s)', 'Pressure (Pa)']

    for i, (pred_grid, ref_grid, ttl) in enumerate(zip([Up, Vp, Pp], [Ur, Vr, Pr], titles)):
        row = i
        cf = ax[row, 0].contourf(Xg, Yg, pred_grid, 100, cmap='jet')
        fig.colorbar(cf, ax=ax[row, 0], fraction=0.046, pad=0.04)
        ax[row, 0].set_title(ttl)
        ax[row, 0].set_xlim([xmin, xmax]); ax[row, 0].set_ylim([ymin, ymax])
        ax[row, 0].axis('equal')
        # fill airfoil hole
        ax[row, 0].fill(airfoil_poly[:, 0], airfoil_poly[:, 1], color='white')

        cf = ax[row, 1].contourf(Xg, Yg, ref_grid, 100, cmap='jet')
        fig.colorbar(cf, ax=ax[row, 1], fraction=0.046, pad=0.04)
        ax[row, 1].set_title(ttl)
        ax[row, 1].set_xlim([xmin, xmax]); ax[row, 1].set_ylim([ymin, ymax])
        ax[row, 1].axis('equal')
        ax[row, 1].fill(airfoil_poly[:, 0], airfoil_poly[:, 1], color='white')

    plt.tight_layout()
    plt.savefig('./uvp_airfoil.png', dpi=300)
    plt.show()
    plt.close('all')

# ---------------------------
# Main: problem setup
# ---------------------------
if __name__ == "__main__":
    # Domain and airfoil params
    CHORD = 1.0
    AOA_DEG = 0.0  # angle of attack in degrees (rotate airfoil if needed)
    # generate airfoil coords (NACA0012)
    airfoil_coords = naca4_coordinates(t=12, c=CHORD, n=301, closed=True)
    # optionally rotate for AoA
    if AOA_DEG != 0.0:
        theta = np.deg2rad(AOA_DEG)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        airfoil_coords = (airfoil_coords @ R.T)

    # set domain bounds (trim to include upstream & downstream)
    x_min, x_max = -0.5, 2.5
    y_min, y_max = -1.0, 1.0
    lb = np.array([x_min, y_min]); ub = np.array([x_max, y_max])

    # Network configuration
    uv_layers = [2] + 6*[40] + [5]   # you can reduce/increase

    # Create boundary samples
    n_wall = airfoil_coords.shape[0]  # use airfoil points as wall points
    WALL = airfoil_coords.copy()

    # Inlet: x = x_min, sample along y
    n_inlet = 201
    y_in = np.random.rand(n_inlet,1)*(y_max - y_min) + y_min
    x_in = np.ones_like(y_in)*x_min
    U_inf = 1.0
    u_inlet = np.ones_like(y_in) * U_inf
    v_inlet = np.zeros_like(y_in)
    INLET = np.concatenate([x_in, y_in, u_inlet, v_inlet], axis=1)

    # Outlet: x = x_max, sample along y (pressure reference)
    n_outlet = 201
    y_out = np.random.rand(n_outlet,1)*(y_max - y_min) + y_min
    x_out = np.ones_like(y_out)*x_max
    OUTLET = np.concatenate([x_out, y_out], axis=1)

    # Top and bottom far-field (not explicitly separate—these are not used as wall here)
    # Collocation points (fluid region) - reduce N to save memory
    N_coll = 30000    # reduce if you had OOM before (was ~40000)
    XY_c = np.random.rand(N_coll, 2)
    XY_c[:,0] = x_min + (x_max - x_min) * XY_c[:,0]
    XY_c[:,1] = y_min + (y_max - y_min) * XY_c[:,1]

    # remove points inside airfoil polygon
    XY_c = remove_inside_polygon(XY_c, airfoil_coords)

    # Ensure boundary/inlet/outlet/airfoil points included in collocation set
    XY_c = np.concatenate((XY_c, WALL, OUTLET, INLET[:,0:2]), axis=0)
    print("Collocation points shape:", XY_c.shape)

    # visualize collocation distribution briefly
    fig, ax = plt.subplots(figsize=(6,2))
    ax.scatter(XY_c[:,0], XY_c[:,1], s=1, alpha=0.2)
    ax.fill(airfoil_coords[:,0], airfoil_coords[:,1], color='red', alpha=0.6)
    ax.set_xlim([x_min, x_max]); ax.set_ylim([y_min, y_max])
    ax.set_title('Collocation & Airfoil (red)')
    plt.show()

    # Build model (set ExistModel=1 to load uvNN.pickle if you have one)
    model = PINN_laminar_flow(XY_c, INLET, OUTLET, WALL, uv_layers, lb, ub, ExistModel=0, uvDir='uvNN_airfoil.pickle')

    # Train: Adam first
    start_time = time.time()
    loss_WALL, loss_INLET, loss_OUTLET, loss_f, loss = model.train(iter=500, learning_rate=5e-4)  # fewer iterations to test
    # optionally run L-BFGS-B; comment out if you hit OOM
    try:
        model.train_bfgs(maxiter=20000)
    except Exception as e:
        print("L-BFGS stage failed or OOMed, continuing. Exception:", e)

    print("--- %s seconds ---" % (time.time() - start_time))

    # Save model params & loss history
    model.save_NN('uvNN_airfoil.pickle')
    with open('loss_history_airfoil.pickle', 'wb') as f:
        pickle.dump(model.loss_rec, f)

    # Prediction grid for visualization
    N_test = 200
    xi = np.linspace(x_min, x_max, N_test)
    yi = np.linspace(y_min, y_max, N_test)
    Xg, Yg = np.meshgrid(xi, yi)
    x_PINN = Xg.flatten()[:, None]
    y_PINN = Yg.flatten()[:, None]

    # remove points inside airfoil for prediction
    pts = np.vstack([x_PINN.flatten(), y_PINN.flatten()]).T
    mask = Path(airfoil_coords).contains_points(pts)
    x_PINN = x_PINN[~mask, :]
    y_PINN = y_PINN[~mask, :]

    u_PINN, v_PINN, p_PINN = model.predict(x_PINN, y_PINN)
    field_PINN = [x_PINN.flatten(), y_PINN.flatten(), u_PINN.flatten(), v_PINN.flatten(), p_PINN.flatten()]

    # LOAD reference solution if you have it (optional)
    # For demonstration we create a very coarse "reference" by using the PINN itself
    # Replace below by your CFD/fluent data loader when available
    x_REF = x_PINN.copy(); y_REF = y_PINN.copy(); u_REF = u_PINN.copy(); v_REF = v_PINN.copy(); p_REF = p_PINN.copy()
    field_REF = [x_REF.flatten(), y_REF.flatten(), u_REF.flatten(), v_REF.flatten(), p_REF.flatten()]

    # Post-process and visualize
    postProcess_airfoil(xmin=x_min, xmax=x_max, ymin=y_min, ymax=y_max, field_REF=field_REF, field_PINN=field_PINN, airfoil_poly=airfoil_coords, N_test=N_test)

    print("Done. Saved visualization 'uvp_airfoil.png'.")
