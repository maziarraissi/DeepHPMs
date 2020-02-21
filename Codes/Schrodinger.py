"""
@author: Maziar Raissi
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from pyDOE import lhs
import time
from plotting import newfig, savefig
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

###############################################################################
############################## Helper Functions ###############################
###############################################################################

def initialize_NN(layers):
    weights = []
    biases = []
    num_layers = len(layers) 
    for l in range(0,num_layers-1):
        W = xavier_init(size=[layers[l], layers[l+1]])
        b = tf.Variable(tf.zeros([1,layers[l+1]],
                        dtype=tf.float32),
                        dtype=tf.float32)
        weights.append(W)
        biases.append(b)        
    return weights, biases
    
def xavier_init(size):
    in_dim = size[0]
    out_dim = size[1]        
    xavier_stddev = np.sqrt(2/(in_dim + out_dim))
    return tf.Variable(tf.truncated_normal([in_dim, out_dim],
                       stddev=xavier_stddev, dtype=tf.float32),
                       dtype=tf.float32)

def neural_net(X, weights, biases):
    num_layers = len(weights) + 1
    H = X
    for l in range(0,num_layers-2):
        W = weights[l]
        b = biases[l]
        H = tf.sin(tf.add(tf.matmul(H, W), b))
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    return Y

###############################################################################
################################ DeepHPM Class ################################
###############################################################################

class DeepHPM:
    def __init__(self, t, x, u, v,
                       x0, u0, v0, tb, X_f,
                       uv_layers, pde_layers,
                       layers,
                       lb_idn, ub_idn,
                       lb_sol, ub_sol):
        
        # Domain Boundary
        self.lb_idn = lb_idn
        self.ub_idn = ub_idn
        
        self.lb_sol = lb_sol
        self.ub_sol = ub_sol
        
        # Init for Identification
        self.idn_init(t, x, u, v, uv_layers, pde_layers)
        
        # Init for Solution
        self.sol_init(x0, u0, v0, tb, X_f, layers)
            
        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
    
    ###########################################################################
    ############################# Identifier ##################################
    ###########################################################################
        
    def idn_init(self, t, x, u, v, uv_layers, pde_layers):
        # Training Data for Identification
        self.t = t
        self.x = x
        self.u = u
        self.v = v
        
        # Layers for Identification
        self.uv_layers = uv_layers
        self.pde_layers = pde_layers
        
        # Initialize NNs for Identification
        self.u_weights, self.u_biases = initialize_NN(uv_layers)
        self.v_weights, self.v_biases = initialize_NN(uv_layers)
        self.pde_u_weights, self.pde_u_biases = initialize_NN(pde_layers)
        self.pde_v_weights, self.pde_v_biases = initialize_NN(pde_layers)
        
        # tf placeholders for Identification
        self.t_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.x_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.v_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.terms_tf = tf.placeholder(tf.float32, shape=[None, pde_layers[0]])
        
        # tf graphs for Identification
        self.idn_u_pred, self.idn_v_pred = self.idn_net_uv(self.t_tf, self.x_tf)
        self.pde_u_pred, self.pde_v_pred = self.net_pde(self.terms_tf)
        self.idn_f_pred, self.idn_g_pred = self.idn_net_fg(self.t_tf, self.x_tf)
        
        # loss for Identification
        self.idn_uv_loss = tf.reduce_sum(tf.square(self.idn_u_pred - self.u_tf)) + \
                           tf.reduce_sum(tf.square(self.idn_v_pred - self.v_tf))
        self.idn_fg_loss = tf.reduce_sum(tf.square(self.idn_f_pred)) + \
                           tf.reduce_sum(tf.square(self.idn_g_pred))
                        
        # Optimizer for Identification
        self.idn_uv_optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.idn_uv_loss,
                                var_list = self.u_weights + self.u_biases + self.v_weights + self.v_biases,
                                method = 'L-BFGS-B',
                                options = {'maxiter': 50000,
                                           'maxfun': 50000,
                                           'maxcor': 50,
                                           'maxls': 50,
                                           'ftol': 1.0*np.finfo(float).eps})
    
        self.idn_fg_optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.idn_fg_loss,
                                var_list = self.pde_u_weights + self.pde_u_biases + self.pde_v_weights + self.pde_v_biases,
                                method = 'L-BFGS-B',
                                options = {'maxiter': 50000,
                                           'maxfun': 50000,
                                           'maxcor': 50,
                                           'maxls': 50,
                                           'ftol': 1.0*np.finfo(float).eps})
    
        self.idn_uv_optimizer_Adam = tf.train.AdamOptimizer()
        self.idn_uv_train_op_Adam = self.idn_uv_optimizer_Adam.minimize(self.idn_uv_loss, 
                                    var_list = self.u_weights + self.u_biases + self.v_weights + self.v_biases)
        
        self.idn_fg_optimizer_Adam = tf.train.AdamOptimizer()
        self.idn_fg_train_op_Adam = self.idn_fg_optimizer_Adam.minimize(self.idn_fg_loss, 
                                    var_list = self.pde_u_weights + self.pde_u_biases + self.pde_v_weights + self.pde_v_biases)
    
    def idn_net_uv(self, t, x):
        X = tf.concat([t,x],1)
        H = 2.0*(X - self.lb_idn)/(self.ub_idn - self.lb_idn) - 1.0
        u = neural_net(H, self.u_weights, self.u_biases)
        v = neural_net(H, self.v_weights, self.v_biases)
        return u, v
    
    def net_pde(self, terms):
        pde_u = neural_net(terms, self.pde_u_weights, self.pde_u_biases)
        pde_v = neural_net(terms, self.pde_v_weights, self.pde_v_biases)
        return pde_u, pde_v
    
    def idn_net_fg(self, t, x):
        u, v = self.idn_net_uv(t, x)
        
        u_t = tf.gradients(u, t)[0]
        v_t = tf.gradients(v, t)[0]
        
        u_x = tf.gradients(u, x)[0]
        v_x = tf.gradients(v, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        v_xx = tf.gradients(v_x, x)[0]
        
        terms = tf.concat([u,v,u_x,v_x,u_xx,v_xx],1)
        
        pde_u, pde_v = self.net_pde(terms)
        
        f = u_t - pde_u
        g = v_t - pde_v
        
        return f, g

    def idn_uv_train(self, N_iter):
        tf_dict = {self.t_tf: self.t, self.x_tf: self.x,
                   self.u_tf: self.u, self.v_tf: self.v}
        
        start_time = time.time()
        for it in range(N_iter):
            
            self.sess.run(self.idn_uv_train_op_Adam, tf_dict)
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.idn_uv_loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                start_time = time.time()
        
        self.idn_uv_optimizer.minimize(self.sess,
                                       feed_dict = tf_dict,
                                       fetches = [self.idn_uv_loss],
                                       loss_callback = self.callback)
        
    def idn_fg_train(self, N_iter):
        tf_dict = {self.t_tf: self.t, self.x_tf: self.x}
        
        start_time = time.time()
        for it in range(N_iter):
            
            self.sess.run(self.idn_fg_train_op_Adam, tf_dict)
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.idn_fg_loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                start_time = time.time()
        
        self.idn_fg_optimizer.minimize(self.sess,
                                       feed_dict = tf_dict,
                                       fetches = [self.idn_fg_loss],
                                       loss_callback = self.callback)

    def idn_predict(self, t_star, x_star):
        
        tf_dict = {self.t_tf: t_star, self.x_tf: x_star}
        
        u_star = self.sess.run(self.idn_u_pred, tf_dict)
        v_star = self.sess.run(self.idn_v_pred, tf_dict)
        
        f_star = self.sess.run(self.idn_f_pred, tf_dict)
        g_star = self.sess.run(self.idn_g_pred, tf_dict)
        
        return u_star, v_star, f_star, g_star
    
    def predict_pde(self, terms_star):
        
        tf_dict = {self.terms_tf: terms_star}
        
        pde_u_star = self.sess.run(self.pde_u_pred, tf_dict)
        pde_v_star = self.sess.run(self.pde_v_pred, tf_dict)
        
        return pde_u_star, pde_v_star
    
    ###########################################################################
    ############################### Solver ####################################
    ###########################################################################
    
    def sol_init(self, x0, u0, v0, tb, X_f, layers):
        # Training Data for Solution
        X0 = np.concatenate((0*x0, x0), 1) # (0, x0)
        X_lb = np.concatenate((tb, 0*tb + self.lb_sol[1]), 1) # (tb, lb[1])
        X_ub = np.concatenate((tb, 0*tb + self.ub_sol[1]), 1) # (tb, ub[1])
        
        self.X_f = X_f # Collocation Points
        self.t0 = X0[:,0:1] # Initial Data (time)
        self.x0 = X0[:,1:2] # Initial Data (space)
        self.t_lb = X_lb[:,0:1] # Boundary Data (time) -- lower boundary
        self.x_lb = X_lb[:,1:2] # Boundary Data (space) -- lower boundary
        self.t_ub = X_ub[:,0:1] # Boundary Data (time) -- upper boundary
        self.x_ub = X_ub[:,1:2] # Boundary Data (space) -- upper boundary
        self.t_f = X_f[:,0:1] # Collocation Points (time)
        self.x_f = X_f[:,1:2] # Collocation Points (space)
        self.u0 = u0 # Boundary Data
        self.v0 = v0 # Boundary Data
        
        # Layers for Solution
        # self.layers = layers
        
        # Initialize NNs for Solution
        # self.u_weights_sol, self.u_biases_sol = initialize_NN(layers)
        # self.v_weights_sol, self.v_biases_sol = initialize_NN(layers)
        
        # tf placeholders for Solution
        self.t0_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.x0_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.u0_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.v0_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.t_lb_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.x_lb_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.t_ub_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.x_ub_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, 1])
        
        # tf graphs for Solution
        self.u0_pred, self.v0_pred, _, _  = self.sol_net_uv(self.t0_tf, self.x0_tf)
        self.u_lb_pred, self.v_lb_pred, self.u_x_lb_pred, self.v_x_lb_pred = self.sol_net_uv(self.t_lb_tf, self.x_lb_tf)
        self.u_ub_pred, self.v_ub_pred, self.u_x_ub_pred, self.v_x_ub_pred = self.sol_net_uv(self.t_ub_tf, self.x_ub_tf)
        self.sol_f_pred, self.sol_g_pred = self.sol_net_fg(self.t_f_tf, self.x_f_tf)
        
        # loss for Solution
        self.sol_loss = tf.reduce_mean(tf.square(self.u0_tf - self.u0_pred)) + \
                        tf.reduce_mean(tf.square(self.v0_tf - self.v0_pred)) + \
                        tf.reduce_mean(tf.square(self.u_lb_pred - self.u_ub_pred)) + \
                        tf.reduce_mean(tf.square(self.v_lb_pred - self.v_ub_pred)) + \
                        tf.reduce_mean(tf.square(self.u_x_lb_pred - self.u_x_ub_pred)) + \
                        tf.reduce_mean(tf.square(self.v_x_lb_pred - self.v_x_ub_pred)) + \
                        tf.reduce_mean(tf.square(self.sol_f_pred)) + \
                        tf.reduce_mean(tf.square(self.sol_g_pred))
        
        # Optimizer for Solution
        self.sol_optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.sol_loss,
                             var_list = self.u_weights + self.u_biases + self.v_weights + self.v_biases,
                             method = 'L-BFGS-B',
                             options = {'maxiter': 50000,
                                        'maxfun': 50000,
                                        'maxcor': 50,
                                        'maxls': 50,
                                        'ftol': 1.0*np.finfo(float).eps})
    
        self.sol_optimizer_Adam = tf.train.AdamOptimizer()
        self.sol_train_op_Adam = self.sol_optimizer_Adam.minimize(self.sol_loss,
                                 var_list = self.u_weights + self.u_biases + self.v_weights + self.v_biases)
    
    def sol_net_uv(self, t, x):
        X = tf.concat([t,x],1)
        H = 2.0*(X - self.lb_sol)/(self.ub_sol - self.lb_sol) - 1.0
        u = neural_net(H, self.u_weights, self.u_biases)
        v = neural_net(H, self.v_weights, self.v_biases)
        u_x = tf.gradients(u, x)[0]
        v_x = tf.gradients(v, x)[0]
        return u, v, u_x, v_x
    
    def sol_net_fg(self, t, x):
        u, v, u_x, v_x = self.sol_net_uv(t,x)
        
        u_t = tf.gradients(u, t)[0]
        v_t = tf.gradients(v, t)[0]
        
        u_xx = tf.gradients(u_x, x)[0]
        v_xx = tf.gradients(v_x, x)[0]
        
        terms = tf.concat([u,v,u_x,v_x,u_xx,v_xx],1)
        
        pde_u, pde_v = self.net_pde(terms)
        
        f = u_t - pde_u
        g = v_t - pde_v

#        f = u_t + 0.5*v_xx + (u**2 + v**2)*v
#        g = v_t - 0.5*u_xx - (u**2 + v**2)*u 

        return f, g
    
    def callback(self, loss):
        print('Loss: %e' % (loss))
        
    def sol_train(self, N_iter):
        tf_dict = {self.t0_tf: self.t0, self.x0_tf: self.x0,
                   self.u0_tf: self.u0, self.v0_tf: self.v0,
                   self.t_lb_tf: self.t_lb, self.x_lb_tf: self.x_lb,
                   self.t_ub_tf: self.t_ub, self.x_ub_tf: self.x_ub,
                   self.t_f_tf: self.t_f, self.x_f_tf: self.x_f}
        
        start_time = time.time()
        for it in range(N_iter):
            
            self.sess.run(self.sol_train_op_Adam, tf_dict)
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.sol_loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                start_time = time.time()
                
        self.sol_optimizer.minimize(self.sess, 
                                    feed_dict = tf_dict,         
                                    fetches = [self.sol_loss], 
                                    loss_callback = self.callback)
    
    def sol_predict(self, t_star, x_star):
        
        u_star = self.sess.run(self.u0_pred, {self.t0_tf: t_star, self.x0_tf: x_star})  
        v_star = self.sess.run(self.v0_pred, {self.t0_tf: t_star, self.x0_tf: x_star})  
        
        f_star = self.sess.run(self.sol_f_pred, {self.t_f_tf: t_star, self.x_f_tf: x_star})
        g_star = self.sess.run(self.sol_g_pred, {self.t_f_tf: t_star, self.x_f_tf: x_star})
        
        return u_star, v_star, f_star, g_star    

###############################################################################
################################ Main Function ################################
###############################################################################

if __name__ == "__main__": 
    
    # Doman bounds
    lb_idn = np.array([0.0, -5.0])
    ub_idn = np.array([np.pi/2, 5.0])
    
    lb_sol = np.array([0.0, -5.0])
    ub_sol = np.array([np.pi/2, 5.0])
    
    ### Load Data ###
    
    data_idn = scipy.io.loadmat('../Data/NLS.mat')
    
    t_idn = data_idn['t'].flatten()[:,None]
    x_idn = data_idn['x'].flatten()[:,None]
    Exact_u_idn = np.real(data_idn['usol'])
    Exact_v_idn = np.imag(data_idn['usol'])
    
    T_idn, X_idn = np.meshgrid(t_idn,x_idn)
    
    keep = 1
    index = int(keep*t_idn.shape[0])
    T_idn = T_idn[:,0:index]
    X_idn = X_idn[:,0:index]
    Exact_u_idn = Exact_u_idn[:,0:index]
    Exact_v_idn = Exact_v_idn[:,0:index]
    Exact_uv_idn = np.sqrt(Exact_u_idn**2 + Exact_v_idn**2)    
    
    t_idn_star = T_idn.flatten()[:,None]
    x_idn_star = X_idn.flatten()[:,None]
    X_idn_star = np.hstack((t_idn_star, x_idn_star))
    u_idn_star = Exact_u_idn.flatten()[:,None]
    v_idn_star = Exact_v_idn.flatten()[:,None]
    
    #
    
    data_sol = scipy.io.loadmat('../Data/NLS.mat')
    
    t_sol = data_sol['t'].flatten()[:,None]
    x_sol = data_sol['x'].flatten()[:,None]
    Exact_u_sol = np.real(data_sol['usol'])
    Exact_v_sol = np.imag(data_sol['usol'])
    Exact_uv_sol = np.sqrt(Exact_u_sol**2 + Exact_v_sol**2)
    
    T_sol, X_sol = np.meshgrid(t_sol,x_sol)
    
    t_sol_star = T_sol.flatten()[:,None]
    x_sol_star = X_sol.flatten()[:,None]
    X_sol_star = np.hstack((t_sol_star, x_sol_star))
    u_sol_star = Exact_u_sol.flatten()[:,None]
    v_sol_star = Exact_v_sol.flatten()[:,None]
    
    ### Training Data ###
    
    # For identification
    N_train = 10000
    
    idx = np.random.choice(t_idn_star.shape[0], N_train, replace=False)    
    t_train = t_idn_star[idx,:]
    x_train = x_idn_star[idx,:]
    u_train = u_idn_star[idx,:]
    v_train = v_idn_star[idx,:]
    
    noise = 0.00
    u_train = u_train + noise*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])
    v_train = v_train + noise*np.std(v_train)*np.random.randn(v_train.shape[0], v_train.shape[1])
    
    # For solution
    N0 = Exact_u_sol.shape[0]
    N_b = Exact_u_sol.shape[1]
    N_f = 20000
        
    idx_x = np.random.choice(x_sol.shape[0], N0, replace=False)
    x0_train = x_sol[idx_x,:]
    u0_train = Exact_u_sol[idx_x,0:1]
    v0_train = Exact_v_sol[idx_x,0:1]
    
    idx_t = np.random.choice(t_sol.shape[0], N_b, replace=False)
    tb_train = t_sol[idx_t,:]
    
    X_f_train = lb_sol + (ub_sol-lb_sol)*lhs(2, N_f)
        
    # Layers
    uv_layers = [2, 50, 50, 50, 50, 1]
    pde_layers = [6, 100, 100, 1]
    
    layers = [2, 50, 50, 50, 50, 1]
    
    # Model
    model = DeepHPM(t_train, x_train, u_train, v_train,
                    x0_train, u0_train, v0_train, tb_train, X_f_train,
                    uv_layers, pde_layers,
                    layers,
                    lb_idn, ub_idn,
                    lb_sol, ub_sol)
    
    # Train the identifier
    model.idn_uv_train(N_iter=0)
        
    model.idn_fg_train(N_iter=0)
    
    u_pred_identifier, v_pred_identifier, f_pred_identifier, g_pred_identifier = model.idn_predict(t_idn_star, x_idn_star)
    
    error_u_identifier = np.linalg.norm(u_idn_star-u_pred_identifier,2)/np.linalg.norm(u_idn_star,2)
    error_v_identifier = np.linalg.norm(v_idn_star-v_pred_identifier,2)/np.linalg.norm(v_idn_star,2)
    print('Error u: %e' % (error_u_identifier))
    print('Error v: %e' % (error_v_identifier))
    
    ### Solution ###
    
    # Train the solver
    model.sol_train(N_iter=0)
        
    u_pred, v_pred, f_pred, g_pred = model.sol_predict(t_sol_star, x_sol_star)
    
    u_pred_idn, v_pred_idn, f_pred_idn, g_pred_idn = model.sol_predict(t_idn_star, x_idn_star)
    
    uv_pred = np.sqrt(u_pred**2 + v_pred**2)
    uv_sol_star = np.sqrt(u_sol_star**2 + v_sol_star**2)
    error_u = np.linalg.norm(u_sol_star-u_pred,2)/np.linalg.norm(u_sol_star,2)
    error_v = np.linalg.norm(v_sol_star-v_pred,2)/np.linalg.norm(v_sol_star,2)
    error_uv = np.linalg.norm(uv_sol_star-uv_pred,2)/np.linalg.norm(uv_sol_star,2)
    print('Error uv: %e' % (error_uv))
    
    error_u_idn = np.linalg.norm(u_idn_star-u_pred_idn,2)/np.linalg.norm(u_idn_star,2)
    error_v_idn = np.linalg.norm(v_idn_star-v_pred_idn,2)/np.linalg.norm(v_idn_star,2)
    
    print('Error u: %e' % (error_u))
    print('Error v: %e' % (error_v))
    print('Error u (idn): %e' % (error_u_idn))
    print('Error v (idn): %e' % (error_v_idn))

    U_pred = griddata(X_sol_star, u_pred.flatten(), (T_sol, X_sol), method='cubic')
    V_pred = griddata(X_sol_star, v_pred.flatten(), (T_sol, X_sol), method='cubic')
    UV_pred = np.sqrt(U_pred**2 + V_pred**2)
    
    ######################################################################
    ############################# Plotting ###############################
    ######################################################################    
    
    fig, ax = newfig(1.0, 0.6)
    ax.axis('off')
    
    ######## Row 2: Pressure #######################
    ########      Predicted p(t,x,y)     ########### 
    gs = gridspec.GridSpec(1, 2)
    gs.update(top=0.8, bottom=0.2, left=0.1, right=0.9, wspace=0.5)
    ax = plt.subplot(gs[:, 0])
    h = ax.imshow(Exact_uv_sol, interpolation='nearest', cmap='jet', 
                  extent=[lb_sol[0], ub_sol[0], lb_sol[1], ub_sol[1]],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title('Exact Dynamics', fontsize = 10)
    
    ########     Exact p(t,x,y)     ########### 
    ax = plt.subplot(gs[:, 1])
    h = ax.imshow(UV_pred, interpolation='nearest', cmap='jet', 
                  extent=[lb_sol[0], ub_sol[0], lb_sol[1], ub_sol[1]], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title('Learned Dynamics', fontsize = 10)
    
    # savefig('./figures/NLS')