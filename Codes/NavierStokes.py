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
        b = tf.Variable(tf.zeros([1,layers[l+1]]), dtype=tf.float32)
        weights.append(W)
        biases.append(b)        
    return weights, biases
    
def xavier_init(size):
    in_dim = size[0]
    out_dim = size[1]        
    xavier_stddev = np.sqrt(2/(in_dim + out_dim))
    return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

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
    def __init__(self, t, x, y, u, v, w,
                       t_b, x_b, y_b, w_b,
                       t_f, x_f, y_f, u_f, v_f,
                       w_layers, pde_layers,
                       layers,
                       lb, ub):
        
        # Domain Boundary
        self.lb = lb
        self.ub = ub
        
        # Init for Identification
        self.idn_init(t, x, y, u, v, w, w_layers, pde_layers)
        
        # Init for Solution
        self.sol_init(t_b, x_b, y_b, w_b,
                      t_f, x_f, y_f, u_f, v_f, layers)
            
        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
    
    ###########################################################################
    ############################# Identifier ##################################
    ###########################################################################
        
    def idn_init(self, t, x, y, u, v, w, w_layers, pde_layers):
        # Training Data for Identification
        self.t = t
        self.x = x
        self.y = y
        self.u = u
        self.v = v
        self.w = w
        
        # Layers for Identification
        self.w_layers = w_layers
        self.pde_layers = pde_layers
        
        # Initialize NNs for Identification
        self.w_weights, self.w_biases = initialize_NN(w_layers)
        self.pde_weights, self.pde_biases = initialize_NN(pde_layers)
        
        # tf placeholders for Identification
        self.t_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.x_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.v_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.w_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.terms_tf = tf.placeholder(tf.float32, shape=[None, pde_layers[0]])
        
        # tf graphs for Identification
        self.idn_w_pred = self.idn_net_w(self.t_tf, self.x_tf, self.y_tf)
        self.pde_pred = self.net_pde(self.terms_tf)
        self.idn_f_pred = self.idn_net_f(self.t_tf, self.x_tf, self.y_tf, self.u_tf, self.v_tf)
        
        # loss for Identification
        self.idn_w_loss = tf.reduce_sum(tf.square(self.idn_w_pred - self.w_tf))
        self.idn_f_loss = tf.reduce_sum(tf.square(self.idn_f_pred))
        
        # Optimizer for Identification
        self.idn_w_optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.idn_w_loss,
                               var_list = self.w_weights + self.w_biases,
                               method = 'L-BFGS-B',
                               options = {'maxiter': 50000,
                                          'maxfun': 50000,
                                          'maxcor': 50,
                                          'maxls': 50,
                                          'ftol': 1.0*np.finfo(float).eps})
    
        self.idn_f_optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.idn_f_loss,
                               var_list = self.pde_weights + self.pde_biases,
                               method = 'L-BFGS-B',
                               options = {'maxiter': 50000,
                                          'maxfun': 50000,
                                          'maxcor': 50,
                                          'maxls': 50,
                                          'ftol': 1.0*np.finfo(float).eps})
    
        self.idn_w_optimizer_Adam = tf.train.AdamOptimizer()
        self.idn_w_train_op_Adam = self.idn_w_optimizer_Adam.minimize(self.idn_w_loss, 
                                   var_list = self.w_weights + self.w_biases)
        
        self.idn_f_optimizer_Adam = tf.train.AdamOptimizer()
        self.idn_f_train_op_Adam = self.idn_f_optimizer_Adam.minimize(self.idn_f_loss, 
                                   var_list = self.pde_weights + self.pde_biases)  
    
    def idn_net_w(self, t, x, y):
        X = tf.concat([t,x,y],1)
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        w = neural_net(H, self.w_weights, self.w_biases)
        return w
    
    def net_pde(self, terms):
        pde = neural_net(terms, self.pde_weights, self.pde_biases)
        return pde
    
    def idn_net_f(self, t, x, y, u, v):
        w = self.idn_net_w(t, x, y)
        
        w_t = tf.gradients(w, t)[0]
        
        w_x = tf.gradients(w, x)[0]
        w_y = tf.gradients(w, y)[0]
        w_xx = tf.gradients(w_x, x)[0]
        w_xy = tf.gradients(w_x, y)[0]
        w_yy = tf.gradients(w_y, y)[0]
        
        terms = tf.concat([u,v,w,w_x,w_y,w_xx,w_xy,w_yy],1)
        
        f = w_t - self.net_pde(terms)
        
        return f

    def idn_w_train(self, N_iter):
        tf_dict = {self.t_tf: self.t, self.x_tf: self.x, self.y_tf: self.y,
                   self.u_tf: self.u, self.v_tf: self.v, self.w_tf: self.w}
        
        start_time = time.time()
        for it in range(N_iter):
            
            self.sess.run(self.idn_w_train_op_Adam, tf_dict)
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.idn_w_loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                start_time = time.time()
        
        self.idn_w_optimizer.minimize(self.sess,
                                      feed_dict = tf_dict,
                                      fetches = [self.idn_w_loss],
                                      loss_callback = self.callback)

    def idn_f_train(self, N_iter):
        tf_dict = {self.t_tf: self.t, self.x_tf: self.x, self.y_tf: self.y,
                   self.u_tf: self.u, self.v_tf: self.v}
        
        start_time = time.time()
        for it in range(N_iter):
            
            self.sess.run(self.idn_f_train_op_Adam, tf_dict)
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.idn_f_loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                start_time = time.time()
        
        self.idn_f_optimizer.minimize(self.sess,
                                      feed_dict = tf_dict,
                                      fetches = [self.idn_f_loss],
                                      loss_callback = self.callback)

    def idn_predict(self, t_star, x_star, y_star):
        
        tf_dict = {self.t_tf: t_star, self.x_tf: x_star, self.y_tf: y_star}
        
        w_star = self.sess.run(self.idn_w_pred, tf_dict)
        
        return w_star
    
    def predict_pde(self, terms_star):
        
        tf_dict = {self.terms_tf: terms_star}
        
        pde_star = self.sess.run(self.pde_pred, tf_dict)
        
        return pde_star
    
    ###########################################################################
    ############################### Solver ####################################
    ###########################################################################
    
    def sol_init(self, t_b, x_b, y_b, w_b,
                       t_f, x_f, y_f, u_f, v_f, layers):
        
        # Training Data for Solution
        self.t_b = t_b # initial and boundary data (time)
        self.x_b = x_b # initial and boundary data (space - x)
        self.y_b = y_b # initial and boundary data (space - y)
        self.w_b = w_b # boundary data (vorticity)
        
        self.t_f = t_f # collocation points (time)
        self.x_f = x_f # collocation points (space - x)
        self.y_f = y_f # collocation points (space - y)
        self.u_f = u_f # collocation points (space - u)
        self.v_f = v_f # collocation points (space - v)
        
        # Layers for Solution
        # self.layers = layers
        
        # Initialize NNs for Solution
        # self.weights, self.biases = initialize_NN(layers)
        
        # tf placeholders for Solution
        self.t_b_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.x_b_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.y_b_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.w_b_tf = tf.placeholder(tf.float32, shape=[None, 1])
        
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.y_f_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.u_f_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.v_f_tf = tf.placeholder(tf.float32, shape=[None, 1])
        
        # tf graphs for Solution
        self.w_b_pred  = self.sol_net_w(self.t_b_tf, self.x_b_tf, self.y_b_tf)
        self.sol_f_pred = self.sol_net_f(self.t_f_tf, self.x_f_tf, self.y_f_tf, self.u_f_tf, self.v_f_tf)
        
        # loss for Solution
        self.sol_loss = tf.reduce_sum(tf.square(self.w_b_tf - self.w_b_pred)) + \
                        tf.reduce_sum(tf.square(self.sol_f_pred))
        
        # Optimizer for Solution
        self.sol_optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.sol_loss,
                             var_list = self.w_weights + self.w_biases,
                             method = 'L-BFGS-B',
                             options = {'maxiter': 50000,
                                        'maxfun': 50000,
                                        'maxcor': 50,
                                        'maxls': 50,
                                        'ftol': 1.0*np.finfo(float).eps})
    
        self.sol_optimizer_Adam = tf.train.AdamOptimizer()
        self.sol_train_op_Adam = self.sol_optimizer_Adam.minimize(self.sol_loss,
                                 var_list = self.w_weights + self.w_biases)
    
    def sol_net_w(self, t, x, y):
        X = tf.concat([t,x,y],1)
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        w = neural_net(H, self.w_weights, self.w_biases)
        return w
    
    def sol_net_f(self, t, x, y, u, v):
        w = self.sol_net_w(t, x, y)
        
        w_t = tf.gradients(w, t)[0]
        
        w_x = tf.gradients(w, x)[0]
        w_y = tf.gradients(w, y)[0]
        
        w_xx = tf.gradients(w_x, x)[0]
        w_xy = tf.gradients(w_x, y)[0]
        w_yy = tf.gradients(w_y, y)[0]
        
        terms = tf.concat([u,v,w,w_x,w_y,w_xx,w_xy,w_yy],1)
        
        f = w_t - self.net_pde(terms)
        
        return f
    
    def callback(self, loss):
        print('Loss: %e' % (loss))
        
    def sol_train(self, N_iter):
        tf_dict = {self.t_b_tf: self.t_b,
                   self.x_b_tf: self.x_b,
                   self.y_b_tf: self.y_b,
                   self.w_b_tf: self.w_b,
                   self.t_f_tf: self.t_f,
                   self.x_f_tf: self.x_f,
                   self.y_f_tf: self.y_f,
                   self.u_f_tf: self.u_f,
                   self.v_f_tf: self.v_f}
        
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
    
    def sol_predict(self, t_star, x_star, y_star):
        
        u_star = self.sess.run(self.w_b_pred, {self.t_b_tf: t_star, self.x_b_tf: x_star, self.y_b_tf: y_star})  
               
        return u_star

###############################################################################
################################ Main Function ################################
###############################################################################

def plot_solution(X_data, w_data, index):
    
    lb = X_data.min(0)
    ub = X_data.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x,y)
    
    W_data = griddata(X_data, w_data.flatten(), (X, Y), method='cubic')
    
    plt.figure(index)
    plt.pcolor(X,Y,W_data, cmap = 'jet')
    plt.colorbar()

if __name__ == "__main__": 

    # Doman bounds
    lb = np.array([0.0, 1, -1.7])
    ub = np.array([30.0, 7.5, 1.7])
    
    ### Load Data ###
    data = scipy.io.loadmat('../Data/cylinder.mat')
        
    t_data = data['t_star']
    X_data = data['X_star']
    U_data = data['U_star']
    w_data = data['w_star']
    
    t_star = np.tile(t_data.T,(2310,1))
    x_star = np.tile(X_data[:,0:1],(1,151))
    y_star = np.tile(X_data[:,1:2],(1,151))
    u_star = U_data[:,0,:]
    v_star = U_data[:,1,:]
    w_star = w_data
    
    t_star = np.reshape(t_star,(-1,1))
    x_star = np.reshape(x_star,(-1,1))
    y_star = np.reshape(y_star,(-1,1))
    u_star = np.reshape(u_star,(-1,1))
    v_star = np.reshape(v_star,(-1,1))
    w_star = np.reshape(w_star,(-1,1))
        
    ### Training Data ###
    
    # For identification
    N_train = 50000
    
    idx = np.random.choice(t_star.shape[0], N_train, replace=False)
    t_train = t_star[idx,:]
    x_train = x_star[idx,:]
    y_train = y_star[idx,:]
    u_train = u_star[idx,:]
    v_train = v_star[idx,:]
    w_train = w_star[idx,:]
        
    # For solution
    N_f = 50000
    
    t_b0 = t_star[t_star == t_star.min()][:,None]
    x_b0 = x_star[t_star == t_star.min()][:,None]
    y_b0 = y_star[t_star == t_star.min()][:,None]
    w_b0 = w_star[t_star == t_star.min()][:,None]
    
    t_b1 = t_star[x_star == x_star.min()][:,None]
    x_b1 = x_star[x_star == x_star.min()][:,None]
    y_b1 = y_star[x_star == x_star.min()][:,None]
    w_b1 = w_star[x_star == x_star.min()][:,None]
    
    t_b2 = t_star[x_star == x_star.max()][:,None]
    x_b2 = x_star[x_star == x_star.max()][:,None]
    y_b2 = y_star[x_star == x_star.max()][:,None]
    w_b2 = w_star[x_star == x_star.max()][:,None]
    
    t_b3 = t_star[y_star == y_star.min()][:,None]
    x_b3 = x_star[y_star == y_star.min()][:,None]
    y_b3 = y_star[y_star == y_star.min()][:,None]
    w_b3 = w_star[y_star == y_star.min()][:,None]
    
    t_b4 = t_star[y_star == y_star.max()][:,None]
    x_b4 = x_star[y_star == y_star.max()][:,None]
    y_b4 = y_star[y_star == y_star.max()][:,None]
    w_b4 = w_star[y_star == y_star.max()][:,None]
    
    t_b_train = np.concatenate((t_b0, t_b1, t_b2, t_b3, t_b4))
    x_b_train = np.concatenate((x_b0, x_b1, x_b2, x_b3, x_b4))
    y_b_train = np.concatenate((y_b0, y_b1, y_b2, y_b3, y_b4))
    w_b_train = np.concatenate((w_b0, w_b1, w_b2, w_b3, w_b4))
    
    idx = np.random.choice(t_star.shape[0], N_train, replace=False)
    t_f_train = t_star[idx,:]
    x_f_train = x_star[idx,:]
    y_f_train = y_star[idx,:]
    u_f_train = u_star[idx,:]
    v_f_train = v_star[idx,:]
    
    # Layers
    w_layers = [3, 200, 200, 200, 200, 1]
    pde_layers = [8, 100, 100, 1]
    
    layers = [3, 200, 200, 200, 200, 1]
    
    # Model
    model = DeepHPM(t_train, x_train, y_train, u_train, v_train, w_train,
                    t_b_train, x_b_train, y_b_train, w_b_train,
                    t_f_train, x_f_train, y_f_train, u_f_train, v_f_train,
                    w_layers, pde_layers,
                    layers,
                    lb, ub)    
        
    # Train the identifier
    model.idn_w_train(N_iter=0)
        
    model.idn_f_train(N_iter=0)
    
    w_pred_identifier = model.idn_predict(t_star, x_star, y_star)
    
    error_w_identifier = np.linalg.norm(w_star-w_pred_identifier,2)/np.linalg.norm(w_star,2)
    print('Error w: %e' % (error_w_identifier))
    
    w_pred_identifier = np.reshape(w_pred_identifier,(-1,151))
    
#    step = 71
#    plot_solution(X_data,w_pred_identifier[:,step],1)
#    plot_solution(X_data,w_data[:,step],2)
#    plot_solution(X_data,np.abs(w_pred_identifier[:,step]-w_data[:,step]),3)
    
    ### Solution ###
    
    # Train the solver
    model.sol_train(N_iter=0)
        
    w_pred = model.sol_predict(t_star, x_star, y_star)
            
    error_w = np.linalg.norm(w_star-w_pred,2)/np.linalg.norm(w_star,2)
    print('Error w: %e' % (error_w))                             

    w_pred = np.reshape(w_pred,(-1,151))

#    step = 71
#    plot_solution(X_data,w_pred[:,step],4)
#    plot_solution(X_data,w_data[:,step],5)
#    plot_solution(X_data,np.abs(w_pred[:,step]-w_data[:,step]),6)
    
    
    ######################################################################
    ############################# Plotting ###############################
    ######################################################################    
    
    snap = 120
    
    lb_plot = X_data.min(0)
    ub_plot = X_data.max(0)
    nn = 200
    x_plot = np.linspace(lb_plot[0], ub_plot[0], nn)
    y_plot = np.linspace(lb_plot[1], ub_plot[1], nn)
    X_plot, Y_plot = np.meshgrid(x_plot,y_plot)
    
    W_data_plot = griddata(X_data, w_data[:,snap].flatten(), (X_plot, Y_plot), method='cubic')
    W_pred_plot = griddata(X_data, w_pred[:,snap].flatten(), (X_plot, Y_plot), method='cubic')
    
    
    fig, ax = newfig(1.0, 0.6)
    ax.axis('off')
    
    ########      Exact     ########### 
    gs = gridspec.GridSpec(1, 2)
    gs.update(top=0.8, bottom=0.2, left=0.1, right=0.9, wspace=0.5)
    ax = plt.subplot(gs[:, 0])
    h = ax.imshow(W_data_plot, interpolation='nearest', cmap='seismic', 
                  extent=[lb_plot[0], ub_plot[0], lb_plot[1], ub_plot[1]],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title('Exact Dynamics', fontsize = 10)
    
    
    ########     Learned     ########### 
    ax = plt.subplot(gs[:, 1])
    h = ax.imshow(W_pred_plot, interpolation='nearest', cmap='seismic', 
                  extent=[lb_plot[0], ub_plot[0], lb_plot[1], ub_plot[1]], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title('Learned Dynamics', fontsize = 10)
    
    # savefig('./figures/NavierStokes', crop = False)
    
    ####### Plotting Vorticity ##################
    
    data = scipy.io.loadmat('../Data/cylinder_vorticity.mat')
    XX = data['XX']
    YY = data['YY']
    WW = data['WW']
    WW[XX**2 + YY**2 < 0.25] = 0
    
    fig, ax = newfig(1.0, 0.65)
    ax.axis('off')
    
    gs0 = gridspec.GridSpec(1, 1)
    gs0.update(top=0.85, bottom=0.2, left=0.25, right=0.8, wspace=0.15)
    
    ax = plt.subplot(gs0[0:1, 0:1])
    h = ax.pcolormesh(XX, YY, WW, cmap='seismic',shading='gouraud', vmin=-5, vmax=5)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Vorticity', fontsize = 10)
    fig.colorbar(h)

    ax.plot([x_star.min(), x_star.max()], [y_star.min(), y_star.min()],'r--')
    ax.plot([x_star.min(), x_star.max()], [y_star.max(), y_star.max()],'r--')
    ax.plot([x_star.min(), x_star.min()], [y_star.min(), y_star.max()],'r--')
    ax.plot([x_star.max(), x_star.max()], [y_star.min(), y_star.max()],'r--')

    # savefig('./figures/Cylinder_vorticity', crop = False)