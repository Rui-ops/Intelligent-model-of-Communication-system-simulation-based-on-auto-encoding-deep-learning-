#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import math
import time
import sys
import matplotlib.cm as cm
import tensorflow as tf
import random


# In[25]:


def tf_print(tmp_var):
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    print(sess.run(tmp_var))


# In[453]:


## START: these are the main parameters to set
P_in_dBm=-2              # input power in dBm
gamma = 1.27            # fiber nonlinearity (set to 0 zero for AWGN or 1.27 for a nonlinear channel)
M = 4                   # constellation size

# main network parameters and optimization parameters (to be modified for performance improvements)
neurons_per_layer = 50 
tx_layers = 3
rx_layers = 3
learning_rate = 0.001
iterations = 50000
stacks = 20
minibatch_size = stacks*M
## END: these are the main parameters to set


# derived channel parameters
channel_uses = 2 # this should be 2: the fiber code will break otherwise
assert(channel_uses==2), "channel uses should be 2"
L=50             # fiber total length
K=20               # number of amplification stages (more layers requires more time)

P_in=10**(P_in_dBm/10)*0.001
Ess=np.sqrt(P_in)
SNR_dB=16
SNR=10**(SNR_dB/10)
sigma2tot=Ess**2/SNR
sigma=np.sqrt(sigma2tot/K)


#sigma = 3.8505e-4*np.sqrt(2)  # N0= h*v*nsp*(G-1) sigma**2 = BW*N0 
#sigma2tot=K*sigma**2
#P_in=10**(P_in_dBm/10)*0.001
#Ess=np.sqrt(P_in)
#SNR=Ess**2/(sigma2tot)
#SNR_dB=10*np.log(SNR)
print(SNR_dB,L)


# In[454]:


#=====================================================#
# Define the components of the computation graph
#=====================================================#
initializer = tf.contrib.layers.xavier_initializer()
# transmitter
W_tx = {}
b_tx = {}
for NN in range(tx_layers):
    if NN == 0:
        in_neurons = M
    else:
        in_neurons = neurons_per_layer
    if NN == tx_layers - 1:
        out_neurons = channel_uses
    else:
        out_neurons = neurons_per_layer
    W_tx[NN] = tf.Variable(initializer([in_neurons, out_neurons]))
    b_tx[NN] = tf.Variable(initializer([1, out_neurons]))
        
# receiver
W_rx = {}
b_rx = {}
for NN in range(rx_layers):
    if NN == 0:
        in_neurons = channel_uses+1
    else:
        in_neurons = neurons_per_layer
    if NN == rx_layers - 1:
        out_neurons = M
    else:
        out_neurons = neurons_per_layer
    W_rx[NN] = tf.Variable(initializer([in_neurons, out_neurons]))
    b_rx[NN] = tf.Variable(initializer([1, out_neurons]))  

# the encoder
def encoder(x):
    for NN in range(tx_layers-1):
        x = tf.nn.tanh(tf.matmul(x, W_tx[NN]) + b_tx[NN])
    x = tf.matmul(x, W_tx[tx_layers-1]) + b_tx[tx_layers-1]
    return x

# the decoder
def decoder(x):
    for NN in range(rx_layers-1):
        x = tf.nn.tanh(tf.matmul(x, W_rx[NN]) + b_rx[NN])
    x = tf.nn.softmax(tf.matmul(x, W_rx[rx_layers-1]) + b_rx[rx_layers-1])
    return x

# the non-dispersive fiber channel  
def fiber_channel(x):
    xr=x[:,0]
    xi=x[:,1]
    for segments in range(1,K+1):               
        s=gamma*(xr**2+xi**2)*L/K        
        xr=xr*tf.cos(s)-xi*tf.sin(s)
        xi=xi*tf.cos(s)+xr*tf.sin(s)
        xr=tf.add(xr,tf.random_normal(tf.shape(xr), mean=0.0, stddev=sigma))
        xi=tf.add(xi,tf.random_normal(tf.shape(xi), mean=0.0, stddev=sigma)) 
    z=tf.stack([xr,xi,xr**2+xi**2]) 
    z=tf.transpose(z) 
    return z

# average transmit power constraint
def normalization(x): # E[|x|^2] = Es
    return Ess*x / tf.sqrt(2*tf.reduce_mean(tf.square(x)))


# In[455]:


_map64_ = dict([('0',(-4,-4)),('1',(-4,-3)),('2',(-4,-1)),('3',(-4,-2)), 
                ('4',(-4,4)),('5',(-4,3)),('6',(-4,1)),('7',(-4,2)),
               ('8',(-3,-4)),('9',(-3,-3)),('10',(-3,-1)),('11',(-3,-2)),
               ('12',(-3,4)),('13',(-3,3)),('14',(-3,1)),('15',(-3,2)),
               ('16',(-1,-4)),('17',(-1,-3)),('18',(-1,-1)),('19',(-1,-2)),
               ('20',(-1,4)),('21',(-1,3)),('22',(-1,1)),('23',(-1,2)),
               ('24',(-2,-4)),('25',(-2,-3)),('26',(-2,-1)),('27',(-2,-2)),
               ('28',(-2,4)),('29',(-2,3)),('30',(-2,1)),('31',(-2,2)),
               ('32',(4,-4)),('33',(4,-3)),('34',(4,-1)),('35',(4,-2)),
               ('36',(4,4)),('37',(4,3)),('38',(4,1)),('39',(4,2)),
               ('40',(3,-4)),('41',(3,-3)),('42',(3,-1)),('43',(3,-2)),
               ('44',(3,4)),('45',(3,3)),('46',(3,1)),('47',(3,2)),
               ('48',(1,-4)),('49',(1,-3)),('50',(1,-1)),('51',(1,-2)),
               ('52',(1,4)),('53',(1,3)),('54',(1,1)),('55',(1,2)),
               ('56',(2,-4)),('57',(2,-3)),('58',(2,-1)),('59',(2,-2)),
               ('60',(2,4)),('61',(2,3)),('62',(2,1)),('63',(2,2))])
_map4_ = dict([('0',(1,1)),('1',(1,-1)),('2',(-1,1)),('3',(-1,-1))])
_map16_ = dict([('0',(3,3)),('1',(1,3)),('2',(-3,3)),('3',(-1,3)),        
              ('7',(-1,1)),('6',(-3,1)),('5',(1,1)),('4',(3,1)),
              ('8',(3,-3)),('9',(1,-3)),('10',(-3,-3)),('11',(-1,-3)),
              ('15',(-1,-1)),('14',(-3,-1)),('13',(1,-1)),('12',(3,-1))])
#用了Grey Mapping


# In[456]:


#=====================================================#
# build the computation graph

X_tilde = tf.placeholder('float', [minibatch_size, M]) # one-hot vectors
enco=tf.placeholder('float', [minibatch_size, 2])

#grid coordinates for visulazing decision regions
resolution=1000
G = tf.placeholder('float', [resolution**2, channel_uses+1])

X = normalization(enco) # minibatch_size x channel_uses
Y = fiber_channel(X)
Z = decoder(Y)
D = decoder(G)
epsilon = 0.000001
loss = -tf.reduce_mean(X_tilde*tf.log(Z+epsilon))
MI=(np.log(M)-loss*M)/np.log(2)
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)


# In[ ]:





# In[457]:


def random_int_list(start, stop, length):
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        random_list.append(random.randint(start, stop))
    return random_list


# In[ ]:





# In[458]:


#=====================================================#
# parameter training
#=====================================================#
start_time = time.time()
unitmatrix = np.eye(M)  
training_set = np.tile(unitmatrix, [stacks, 1])
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)
MI_tmp=0
totalloss=[]
enco_set=[]
for i in range(0,stacks):
    for j in range(0,M):
         enco_set.append(_map4_.get(str(j)))
enco_set=np.array(enco_set)
for i in range(1, iterations+1):
    _, loss_tmp, MI_tmp = sess.run([train, loss, MI], feed_dict={X_tilde: training_set, enco: enco_set})
    totalloss=np.append(totalloss, loss_tmp)
    if i%1000==0 or i==1:
        print('iteration ', i, ': loss = ', loss_tmp, '; Mutual information [bits] = ', MI_tmp)        
elapsed = time.time() - start_time
print("{0:.2f} seconds".format(elapsed))


# In[459]:


# BER Calculation
test_length=100000
X_t = tf.placeholder('float', [test_length, 2]) # one-hot vectors


Xt = normalization(X_t) # minibatch_size x channel_uses
Yt = fiber_channel(Xt)
Zt = decoder(Yt)


# In[460]:


N=test_length
test_data=random_int_list(0, M-1, N)
test_enco=[]
for i in range(0,N):
    test_enco.append(_map4_[str(test_data[i])])
    
[constellation,receive_points,decoded] = sess.run([Xt,Yt,Zt],feed_dict={X_t:test_enco})


# In[461]:


pred_output = np.argmax(decoded,axis=1)
no_errors = (pred_output != test_data)
no_errors =  no_errors.astype(int).sum()
ber = no_errors 
print ('BER:',ber/N)
#print(pred_output)


# In[ ]:





# In[ ]:




