
# coding: utf-8

# In[1]:


import dynet as dy
import numpy as np


# # `dyNet` example: `spam` v. `ham`

# ## import features

# In[2]:


import sys
sys.path.append("..")
import utils as u


# In[3]:


# change this string to match the path on your computer
path_to_root = "/Users/mcapizzi/Github/dynet_tutorial/"


# In[4]:


trainX, trainY, testX, testY = u.import_data(path_to_root)


# In[5]:


trainX.shape, trainY.shape


# In[6]:


testX.shape, testY.shape


# In[7]:


testY


# In[8]:


np.nonzero(testX[20])


# In[9]:


trainX[20][47]


# ## build architecture
# image from here: http://jrmeyer.github.io/tutorial/2016/02/01/TensorFlow-Tutorial.html <br>
# text from here: https://www.tensorflow.org/tutorials/wide#how_logistic_regression_works

# ![logistic_regression](images/logistic_regression.png)

# ![logistic_regression_math](images/logistic_regression_math.png)

# ### initialize empty model
# 
# See http://dynet.readthedocs.io/en/latest/python_ref.html#parametercollection

# In[10]:


lr_model = dy.ParameterCollection()   # used to be called dy.Model()
lr_model


# ### dimensions

# In[11]:


# size of input (2955)
input_size = trainX.shape[1]
# size of output
output_size = 1


# ### weight matrices and bias vectors

# #### paramater `initializer`
# See http://dynet.readthedocs.io/en/latest/python_ref.html#parameters-initializers

# In[12]:


initializer = dy.GlorotInitializer(gain=4.0)


# In[13]:


# W (input x output) as a Parameters object
pW = lr_model.add_parameters(
    (input_size, output_size),
    init=initializer
)
type(pW), type(dy.parameter(pW))


# In[14]:


# check the shape of the Expression
dy.parameter(pW).npvalue().shape


# In[15]:


# b (1 x output) as a Parameters object
pb = lr_model.add_parameters(
    (1, output_size),
    init=initializer
)
# check the shape
dy.parameter(pb).npvalue().shape


# ### forward operations

# In[16]:


def forward_pass(x):
    """
    This function will wrap all the steps of the forward pass
    :param x: the input
    """
    # convert input to Expression (this step must happen here b/c of autobatching)
    x = dy.inputTensor(x)
    # convert Parameters to Expressions
    W = dy.parameter(pW)
    b = dy.parameter(pb)
    affine_transformation = x * W + b          
    # calculate and return the sigmoid activation
    return dy.logistic(affine_transformation)


# ### training

# #### initializing a `trainer`
# See http://dynet.readthedocs.io/en/latest/python_ref.html#optimizers

# In[17]:


trainer = dy.SimpleSGDTrainer(
    m=lr_model,
    learning_rate=0.01
)


# ### autobatching
# See http://dynet.readthedocs.io/en/latest/minibatch.html#

# In[19]:


dyparams = dy.DynetParams()
dyparams.set_autobatch(True)
dyparams.set_random_seed(1978)
dyparams.init()


# #### one `epoch`

# In[20]:


# store original values of W (for comparison)
original_W = dy.parameter(pW).npvalue()
# begin a clean computational graph
dy.renew_cg()
# initialize list to capture individual losses
losses = []


# In[21]:


# iterate through the dataset
for i in range(trainX.shape[0]):
    # prepare input
    x = np.expand_dims(trainX[i], axis=0)   # must make it a vector with dimensions (1 x voc_size)
    # prepare output
    y = dy.scalarInput(trainY[i])
    # make a forward pass
    pred = forward_pass(x)
    # calculate loss for each example
    loss = dy.binary_log_loss(pred, y) 
    losses.append(loss)


# In[22]:


# get total loss for dataset
total_loss = dy.esum(losses)
# apply the calculations of the computational graph
total_loss.forward()
# calculate loss to backpropogate
total_loss.backward()
# update parameters with backpropogated error
trainer.update()


# In[23]:


# confirm that parameters updated
dy.renew_cg()
print("change in W parameter values: {}".format(
    np.sum(original_W - dy.parameter(pW).npvalue())
))


# ### testing

# #### make a single prediction

# In[24]:


pred = forward_pass(np.expand_dims(testX[0], axis=0))
print(pred.value())


# #### get predictions on entire test set

# In[25]:


all_preds = []
dy.renew_cg()
for i in range(testX.shape[0]):
    x = np.expand_dims(testX[i], axis=0)
    pred = forward_pass(x)
    all_preds.append(pred.value())
original_preds = all_preds


# In[26]:


print(original_preds)


# In[27]:


def check_score(pred, true_y):
    # convert pred to hard label
    label = 1 if pred >= 0.5 else 0
    # compare to true_y
    return 1 if label == true_y else 0


# In[28]:


def get_accuracy(list_of_scores):
    return float(sum(list_of_scores) / len(list_of_scores))


# In[29]:


accuracy = get_accuracy([check_score(p, y) for p,y in zip(all_preds, list(testY))])
accuracy


# ### multiple epochs and minibatches

# In[30]:


num_epochs = 800
batch_size = 128
num_batches = int(np.ceil(trainX.shape[0] / batch_size))
num_batches


# In[31]:


# bookeeping
original_W = dy.parameter(pW).npvalue()
epoch_losses = []
all_accuracies = []


# In[32]:


# iterate through epochs
for i in range(num_epochs):
    epoch_loss = []
    # reporting
    if i % 100 == 0:
        print("epoch {}".format(i+1))
    # shuffle dataset
    np.random.seed(i)
    np.random.shuffle(trainX)
    np.random.seed(i)           # make sure to reset seed again to keep labels and data together!
    np.random.shuffle(trainY)
    # iterate through batches
    for j in range(num_batches):
        # begin a clean computational graph *at beginning of each batch*
        dy.renew_cg()
        losses = []
        # build the batch
        batchX = trainX[j*batch_size:(j+1)*batch_size]
        batchY = trainY[j*batch_size:(j+1)*batch_size]
        # iterate through the batch
        for k in range(batchX.shape[0]):
            # prepare input
            x = np.expand_dims(batchX[k], axis=0)
            # prepare output
            y = dy.scalarInput(batchY[k])
            # make a forward pass
            pred = forward_pass(x)
            # calculate loss for each example
            loss = dy.binary_log_loss(pred, y)  
            losses.append(loss)
        # get total loss for batch
        total_loss = dy.esum(losses)
        # apply the calculations of the computational graph
        total_loss.forward()
        # calculate loss to backpropogate
        total_loss.backward()
        # update parameters with backpropogated error
        trainer.update()
        # record batch loss
        epoch_loss.append(total_loss.npvalue())
    # record epoch loss
    epoch_losses.append(np.sum(epoch_loss))
    # check performance on test set
    all_preds = []
    dy.renew_cg()
    for i in range(testX.shape[0]):
        x = np.expand_dims(testX[i], axis=0)
        pred = forward_pass(x)
        all_preds.append(pred.value())
    accuracy = get_accuracy([check_score(p, y) for p,y in zip(all_preds, list(testY))])
    all_accuracies.append(accuracy)
# confirm that parameters updated
dy.renew_cg()
print("change in W parameter values: {}".format(
    np.sum(original_W - dy.parameter(pW).npvalue())
))


# ### visualize loss and accuracy

# In[33]:


import matplotlib.pyplot as plt
plt.plot(epoch_losses)
plt.show()


# In[34]:


plt.plot(all_accuracies)
plt.show()


# ### testing

# In[35]:


all_preds = []
dy.renew_cg()
for i in range(testX.shape[0]):
    x = np.expand_dims(testX[i], axis=0)
    pred = forward_pass(x)
    all_preds.append(pred.value())


# In[36]:


accuracy = get_accuracy([check_score(p, y) for p,y in zip(all_preds, list(testY))])
accuracy

