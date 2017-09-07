
# coding: utf-8

# In[1]:


import numpy as np
import dynet_config
dynet_config.set(
    mem=2048,          # can probably get away with 1024
    autobatch=True,    # utilize autobatching
    random_seed=1978   # simply for reproducibility here
)
import dynet as dy


# # `dyNet` example: `spam` v. `ham`

# In[2]:


dy.__version__


# ## import features

# In[3]:


import sys
sys.path.append("..")
import utils as u


# In[4]:


# change this string to match the path on your computer
path_to_root = "/Users/mcapizzi/Github/dynet_tutorial/"


# In[5]:


trainX, trainY, testX, testY = u.import_data(path_to_root)


# In[6]:


trainX.shape, trainY.shape


# In[7]:


testX.shape, testY.shape


# The labels are either `1` or `0` where `1=Spam` and `0=Ham`

# In[8]:


testY


# The data is a `matrix` where each `row` is a document, and each `column` is a "transformed" count of how many times each word in the vocabulary appeared in that document.

# In[9]:


np.nonzero(testX[23])


# In[10]:


trainX[23][77]


# ## build architecture
# all images borrowed from here: http://u.cs.biu.ac.il/~yogo/nnlp.pdf (now a book!)

# ![goldberg_nn](images/goldberg_ff.png)

# ![goldberg_ff_math](images/goldberg_ff_math.png)

# ### initialize empty model
# 
# See http://dynet.readthedocs.io/en/latest/python_ref.html#parametercollection
# 
# The first thing to be done is initialize the `ParameterCollection()` which will house all the parameters that will be updated during training.

# In[11]:


feed_forward_model = dy.ParameterCollection()   # used to be called dy.Model()
feed_forward_model


# ### dimensions
# 
# You have a decision on what size you want the `hidden` layer to be.
# 
# ![goldberg_math_simple](images/goldberg_ff_math_simple.png)

# In[12]:


# size of input (2955)
input_size = trainX.shape[1]
################
# HYPERPARAMETER
################
# size of hidden layer
hidden_size = 200


# #### paramater `initializer`
# See http://dynet.readthedocs.io/en/latest/python_ref.html#parameters-initializers
# 
# Next we need to "initialize" the parameter values.  `GlorotInitializer` is a pretty standard approach *however* the `gain` parameter depends on the type of `activation` being used.

# In[13]:


################
# HYPERPARAMETER
################
initializer = dy.GlorotInitializer(gain=4.0)


# You'll notice that the objects are `_dynet.Parameters` and *not* `expressions` until you "wrap" them with `dy.parameter()`

# In[14]:


# W_1 (input x hidden) as a Parameters object
pW_1 = feed_forward_model.add_parameters(
    (input_size, hidden_size),
    init=initializer
)
type(pW_1), type(dy.parameter(pW_1))


# In[15]:


# check the shape of the Expression
dy.parameter(pW_1).npvalue().shape


# In[16]:


# b_1 (1 x hidden) as a Parameters object
pb_1 = feed_forward_model.add_parameters(
    (1, hidden_size),
    init=initializer
)
# check the shape
dy.parameter(pb_1).npvalue().shape


# In[17]:


# W_2 (hidden x output) as a Parameters object
pW_2 = feed_forward_model.add_parameters(
    (hidden_size, 1),
    init=initializer
)
# check the shape
dy.parameter(pW_2).npvalue().shape


# In[18]:


# b_2 (1 x output) as a Paramters object
pb_2 = feed_forward_model.add_parameters(
    (1, 1),
    init=initializer
)
# check the shape
dy.parameter(pb_2).npvalue().shape


# ### forward operations
# ![goldberg_math_simple](images/goldberg_ff_math_simple.png)
# 
# The only real choice is the type of `activation`.  See here for your choices: http://dynet.readthedocs.io/en/latest/operations.html

# In[19]:


def forward_pass(x):
    """
    This function will wrap all the steps of the forward pass
    :param x: the input
    """
    # convert input to Expression (this step must happen here b/c of autobatching)
    x = dy.inputTensor(x)
    # convert Parameters to Expressions
    W_1 = dy.parameter(pW_1)
    b_1 = dy.parameter(pb_1)
    W_2 = dy.parameter(pW_2)
    b_2 = dy.parameter(pb_2)
    # calculate the first hidden layer
    hidden = x * W_1 + b_1          
    ################
    # HYPERPARAMETER
    ################
    # calculate the sigmoid activation  (or RELU, SELU, ELU, tanh, etc...)
    hidden_activation = dy.logistic(hidden)    
    # calculate the output layer
    output = hidden_activation * W_2 + b_2
    # return the sigmoid of the output
    return dy.logistic(output)


# ### training

# #### initializing a `trainer`
# See http://dynet.readthedocs.io/en/latest/python_ref.html#optimizers
# 
# This decision is a big one.  It relates to what "optimizer" will be used to update the parameters.  Here I've chosen a *very simple* `trainer`, however the default `learning_rate` is almost never a good one.

# In[20]:


################
# HYPERPARAMETER
################
trainer = dy.SimpleSGDTrainer(
    m=feed_forward_model,
    learning_rate=0.01
)


# ### autobatching
# See http://dynet.readthedocs.io/en/latest/minibatch.html# <br>
# and the technical details here: https://arxiv.org/pdf/1705.07860.pdf
# 
# This is one of the real advantages of `dyNet`.  It's "overkill" for this example, but will become hugely valuable when training `recurrent neural networks` (`RNNs`).

# In[21]:


import dynet_config
dynet_config.set(
    mem=2048,          # can probably get away with 1024
    autobatch=True,    # utilize autobatching
    random_seed=1978   # simply for reproducibility here
)
import dynet as dy


# #### one `epoch`
# 
# Let's walk through *one* epoch (where our model sees all of our data *one* time).
# 
# The most important step is `dy.renew_cg()` which starts off a "clean" computational graph.
# 

# In[22]:


# store original values of W_1
original_W1 = dy.parameter(pW_1).npvalue()
# begin a clean computational graph
dy.renew_cg()
# initialize list to capture individual losses
losses = []


# `autobatching` allows us to feed each datapoint in one at a time, and `dyNet` will figure out how to "optimize" the operations.  Let's iterate through our training data.

# In[23]:


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


# Now let's accumulate the loss and backpropogate it.

# In[24]:


# get total loss for dataset
total_loss = dy.esum(losses)
# apply the calculations of the computational graph
total_loss.forward()
# calculate loss to backpropogate
total_loss.backward()
# update parameters with backpropogated error
trainer.update()


# Let's make sure that our parameter `W_1` has been updated (e.g. it "learned" something).

# In[25]:


# confirm that parameters updated
dy.renew_cg()
print("change in W_1 parameter values: {}".format(
    np.sum(original_W1 - dy.parameter(pW_1).npvalue())
))


# ### testing

# #### make a single prediction
# 
# Let's see how our model does on a single document.  The `output` can be understood as the probability the document is `Spam`.

# In[26]:


pred = forward_pass(np.expand_dims(testX[0], axis=0))
print(pred.value())


# #### get predictions on entire test set
# 
# Let's look across the entire dataset.

# In[27]:


all_preds = []
dy.renew_cg()
for i in range(testX.shape[0]):
    x = np.expand_dims(testX[i], axis=0)
    pred = forward_pass(x)
    all_preds.append(pred.value())
original_preds = all_preds


# You'll notice that the output is pretty much the same for *all* documents.  Not suprising since the model only saw each document only once.

# In[28]:


print(original_preds)


# In[29]:


def check_score(pred, true_y):
    # convert pred to hard label
    label = 1 if pred >= 0.5 else 0
    # compare to true_y
    return 1 if label == true_y else 0


# In[30]:


def get_accuracy(list_of_scores):
    return float(sum(list_of_scores) / len(list_of_scores))


# And since we predicted `0` for all documents, then our accuracy is simply matching the distribution of the data.

# In[31]:


accuracy = get_accuracy([check_score(p, y) for p,y in zip(all_preds, list(testY))])
accuracy


# ### multiple epochs and minibatches

# We need to run the model through the data many, many more times before it can learn anything meaningful.  
# 
# So we need to decide (1) how many `epochs` (times through the data), and (2) how many datapoints we want to show the model at once.  If `batch_size=len(data)` then we show the model *all* the data to the model at one time (and only one parameter update is made).  If `batch_size=1` then we only show the model one item at a time.

# In[32]:


################
# HYPERPARAMETER
################
num_epochs = 1000
################
# HYPERPARAMETER
################
batch_size = 128
################
# HYPERPARAMETER
################
num_batches = int(np.ceil(trainX.shape[0] / batch_size))
num_batches


# In[33]:


# bookeeping
original_W1 = dy.parameter(pW_1).npvalue()
epoch_losses = []
all_accuracies = []


# Below is code for iterating through multiple `epoch`s of the data.

# In[34]:


# iterate through epochs
for i in range(num_epochs):
    epoch_loss = []
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
        # applies the calculations of the computational graph
        total_loss.forward()
        # calculates loss to backpropogate
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
print("change in W_1 parameter values: {}".format(
    np.sum(original_W1 - dy.parameter(pW_1).npvalue())
))


# ### visualize loss and accuracy

# In[35]:


import matplotlib.pyplot as plt
plt.plot(epoch_losses)
plt.show()


# In[36]:


plt.plot(all_accuracies)
plt.show()


# ### testing

# In[37]:


all_preds = []
dy.renew_cg()
for i in range(testX.shape[0]):
    x = np.expand_dims(testX[i], axis=0)
    pred = forward_pass(x)
    all_preds.append(pred.value())


# Not surpisingly, the model now learns how to distinguish different documents, and so the predictions range all over.

# In[38]:


print(all_preds)


# And our accuracy is fantastic!

# In[39]:


accuracy = get_accuracy([check_score(p, y) for p,y in zip(all_preds, list(testY))])
accuracy

