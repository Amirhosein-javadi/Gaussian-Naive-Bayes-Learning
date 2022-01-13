#!/usr/bin/env python
# coding: utf-8

# <div align=center>
# 		
# <p></p>
# <p></p>
# <font size=5>
# In the Name of God
# <font/>
# <p></p>
#  <br/>
#     <br/>
#     <br/>
# <font color=#FF7500>
# Sharif University of Technology - Departmenet of Computer Engineering
# </font>
# <p></p>
# <font color=blue>
# Artifical Intelligence - Dr. Mohammad Hossein Rohban
# </font>
# <br/>
# <br/>
# Fall 2021
# 
# </div>
# 
# <hr/>
# 		<div align=center>
# 		    <font color=red size=6>
# 			    <br />
# Practical Assignment 4 Naive Bayes
#             	<br/>
# 			</font>
#     <br/>
#     <br/>
# <font size=4> 
#                 <br/><b>
#               Cheating is Strongly Prohibited
#                 </b><br/><br/>
#                 <font color=red>
# Please run all the cells.
#      </font>
# </font>
#                 <br/>
#     </div>

# # Personal Data

# In[1]:


# Set your student number
student_number = 97101489
Name = 'Amirhosein'
Last_Name = 'Javadi'


# # Rules
# - You **are** allowed to add or remove cells.
# - By running the cell below, you can see if your jupyter file is accepted or not. This cell will also **generate a python file which you'll have to upload to Quera** (as well as your jupyter file). The python file will later be validated and if the code in both files doesn't match, **your Practical Assignment won't be graded**.

# In[2]:


# remember to save your jupyter file before running this script
from Helper_codes.validator import *

python_code = extract_python("./Q1.ipynb")
with open(f'python_code_Q1_{student_number}.py', 'w') as file:
    file.write(python_code)


# # Gaussian Naive Bayes (40 Points)

# <font size=4>
# Author: Kimia Noorbakhsh
# 			<br/>
#                 <font color=red>
# Please run all the cells.
#      </font>
# </font>
#                 <br/>
#     </div>

# In this assignment, you are going to implement a Naive Bayes Classifier for the MNIST Dataset (Well, of course, **from scratch**!). The MNIST data set is a vast database of handwritten digits that is commonly used to form various image processing systems. 
# 
# Please note the following before you continue:
# - After implementing your Classifier, train your model on the **train** section of the MNIST dataset and validate your model by testing it on the test set.
# - Note that if you use any of the **test** images during training or for improving the accuracy, you will not earn any points for this assignment. 
# - Do not forget to use **Laplace Smoothing** to avoid overfitting.

# Recall Bayes rule:
#     $$P(c|x) =  \frac{P(x|c)P(c)}{P(x)} \;\;\;\;(1)$$
#     
# Here $x$ stands for the image, or more precisely, the pixel values of the formatted image as a vector, and $c$ stands for the number, which can be 0, 1, ..., 9. We can read the left side $P(c|x)$ as "the probability of the class being $c$ given the $x$" data (posterior). We can read the right side $P(x|c)$ as "the probability of $x$ data being in the $c$" class (likelihood). We care about the value of $c$. It tells us "what number" this picture is. The chosen class is simply the one with the highest probability for this data:
# $$c^* = argmax_{c}P(c|x)$$
# Now, we can ignore $P(x)$ in equation (1) (why?). Using this information, we can simplify our problem so that, in order to choose “which digit” given an image, all we need to do is calculate this argmax (P(x) is removed):
# $$c^* = argmax_{c}P(x|c)P(c)$$
# Now, we need to think about how to calculate $P(c)$, and $P(x|c)$. We leave this section for you to think about ^_^. But as a guide for $P(x|c)$, read the following. 
# 
# Remember that pixels represent the intensity of light, and that the intensity of light is in fact continuous. A first reasonable estimation to model continuous data is the multivariate Gaussian or multivariate Normal. We can write:
# $$P(x|c) = \frac{1}{\sqrt{(2\pi)^{D}|\Sigma|}}\exp(-\frac{1}{2}(x - \mu)^{T}\Sigma^{-1}(x-\mu))$$
# Note that because probabilities are very small when dimensionality is high, we are going to work with log-probability rather than probability. So instead of getting numbers that are very close to 0, which is inaccurate when you use a computer to represent them, we're just going to get negative numbers. The log-probability can be represented as ($D$ is the dimentionality):
# $$\log{P(x|c) = -\frac{D}{2}\ln(2\pi)-\frac{1}{2}\ln|\Sigma|-\frac{1}{2}(x - \mu)^{T}\Sigma^{-1}(x-\mu)}$$
# To calculate $\mu$ and $\Sigma$, you can use the **sample** mean and covariance (see [here.](https://en.wikipedia.org/wiki/Sample_mean_and_covariance)) Also note that to get the argmax over $P(x|c)P(c)$, we can choose the digit class using:
# $$c^* = argmax_{c}(\log P(x|c)+\log P(c))$$
# Now, let's dive into implementing a **Gaussian Naive Bayes Classifier.**

# ## Loading data

# For your convineince, use the following cells to access the data. 

# In[196]:


#!pip install torchvision
#!pip install numpy
# and other libraries you might need

from torchvision import datasets
import numpy as np
import math

from matplotlib import pyplot as plt


# In[197]:


train_data = datasets.MNIST('./data', train=True, download=True)
test_data  = datasets.MNIST('./data', train=False, download=True)

train_images = np.array(train_data.data)
train_labels = np.array(train_data.targets)
test_images = np.array(test_data.data)
test_labels = np.array(test_data.targets)


# ## Training the Model

# In[198]:


class Bayes:
    def train(self, train_images, train_labels):
        k1 = np.size(train_labels)
        k2 = 1000
        dim = 784    # 28 * 28
        self.gaussian = {}
        self.label_pros = {}
        self.labels = set(train_labels)
        smoothing_factor = 1
        self.N = np.zeros(10)
        self.mu_est = np.float64(np.zeros([10,dim]))
        self.sigma_est = np.float64(np.zeros([10,dim,dim]))
        for i in range(10):
            self.N[i] = np.sum(train_labels==i)
        for i in range(k1):
            img = np.float64(np.reshape(train_images[i],(dim)))
            self.mu_est[train_labels[i],:] += img/(self.N[train_labels[i]])
        
        for i in range(k2):
            img = np.float64(np.reshape(train_images[i],(dim)))
            img_meanless = img - self.mu_est[train_labels[i],:] 
            im_shaped = np.reshape(img_meanless,(dim,1))
            temp = np.matmul(im_shaped,im_shaped.T)              
            self.sigma_est[train_labels[i],:,:] += temp /(self.N[train_labels[i]]-1)
                            
        for i in range(10):
            self.sigma_est[i] += smoothing_factor * np.identity(dim)  
        pass
    
    def calc_accuracy(self, images, labels):
        n = len(images)
        dim = 784    # 28 * 28
        decision = np.zeros_like(labels)   
        for i in range(n):
            im = np.float64(np.reshape(images[i],(1,dim)))
            decision[i] = self.predict_labels(im)
        presition = np.sum(decision[:n]==labels[:n]) / n
        print(presition)
        return presition
    
    def predict_labels(self, images):
        px_c = np.zeros(10)     
        for j in range(10):
            im_meanless = images - self.mu_est[j]
            a1 = -(dim/2) * np.log(2 * math.pi)
            a2 = -(1/2) * np.log(np.linalg.det(self.sigma_est[j]))
            a3 = -(1/2) * np.matmul(np.matmul(im_meanless,np.linalg.inv(self.sigma_est[j])),im_meanless.T)
            px_c[j] = a1+a2+a3
        decision = px_c.argmax()
        
        #################################################
        # 𝑃(𝑐)) is aproximatly uniformely distrivuted.  #
        # So I didn't add log𝑃(𝑐)) in optimization and  #
        # I've got a good result                        #
        #################################################
        return decision


# In[199]:


network = Bayes()
network.train(train_images, train_labels)
#network.smoothing(0.5)


# ## Model Evaluation

# In[ ]:


print("Accuracy on test data (%) : " + str(network.calc_accuracy(test_images, test_labels) * 100))

