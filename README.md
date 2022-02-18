# Gaussian Naive Bayes Learning

Recall Bayes rule:

<img src="https://render.githubusercontent.com/render/math?math=P(c|x) =  \frac{P(x|c)P(c)}{P(x)}">
    
Here x stands for the image, or more precisely, the pixel values of the formatted image as a vector, and c stands for the number, which can be 0, 1, ..., 9. We can read the left side P(c|x) as "the probability of the class being $c$ given the x" data. We can read the right side P(x|c) as "the probability of x data being in the c" class. We care about the value of c. It tells us "what number" this picture is. The chosen class is simply the one with the highest probability for this data:

<img src="https://render.githubusercontent.com/render/math?math=c^* = argmax_{c}P(c|x)">

Now, we can ignore P(x) in equation. Using this information, we can simplify our problem so that, in order to choose “which digit” given an image, all we need to do is calculate this argmax:

<img src="https://render.githubusercontent.com/render/math?math=c^* = argmax_{c}P(x|c)P(c)">

Now, we need to think about how to calculate P(c), and P(x|c).

Remember that pixels represent the intensity of light, and that the intensity of light is in fact continuous. A first reasonable estimation to model continuous data is the multivariate Gaussian or multivariate Normal. We can write:

<img src="https://render.githubusercontent.com/render/math?math=P(x|c) = \frac{1}{\sqrt{(2\pi)^{D}|\Sigma|}}\exp(-\frac{1}{2}(x - \mu)^{T}\Sigma^{-1}(x-\mu))">

Note that because probabilities are very small when dimensionality is high, we are going to work with log-probability rather than probability. So instead of getting numbers that are very close to 0, which is inaccurate when you use a computer to represent them, we're just going to get negative numbers. The log-probability can be represented as (D is the dimentionality):

<img src="https://render.githubusercontent.com/render/math?math=\log{P(x|c) = -\frac{D}{2}\ln(2\pi)-\frac{1}{2}\ln|\Sigma|-\frac{1}{2}(x - \mu)^{T}\Sigma^{-1}(x-\mu)}">

To calculate $\mu$ and $\Sigma$, you can use the **sample** mean and covariance (see [here.](https://en.wikipedia.org/wiki/Sample_mean_and_covariance)) 
Also note that to get the argmax over P(x|c)P(c), we can choose the digit class using:

<img src="https://render.githubusercontent.com/render/math?math=c^* = argmax_{c}(\log P(x|c)+\log P(c))">

Now, let's dive into implementing a **Gaussian Naive Bayes Classifier.**
