## Lab days

This guide provides ideas on what you might want to consider performing during your lab.

**Task 1: Task-distribution file, organisation of gitlab**
Alongside your documentation, please submit a file which shows who of you has contributed which part of the tasks.
Throughout the lab you work in individual branches but your final submission of all relevant files should be together in one branch marked as submission.


**Task 2:** *Load the files and generate a training, validation, and test set. Each set contains the energy-band-image, the redshift of the cluster, and its mass.*

**Task 3: Neural network - regression**
Buid a neural network which can take these images as input and predicts just the mass. Train this vision network and check the performance by looking at the training and validation error. For your network you should check the scatter of the true mass vs the predicted mass on a scatter plot.

Most likely your network is not performing well at this stage. Now we need to discuss on how to improve upon our predictions? You should discuss as a team which strategies you can think about improving the prediction. For each of these strategies you should also think about how you would go about testing them.

**Task 4 (group): Neural network - regression improvement**
Discuss in the group how the performance can be improved. Make a list and everybody can try one such potential improvement.

**Task 5 (individual): Neural network - regression improvement**
Each of you should implement at least one and test one of the ways to improve performance. In your report you should describe your hyperparameter scan and discuss the performance. This includes comments on the training behaviour and the generalisation behaviour after training. You should store the weights of your networks such that you can re-use them later.
*If time permits, we shall discuss how to implement the code and the hyperparameter scan efficiently.*

Finally, we are interested in estimating the error associated to the predictions of our neural network. To do this, we are interested in predicting the mean and standard deviation of a Gaussian for any given image. Given this assumption of a Gaussian, we can evaluate the conditional probability $`p(y|x,\theta)`$ where $`y`$ denotes the mass in our dataset, $x$ the image, and $`\theta`$ summarises our neural network parameters. In particular we obtain:
```math
p(y|x,\theta)=\frac{1}{\sqrt{2\pi\sigma(x,\theta)^2}}\exp{\left(\frac{(y-f(x,\theta))^2}{2\sigma(x,\theta)^2}\right)}.
```

We will be interested in implementing the log-likelihood of this quantity for our best-performing neural network.

**Task 6:** *Implement this modified loss which you have derived in your preparation to the lab. You might find [this documentation](https://keras.io/api/losses/) and our [mlandpythonbasics.md](guides/mlandpythonbasics.md) useful in the process.*

**Task 7:** *Modify your network such that it predicts the mean and standard deviation associated to a single cluster. Does your network train appropriately?*


**Task 8: Re-training your best network with log-likelihood loss**
As the final test we take our best neural network, and re-train it with our log-likelihood loss. If none of your neural networks has produced reasonable results, you will be provided with a working neural network, and you can work with that respective neural network.

Visualise your results as a scatter plot where you either show the prediction uncertainty in the color or with respective error bars.
