# GaussianProcessRegression
Gaussian Process Regression (GPR) is adopted to choose the most likely solution for $B^0$ meson to resolve the quadratic ambiguity. GPR is utilised through TensorFlow and due to large datasets GPU is used for training. 

To implement the sGPR model for the B meson system, several hyper-parameter (training variables) choices were optimised. We adopt the Matern covariance function as a flexible general class of kernels. To ensure computational feasibility without sacrificing accuracy, the number of inducing points was fixed at 4000, providing a robust approximation of the underlying function across the large dataset.
The training sample (input features) is derived from MC sample for $B^0\to D^*\mu\nu$ after all selection stages applied. To teach the topology of our kinematic space by encapsulating all the necessary information for the sGPR to model the prediction of true magnitude of $B^0$ meson spatial momentum, the following kinematic
variables are trained:
- spatial components of $D^*$ meson momentum vector
- spatial components of $\mu$ momentum vector
- spatial components of $B^0$ meson flight distance vector

## Workflow
- It builds and saves necessary kinematic variables to be used at training: ```prep_data.py```
- Train the kinematic variables and get model parameters from subset: ```training.py```
- Apply the model to whole dataset to predict the most likely solution: ```apply_model.py```

## Results

![Prediction](images/b0_PResolution_log.pdf)

The model improves the reconstruction resolution by approximately 40% comparing to the generic approach in this area.
