# GaussianProcessRegression
Gaussian Process Regression (GPR) is adopted to choose the most likely solution for B0 meson to resolve the quadratic ambiguity. GPR is utilised through TensorFlow and due to large datasets GPU is used for training. 

To implement the sGPR model for the B meson system, several hyper-parameter (training variables) choices were optimised. We adopt the Matern covariance function as a flexible general class of kernels. To ensure computational feasibility without sacrificing accuracy, the number of inducing points was fixed at 4000, providing a robust approximation of the underlying function across the large dataset.
The training sample (input features) is derived from MC sample for B0\to D*\mu\nu after all selection stages applied. To teach the topology of our kinematic space by encapsulating all the necessary information for the sGPR to model the prediction of true magnitude of B0 meson spatial momentum, the following kinematic
variables are trained:
- spatial components of D* meson momentum vector
- spatial components of $\mu$ momentum vector
- spatial components of B0 meson flight distance vector
