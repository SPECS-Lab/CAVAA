# Adaptive layer

Responsible: Adrian Fernandez Amil (adrian.fernandezamil@donders.ru.nl)

CAVAA's adaptive layer is a hippocampal modeled based on a sparse (convolutional) autoencoder with an orthonormal activity constraint in the latent space to promote the formation of 'place cells'. The model receives images (currently of 84 x 84 x 3), and outputs a sparse, high-dimensional embedding that contains high spatial information. The model was presented in [Amil, A. F., Freire, I. T., & Verschure, P. F. (2024). Discretization of continuous input spaces in the hippocampal autoencoder. arXiv preprint arXiv:2405.14600](https://arxiv.org/abs/2405.14600).

This folder also includes a whole suite of functions to train the model, analyze its internal representations, and compute metrics related to space.
