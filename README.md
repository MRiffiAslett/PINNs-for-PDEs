# Physics-Informed Neural Networks for Solving PDEs

This project implements a Physics-Informed Neural Network (PINN) for solving a simple Partial Differential Equation (PDE). The approach uses deep learning to understand both the data and the underlying physics of the system, as described in the equation.

## Problem

The Partial Differential Equation we consider is of the form:

∂u/∂t = ∂²u/∂x²,

with the boundary conditions u(t, 0) = u(t, 1) = 0, and the initial condition u(0, x) = sin(pi*x).

The analytical solution to this PDE is u(t, x) = exp(-pi²t)sin(pi*x), which is used to generate synthetic data for training and testing the PINN.

## Method

We use a PINN that has been implemented using the TensorFlow library. The neural network is trained to output a single value of u given t and x, and the loss function includes a term for the residual of the PDE, which the network learns to minimize. This encourages the network to learn not just the data, but also the underlying physics as described by the PDE.

## Files

* `pinn_pde_solver.py`: This is the main Python script that implements and trains the PINN.
* `requirements.txt`: This file lists the Python dependencies needed to run the script.

## Installation

1. Clone the repository:

\`\`\`bash
git clone <your-repo-url>
cd <your-repo-name>
\`\`\`

2. Install the required dependencies:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Usage

Run the Python script:

\`\`\`bash
python pinn_pde_solver.py
\`\`\`

This will train the network and print the loss at every 100 epochs. After training, it will output the predicted values for a set of test inputs.

## Results

The script prints out the loss during training to show the progress of the training. After training, it tests the trained network on a set of test inputs and prints out the predicted values.

## Future Work

This is a simple example for illustrative purposes. In a practical application
