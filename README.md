## TF-OSQP

[![TF](https://aleen42.github.io/badges/src/tensorflow.svg)](https://www.tensorflow.org/) ![pyversions](https://img.shields.io/pypi/pyversions/tf-osqp) ![pipversion](https://img.shields.io/pypi/v/tf-osqp) ![status](https://img.shields.io/pypi/status/tf-osqp) [![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/) ![wheel](https://img.shields.io/pypi/format/tf-osqp) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


Tensorflow 2.x implementation of [OSQP](https://osqp.org/).  

**This is not an official OSQP implementation.** You can find official versions [here](https://github.com/oxfordcontrol/osqp).

From their page: 

The OSQP (Operator Splitting Quadratic Program) solver is a numerical
optimization package for solving problems in the form

    minimize        0.5 x' P x + q' x

    subject to      l <= A x <= u

where ``x in R^n`` is the optimization variable. The objective function
is defined by a positive semidefinite matrix ``P in S^n_+`` and vector
``q in R^n``. The linear constraints are defined by matrix
``A in R^{m x n}`` and vectors ``l in R^m U {-inf}^m``,
``u in R^m U {+inf}^m``.



## Motivation
This project was created as a complement to a new TF2 implementation for the Revised Greedy Expansion algorithm used to find the convex hull of high dimensional vectors. Locating the distance of a point from a convex hull requires the solution of linearly constrained quadratic equations.

Code for that project will be made available soon.


## Features

### Live in latest release
**Full TF 2.x compatibility:** All computations use native TF ops

**Python side effects absent:** Every constant and variable coded as named instances of ```tf.constant``` and ```tf.Variable```

**Well documented:** Docstrings available for all function with constraints and meanings explained for parameters

### Future releases
**Graph mode execution:** Leverage tf.function() to execute code in Graph mode on TF2

**Training parameter assignment:** Ability to modify training parameters extended to every single parameter

**Full OSQP feature set:** Remaining features like solution polishing added


## Dependencies

| Dependency   | Description | Version   | License
| ------------- |:-------------:| :----:|:---:|
| TensorFlow  | Open source symbolic math library for ML    | >= 2.4rc3 | Apache 2.0 
| python      | Base python software | >= 3.6      | PSF
| numpy | Used to define nan and inf constants | >= 1.18      | NumPy License


## Installation

#### Latest PyPI release

	pip install tf_osqp


## Usage

Having defined ```tf.float32``` matrices P, q, A, l, u, use the code as follows:

	from tf_osqp.tf_osqp import solver
	
    quad_solver = solver()
    quad_solver.set_problem(P, q, A, l, u)
    
    x, y = quad_solver.solve()

Here, x will be the solution at optima, y will be the corresponding Lagrangian


## Contribute

### Product backlog
Current implementation is the skeletal engine written to get the work done. It is some way away from full potential.  
- Refactor code to tf.function() to capitalize on speedups from Graph execution
	- Current version uses Eager execution. This constrains performance and speeds are similar to numpy implementation   
- Add remaining OSQP functionality, e.g., solution polishing
- Make API freer to use, including option for warm starts and changing training parameters between iterations

### Code style

Code is written with PEP-8 guidelines - recommended for PRs.

## Citations

1. G. Banjac, P. Goulart, B. Stellato, and S. Boyd, Infeasibility detection in the alternating direction method of multipliers for convex optimization, Journal of Optimization Theory and Applications 183(2019), no. 2, 490–519. 
2. G. Banjac, B. Stellato, N. Moehle, P. Goulart, A. Bemporad, and S. Boyd, Embedded codegeneration using the OSQP solver, IEEE Conference on Decision and Control (CDC), 2017. 
3. M. Schubiger, G. Banjac, and J. Lygeros, GPU acceleration of ADMM for large-scale quadratic programming, Journal of Parallel and Distributed Computing 144(2020), 55–67.
4. B. Stellato, G. Banjac, P. Goulart, A. Bemporad, and S. Boyd, OSQP: an operator splitting solver for quadratic programs, Mathematical Programming Computation 12(2020), no. 4,637–672.
5. B. Stellato, V. V. Naik, A. Bemporad, P. Goulart, and S. Boyd, Embedded mixed-integer quadratic optimization using the OSQP solver, European Control Conference (ECC), 2018

The original pure python source code for OSQP can be found on their repo [here](https://github.com/oxfordcontrol/osqp-python/tree/master/modulepurepy).

Current implementation in this repo can execute on GPUs but the code hasn't been optimized to offer large speed ups yet. To see better performance on GPUs, the recommendation is to OSQP's official CUDA implementation: [cuosqp](https://github.com/oxfordcontrol/cuosqp) 

## License
Apache 2.0 