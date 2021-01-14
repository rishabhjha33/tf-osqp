"""
Engine for OSQP solution through tensorflow.

Created by @pachees

"""

import numpy as np
import tensorflow as tf
import warnings


physical_devices = tf.config.list_physical_devices('GPU')

gpu_warning = """Computing on GPU with some TF versions raise an error without memory growth enabled (performed automatically by tf_osqp). Please report the issue on GitHub if your console crashes on tensorflow-gpu """

try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    warnings.warn(gpu_warning, UserWarning)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

class solver(tf.Module):
    """Base class encapsulating entire tf_osqp library.
        
    Class is used to first formulate the problem by setting the values of 
    P q A l u and then handles optimization for the problem iteratively until
    either a termination is reached or maximum iterations completed.
    """

    def __init__(self):
        """Initialize the problem with default values of constants.
        
        Create variables and set values based on default osqp values.
        
        Returns
        -------
        None.
        """

        # Parameters
        self.rho = tf.constant(value=0.1, name='rho', dtype=tf.float32)
        self.sigma = tf.constant(value=1e-6, name='sigma', dtype=tf.float32)
        self.eps_abs = tf.constant(
            value=1e-4, name='eps_abs', dtype=tf.float32)
        self.eps_rel = tf.constant(
            value=1e-4, name='eps_rel', dtype=tf.float32)
        self.eps_prim_inf = tf.constant(
            value=0.0001, name='eps_prim_inf', dtype=tf.float32)
        self.eps_dual_inf = tf.constant(
            value=0.0001, name='eps_dual_inf', dtype=tf.float32)
        self.alpha = tf.constant(value=1.6, name='alpha', dtype=tf.float32)
        self.adaptive_rho_interval = tf.constant(
            value=200, name='adaptive_rho_interval', dtype=tf.int32)
        self.adaptive_rho_tolerance = tf.constant(
            value=5, name='adaptive_rho_tolerance', dtype=tf.float32)

        # Max iterations for looping
        self.max_iters = tf.constant(
            value=2000, name='max_iters', dtype=tf.int32)

        # Parameter bounds
        self.RHO_MIN = tf.constant(
            value=0.000001, name='RHO_MIN', dtype=tf.float32)
        self.RHO_MAX = tf.constant(
            value=1_000_000, name='RHO_MAX', dtype=tf.float32)
        self.RHO_EQ_OVER_RHO_INEQ = tf.constant(
            value=1_000, name='RHO_EQ_OVER_RHO_INEQ', dtype=tf.float32)
        self.RHO_TOL = tf.constant(
            value=0.0001, name='RHO_TOL', dtype=tf.float32)

        # OSQP Infinity
        self.OSQP_INFTY = tf.constant(
            value=1e+30, name='OSQP_INFTY', dtype=tf.float32)
        #inf = tf.constant(value=np.inf, name='np_inf', dtype=tf.float32)
        self.inf = np.inf

        # OSQP Nan
        self.OSQP_NAN = tf.constant(
            value=np.nan, name='OSQP_NAN', dtype=tf.float32)
        self.divbyzero = tf.constant(
            value=1e-10, name='divbyzero', dtype=tf.float32)

        # Scaling
        self.MIN_SCALING = tf.constant(
            value=0.0001, name='MIN_SCALING', dtype=tf.float32)
        self.MAX_SCALING = tf.constant(
            value=10_000, name='MAX_SCALING', dtype=tf.float32)
        
    def set_problem(self, P, q, A, l, u):
        """Set up the problem based on P, q, A, l, u.
        Parameters
        ----------
        P : tf.float32 matrix - shape = n_x X n_x
            Positive semidefinite matrix to define quadratic coefficients 
            of x in the optimization equation.
        q : tf.float32 vector - shape = n_x
            Vector to define the linear terms of x in the optimization equation.
        A : tf.float32 matrix - shape = n_constraints X n_x
            Defining the coefficients of all linear constraints on x.
        l : tf.float32 vector - shape = n_contraints
            Lower bound of each contraint defined by A.
        u : tf.float32 vector - shape = n_contraints
            Upper bound of each constraint defined by A.

        Returns
        -------
        None.
        """

        self.P = P
        self.q = q
        self.A = A
        self.l = l
        self.u = u

        self.n = tf.constant(value=P.shape[0], dtype=tf.int32, shape=[])
        self.m = tf.constant(value=A.shape[0], dtype=tf.int32, shape=[])

        self.x = tf.Variable(initial_value=tf.zeros(self.n), trainable=True, shape=[
                             self.n], validate_shape=True, name='osqp_x')
        self.z = tf.Variable(initial_value=tf.zeros(self.m), trainable=True, shape=[
                             self.m], validate_shape=True, name='osqp_z')
        self.x_prev = tf.Variable(initial_value=tf.zeros(self.n), trainable=True, shape=[
                                  self.n], validate_shape=True, name='osqp_x_prev')
        self.z_prev = tf.Variable(initial_value=tf.zeros(self.m), trainable=True, shape=[
                                  self.m], validate_shape=True, name='osqp_z_prev')

        self.xz_tilde = tf.Variable(initial_value=tf.zeros(
            (self.n+self.m)), trainable=True, shape=[self.n+self.m], validate_shape=True, name='osqp_xz_tilde')
        self.y = tf.Variable(initial_value=tf.zeros(self.m), trainable=True, shape=[
                             self.m], validate_shape=True, name='osqp_y')

        # set rho vec
        rho_vec_ = tf.concat(values=[tf.fill(dims=[self.m-1, ], value=self.rho),
                                     tf.fill(dims=[1, ], value=self.RHO_EQ_OVER_RHO_INEQ * self.rho)], axis=0)
        rho_inv_vec_ = tf.math.reciprocal(rho_vec_)

        self.rho_vec = tf.Variable(initial_value=rho_vec_, trainable=True, shape=[
                                   self.m], validate_shape=True, name='osqp_rho_vec')
        self.rho_inv_vec = tf.Variable(initial_value=rho_inv_vec_, trainable=True, shape=[
                                       self.m], validate_shape=True, name='osqp_rho_inv_vec')

        self.delta_y = tf.zeros_like(self.z)
        self.delta_x = tf.zeros_like(self.x)

        self.z_part = tf.zeros_like(self.z)
        self.x_part = tf.zeros_like(self.x)

        self.pri_check = tf.Variable(
            initial_value=False, name='pri_check', dtype=tf.bool)
        self.dua_check = tf.Variable(
            initial_value=False, name='dua_check', dtype=tf.bool)
        self.prim_inf_check = tf.Variable(
            initial_value=False, name='prim_inf_check', dtype=tf.bool)
        self.dual_inf_check = tf.Variable(
            initial_value=False, name='dual_inf_check', dtype=tf.bool)

    def check_continuation(self):
        """Check if any criterion for termination has been met yet.
        
        Returning true allows next iteration to happen 

        Returns
        -------
        bool
            Whether to continue to the next iteration.
        str
            Reason for termination, if needed.
        """
        
        self.pri_check.assign(False)
        self.dua_check.assign(False)
        self.prim_inf_check.assign(False)
        self.dual_inf_check.assign(False)

        # Compute residuals
        self.pri_res = tf.norm(tf.tensordot(
            self.A, self.x, axes=1) - self.z, ord=self.inf)
        self.dua_res = tf.norm(tf.tensordot(self.P, self.x, axes=1) + self.q +
                               tf.tensordot(tf.transpose(self.A), self.y, axes=1), ord=self.inf)

        # Check ki kahi yeh non convex toh nahi ono
        if (self.pri_res > self.OSQP_INFTY) or (self.dua_res > self.OSQP_INFTY):
            return False, 'OSQP_INFTY'

        # Compute primal tolerance
        max_rel_eps = tf.math.maximum(tf.norm(tf.tensordot(self.A, self.x, axes=1), ord=self.inf),
                                      tf.norm(self.z, ord=self.inf))
        eps_pri = self.eps_abs + self.eps_rel * max_rel_eps

        if self.pri_res < eps_pri:
            self.pri_check.assign(True)
        else:
            # Check primal infeasibility
            norm_delta_y = tf.norm(self.delta_y, ord=self.inf)

            if norm_delta_y > self.eps_prim_inf:
                Atdelta_y = tf.tensordot(
                    tf.transpose(self.A), self.delta_y, axes=1)
                self.prim_inf_check.assign(
                    tf.norm(Atdelta_y, self.inf) < self.eps_prim_inf * norm_delta_y)
            else:
                self.prim_inf_check.assign(False)

        # Compute dual tolerance
        max_rel_eps = tf.math.maximum(tf.norm(tf.tensordot(tf.transpose(self.A), self.y, axes=1), ord=self.inf),
                                      tf.math.maximum(tf.norm(tf.tensordot(self.P, self.x, axes=1), ord=self.inf),
                                                      tf.norm(self.q, self.inf)))

        eps_dua = self.eps_abs + self.eps_rel * max_rel_eps

        if self.dua_res < eps_dua:
            self.dua_check.assign(True)
        else:
            # Check dual infeasibility
            norm_delta_x = tf.norm(self.delta_x, self.inf)
            scale_cost = 1.0

            # Prevent 0 division
            if norm_delta_x > self.eps_dual_inf:

                # First check q'* delta_x < 0
                if tf.tensordot(self.q, self.delta_x, axes=1) < - scale_cost * self.eps_dual_inf * norm_delta_x:
                    # Compute P * delta_x
                    Pdelta_x = tf.tensordot(self.P, self.delta_x, axes=1)

                    # No all checks managed to pass. Problem not dual infeasible
                    # brought before the next ifs
                    self.dual_inf_check.assign(False)

                    # Check if ||P * delta_x|| = 0
                    if tf.norm(Pdelta_x, self.inf) < scale_cost * self.eps_dual_inf * norm_delta_x:
                        # All conditions passed -> dual infeasible
                        # brought from after the loop to before
                        self.dual_inf_check.assign(True)

                        # Compute A * delta_x
                        Adelta_x = tf.tensordot(self.A, self.delta_x, axes=1)

                        for i in tf.range(self.m):
                            # De Morgan's Law applied to negate
                            # conditions on A * delta_x
                            if ((self.u[i] < self.OSQP_INFTY * self.MIN_SCALING) and
                                (Adelta_x[i] > self.eps_dual_inf * norm_delta_x)) \
                                or ((self.l[i] > -self.OSQP_INFTY * self.MIN_SCALING) and
                                    (Adelta_x[i] < -self.eps_dual_inf * norm_delta_x)):
                                # At least one condition not satisfied
                                self.dual_inf_check.assign(False)
                                break
        
        # Compare residuals and determine solver status
        if self.pri_check & self.dua_check:
            return False, 'pdcheck'
        elif self.prim_inf_check:
            # Store original certificate
            return False, 'pinfcheck'
        elif self.dual_inf_check:
            return False, 'dinfcheck'
        else:
            return True, 'true'

    def adapt_rho(self):
        """Adapts the value of rho based on current value of residuals.
        
        Rho helps control the size of update of x in each iteration. This 
        function helps control how large rho is, depending on how large the 
        residuals are in the last iteration.

        Returns
        -------
        None.

        """
        # Adapt rho

        # Compute normalized residuals
        norm_pri_res = self.pri_res / tf.math.maximum(tf.norm(tf.tensordot(self.A, self.x, axes=1), self.inf),
                                                      tf.norm(self.z, self.inf)) + self.divbyzero

        norm_dua_res = self.dua_res / tf.math.maximum(tf.norm(tf.tensordot(tf.transpose(self.A), self.y, axes=1), self.inf),
                                                      tf.math.maximum(tf.norm(tf.tensordot(self.P, self.x, axes=1), self.inf),
                                                                      tf.norm(self.q, self.inf))) + self.divbyzero

        # Compute new rho
        rho_new_ = self.rho * \
            tf.sqrt(tf.divide(norm_pri_res, (norm_dua_res + self.divbyzero)))
        rho_new = tf.math.minimum(tf.math.maximum(
            rho_new_, self.RHO_MIN), self.RHO_MAX)

        if rho_new > self.adaptive_rho_tolerance * self.rho or \
                rho_new < 1.0 / self.adaptive_rho_tolerance * self.rho:
            self.rho = rho_new
            # Set new rho_vec and rho_inv_vec
            self.rho_vec = tf.concat(values=[tf.fill(dims=[self.m-1, ], value=self.rho),
                                             tf.fill(dims=[1, ], value=self.RHO_EQ_OVER_RHO_INEQ * self.rho)],
                                     axis=0)
            self.rho_inv_vec = tf.math.reciprocal(self.rho_vec)

            self.LU, self.perm = self.formulate_LUperm()

    def formulate_LUperm(self):
        """Formulates LU factorization for next round of solution.     
        
        Called during set up and when sigma or rho changes.

        Returns
        -------
        LU : tf.float32 tensor
            Lower triangular and upper triangular factors
        perm : tf.float32 tensor
            Permutation matrix for this case
        """
        # Solve linear system
        KKT = tf.concat([tf.concat([self.P + self.sigma * tf.eye(self.n),
                                    tf.transpose(self.A)],
                                   axis=1),  # dim: n * (n + m)
                         tf.concat([self.A,
                                    -tf.linalg.diag(self.rho_inv_vec)],
                                   axis=1)],  # dim: m * (n + m)
                        axis=0)  # dim: (n+m) * (n+m)
        LU, perm = tf.linalg.lu(KKT)
        return LU, perm

    def objective(self):
        """Computes current value of objective at call.

        Returns
        -------
        tf.float32
            The current value of objective
        """
        return (1/2)*tf.matmul(tf.matmul(self.x[:, tf.newaxis],
                                         self.P, transpose_a=True),
                               self.x[:, tf.newaxis])[0][0] + tf.matmul(tf.transpose(self.q[:, tf.newaxis]),
                                                                        self.x[:, tf.newaxis])[0][0]

    def solve(self):
        """Solves the problem set currently.
        
        Solves for the optimization objective iteratively until a termination
        criteria is satisfied

        Returns
        -------
        x : tf.float32 vector
            Current approximate solution when termination was reached
        y : tf.float32 vector
            Lagrangian vector for x
        """

        self.LU, self.perm = self.formulate_LUperm()

        for i in tf.range(1, self.max_iters + 1):

            self.x_prev.assign(self.x)
            self.z_prev.assign(self.z)

            # 1. Update_xz_tilde.

            # Compute RHS and store in xz_tilde
            self.x_part = self.sigma * self.x_prev - self.q
            self.z_part = self.z_prev - self.rho_inv_vec * self.y
            self.xz_tilde.assign(
                tf.concat([self.x_part, self.z_part], axis=0), name='osqp_update_xz_tilde')

            self.xz_tilde.assign(tf.linalg.lu_solve(lower_upper=self.LU,
                                                    perm=self.perm,
                                                    # tf documentation for lu_solve
                                                    rhs=self.xz_tilde[:,
                                                                      tf.newaxis],
                                                    name='osqp_xz_tilde_solved')[:, 0])

            # Update z_tilde (from paper: recover this from nu (the v like symbol))
            z_part2 = self.z_prev + self.rho_inv_vec * \
                (self.xz_tilde[self.n:] - self.y)
            self.xz_tilde.assign(tf.concat([self.xz_tilde[:self.n],
                                            z_part2],
                                           axis=0))

            # 2. Update x and z.

            # Update x.
            self.x = self.alpha * \
                self.xz_tilde[:self.n] + (1.0 - self.alpha) * self.x_prev
            self.delta_x = self.x - self.x_prev

            # Update z.
            z_ = self.alpha * \
                self.xz_tilde[self.n:] + (1.0 - self.alpha) * \
                self.z_prev + self.rho_inv_vec * self.y
            # project x to [l,u] by bounding out of bound values
            self.z = tf.math.minimum(tf.math.maximum(z_, self.l), self.u)

            # 3. Update y.
            self.delta_y = self.rho_vec * \
                (self.alpha * self.xz_tilde[self.n:] +
                 (1.0 - self.alpha) * self.z_prev - self.z)
            self.y = self.y + self.delta_y

            continuing, case = self.tf_check_continuation()

            if not continuing:
                tf.print(case)
                break

            if i % self.adaptive_rho_interval == 0:
                self.adapt_rho()

        return self.x, self.y
