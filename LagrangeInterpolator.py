import openmdao.api as om
import numpy as np


class CalculateBarycentricWeights(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_interp_from_nodes')

    def setup(self):
        num_interp_from_nodes = self.options['num_interp_from_nodes']

        self.add_input('data_x', shape=(num_interp_from_nodes,))

        self.add_output('weights', shape=(num_interp_from_nodes,))

        self.declare_partials('weights', 'data_x')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        x = inputs['data_x']

        spacings = x.reshape(-1, 1) - x  # Do xj - xk for j = 0, ..., n
        spacings = spacings[~np.eye(spacings.shape[0], dtype=bool)].reshape(spacings.shape[0], -1)  # Remove diagonal indices, so becomes xj-xk for all j except j=k
        weights = 1/np.prod(spacings, axis=1)
        outputs['weights'] = weights

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        n = self.options['num_interp_from_nodes']
        x = inputs['data_x']

        # What follows is pure pattern matching, I have no idea why this really produces the correct derivatives
        spacings = x.reshape(-1, 1) - x  # Do xj - xk for j = 0, ..., n
        spacings = spacings[~np.eye(spacings.shape[0], dtype=bool)].reshape(spacings.shape[0], -1)  # Remove diagonal indices, so becomes xj-xk for all j except j=k

        tiled = np.tile(spacings, n - 1)  # Repeat the array for calculating the derivatives
        tiled[:, np.arange(n - 1) * n] = tiled[:, np.arange(n-1)*n]**2  # Square the factors that need squaring
        tiled.reshape(-1, n-1, n-1)  # Shape for product operation

        dwi_dxj = 1/(np.prod(tiled.reshape(-1, n-1, n-1), axis=2))  # Perform product operation, this gives us the off-diagonal entries in the subjacobian
        dwj_dxj = np.sum(-dwi_dxj, axis=1)  # Negate then sum these to get the diagonal entries

        dw_dx = np.zeros((n, n))
        dw_dx[:, :-1] += np.tril(dwi_dxj, -1)  # This code found at https://stackoverflow.com/questions/34640169/what-is-the-fastest-way-to-insert-elements-diagonally-in-2d-numpy-array
        dw_dx[:, 1:] += np.triu(dwi_dxj, 0)
        np.fill_diagonal(dw_dx, dwj_dxj)
        partials['weights', 'data_x'] = dw_dx


class PerformInterpolation(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_interp_from_nodes')
        self.options.declare('num_interp_to_nodes')

    def setup(self):
        num_interp_from_nodes = self.options['num_interp_from_nodes']
        num_interp_to_nodes = self.options['num_interp_to_nodes']

        self.add_input('data_x', shape=(num_interp_from_nodes,))
        self.add_input('data_y', shape=(num_interp_from_nodes,))
        self.add_input('weights', shape=(num_interp_from_nodes,))
        self.add_input('interp_x', shape=(num_interp_to_nodes,))

        self.add_output('interp_y', shape=(num_interp_to_nodes,))

        self.declare_partials('interp_y', 'data_x')
        self.declare_partials('interp_y', 'data_y')
        self.declare_partials('interp_y', 'weights')
        self.declare_partials('interp_y', 'interp_x',
                              rows=np.arange(num_interp_to_nodes),
                              cols=np.arange(num_interp_to_nodes))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        x_f = inputs['data_x']  # x from
        y_f = inputs['data_y']
        w = inputs['weights']
        x_t = inputs['interp_x']  # x to

        dist = x_t.reshape(-1, 1) - x_f
        dist = np.where(np.abs(dist) < 1e-5, 1e-5, dist)  # This catches any singularities when the interpolating coordinate is the same as a data coordinate, prevents division by 0

        weight_dist = w/dist  # Weight divided by distance
        y_t = np.sum(weight_dist*y_f, axis=1)/np.sum(weight_dist, axis=1)
        outputs['interp_y'] = y_t

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        x_f = inputs['data_x']
        y_f = inputs['data_y']
        w = inputs['weights']
        x_t = inputs['interp_x']

        dist = x_t.reshape(-1, 1) - x_f
        dist = np.where(np.abs(dist) < 1e-9, 1e-9, dist)

        weight_dist = w/dist  # To save repeated calculation
        weight_dist_squared = w/dist**2

        f = np.sum(weight_dist*y_f, axis=1).reshape(-1, 1)  # Denominator, f as in quotation rule notation
        g = np.sum(weight_dist, axis=1).reshape(-1, 1)   # Denominator, g as in quotation rule notation
        one_over_g_squared = 1/g**2

        df_dx_t = np.sum(-y_f*weight_dist_squared, axis=1)
        dg_dx_t = np.sum(-weight_dist_squared, axis=1)
        partials['interp_y', 'interp_x'] = ((g.reshape(-1)*df_dx_t - f.reshape(-1)*dg_dx_t)*one_over_g_squared.reshape(-1))  # Quotient rule

        dg_dx_f = weight_dist_squared
        df_dx_f = y_f*dg_dx_f  # Tiny bit faster to reuse the result like this, probably
        partials['interp_y', 'data_x'] = (g*df_dx_f - f*dg_dx_f)*one_over_g_squared

        # dg_dy_f = 0
        df_dy_f = weight_dist
        partials['interp_y', 'data_y'] = df_dy_f/g

        dg_dw = 1/dist
        df_dw = y_f*dg_dw
        partials['interp_y', 'weights'] = (g*df_dw - f*dg_dw)*one_over_g_squared


class LagrangeInterpolatorGroup(om.Group):

    def initialize(self):
        self.options.declare('num_interp_from_nodes')
        self.options.declare('num_interp_to_nodes')

    def setup(self):
        num_interp_from_nodes = self.options['num_interp_from_nodes']
        num_interp_to_nodes = self.options['num_interp_to_nodes']

        self.add_subsystem('CalculateBarycentricWeights', CalculateBarycentricWeights(num_interp_from_nodes=num_interp_from_nodes),
                           promotes_inputs=['data_x'],
                           promotes_outputs=['weights'])

        self.add_subsystem('PerformInterpolation', PerformInterpolation(num_interp_from_nodes=num_interp_from_nodes,
                                                                        num_interp_to_nodes=num_interp_to_nodes),
                           promotes_inputs=['data_x',
                                            'data_y',
                                            'interp_x',
                                            'weights'],
                           promotes_outputs=['interp_y'])


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    n_from = 50
    n_to = 7

    p = om.Problem(model=om.Group())

    p.model.add_subsystem('LagrangeInterpolatorGroup', LagrangeInterpolatorGroup(num_interp_from_nodes=n_from,
                                                                                 num_interp_to_nodes=n_to),
                          promotes_inputs=['data_x',
                                           'data_y',
                                           'interp_x'],
                          promotes_outputs=['interp_y'])

    p.model.set_input_defaults('data_x', np.linspace(0, 1, n_from))
    p.model.set_input_defaults('data_y', np.cos(2*np.pi*np.linspace(0, 1, n_from)))
    p.model.set_input_defaults('interp_x', np.linspace(0, 1, n_to))

    p.setup()

    with np.printoptions(linewidth=1024, edgeitems=20, formatter={'float_kind': '{:5.5f}'.format}):
        # formatter={'float_kind': '{:5.2f}'.format})
        p.check_partials(show_only_incorrect=False, compact_print=False)

    p.run_model()

    plt.plot(p.get_val('data_x'), p.get_val('data_y'), marker='x', label='Source data')
    plt.plot(p.get_val('interp_x'), p.get_val('interp_y'), marker='+', label='Interpolated data')
    plt.legend()
