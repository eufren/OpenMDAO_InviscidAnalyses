import openmdao.api as om
import numpy as np


class CumulativeIntegrateTrapeziums(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('n', types=int)

    def setup(self):
        n = self.options['n']

        self.add_input('x', shape=(n,))
        self.add_input('x0', 0)
        self.add_input('y', shape=(n,))
        self.add_input('y0', 0)

        self.add_output('cumulative_integral', shape=(n,))

        self.declare_partials('cumulative_integral', 'x0')
        stops = np.repeat(n, n - 1) # Adapted from https://codereview.stackexchange.com/questions/83018/vectorized-numpy-version-of-arange-with-multiple-start-stop
        starts = np.arange(1, n)
        l = stops - starts
        self.declare_partials('cumulative_integral', 'x',
                              rows=np.concatenate((np.arange(n), np.repeat(stops - l.cumsum(), l) + np.arange(l.sum()))),
                              cols=np.concatenate((np.arange(n), np.repeat(np.arange(n-1), np.arange(n-1, 0, -1)))))
        self.declare_partials('cumulative_integral', 'y0')
        self.declare_partials('cumulative_integral', 'y',
                              rows=np.concatenate((np.arange(n), np.repeat(stops - l.cumsum(), l) + np.arange(l.sum()))),
                              cols=np.concatenate((np.arange(n), np.repeat(np.arange(n - 1), np.arange(n - 1, 0, -1)))))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        n = self.options['n']

        x = inputs['x']
        x0 = inputs['x0']
        y = inputs['y']
        y0 = inputs['y0']

        A = np.zeros(n)
        A[0] = (x[0]-x0)*(y[0]+y0)/2
        A[1:] = (x[1:] - x[:-1])*(y[:-1] + y[1:])/2
        outputs['cumulative_integral'] = np.cumsum(A)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        n = self.options['n']

        x = inputs['x']
        x0 = inputs['x0']
        y = inputs['y']
        y0 = inputs['y0']

        # w.r.t x
        dIi_dxi = np.zeros(n)  # This is dIi_dxj for j = i
        dIi_dxi[0] = (y0 + y[0])/2
        dIi_dxi[1:] = (y[:-1] + y[1:])/2

        dIi_dxj0 = (y0 - y[1])/2  # x0 as in x[0], not x0. This is dIi_dxj for j = 0, j != i
        dIi_dxj0 = np.repeat(dIi_dxj0, n-1)

        dIi_dxj = (y[:-2] - y[2:])/2  # This is dIi_dxj for j != 0, j != i
        dIi_dxj = np.repeat(dIi_dxj, np.arange(n-2, 0, -1))

        dI_dx = np.concatenate((dIi_dxi, dIi_dxj0, dIi_dxj))

        # w.r.t x0
        dI_dx0 = -(y[0] + y0) / 2

        # w.r.t y
        dIi_dyi = np.zeros(n)  # This is dIi_dyj for j = i
        dIi_dyi[0] = (x[0] - x0) / 2
        dIi_dyi[1:] = (x[1:] - x[:-1])/2

        dIi_dyj0 = (x[1] - x0)/2  # yj0 as in y[0], not y0. This is dIi_dyj for j = 0, j != i
        dIi_dyj0 = np.repeat(dIi_dyj0, n-1)

        dIi_dyj = (x[2:] - x[:-2])/2  # This is dIi_dyj for j != 0, j != i
        dIi_dyj = np.repeat(dIi_dyj, np.arange(n-2, 0, -1))

        dI_dy = np.concatenate((dIi_dyi, dIi_dyj0, dIi_dyj))

        # w.r.t y0
        dI_dy0 = (x[0]-x0)/2

        partials['cumulative_integral', 'x'] = dI_dx
        partials['cumulative_integral', 'x0'] = dI_dx0
        partials['cumulative_integral', 'y'] = dI_dy
        partials['cumulative_integral', 'y0'] = dI_dy0
