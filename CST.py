import openmdao.api as om
import numpy as np
from scipy.special import comb


class GenerateXCoordinates(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_points', types=int)

    def setup(self):
        num_points = self.options['num_points']

        self.add_output('x_coordinates', shape=(num_points,))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        num_points = self.options['num_points']

        outputs['x_coordinates'] = 0.5*np.cos(np.pi * (np.linspace(0, 0.9999, num_points + 1) + 1))[1:] + 0.5


class ClassFunction(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_points', types=int)

    def setup(self):
        num_points = self.options['num_points']

        self.add_input('x_coordinates', shape=(num_points,))

        self.add_output('class_function', shape=(num_points,))

        self.declare_partials('class_function', 'x_coordinates',
                              rows=np.arange(num_points),
                              cols=np.arange(num_points))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        x = inputs['x_coordinates']

        outputs['class_function'] = np.sqrt(x)*(1-x)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        x = inputs['x_coordinates']

        partials['class_function', 'x_coordinates'] = -((3*x)-1)/(2*np.sqrt(x))  # Not really necessary, as x-coordinates shouldn't change.


class BernsteinPolynomials(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_points', types=int)
        self.options.declare('num_weights', types=int)

    def setup(self):
        num_points = self.options['num_points']
        num_weights = self.options['num_weights']

        self.add_input('x_coordinates', shape=(num_points,))
        self.add_input('weights', shape=(num_weights,))

        self.add_output('bernstein_polynomials', shape=(num_weights, num_points))

        self.declare_partials('bernstein_polynomials', 'x_coordinates',
                              rows=np.arange(num_points*num_weights),
                              cols=np.tile(np.arange(num_points), num_weights))
        self.declare_partials('bernstein_polynomials', 'weights',
                              rows=np.arange(num_points*num_weights),
                              cols=np.repeat(np.arange(num_weights), num_points))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        num_weights = self.options['num_weights']
        x = inputs['x_coordinates']
        weights = inputs['weights']

        r = np.arange(num_weights).reshape(-1, 1)
        weights = weights.reshape(-1, 1)
        bernstein_polynomials = weights*comb(num_weights-1, r)*(x**r)*((1-x)**(num_weights-1-r))

        outputs['bernstein_polynomials'] = bernstein_polynomials

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        num_weights = self.options['num_weights']
        x = inputs['x_coordinates']
        weights = inputs['weights']

        r = np.arange(num_weights).reshape(-1, 1)
        weights = weights.reshape(-1, 1)
        a = weights*comb(num_weights-1, r)
        db_dx = a*r*((1-x)**(num_weights-1-r))*(x**(r-1)) \
            - a*(num_weights-1-r)*((1-x)**(num_weights-2-r))*(x**r)

        partials['bernstein_polynomials', 'x_coordinates'] = db_dx.reshape(-1)  # This has a divide by zero, but x_coordinates shouldn't change anyway. Not an issue.
        partials['bernstein_polynomials', 'weights'] = (comb(num_weights-1, r)*(x**r)*((1-x)**(num_weights-1-r))).reshape(-1)


class SummedBernsteinPolynomials(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_points', types=int)
        self.options.declare('num_weights', types=int)

    def setup(self):
        num_points = self.options['num_points']
        num_weights = self.options['num_weights']

        self.add_input('bernstein_polynomials', shape=(num_weights, num_points))

        self.add_output('summed_bernstein_polynomials', shape=(num_points,))

        self.declare_partials('summed_bernstein_polynomials', 'bernstein_polynomials',
                              rows=np.tile(np.arange(num_points), num_weights),
                              cols=np.arange(num_points*num_weights),
                              val=np.ones(num_points*num_weights))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        bernstein_polynomials = inputs['bernstein_polynomials']

        outputs['summed_bernstein_polynomials'] = np.sum(bernstein_polynomials, axis=0)


class TrailingEdgeWedge(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_points', types=int)

    def setup(self):
        num_points = self.options['num_points']

        self.add_input('x_coordinates', shape=(num_points,))
        self.add_input('te_weight')

        self.add_output('trailing_edge_wedge', shape=(num_points,))

        self.declare_partials('trailing_edge_wedge', 'x_coordinates',
                              rows=np.arange(num_points),
                              cols=np.arange(num_points))
        self.declare_partials('trailing_edge_wedge', 'te_weight')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        x = inputs['x_coordinates']
        te_weight = inputs['te_weight']

        outputs['trailing_edge_wedge'] = te_weight * x

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        x = inputs['x_coordinates']
        te_weight = inputs['te_weight']

        partials['trailing_edge_wedge', 'x_coordinates'] = te_weight
        partials['trailing_edge_wedge', 'te_weight'] = x


class LeadingEdgeShaping(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_points', types=int)
        self.options.declare('num_weights', types=int)

    def setup(self):
        num_points = self.options['num_points']

        self.add_input('x_coordinates', shape=(num_points,))
        self.add_input('le_weight')

        self.add_output('leading_edge_shape', shape=(num_points,))

        self.declare_partials('leading_edge_shape', 'x_coordinates',
                              rows=np.arange(num_points),
                              cols=np.arange(num_points))
        self.declare_partials('leading_edge_shape', 'le_weight')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        num_weights = self.options['num_weights']
        x = inputs['x_coordinates']
        le_weight = inputs['le_weight']

        outputs['leading_edge_shape'] = x*np.sqrt(1-x)*le_weight*((1-x)**(num_weights-1))

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        num_weights = self.options['num_weights']
        x = inputs['x_coordinates']
        le_weight = inputs['le_weight']

        partials['leading_edge_shape', 'x_coordinates'] = le_weight * ((1-x)**(num_weights-1.5)) * ((((2*num_weights)+1)*x)-2) / (-2)
        partials['leading_edge_shape', 'le_weight'] = x*np.sqrt(1-x)*((1-x)**(num_weights-1))


class GenerateSurface(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_points', types=int)

    def setup(self):
        num_points = self.options['num_points']

        self.add_input('x_coordinates', shape=(num_points,))
        self.add_input('class_function', shape=(num_points,))
        self.add_input('summed_bernstein_polynomials', shape=(num_points,))
        self.add_input('trailing_edge_wedge', shape=(num_points,))
        self.add_input('leading_edge_shape', shape=(num_points,))

        self.add_output('surface', shape=(2, num_points))

        range_num_points = np.arange(num_points)  # These get used so often right below, best to precompute them.
        ones_num_points = np.ones(num_points)

        self.declare_partials('surface', 'x_coordinates',
                              rows=range_num_points,
                              cols=range_num_points,
                              val=ones_num_points)
        self.declare_partials('surface', 'class_function',
                              rows=range_num_points+num_points,
                              cols=range_num_points)
        self.declare_partials('surface', 'summed_bernstein_polynomials',
                              rows = range_num_points + num_points,
                              cols = range_num_points,
                              val = ones_num_points)
        self.declare_partials('surface', 'trailing_edge_wedge',
                              rows=range_num_points + num_points,
                              cols=range_num_points,
                              val=ones_num_points)
        self.declare_partials('surface', 'leading_edge_shape',
                              rows=range_num_points + num_points,
                              cols=range_num_points,
                              val=ones_num_points)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        class_function = inputs['class_function']
        x_coordinates = inputs['x_coordinates']
        summed_bernstein_polynomials = inputs['summed_bernstein_polynomials']
        trailing_edge_wedge = inputs['trailing_edge_wedge']
        leading_edge_shape = inputs['leading_edge_shape']

        y_coordinates = class_function*summed_bernstein_polynomials + trailing_edge_wedge + leading_edge_shape
        surface = np.vstack((x_coordinates, y_coordinates))

        outputs['surface'] = surface

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        class_function = inputs['class_function']
        summed_bernstein_polynomials = inputs['summed_bernstein_polynomials']

        partials['surface', 'class_function'] = summed_bernstein_polynomials
        partials['surface', 'summed_bernstein_polynomials'] = class_function


class CSTSurface(om.Group):

    def initialize(self):
        self.options.declare('num_points', types=int)
        self.options.declare('num_weights', types=int)

    def setup(self):
        num_points = self.options['num_points']
        num_weights = self.options['num_weights']

        self.add_subsystem('GenerateXCoordinates', GenerateXCoordinates(num_points=num_points),
                           promotes_outputs=['x_coordinates'])

        self.add_subsystem('ClassFunction', ClassFunction(num_points=num_points),
                           promotes_inputs=['x_coordinates'],
                           promotes_outputs=['class_function'])

        self.add_subsystem('BernsteinPolynomials', BernsteinPolynomials(num_points=num_points, num_weights=num_weights),
                           promotes_inputs=['x_coordinates',
                                            'weights'],
                           promotes_outputs=['bernstein_polynomials'])

        self.add_subsystem('SummedBernsteinPolynomials', SummedBernsteinPolynomials(num_points=num_points, num_weights=num_weights),
                           promotes_inputs=['bernstein_polynomials'],
                           promotes_outputs=['summed_bernstein_polynomials'])

        self.add_subsystem('TrailingEdgeWedge', TrailingEdgeWedge(num_points=num_points),
                           promotes_inputs=['x_coordinates',
                                            'te_weight'],
                           promotes_outputs=['trailing_edge_wedge'])

        self.add_subsystem('LeadingEdgeShaping', LeadingEdgeShaping(num_points=num_points, num_weights=num_weights),
                           promotes_inputs=['x_coordinates',
                                            'le_weight'],
                           promotes_outputs=['leading_edge_shape'])

        self.add_subsystem('GenerateSurface', GenerateSurface(num_points=num_points),
                           promotes_inputs=['x_coordinates',
                                            'class_function',
                                            'summed_bernstein_polynomials',
                                            'trailing_edge_wedge',
                                            'leading_edge_shape'],
                           promotes_outputs=['surface'])


class CombineSurfaces(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_points', types=int)
        self.options.declare('num_upper_points', types=int)
        self.options.declare('num_lower_points', types=int)

    def setup(self):
        num_points = self.options['num_points']
        num_upper_points = self.options['num_upper_points']
        num_lower_points = self.options['num_lower_points']

        self.add_input('upper_surface', shape=(2, num_upper_points))
        self.add_input('lower_surface', shape=(2, num_lower_points))

        self.add_output('aerofoil_ccw_coordinates', shape=(2, (num_upper_points+num_lower_points)))

        self.declare_partials('aerofoil_ccw_coordinates', 'upper_surface',
                              rows=np.concatenate((np.arange(num_upper_points), np.arange(num_upper_points)+num_points)),
                              cols=np.concatenate((np.arange(num_upper_points-1, -1, -1), np.arange((2*num_upper_points)-1, num_upper_points-1, -1))),
                              val=np.ones(2*num_upper_points))
        self.declare_partials('aerofoil_ccw_coordinates', 'lower_surface',
                              rows=np.concatenate((np.arange(num_lower_points)+num_upper_points, np.arange(num_lower_points)+num_points+num_upper_points)),
                              cols=np.concatenate((np.arange(num_lower_points), np.arange(num_lower_points, 2*num_lower_points))),
                              val=np.ones(2*num_lower_points))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        upper_surface = inputs['upper_surface']
        lower_surface = inputs['lower_surface']

        outputs['aerofoil_ccw_coordinates'] = np.concatenate((np.flip(upper_surface, axis=1), lower_surface), axis=1)


class CSTAerofoil(om.Group):

    def initialize(self):
        self.options.declare('num_points', types=int)
        self.options.declare('num_upper_weights', types=int)
        self.options.declare('num_lower_weights', types=int)

    def setup(self):
        num_points = self.options['num_points']
        num_upper_weights = self.options['num_upper_weights']
        num_lower_weights = self.options['num_lower_weights']

        if num_points % 2 == 0:
            num_upper_points = num_points // 2
            num_lower_points = num_points // 2

        if num_points % 2 == 1:
            num_upper_points = num_points // 2 + 1
            num_lower_points = num_points // 2

        self.add_subsystem('UpperSurface', CSTSurface(num_points=num_upper_points, num_weights=num_upper_weights),
                           promotes_inputs=[('weights', 'upper_weights'),
                                            'le_weight',
                                            'te_weight'],
                           promotes_outputs=[('surface', 'upper_surface')])

        self.add_subsystem('LowerSurface', CSTSurface(num_points=num_lower_points, num_weights=num_lower_weights),
                           promotes_inputs=[('weights', 'lower_weights'),
                                            'le_weight',
                                            'te_weight'],
                           promotes_outputs=[('surface', 'lower_surface')])

        self.add_subsystem('CombineSurfaces', CombineSurfaces(num_points=num_points, num_upper_points=num_upper_points,
                                                              num_lower_points=num_lower_points),
                           promotes_inputs=['upper_surface',
                                            'lower_surface'],
                           promotes_outputs=['aerofoil_ccw_coordinates'])


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    num_points = 100
    num_upper_weights = 3
    num_lower_weights = 3
    p = om.Problem(model=om.Group())
    p.model.add_subsystem('CSTAerofoil',
                          CSTAerofoil(num_points=num_points,
                                      num_upper_weights=num_upper_weights,
                                      num_lower_weights=num_lower_weights),
                          promotes_inputs=['upper_weights',
                                           'lower_weights',
                                           'le_weight',
                                           'te_weight'],
                          promotes_outputs=['aerofoil_ccw_coordinates'])

    p.model.set_input_defaults('upper_weights', [0.206, 0.2728, 0.2292])
    p.model.set_input_defaults('lower_weights', [-0.1294, -0.0036, -0.0666])
    p.model.set_input_defaults('le_weight', 0)
    p.model.set_input_defaults('te_weight', 0)

    p.setup()

    p.run_model()

    coords = p.get_val('aerofoil_ccw_coordinates')
    x = coords[0, :]
    y = coords[1, :]
    plt.plot(x,y)

    with np.printoptions(linewidth=2024, edgeitems=20, formatter={'float_kind': '{:5.5f}'.format}):
          #formatter={'float_kind': '{:5.2f}'.format})
          p.check_partials(show_only_incorrect=True, compact_print=True, method='fd')