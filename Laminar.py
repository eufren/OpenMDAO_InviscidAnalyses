import openmdao.api as om
import numpy as np
from Utilities import CumulativeIntegrateTrapeziums

# Closure relations and theory from Low Speed Aerodynamics
# TODO: ADD PARTIALS


class CalculateNodePositions(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']

        self.add_input('region_length')

        self.add_output('node_positions', shape=(num_nodes,))

        self.declare_partials('node_positions', 'region_length')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        num_nodes = self.options['num_nodes']

        l = inputs['region_length']

        outputs['node_positions'] = np.linspace(0, l, num_nodes)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        num_nodes = self.options['num_nodes']

        partials['node_positions', 'region_length'] = np.arange(num_nodes)/(num_nodes-1)


class CalculateMomentumThicknessReynolds(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']

        self.add_input('momentum_thickness', shape=(num_nodes, ))
        self.add_input('kinematic_viscosity')
        self.add_input('node_external_tangential_velocities', shape=(num_nodes, ))

        self.add_output('Re_theta', shape=(num_nodes,))

        self.declare_partials('Re_theta', 'momentum_thickness',
                              rows=np.arange(num_nodes),
                              cols=np.arange(num_nodes))
        self.declare_partials('Re_theta', 'kinematic_viscosity')
        self.declare_partials('Re_theta', 'node_external_tangential_velocities',
                              rows=np.arange(num_nodes),
                              cols=np.arange(num_nodes))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        theta = inputs['momentum_thickness']
        nu = inputs['kinematic_viscosity']
        ue = inputs['node_external_tangential_velocities']

        outputs['Re_theta'] = ue*theta/nu

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        theta = inputs['momentum_thickness']
        nu = inputs['kinematic_viscosity']
        ue = inputs['node_external_tangential_velocities']

        partials['Re_theta', 'momentum_thickness'] = ue/nu
        partials['Re_theta', 'kinematic_viscosity'] = -ue*theta/(nu**2)
        partials['Re_theta', 'node_external_tangential_velocities'] = theta/nu


class CalculateMomentumThicknessIntegrand(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']

        self.add_input('node_external_tangential_velocities', shape=(num_nodes,))

        self.add_output('ue_pow_5', shape=(num_nodes,))

        self.declare_partials('ue_pow_5', 'node_external_tangential_velocities',
                              rows=np.arange(num_nodes),
                              cols=np.arange(num_nodes))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ue = inputs['node_external_tangential_velocities']

        outputs['ue_pow_5'] = ue**5

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        ue = inputs['node_external_tangential_velocities']

        partials['ue_pow_5', 'node_external_tangential_velocities'] = 5*ue**4


class CalculateMomentumThickness(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']

        self.add_input('thwaites_integral', shape=(num_nodes,))
        self.add_input('kinematic_viscosity')
        self.add_input('node_external_tangential_velocities', shape=(num_nodes,))

        self.add_output('momentum_thickness', shape=(num_nodes,))

        self.declare_partials('momentum_thickness', 'thwaites_integral',
                              rows=np.arange(num_nodes),
                              cols=np.arange(num_nodes))
        self.declare_partials('momentum_thickness', 'kinematic_viscosity')
        self.declare_partials('momentum_thickness', 'node_external_tangential_velocities',
                              rows=np.arange(num_nodes),
                              cols=np.arange(num_nodes))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        integral = inputs['thwaites_integral']
        nu = inputs['kinematic_viscosity']
        ue = inputs['node_external_tangential_velocities']

        theta2ue6 = 0.45*nu*integral

        theta = np.sqrt(theta2ue6/ue**6)
        theta[0] = 1e-6  # This helps with avoiding divide by zeroes later

        outputs['momentum_thickness'] = theta

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        integral = inputs['thwaites_integral']
        nu = inputs['kinematic_viscosity']
        ue = inputs['node_external_tangential_velocities']

        partials['momentum_thickness', 'thwaites_integral'][1:] = (np.sqrt(0.45*nu/(4*integral[1:]*ue[1:]**6)))
        partials['momentum_thickness', 'thwaites_integral'][0] = 0
        partials['momentum_thickness', 'kinematic_viscosity'] = np.sqrt(0.45*integral/(4*nu*ue**6))
        partials['momentum_thickness', 'kinematic_viscosity'][0] = 0
        partials['momentum_thickness', 'node_external_tangential_velocities'] = -3*np.sqrt(0.45*integral*nu/ue**8)
        partials['momentum_thickness', 'node_external_tangential_velocities'][0] = 0


class CalculateLambda(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']

        self.add_input('momentum_thickness', shape=(num_nodes,))
        self.add_input('kinematic_viscosity')
        self.add_input('due_ds', shape=(num_nodes,))

        self.add_output('lambda', shape=(num_nodes,))

        self.declare_partials('lambda', 'momentum_thickness',
                              rows=np.arange(num_nodes),
                              cols=np.arange(num_nodes))
        self.declare_partials('lambda', 'kinematic_viscosity')
        self.declare_partials('lambda', 'due_ds',
                              rows=np.arange(num_nodes),
                              cols=np.arange(num_nodes))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        theta = inputs['momentum_thickness']
        nu = inputs['kinematic_viscosity']
        due_ds = inputs['due_ds']

        outputs['lambda'] = theta**2 * (1/nu) * due_ds

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        theta = inputs['momentum_thickness']
        nu = inputs['kinematic_viscosity']
        due_ds = inputs['due_ds']

        partials['lambda', 'momentum_thickness'] = 2*due_ds*theta/nu
        partials['lambda', 'kinematic_viscosity'] = -(theta**2)*due_ds/nu**2
        partials['lambda', 'due_ds'] = (theta**2)/nu


class ComputeLandH(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']

        self.add_input('lambda', shape=(num_nodes,))

        self.add_output('l', shape=(num_nodes,))
        self.add_output('H', shape=(num_nodes,))

        self.declare_partials('l', 'lambda',
                              rows=np.arange(num_nodes),
                              cols=np.arange(num_nodes))
        self.declare_partials('H', 'lambda',
                              rows=np.arange(num_nodes),
                              cols=np.arange(num_nodes))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        lamb = inputs['lambda']

        #ones = np.ones_like(lamb)
        zeros = np.zeros_like(lamb)

        # l_and_H = np.where(lamb <= -0.1, [(0.22 + 1.402*(-0.1) + (0.018*-0.1)/(-0.1+0.107))*ones, (2.088 + 0.0731/(-0.1+0.14))*ones], [zeros, zeros])\
        #         + np.where((-0.1<lamb)&(lamb<=0), [0.22 + 1.402*lamb + (0.018*lamb)/(lamb+0.107), 2.088 + 0.0731/(lamb+0.14)], [zeros, zeros])\
        #         + np.where((0<lamb)&(lamb<=0.1), [0.22 + 1.57*lamb - 1.8*(lamb**2), 2.61 - 3.75*lamb + 5.24*(lamb**2)], [zeros, zeros]) \
        #         + np.where(lamb>=0.1, [(0.22 + 1.57*0.1 - 1.8*0.01)*ones, (2.61 - 3.75*0.1 + 5.24*0.01)*ones], [zeros, zeros]) \

        l_and_H = np.where(lamb<=0, [0.22 + 1.402*lamb + (0.018*lamb)/(lamb+0.107), 2.088 + 0.0731/(lamb+0.14)], [zeros, zeros])\
                + np.where(0<lamb, [0.22 + 1.57*lamb - 1.8*(lamb**2), 2.61 - 3.75*lamb + 5.24*(lamb**2)], [zeros, zeros])

        outputs['l'] = l_and_H[0, :]
        outputs['H'] = l_and_H[1, :]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        lamb = inputs['lambda']

        #ones = np.ones_like(lamb)
        zeros = np.zeros_like(lamb)

        dl_and_dH = np.where(lamb<=0, [(0.018/(lamb+0.107))-(0.018*lamb/(lamb+0.107)**2)+1.402, -0.0731/(lamb+0.14)**2], [zeros, zeros])\
                  + np.where(0<lamb, [1.57 - 3.6*lamb, 10.48*lamb - 3.75], [zeros, zeros])

        partials['l', 'lambda'] = dl_and_dH[0, :]
        partials['H', 'lambda'] = dl_and_dH[1, :]


class CalculateSkinFrictionCoefficient(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']

        self.add_input('l', shape=(num_nodes,))
        self.add_input('Re_theta', shape=(num_nodes,))

        self.add_output('Cf', shape=(num_nodes,))

        self.declare_partials('Cf', 'l',
                              rows=np.arange(num_nodes),
                              cols=np.arange(num_nodes))
        self.declare_partials('Cf', 'Re_theta',
                              rows=np.arange(num_nodes),
                              cols=np.arange(num_nodes))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        l = inputs['l']
        Re_theta = inputs['Re_theta']

        outputs['Cf'] = 2*l/Re_theta

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        l = inputs['l']
        Re_theta = inputs['Re_theta']

        partials['Cf', 'l'] = 2/Re_theta
        partials['Cf', 'Re_theta'] = -2*l/Re_theta**2


class CalculateDisplacementThickness(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']

        self.add_input('momentum_thickness', shape=(num_nodes,))
        self.add_input('H', shape=(num_nodes,))

        self.add_output('displacement_thickness', 0.001, shape=(num_nodes,))

        self.declare_partials('displacement_thickness', 'momentum_thickness',
                              rows=np.arange(num_nodes),
                              cols=np.arange(num_nodes))
        self.declare_partials('displacement_thickness', 'H',
                              rows=np.arange(num_nodes),
                              cols=np.arange(num_nodes))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        theta = inputs['momentum_thickness']
        H = inputs['H']

        outputs['displacement_thickness'] = H*theta

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        theta = inputs['momentum_thickness']
        H = inputs['H']

        partials['displacement_thickness', 'momentum_thickness'] = H
        partials['displacement_thickness', 'H'] = theta


class CalculateGradients(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']

        self.add_input('region_length')
        self.add_input('node_external_tangential_velocities', shape=(num_nodes,))

        self.add_output('due_ds', shape=(num_nodes,))

        sparse_rows = np.concatenate(([0, 0], np.arange(num_nodes - 2) + 1, np.arange(num_nodes - 2) + 1, [num_nodes-1, num_nodes-1]))
        sparse_cols = np.concatenate(([0, 1], np.arange(num_nodes - 2), np.arange(num_nodes - 2) + 2, [num_nodes-2, num_nodes-1]))

        self.declare_partials('due_ds', 'node_external_tangential_velocities',
                              rows=sparse_rows,
                              cols=sparse_cols)
        self.declare_partials('due_ds', 'region_length')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        num_nodes = self.options['num_nodes']
        l = inputs['region_length']
        ue = inputs['node_external_tangential_velocities']

        h = (l/(num_nodes-1))[0]  # Have to take this index because openMDAO passes it in as an array which upsets numpy later

        outputs['due_ds'] = np.gradient(ue, h)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        num_nodes = self.options['num_nodes']
        l = inputs['region_length']
        ue = inputs['node_external_tangential_velocities']

        h = (l/(num_nodes-1))[0]

        due_ds = np.gradient(ue, h)

        edge_val_1 = -((num_nodes-1)/(l))
        edge_val_2 = -edge_val_1
        internal_val_1 = -((num_nodes-1)/(2*l))
        internal_val_2 = -internal_val_1
        monovariable_gradient_partials = np.concatenate((edge_val_1, edge_val_2,
                                                         np.repeat(internal_val_1, num_nodes-2),
                                                         np.repeat(internal_val_2, num_nodes-2),
                                                         edge_val_1, edge_val_2))

        partials['due_ds', 'node_external_tangential_velocities'] = monovariable_gradient_partials
        partials['due_ds', 'region_length'] = -due_ds/l


class LaminarGroup(om.Group):

    def initialize(self):
        self.options.declare('num_nodes')

    def setup(self):
        num_nodes = self.options['num_nodes']

        self.add_subsystem('CalculateNodePositions', CalculateNodePositions(num_nodes=num_nodes),
                           promotes_inputs=['region_length'],
                           promotes_outputs=['node_positions'])

        self.add_subsystem('CalculateGradients', CalculateGradients(num_nodes=num_nodes),
                           promotes_inputs=['region_length',
                                            'node_external_tangential_velocities'],
                           promotes_outputs=['due_ds'])

        self.add_subsystem('CalculateMomentumThicknessIntegrand', CalculateMomentumThicknessIntegrand(num_nodes=num_nodes),
                           promotes_inputs=['node_external_tangential_velocities'],
                           promotes_outputs=['ue_pow_5'])

        self.add_subsystem('IntegrateThwaites', CumulativeIntegrateTrapeziums(n=num_nodes),
                           promotes_inputs=[('x', 'node_positions'),
                                            ('y', 'ue_pow_5')],
                           promotes_outputs=[('cumulative_integral', 'thwaites_integral')])

        self.add_subsystem('CalculateMomentumThickness', CalculateMomentumThickness(num_nodes=num_nodes),
                           promotes_inputs=['thwaites_integral',
                                            'kinematic_viscosity',
                                            'node_external_tangential_velocities'],
                           promotes_outputs=['momentum_thickness'])

        self.add_subsystem('CalculateLambda', CalculateLambda(num_nodes=num_nodes),
                           promotes_inputs=['momentum_thickness',
                                            'kinematic_viscosity',
                                            'due_ds'],
                           promotes_outputs=['lambda'])

        self.add_subsystem('ComputeLandH', ComputeLandH(num_nodes=num_nodes),
                           promotes_inputs=['lambda'],
                           promotes_outputs=['l',
                                             'H'])

        self.add_subsystem('CalculateMomentumThicknessReynolds', CalculateMomentumThicknessReynolds(num_nodes=num_nodes),
                           promotes_inputs=['momentum_thickness',
                                            'kinematic_viscosity',
                                            'node_external_tangential_velocities'],
                           promotes_outputs=['Re_theta'])

        self.add_subsystem('CalculateSkinFrictionCoefficient', CalculateSkinFrictionCoefficient(num_nodes=num_nodes),
                           promotes_inputs=['l',
                                            'Re_theta'],
                           promotes_outputs=['Cf'])

        self.add_subsystem('CalculateDisplacementThickness', CalculateDisplacementThickness(num_nodes=num_nodes),
                           promotes_inputs=['momentum_thickness',
                                            'H'],
                           promotes_outputs=['displacement_thickness'])

        self.linear_solver = om.LinearRunOnce()
        self.nonlinear_solver = om.NonlinearRunOnce()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    numero_nodeso = 10

    p = om.Problem(model=om.Group())

    p.model.add_subsystem('LaminarGroup',
                          LaminarGroup(num_nodes=numero_nodeso),
                          promotes_inputs=[('region_length', 'upper_length'),
                                           'kinematic_viscosity',
                                           ('node_external_tangential_velocities', 'upper_external_tangential_velocities')],
                          promotes_outputs=[('momentum_thickness', 'upper_momentum_thickness'),
                                            ('displacement_thickness', 'upper_displacement_thickness'),
                                            ('H', 'upper_H'),
                                            ('Cf', 'upper_Cf'),
                                            ('node_positions', 'upper_node_positions')])

    total_length = 2
    p.model.set_input_defaults('kinematic_viscosity', 1.81e-5)
    p.model.set_input_defaults('upper_length', 0.4)
    p.model.set_input_defaults('upper_external_tangential_velocities', np.linspace(1, 10, 2*numero_nodeso)[:numero_nodeso])

    p.setup()
    p.run_model()

    with np.printoptions(linewidth=1024, edgeitems=20, formatter={'float_kind': '{:5.5f}'.format}):
         # formatter={'float_kind': '{:5.2f}'.format})
         p.check_partials(show_only_incorrect=True, compact_print=False, method='fd')

    node_positions = p.get_val('upper_node_positions')
    Cf = p.get_val('upper_Cf')
    d1 = p.get_val('upper_displacement_thickness')
    d2 = p.get_val('upper_momentum_thickness')

    fig, axs = plt.subplots(3, 1)
    axs[0].plot(node_positions, d1, label='Displacement Thickness')
    axs[0].set_title('Displacement Thickness')
    axs[0].legend()
    axs[1].plot(node_positions, d2, label='Momentum Thickness')
    axs[1].set_title('Momentum Thickness')
    axs[1].legend()
    axs[2].plot(node_positions, Cf, label='Cf')
    axs[2].set_title('Skin Friction Coefficient')
    #axs[2].set_ylim(-0.0025, max(Cf))*2)
    axs[2].legend()
    plt.show()
