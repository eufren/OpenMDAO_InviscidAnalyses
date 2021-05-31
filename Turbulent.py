import openmdao.api as om
import numpy as np

# Closure relations mostly taken from https://www.rug.nl/research/portal/files/14407586/root.pdf


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


class CalculateGreenCf0(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']

        self.add_input('Re_theta', shape=(num_nodes,))

        self.add_output('Cf0', shape=(num_nodes,), lower=0.0001)

        self.declare_partials('Cf0', 'Re_theta',
                              rows=np.arange(num_nodes),
                              cols=np.arange(num_nodes))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        Re_theta = inputs['Re_theta']

        outputs['Cf0'] = (0.01013/(np.log10(Re_theta)-1.02))-0.00075

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        Re_theta = inputs['Re_theta']

        partials['Cf0', 'Re_theta'] = -0.01013/(np.log(10)*Re_theta*(np.log(Re_theta)/np.log(10) - 51/50)**2)


class CalculateGreenH0(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']

        self.add_input('Cf0', shape=(num_nodes,))

        self.add_output('H0', shape=(num_nodes,))

        self.declare_partials('H0', 'Cf0',
                              rows=np.arange(num_nodes),
                              cols=np.arange(num_nodes))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        Cf0 = inputs['Cf0']

        outputs['H0'] = (1/(1 - 6.55*np.sqrt(Cf0/2)))

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        Cf0 = inputs['Cf0']

        partials['H0', 'Cf0'] = 6.55/(2**1.5 * (1-6.55*np.sqrt(Cf0/2))**2 * np.sqrt(Cf0))


class CalculateGreenCfs(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']

        self.add_input('Cf0', shape=(num_nodes,))
        self.add_input('H0', shape=(num_nodes,))
        self.add_input('H', shape=(num_nodes,))

        self.add_output('Cfs', shape=(num_nodes,))

        self.declare_partials('Cfs', 'Cf0',
                              rows=np.arange(num_nodes),
                              cols=np.arange(num_nodes))
        self.declare_partials('Cfs', 'H0',
                              rows=np.arange(num_nodes),
                              cols=np.arange(num_nodes))
        self.declare_partials('Cfs', 'H',
                              rows=np.arange(num_nodes),
                              cols=np.arange(num_nodes))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        Cf0 = inputs['Cf0']
        H0 = inputs['H0']
        H = inputs['H']

        outputs['Cfs'] = Cf0*(0.9/((H/H0)-0.4) - 0.5)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        Cf0 = inputs['Cf0']
        H0 = inputs['H0']
        H = inputs['H']

        partials['Cfs', 'Cf0'] = 0.9/((H/H0)-0.4) - 0.5
        partials['Cfs', 'H0'] = Cf0*(0.9*H/(H0**2 * ((H/H0)-0.4)**2))
        partials['Cfs', 'H'] = Cf0*(-0.9/(H0* ((H/H0)-0.4)**2))


class CalculateHouwinkVeldmanH1(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']

        self.add_input('H', shape=(num_nodes,))

        self.add_output('H1', shape=(num_nodes,))

        self.declare_partials('H1', 'H',
                              rows=np.arange(num_nodes),
                              cols=np.arange(num_nodes))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        H = inputs['H']

        ht = np.where(H >= 2.732, 0.5 * (H - 2.732) + 2.732, 0)

        H1 = np.where(H < 2.732, ((0.5*H + 1)*H)/(H-1), 0) \
           + np.where(ht < 4, ((0.5*ht + 1)*ht)/(ht-1), 0) \
           + np.where(ht >= 4, 1.75 + (5.52273*ht)/(ht+5.818181), 0)

        outputs['H1'] = H1

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        H = inputs['H']

        ht = np.where(H >= 2.732, 0.5 * (H - 2.732) + 2.732, 0)
        # dht_dH = 0.5, shows up as factor

        # For some reason, complex step thinks the derivative for the first one is (0.5*(H**2) - H - 1)/((H - 1)**2) + 0.5, but i have no idea where the +0.5 comes from
        dH1_dH = np.where(H < 2.732, (0.5*(H**2) - H - 1)/((H - 1)**2), 0) \
               + np.where(ht < 4, 0.5 * (0.5*(ht**2) - ht - 1) / ((ht - 1)**2), 0) \
               + np.where(ht >= 4, 0.5 * 32.13224275413/(ht + 5.818181)**2, 0)

        partials['H1', 'H'] = dH1_dH


class CalculateEntrainmentCoefficient(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']

        self.add_input('H1', shape=(num_nodes,))

        self.add_output('CE', shape=(num_nodes,))

        self.declare_partials('CE', 'H1',
                              rows=np.arange(num_nodes),
                              cols=np.arange(num_nodes))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        H1 = inputs['H1']

        outputs['CE'] = 0.036*(H1-3)**-0.6169

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        H1 = inputs['H1']

        partials['CE', 'H1'] = -0.0222084/((H1-3)**1.6169)


class CalculateGradients(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']

        self.add_input('region_length')
        self.add_input('node_external_tangential_velocities', shape=(num_nodes,))
        self.add_input('H1', shape=(num_nodes,))
        self.add_input('momentum_thickness', shape=(num_nodes,))

        self.add_output('due_ds', shape=(num_nodes,))
        self.add_output('dtheta_ds', shape=(num_nodes,))
        self.add_output('dueH1theta_ds', shape=(num_nodes,))

        sparse_rows = np.concatenate(([0, 0], np.arange(num_nodes - 2) + 1, np.arange(num_nodes - 2) + 1, [num_nodes-1, num_nodes-1]))
        sparse_cols = np.concatenate(([0, 1], np.arange(num_nodes - 2), np.arange(num_nodes - 2) + 2, [num_nodes-2, num_nodes-1]))

        self.declare_partials('due_ds', 'node_external_tangential_velocities',
                              rows=sparse_rows,
                              cols=sparse_cols)
        self.declare_partials('due_ds', 'region_length')
        self.declare_partials('dtheta_ds', 'momentum_thickness',
                              rows=sparse_rows,
                              cols=sparse_cols)
        self.declare_partials('dtheta_ds', 'region_length')
        self.declare_partials('dueH1theta_ds', 'node_external_tangential_velocities',
                              rows=sparse_rows,
                              cols=sparse_cols)
        self.declare_partials('dueH1theta_ds', 'H1',
                              rows=sparse_rows,
                              cols=sparse_cols)
        self.declare_partials('dueH1theta_ds', 'momentum_thickness',
                              rows=sparse_rows,
                              cols=sparse_cols)
        self.declare_partials('dueH1theta_ds', 'region_length')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        num_nodes = self.options['num_nodes']
        l = inputs['region_length']
        ue = inputs['node_external_tangential_velocities']
        H1 = inputs['H1']
        theta = inputs['momentum_thickness']

        h = (l/(num_nodes-1))[0]  # Have to take this index because openMDAO passes it in as an array which upsets numpy later
        ueH1theta = ue*H1*theta

        outputs['due_ds'] = np.gradient(ue, h)
        outputs['dtheta_ds'] = np.gradient(theta, h)
        outputs['dueH1theta_ds'] = np.gradient(ueH1theta, h)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        num_nodes = self.options['num_nodes']
        l = inputs['region_length']
        ue = inputs['node_external_tangential_velocities']
        H1 = inputs['H1']
        theta = inputs['momentum_thickness']

        h = (l/(num_nodes-1))[0]
        ueH1theta = ue*H1*theta

        due_ds = np.gradient(ue, h)
        dtheta_ds = np.gradient(theta, h)
        dueH1theta_ds = np.gradient(ueH1theta, h)


        edge_val_1 = -((num_nodes-1)/(l))
        edge_val_2 = -edge_val_1
        internal_val_1 = -((num_nodes-1)/(2*l))
        internal_val_2 = -internal_val_1

        partials['due_ds', 'node_external_tangential_velocities'] = np.concatenate((edge_val_1, edge_val_2,
                                                                                    np.repeat(internal_val_1, num_nodes-2),
                                                                                    np.repeat(internal_val_2, num_nodes-2),
                                                                                    edge_val_1, edge_val_2))
        partials['due_ds', 'region_length'] = -due_ds/l
        partials['dtheta_ds', 'momentum_thickness'] = np.concatenate((edge_val_1, edge_val_2,
                                                                      np.repeat(internal_val_1, num_nodes-2),
                                                                      np.repeat(internal_val_2, num_nodes-2),
                                                                      edge_val_1, edge_val_2))
        partials['dtheta_ds', 'region_length'] = -dtheta_ds/l
        partials['dueH1theta_ds', 'node_external_tangential_velocities'] = np.concatenate((edge_val_1*H1[0]*theta[0], edge_val_2*H1[1]*theta[1],
                                                                                           internal_val_1*H1[:-2]*theta[:-2],
                                                                                           internal_val_2*H1[2:]*theta[2:],
                                                                                           edge_val_1*H1[-2]*theta[-2], edge_val_2*H1[-1]*theta[-1]))
        partials['dueH1theta_ds', 'H1'] = np.concatenate((edge_val_1*ue[0]*theta[0], edge_val_2*ue[1]*theta[1],
                                                          internal_val_1*ue[:-2]*theta[:-2],
                                                          internal_val_2*ue[2:]*theta[2:],
                                                          edge_val_1*ue[-2]*theta[-2], edge_val_2*ue[-1]*theta[-1]))
        partials['dueH1theta_ds', 'momentum_thickness'] = np.concatenate((edge_val_1*H1[0]*ue[0], edge_val_2*H1[1]*ue[1],
                                                                          internal_val_1*ue[:-2]*H1[:-2],
                                                                          internal_val_2*ue[2:]*H1[2:],
                                                                          edge_val_1*ue[-2]*H1[-2], edge_val_2*ue[-1]*H1[-1]))
        partials['dueH1theta_ds', 'region_length'] = -dueH1theta_ds/l


class TurbulentBoundaryLayer(om.ImplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']

        self.add_input('dtheta_ds', shape=(num_nodes,))
        self.add_input('dueH1theta_ds', shape=(num_nodes,))
        self.add_input('due_ds', shape=(num_nodes,))
        self.add_input('node_external_tangential_velocities', shape=(num_nodes,))
        self.add_input('Cfs', shape=(num_nodes,))
        self.add_input('CE', shape=(num_nodes,))
        self.add_input('initial_H')
        self.add_input('initial_momentum_thickness')

        self.add_output('H', 1.5, shape=(num_nodes,), lower=1.0001)
        self.add_output('momentum_thickness', 0.001, shape=(num_nodes,), lower=0.0001)

    def setup_partials(self):
        num_nodes = self.options['num_nodes']

        self.declare_partials('H', 'node_external_tangential_velocities',
                              rows=np.arange(num_nodes),
                              cols=np.arange(num_nodes))
        self.declare_partials('H', 'dueH1theta_ds',
                              rows=np.arange(num_nodes),
                              cols=np.arange(num_nodes))
        self.declare_partials('H', 'CE',
                              rows=np.arange(num_nodes),
                              cols=np.arange(num_nodes))
        self.declare_partials('H', 'initial_H',
                              rows=[0],
                              cols=[0],
                              val=-1)
        self.declare_partials('H', 'H',
                              rows=[0],
                              cols=[0],
                              val=1)

        self.declare_partials('momentum_thickness', 'node_external_tangential_velocities',
                              rows=np.arange(num_nodes),
                              cols=np.arange(num_nodes))
        self.declare_partials('momentum_thickness', 'due_ds',
                              rows=np.arange(num_nodes),
                              cols=np.arange(num_nodes))
        self.declare_partials('momentum_thickness', 'dtheta_ds',
                              rows=np.arange(num_nodes),
                              cols=np.arange(num_nodes))
        self.declare_partials('momentum_thickness', 'Cfs',
                              rows=np.arange(num_nodes),
                              cols=np.arange(num_nodes))
        self.declare_partials('momentum_thickness', 'H',
                              rows=np.arange(num_nodes),
                              cols=np.arange(num_nodes))
        self.declare_partials('momentum_thickness', 'initial_momentum_thickness',
                              rows=[0],
                              cols=[0],
                              val=-1)
        self.declare_partials('momentum_thickness', 'momentum_thickness',
                              rows=np.arange(num_nodes),
                              cols=np.arange(num_nodes))

    def apply_nonlinear(self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None):
        dueH1theta_ds = inputs['dueH1theta_ds']
        due_ds = inputs['due_ds']
        dtheta_ds = inputs['dtheta_ds']
        ue = inputs['node_external_tangential_velocities']
        Cf = inputs['Cfs']
        CE = inputs['CE']
        initial_H = inputs['initial_H']
        initial_theta = inputs['initial_momentum_thickness']
        theta = outputs['momentum_thickness']
        H = outputs['H']

        residuals['momentum_thickness'][0] = theta[0] - initial_theta
        residuals['momentum_thickness'][1:] = (dtheta_ds + (theta/ue)*due_ds*(2+H) - 0.5*Cf)[1:]
        residuals['H'][0] = H[0] - initial_H
        residuals['H'][1:] = (dueH1theta_ds/ue - CE)[1:]

    # solve_nonlinear(inputs, outputs) is where the marching calculations would go

    def linearize(self, inputs, outputs, jacobian, discrete_inputs=None, discrete_outputs=None):
        num_nodes = self.options['num_nodes']

        dueH1theta_ds = inputs['dueH1theta_ds']
        due_ds = inputs['due_ds']
        ue = inputs['node_external_tangential_velocities']
        theta = outputs['momentum_thickness']
        H = outputs['H']

        jacobian['H', 'node_external_tangential_velocities'][0] = 0
        jacobian['H', 'node_external_tangential_velocities'][1:] = (-dueH1theta_ds/(ue**2))[1:]
        jacobian['H', 'dueH1theta_ds'][0] = 0
        jacobian['H', 'dueH1theta_ds'][1:] = 1/ue[1:]
        jacobian['H', 'CE'][0] = 0
        jacobian['H', 'CE'][1:] = -np.ones(num_nodes-1)

        jacobian['momentum_thickness', 'node_external_tangential_velocities'][0] = 0
        jacobian['momentum_thickness', 'node_external_tangential_velocities'][1:] = -((theta / (ue**2)) * due_ds * (2 + H))[1:]
        jacobian['momentum_thickness', 'due_ds'][0] = 0
        jacobian['momentum_thickness', 'due_ds'][1:] = ((theta/ue)*(2+H))[1:]
        jacobian['momentum_thickness', 'dtheta_ds'][0] = 0
        jacobian['momentum_thickness', 'dtheta_ds'][1:] = np.ones(num_nodes-1)
        jacobian['momentum_thickness', 'Cfs'][0] = 0
        jacobian['momentum_thickness', 'Cfs'][1:] = -0.5*np.ones(num_nodes-1)
        jacobian['momentum_thickness', 'H'][0] = 0
        jacobian['momentum_thickness', 'H'][1:] = ((theta/ue)*due_ds)[1:]
        jacobian['momentum_thickness', 'momentum_thickness'][0] = 1
        jacobian['momentum_thickness', 'momentum_thickness'][1:] = ((1/ue)*due_ds*(2+H))[1:]


class CalculateDisplacementThickness(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']

        self.add_input('momentum_thickness', shape=(num_nodes,))
        self.add_input('H', shape=(num_nodes,))

        self.add_output('displacement_thickness', shape=(num_nodes,))

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


class TurbulentGroup(om.Group):

    def initialize(self):
        self.options.declare('num_nodes')

    def setup(self):
        num_nodes = self.options['num_nodes']

        self.add_subsystem('CalculateNodePositions', CalculateNodePositions(num_nodes=num_nodes),
                           promotes_inputs=['region_length'],
                           promotes_outputs=['node_positions'])

        self.add_subsystem('CalculateMomentumThicknessReynolds', CalculateMomentumThicknessReynolds(num_nodes=num_nodes),
                           promotes_inputs=['momentum_thickness',
                                            'kinematic_viscosity',
                                            'node_external_tangential_velocities'],
                           promotes_outputs=['Re_theta'])

        self.add_subsystem('CalculateGreenCf0', CalculateGreenCf0(num_nodes=num_nodes),
                           promotes_inputs=['Re_theta'],
                           promotes_outputs=['Cf0'])

        self.add_subsystem('CalculateGreenH0', CalculateGreenH0(num_nodes=num_nodes),
                           promotes_inputs=['Cf0'],
                           promotes_outputs=['H0'])

        self.add_subsystem('CalculateGreenCfs', CalculateGreenCfs(num_nodes=num_nodes),
                           promotes_inputs=['Cf0',
                                            'H0',
                                            'H'],
                           promotes_outputs=['Cfs'])

        self.add_subsystem('CalculateHouwinkVeldmanH1', CalculateHouwinkVeldmanH1(num_nodes=num_nodes),
                           promotes_inputs=['H'],
                           promotes_outputs=['H1'])

        self.add_subsystem('CalculateEntrainmentCoefficient', CalculateEntrainmentCoefficient(num_nodes=num_nodes),
                           promotes_inputs=['H1'],
                           promotes_outputs=['CE'])

        self.add_subsystem('CalculateGradients', CalculateGradients(num_nodes=num_nodes),
                           promotes_inputs=['region_length',
                                            'node_external_tangential_velocities',
                                            'momentum_thickness',
                                            'H1'],
                           promotes_outputs=['due_ds',
                                             'dtheta_ds',
                                             'dueH1theta_ds'])

        self.add_subsystem('TurbulentBoundaryLayer', TurbulentBoundaryLayer(num_nodes=num_nodes),
                           promotes_inputs=['node_external_tangential_velocities',
                                            'due_ds',
                                            'dtheta_ds',
                                            'dueH1theta_ds',
                                            'Cfs',
                                            'CE',
                                            'initial_momentum_thickness',
                                            'initial_H'],
                           promotes_outputs=['momentum_thickness',
                                             'H'])

        self.add_subsystem('CalculateDisplacementThickness', CalculateDisplacementThickness(num_nodes=num_nodes),
                           promotes_inputs=['momentum_thickness',
                                            'H'],
                           promotes_outputs=['displacement_thickness'])

        self.linear_solver = om.DirectSolver()
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, maxiter=30, iprint=2)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    numero_nodeso = 50

    p = om.Problem(model=om.Group())

    p.model.add_subsystem('TurbulentGroup',
                          TurbulentGroup(num_nodes=numero_nodeso),
                          promotes_inputs=[('region_length', 'upper_length'),
                                           'kinematic_viscosity',
                                           ('node_external_tangential_velocities',
                                            'upper_external_tangential_velocities'),
                                           'initial_momentum_thickness',
                                           'initial_H'],
                          promotes_outputs=[('momentum_thickness', 'upper_momentum_thickness'),
                                            ('displacement_thickness', 'upper_displacement_thickness'),
                                            ('H', 'upper_H'),
                                            ('Cfs', 'upper_Cf'),
                                            ('node_positions', 'upper_node_positions')])

    total_length = 2
    p.model.set_input_defaults('kinematic_viscosity', 1.81e-5)
    p.model.set_input_defaults('upper_length', 1)
    p.model.set_input_defaults('upper_external_tangential_velocities', np.repeat(1, numero_nodeso))
    p.model.set_input_defaults('initial_H', 1.01)
    p.model.set_input_defaults('initial_momentum_thickness', 0.0008)

    p.setup()

    # with np.printoptions(linewidth=1024, edgeitems=20, formatter={'float_kind': '{:5.5f}'.format}):
    #      # formatter={'float_kind': '{:5.2f}'.format})
    #      p.check_partials(show_only_incorrect=True, compact_print=False, method='fd')

    p.run_model()

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

