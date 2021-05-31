import openmdao.api as om
import numpy as np

# Based on https://sci-hub.se/https://arc.aiaa.org/doi/10.2514/3.9789


class CalculateExplicitClosures(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']

        self.add_input('H', 1.1, shape=(num_nodes,))

        self.add_output('dntilde_dRe_theta', shape=(num_nodes,))
        self.add_output('l', shape=(num_nodes,))
        self.add_output('m', shape=(num_nodes,))

        self.declare_partials('dntilde_dRe_theta', 'H',
                              rows=np.arange(num_nodes),
                              cols=np.arange(num_nodes))
        self.declare_partials('l', 'H',
                              rows=np.arange(num_nodes),
                              cols=np.arange(num_nodes))
        self.declare_partials('m', 'H',
                              rows=np.arange(num_nodes),
                              cols=np.arange(num_nodes))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        H = inputs['H']

        outputs['dntilde_dRe_theta'] = 0.01*np.sqrt((2.4*H - 3.7 + 2.5*np.tanh(1.5*H - 4.65))**2 + 0.25)
        outputs['l'] = l = (6.54*H - 14.07) / H**2
        outputs['m'] = (0.058*(H-4)**2/(H-1) - 0.068)/l

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        H = inputs['H']

        partials['dntilde_dRe_theta', 'H'] = (0.01*(3.75*np.cosh(1.5*H-4.65)**-2+2.4)*(2.5*np.tanh(1.5*H-4.65)+2.4*H-3.7))/np.sqrt((2.5*np.tanh(1.5*H-4.65)+2.4*H-3.7)**2+0.25)
        partials['l', 'H'] = -(6.54*H-28.14)/(H**3)
        partials['m', 'H'] = (H*(0.75864*H**4 -7.06542*H**3 +25.19328*H**2 -42.98328*H +28.02744))/((H-1)**2*(6.54*H-14.07)**2)


class Calculate_dntilde_ds(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']

        self.add_input('transition_distance')
        self.add_input('log_max_amplification_ratio', shape=(num_nodes,))

        self.add_output('dntilde_ds', shape=(num_nodes,))

        self.declare_partials('dntilde_ds', 'log_max_amplification_ratio',
                              rows=np.concatenate(([0, 0], np.arange(num_nodes - 2) + 1, np.arange(num_nodes - 2) + 1, [num_nodes-1, num_nodes-1])),
                              cols=np.concatenate(([0, 1], np.arange(num_nodes - 2), np.arange(num_nodes - 2) + 2, [num_nodes-2, num_nodes-1])))
        self.declare_partials('dntilde_ds', 'transition_distance')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        num_nodes = self.options['num_nodes']
        L = inputs['transition_distance']
        ntilde = inputs['log_max_amplification_ratio']

        h = (L/(num_nodes-1))[0]  # Have to take this index because openMDAO passes it in as an array which upsets numpy later

        outputs['dntilde_ds'] = np.gradient(ntilde, h)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        num_nodes = self.options['num_nodes']
        L = inputs['transition_distance']
        ntilde = inputs['log_max_amplification_ratio']

        h = (L/(num_nodes-1))[0]

        dntilde_ds = np.gradient(ntilde, h)

        edge_val_1 = -((num_nodes-1)/(L))
        edge_val_2 = -edge_val_1
        internal_val_1 = -((num_nodes-1)/(2*L))
        internal_val_2 = -internal_val_1

        partials['dntilde_ds', 'log_max_amplification_ratio'] = np.concatenate((edge_val_1, edge_val_2,
                                                                                    np.repeat(internal_val_1, num_nodes-2),
                                                                                    np.repeat(internal_val_2, num_nodes-2),
                                                                                    edge_val_1, edge_val_2))
        partials['dntilde_ds', 'transition_distance'] = -dntilde_ds/L


class ImplicitTransition(om.ImplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('transition_log_max_amplification_ratio', types=float)

    def setup(self):
        num_nodes = self.options['num_nodes']

        self.add_input('dntilde_dRe_theta', shape=(num_nodes,))
        self.add_input('l', shape=(num_nodes,))
        self.add_input('m', shape=(num_nodes,))
        self.add_input('dntilde_ds', shape=(num_nodes,))
        self.add_input('momentum_thickness', shape=(num_nodes,))

        self.add_output('log_max_amplification_ratio', np.linspace(0, 8.9, num_nodes), shape=(num_nodes,))
        self.add_output('transition_distance')

    def setup_partials(self):
        num_nodes = self.options['num_nodes']

        self.declare_partials('log_max_amplification_ratio', 'dntilde_dRe_theta',
                              rows=np.arange(num_nodes),
                              cols=np.arange(num_nodes))
        self.declare_partials('log_max_amplification_ratio', 'l',
                              rows=np.arange(num_nodes),
                              cols=np.arange(num_nodes))
        self.declare_partials('log_max_amplification_ratio', 'm',
                              rows=np.arange(num_nodes),
                              cols=np.arange(num_nodes))
        self.declare_partials('log_max_amplification_ratio', 'dntilde_ds',
                              rows=np.arange(num_nodes),
                              cols=np.arange(num_nodes))
        self.declare_partials('log_max_amplification_ratio', 'momentum_thickness',
                              rows=np.arange(num_nodes),
                              cols=np.arange(num_nodes))
        self.declare_partials('log_max_amplification_ratio', 'log_max_amplification_ratio',
                              rows=[0],
                              cols=[0],
                              val=1)

        self.declare_partials('transition_distance', 'log_max_amplification_ratio',
                              rows=[0],
                              cols=[num_nodes-1],
                              val=1)

    def apply_nonlinear(self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None):
        ntilde_crit = self.options['transition_log_max_amplification_ratio']
        l = inputs['l']
        m = inputs['m']
        dntilde_dRe_theta = inputs['dntilde_dRe_theta']
        dntilde_ds = inputs['dntilde_ds']
        theta = inputs['momentum_thickness']
        #L = outputs['transition_distance']
        ntilde = outputs['log_max_amplification_ratio']

        residuals['log_max_amplification_ratio'][0] = ntilde[0]  # Initial condition, ntilde(0) = 0
        residuals['log_max_amplification_ratio'][1:] = (dntilde_dRe_theta*0.5*(m+1)*l/theta - dntilde_ds)[1:]
        residuals['transition_distance'] = ntilde[-1] - ntilde_crit

    def linearize(self, inputs, outputs, jacobian, discrete_inputs=None, discrete_outputs=None):
        num_nodes = self.options['num_nodes']

        l = inputs['l']
        m = inputs['m']
        dntilde_dRe_theta = inputs['dntilde_dRe_theta']
        theta = inputs['momentum_thickness']

        jacobian['log_max_amplification_ratio', 'dntilde_dRe_theta'][0] = 0
        jacobian['log_max_amplification_ratio', 'dntilde_dRe_theta'][1:] = (0.5*(m+1)*l/theta)[1:]
        jacobian['log_max_amplification_ratio', 'l'][0] = 0
        jacobian['log_max_amplification_ratio', 'l'][1:] = (dntilde_dRe_theta*0.5*(m+1)*1/theta)[1:]
        jacobian['log_max_amplification_ratio', 'm'][0] = 0
        jacobian['log_max_amplification_ratio', 'm'][1:] = (dntilde_dRe_theta*0.5*l/theta)[1:]
        jacobian['log_max_amplification_ratio', 'momentum_thickness'][0] = 0
        jacobian['log_max_amplification_ratio', 'momentum_thickness'][1:] = (-dntilde_dRe_theta*0.5*(m+1)*l/(theta**2))[1:]
        jacobian['log_max_amplification_ratio', 'dntilde_ds'][0] = 0
        jacobian['log_max_amplification_ratio', 'dntilde_ds'][1:] = -np.ones(num_nodes-1)


class TransitionGroup(om.Group):

    def initialize(self):
        self.options.declare('num_nodes')
        self.options.declare('transition_log_max_amplification_ratio', types=float)

    def setup(self):
        num_nodes = self.options['num_nodes']
        transition_log_max_amplification_ratio = self.options['transition_log_max_amplification_ratio']

        self.add_subsystem('CalculateExplicitClosures', CalculateExplicitClosures(num_nodes=num_nodes),
                           promotes_inputs=['H'],
                           promotes_outputs=['dntilde_dRe_theta',
                                             'l',
                                             'm'])

        self.add_subsystem('Calculate_dntilde_ds', Calculate_dntilde_ds(num_nodes=num_nodes),
                           promotes_inputs=['log_max_amplification_ratio',
                                            'transition_distance'],
                           promotes_outputs=['dntilde_ds'])

        self.add_subsystem('ImplicitTransition', ImplicitTransition(num_nodes=num_nodes, transition_log_max_amplification_ratio=transition_log_max_amplification_ratio),
                           promotes_inputs=['dntilde_dRe_theta',
                                            'l',
                                            'm',
                                            'dntilde_ds',
                                            'momentum_thickness'],
                           promotes_outputs=['transition_distance',
                                             'log_max_amplification_ratio'])

        self.linear_solver = om.DirectSolver()
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, maxiter=30, iprint=2)


if __name__ == "__main__":
    p = om.Problem(model=om.Group())

    p.model.add_subsystem('InviscidSubsystem',
                          TransitionGroup(num_nodes=20),
                          promotes_inputs=['H',
                                           'momentum_thickness'],
                          promotes_outputs=['dntilde_dRe_theta'])
    p.model.set_input_defaults('theta', [1.00000000e-05, 3.48148878e-04, 4.82571207e-04, 5.81496392e-04,
           6.65609732e-04, 7.39808467e-04, 8.07727764e-04, 8.69899587e-04,
           9.28499435e-04, 9.83042021e-04, 1.03536180e-03, 1.08450334e-03,
           1.13222524e-03, 1.17728391e-03, 1.22145469e-03, 1.26328656e-03,
           1.30460947e-03, 1.34380744e-03, 1.38278159e-03, 1.41977625e-03])
    p.setup()

    with np.printoptions(linewidth=1024, edgeitems=20, formatter={'float_kind': '{:5.5f}'.format}):
        # formatter={'float_kind': '{:5.2f}'.format})
        p.check_partials(method='fd', form='central', show_only_incorrect=False, compact_print=False)

