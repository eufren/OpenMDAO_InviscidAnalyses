import openmdao.api as om
from openmdao.components.interp_util.interp import InterpND
import numpy as np


# Component that takes transitional b.l node locations and combines them into CW series around TE
# Component that takes interpolated ue and splits them into the different boundary layers

class NodePositionAssembler(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_upper_lam_nodes', types=int)
        self.options.declare('num_upper_turb_nodes', types=int)
        self.options.declare('num_lower_lam_nodes', types=int)
        self.options.declare('num_lower_turb_nodes', types=int)

    def setup(self):
        num_upper_lam_nodes = self.options['num_upper_lam_nodes']
        num_upper_turb_nodes = self.options['num_upper_turb_nodes']
        num_lower_lam_nodes = self.options['num_lower_lam_nodes']
        num_lower_turb_nodes = self.options['num_lower_turb_nodes']
        num_viscous_nodes = num_upper_lam_nodes + num_upper_turb_nodes + num_lower_lam_nodes + num_lower_turb_nodes

        self.add_input('upper_laminar_node_positions', shape=(num_upper_lam_nodes,))
        self.add_input('upper_turbulent_node_positions', shape=(num_upper_turb_nodes,))
        self.add_input('lower_laminar_node_positions', shape=(num_lower_lam_nodes,))
        self.add_input('lower_turbulent_node_positions', shape=(num_lower_turb_nodes,))

        self.add_output('viscous_node_positions', shape=(num_viscous_nodes,))

        self.declare_partials('viscous_node_positions', 'lower_turbulent_node_positions',
                              rows=np.arange(num_lower_turb_nodes),
                              cols=np.arange(num_lower_turb_nodes),
                              val=1)
        self.declare_partials('viscous_node_positions', 'lower_laminar_node_positions',
                              rows=np.arange(num_lower_turb_nodes, num_lower_turb_nodes+num_lower_lam_nodes),
                              cols=np.arange(num_lower_lam_nodes),
                              val=1)
        self.declare_partials('viscous_node_positions', 'upper_laminar_node_positions',
                              rows=np.arange(num_lower_turb_nodes+num_lower_lam_nodes, num_lower_turb_nodes+num_lower_lam_nodes+num_upper_lam_nodes),
                              cols=np.arange(num_upper_lam_nodes),
                              val=1)
        self.declare_partials('viscous_node_positions', 'upper_turbulent_node_positions',
                              rows=np.arange(num_lower_turb_nodes+num_lower_lam_nodes+num_upper_lam_nodes, num_viscous_nodes),
                              cols=np.arange(num_upper_turb_nodes),
                              val=1)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        s_ul = inputs['upper_laminar_node_positions']
        s_ut = inputs['upper_turbulent_node_positions']
        s_ll = inputs['lower_laminar_node_positions']
        s_lt = inputs['lower_turbulent_node_positions']

        outputs['viscous_node_positions'] = np.concatenate((s_lt, s_ll, s_ul, s_ut))


class InterpolatedUeDissasembler(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_inviscid_nodes', types=int)
        self.options.declare('num_upper_lam_nodes', types=int)
        self.options.declare('num_upper_turb_nodes', types=int)
        self.options.declare('num_lower_lam_nodes', types=int)
        self.options.declare('num_lower_turb_nodes', types=int)

    def setup(self):
        num_upper_lam_nodes = self.options['num_upper_lam_nodes']
        num_upper_turb_nodes = self.options['num_upper_turb_nodes']
        num_lower_lam_nodes = self.options['num_lower_lam_nodes']
        num_lower_turb_nodes = self.options['num_lower_turb_nodes']
        num_viscous_nodes = num_upper_lam_nodes + num_upper_turb_nodes + num_lower_lam_nodes + num_lower_turb_nodes

        self.add_input('interpolated_external_tangential_velocities', shape=(num_viscous_nodes,))

        self.add_output('upper_laminar_external_tangential_velocities', shape=(num_upper_lam_nodes,))
        self.add_output('upper_turbulent_external_tangential_velocities', shape=(num_upper_turb_nodes,))
        self.add_output('lower_laminar_external_tangential_velocities', shape=(num_lower_lam_nodes,))
        self.add_output('lower_turbulent_external_tangential_velocities', shape=(num_lower_turb_nodes,))

        self.declare_partials('lower_turbulent_external_tangential_velocities', 'interpolated_external_tangential_velocities',
                              rows=np.arange(num_lower_turb_nodes),
                              cols=np.arange(num_lower_turb_nodes),
                              val=1)
        self.declare_partials('lower_laminar_external_tangential_velocities', 'interpolated_external_tangential_velocities',
                              rows=np.arange(num_lower_lam_nodes),
                              cols=np.arange(num_lower_turb_nodes, num_lower_turb_nodes+num_lower_lam_nodes),
                              val=1)
        self.declare_partials('upper_laminar_external_tangential_velocities', 'interpolated_external_tangential_velocities',
                              rows=np.arange(num_upper_lam_nodes),
                              cols=np.arange(num_lower_turb_nodes+num_lower_lam_nodes, num_lower_turb_nodes+num_lower_lam_nodes+num_upper_lam_nodes),
                              val=1)
        self.declare_partials('upper_turbulent_external_tangential_velocities', 'interpolated_external_tangential_velocities',
                              rows=np.arange(num_upper_turb_nodes),
                              cols=np.arange(num_lower_turb_nodes+num_lower_lam_nodes+num_upper_lam_nodes, num_viscous_nodes),
                              val=1)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        num_upper_lam_nodes = self.options['num_upper_lam_nodes']
        num_upper_turb_nodes = self.options['num_upper_turb_nodes']
        num_lower_lam_nodes = self.options['num_lower_lam_nodes']
        num_lower_turb_nodes = self.options['num_lower_turb_nodes']
        num_viscous_nodes = num_lower_turb_nodes+num_lower_lam_nodes+num_upper_lam_nodes+num_upper_turb_nodes

        ue_int = inputs['interpolated_external_tangential_velocities']

        ue_lt = ue_int[0:num_lower_turb_nodes]
        ue_ll = ue_int[num_lower_turb_nodes:num_lower_turb_nodes+num_lower_lam_nodes]
        ue_ul = ue_int[num_lower_turb_nodes+num_lower_lam_nodes:num_lower_turb_nodes+num_lower_lam_nodes+num_upper_lam_nodes]
        ue_ut = ue_int[num_lower_turb_nodes+num_lower_lam_nodes+num_upper_lam_nodes:num_viscous_nodes]

        outputs['lower_turbulent_external_tangential_velocities'] = ue_lt
        outputs['lower_laminar_external_tangential_velocities'] = ue_ll
        outputs['upper_laminar_external_tangential_velocities'] = ue_ul
        outputs['upper_turbulent_external_tangential_velocities'] = ue_ut


class FindInviscidPositions(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_inviscid_nodes', types=int)

    def setup(self):
        num_inviscid_nodes = self.options['num_inviscid_nodes']
        self.add_input('panel_lengths', shape=num_inviscid_nodes)
        self.add_output('inviscid_node_positions', shape=num_inviscid_nodes)

        self.declare_partials('inviscid_node_positions', 'panel_lengths')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        l = inputs['panel_lengths']

        tril_l = np.tril(l)
        tril_l[np.diag_indices_from(tril_l)] *= 0.5
        posns = np.sum(tril_l, axis=1)
        outputs['inviscid_node_positions'] = posns

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        l = inputs['panel_lengths']

        dpos_dl = np.tril(np.ones_like(l))
        dpos_dl[np.diag_indices_from(dpos_dl)] *= 0.5

        partials['inviscid_node_positions', 'panel_lengths'] = dpos_dl


class InterpolateVels(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_interp_from_nodes')
        self.options.declare('num_interp_to_nodes')

    def setup(self):
        num_interp_from_nodes = self.options['num_interp_from_nodes']
        num_interp_to_nodes = self.options['num_interp_to_nodes']

        self.add_input('data_y', shape=(num_interp_from_nodes,))
        self.add_input('data_x', shape=(num_interp_from_nodes,))
        self.add_input('interp_x', shape=(num_interp_to_nodes,))

        self.add_output('interp_y', shape=(num_interp_to_nodes,))

        self.declare_partials('interp_y', 'data_y', method='fd')
        self.declare_partials('interp_y', 'data_x', method='fd')
        self.declare_partials('interp_y', 'interp_x',
                              rows=np.arange(num_interp_to_nodes),
                              cols=np.arange(num_interp_to_nodes))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        y_f = inputs['data_y']
        x_f = inputs['data_x']
        x_t = inputs['interp_x']

        interp = InterpND(method='slinear', points=x_f, values=y_f)
        interp.extrapolate = True
        y_t = interp.interpolate(x_t)

        outputs['interp_y'] = y_t

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        y_f = inputs['data_y']
        x_f = inputs['data_x']
        x_t = inputs['interp_x']

        interp = InterpND(method='slinear', points=x_f, values=y_f)
        interp.extrapolate = True
        interp._compute_d_dvalues = True
        y_t, dyt_dxt = interp.interpolate(x_t, compute_derivative=True)

        partials['interp_y', 'interp_x'] = np.squeeze(dyt_dxt)


class ITVInterpolatorGroup(om.Group):

    def initialize(self):
        self.options.declare('num_inviscid_nodes', types=int)
        self.options.declare('num_upper_lam_nodes', types=int)
        self.options.declare('num_upper_turb_nodes', types=int)
        self.options.declare('num_lower_lam_nodes', types=int)
        self.options.declare('num_lower_turb_nodes', types=int)

    def setup(self):
        num_inviscid_nodes = self.options['num_inviscid_nodes']
        num_upper_lam_nodes = self.options['num_upper_lam_nodes']
        num_upper_turb_nodes = self.options['num_upper_turb_nodes']
        num_lower_lam_nodes = self.options['num_lower_lam_nodes']
        num_lower_turb_nodes = self.options['num_lower_turb_nodes']
        num_viscous_nodes = num_upper_lam_nodes + num_upper_turb_nodes + num_lower_lam_nodes + num_lower_turb_nodes

        self.add_subsystem('NodePositionAssembler', NodePositionAssembler(num_upper_lam_nodes=num_upper_lam_nodes, num_upper_turb_nodes=num_upper_turb_nodes,
                                                                          num_lower_lam_nodes=num_lower_lam_nodes, num_lower_turb_nodes=num_lower_turb_nodes),
                           promotes_inputs=['upper_laminar_node_positions',
                                            'upper_turbulent_node_positions',
                                            'lower_laminar_node_positions',
                                            'lower_turbulent_node_positions'],
                           promotes_outputs=['viscous_node_positions'])

        self.add_subsystem('FindInviscidPositions', FindInviscidPositions(num_inviscid_nodes=num_inviscid_nodes),
                           promotes_inputs=['panel_lengths'],
                           promotes_outputs=['inviscid_node_positions'])

        self.add_subsystem('InterpolateVels', InterpolateVels(num_interp_from_nodes=num_inviscid_nodes, num_interp_to_nodes=num_viscous_nodes),
                           promotes_inputs=[('data_x', 'inviscid_node_positions'),
                                            ('data_y', 'external_tangential_velocities'),
                                            ('interp_x', 'viscous_node_positions')],
                           promotes_outputs=[('interp_y', 'interpolated_external_tangential_velocities')])

        self.add_subsystem('InterpolatedUeDissasembler', InterpolatedUeDissasembler(num_upper_lam_nodes=num_upper_lam_nodes, num_upper_turb_nodes=num_upper_turb_nodes,
                                                                                    num_lower_lam_nodes=num_lower_lam_nodes, num_lower_turb_nodes=num_lower_turb_nodes),
                           promotes_inputs=['interpolated_external_tangential_velocities'],
                           promotes_outputs=['lower_turbulent_external_tangential_velocities',
                                             'lower_laminar_external_tangential_velocities',
                                             'upper_laminar_external_tangential_velocities',
                                             'upper_turbulent_external_tangential_velocities'])

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    n_invisc = 10
    n_lam_upper = 3
    n_lam_lower = 4
    n_turb_upper = 5
    n_turb_lower = 6
    n_visc = n_turb_upper+n_lam_upper+n_lam_lower+n_turb_lower

    p = om.Problem(model=om.Group())

    p.model.add_subsystem('ITVInterpolatorGroup',
                          ITVInterpolatorGroup(num_inviscid_nodes=n_invisc,
                                               num_upper_lam_nodes=n_lam_upper, num_upper_turb_nodes=n_turb_upper,
                                               num_lower_lam_nodes=n_lam_lower, num_lower_turb_nodes=n_turb_lower),
                          promotes_inputs=['panel_lengths',
                                           'external_tangential_velocities',
                                           'upper_laminar_node_positions',
                                           'upper_turbulent_node_positions',
                                           'lower_laminar_node_positions',
                                           'lower_turbulent_node_positions'],
                          promotes_outputs=['interpolated_external_tangential_velocities',
                                            'lower_turbulent_external_tangential_velocities',
                                            'lower_laminar_external_tangential_velocities',
                                            'upper_laminar_external_tangential_velocities',
                                            'upper_turbulent_external_tangential_velocities',
                                            'viscous_node_positions'])

    p.model.set_input_defaults('panel_lengths', np.linspace(0, 1, n_invisc))
    p.model.set_input_defaults('external_tangential_velocities', np.cos(2*np.pi*np.linspace(0, 1, n_invisc)))
    p.model.set_input_defaults('lower_turbulent_node_positions', np.linspace(0.1, 0.9, n_visc-3)[0:n_turb_lower])
    p.model.set_input_defaults('lower_laminar_node_positions', np.linspace(0.1, 0.9, n_visc-3)[n_turb_lower-1:n_turb_lower+n_lam_lower-1])
    p.model.set_input_defaults('upper_laminar_node_positions', np.linspace(0.1, 0.9, n_visc-3)[n_turb_lower+n_lam_lower-2:n_turb_lower+n_lam_lower+n_lam_upper-2])
    p.model.set_input_defaults('upper_turbulent_node_positions', np.linspace(0.1, 0.9, n_visc-3)[n_turb_lower+n_lam_lower+n_lam_upper-3:n_turb_lower+n_lam_lower+n_lam_upper+n_turb_upper-3])

    p.setup()

    with np.printoptions(linewidth=1024, edgeitems=20, formatter={'float_kind': '{:5.5f}'.format}):
        # formatter={'float_kind': '{:5.2f}'.format})
        p.check_partials(show_only_incorrect=True, compact_print=False, method='fd')

    p.run_model()

    plt.plot(p.get_val('panel_lengths'), p.get_val('external_tangential_velocities'), linestyle=' ', marker='x', label='Inviscid Velocities')
    plt.plot(p.get_val('lower_turbulent_node_positions'), p.get_val('lower_turbulent_external_tangential_velocities'), marker='+', label='Lower Turbulent')
    plt.plot(p.get_val('lower_laminar_node_positions'), p.get_val('lower_laminar_external_tangential_velocities'), marker='+', label='Lower Laminar')
    plt.plot(p.get_val('upper_laminar_node_positions'), p.get_val('upper_laminar_external_tangential_velocities'), marker='+', label='Upper Laminar')
    plt.plot(p.get_val('upper_turbulent_node_positions'), p.get_val('upper_turbulent_external_tangential_velocities'), marker='+', label='Upper Turbulent')
    plt.legend()
    plt.show()
