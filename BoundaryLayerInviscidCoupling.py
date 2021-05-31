import openmdao.api as om
import numpy as np

import TransitionalBoundaryLayer as transitionalboundarylayer
import Inviscid as inviscid
import ITVInterpolator as itvinterpolator

# Use stagnation point position to locate two boundary layers
# Do some kind of magic to account for lower boundary layer going against the surface direction
# Run 1 way coupled analysis (use inviscid velocity as edge velocity to drive boundary layers)


class GenerateBLNodes(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_inviscid_nodes')
        self.options.declare('num_upper_lam_nodes')
        self.options.declare('num_upper_turb_nodes')
        self.options.declare('num_lower_lam_nodes')
        self.options.declare('num_lower_turb_nodes')

    def setup(self):
        num_inviscid_nodes = self.options['num_inviscid_nodes']
        num_upper_lam_nodes = self.options['num_upper_lam_nodes']
        num_upper_turb_nodes = self.options['num_upper_turb_nodes']
        num_lower_lam_nodes = self.options['num_lower_lam_nodes']
        num_lower_turb_nodes = self.options['num_lower_turb_nodes']

        self.add_input('panel_lengths', shape=(num_inviscid_nodes,))
        self.add_input('stagnation_point_position', 1)
        self.add_input('upper_transition_length', 0.05)
        self.add_input('lower_transition_length', 0.05)

        self.add_output('upper_length')
        self.add_output('lower_length')
        self.add_output('upper_laminar_node_positions', shape=(num_upper_lam_nodes,))
        self.add_output('upper_turbulent_node_positions', shape=(num_upper_turb_nodes,))
        self.add_output('lower_laminar_node_positions', shape=(num_lower_lam_nodes,))
        self.add_output('lower_turbulent_node_positions', shape=(num_lower_turb_nodes,))

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        num_upper_lam_nodes = self.options['num_upper_lam_nodes']
        num_upper_turb_nodes = self.options['num_upper_turb_nodes']
        num_lower_lam_nodes = self.options['num_lower_lam_nodes']
        num_lower_turb_nodes = self.options['num_lower_turb_nodes']

        l = inputs['panel_lengths']
        s0 = inputs['stagnation_point_position']
        tru = inputs['upper_transition_length']
        trl = inputs['lower_transition_length']

        perimeter = np.sum(l)

        outputs['upper_length'] = perimeter - s0
        outputs['lower_length'] = s0
        outputs['upper_laminar_node_positions'] = np.linspace(s0, s0+tru, num_upper_lam_nodes)
        outputs['upper_turbulent_node_positions'] = np.linspace(s0+tru, perimeter, num_upper_turb_nodes)
        outputs['lower_laminar_node_positions'] = np.linspace(trl, s0, num_lower_lam_nodes)
        outputs['lower_turbulent_node_positions'] = np.linspace(0, trl, num_lower_turb_nodes)


class BoundaryLayerCoupling(om.Group):

    def initialize(self):
        self.options.declare('num_base_points')
        self.options.declare('num_upper_lam_nodes')
        self.options.declare('num_upper_turb_nodes')
        self.options.declare('num_lower_lam_nodes')
        self.options.declare('num_lower_turb_nodes')

    def setup(self):
        num_base_points = self.options['num_base_points']
        num_inviscid_nodes = num_base_points-1
        num_upper_lam_nodes = self.options['num_upper_lam_nodes']
        num_upper_turb_nodes = self.options['num_upper_turb_nodes']
        num_lower_lam_nodes = self.options['num_lower_lam_nodes']
        num_lower_turb_nodes = self.options['num_lower_turb_nodes']

        self.add_subsystem('InviscidSubsystem',
                           inviscid.InviscidGroup(num_points=num_base_points),
                           promotes_inputs=['aerofoil_ccw_coordinates', 'alpha'],
                           promotes_outputs=['external_tangential_velocities',
                                             'pressure_coefficients',
                                             'stagnation_point_position',
                                             'panel_lengths'])

        self.add_subsystem('GenerateBLNodes', GenerateBLNodes(num_inviscid_nodes=num_inviscid_nodes,
                                                              num_upper_lam_nodes=num_upper_lam_nodes, num_upper_turb_nodes=num_upper_turb_nodes,
                                                              num_lower_lam_nodes=num_lower_lam_nodes, num_lower_turb_nodes=num_lower_turb_nodes),
                           promotes_inputs=['panel_lengths',
                                            'stagnation_point_position',
                                            'upper_transition_length',
                                            'lower_transition_length'],
                           promotes_outputs=['upper_length',
                                             'lower_length',
                                             'upper_laminar_node_positions',
                                             'upper_turbulent_node_positions',
                                             'lower_laminar_node_positions',
                                             'lower_turbulent_node_positions'])

        self.add_subsystem('ITVInterpolatorGroup',
                           itvinterpolator.ITVInterpolatorGroup(num_inviscid_nodes=num_inviscid_nodes,
                                                                num_upper_lam_nodes=num_upper_lam_nodes, num_upper_turb_nodes=num_upper_turb_nodes,
                                                                num_lower_lam_nodes=num_lower_lam_nodes, num_lower_turb_nodes=num_lower_turb_nodes),
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
                                             'viscous_node_positions',
                                             'inviscid_node_positions'])

        self.add_subsystem('InvertLowerLamVelocities', om.ExecComp('neg_lower_laminar_external_tangential_velocities = -lower_laminar_external_tangential_velocities[::-1]', lower_laminar_external_tangential_velocities=np.ones(num_lower_lam_nodes)),
                           promotes_inputs=['lower_laminar_external_tangential_velocities'],
                           promotes_outputs=['neg_lower_laminar_external_tangential_velocities'])

        self.add_subsystem('InvertLowerTurbVelocities', om.ExecComp('neg_lower_turbulent_external_tangential_velocities = -lower_turbulent_external_tangential_velocities[::-1]', lower_turbulent_external_tangential_velocities=np.ones(num_lower_turb_nodes)),
                           promotes_inputs=['lower_turbulent_external_tangential_velocities'],
                           promotes_outputs=['neg_lower_turbulent_external_tangential_velocities'])

        self.add_subsystem('UpperBoundaryLayer', transitionalboundarylayer.TransitionalBoundaryLayerGroup(num_lam_nodes=num_upper_lam_nodes, num_turb_nodes=num_upper_turb_nodes),
                           promotes_inputs=[('total_length', 'upper_length'),
                                            ('lam_node_external_tangential_velocities', 'upper_laminar_external_tangential_velocities'),
                                            ('turb_node_external_tangential_velocities', 'upper_turbulent_external_tangential_velocities'),
                                            'kinematic_viscosity'],
                           promotes_outputs=[('lam_momentum_thickness', 'upper_lam_momentum_thickness'),
                                             ('turb_momentum_thickness', 'upper_turb_momentum_thickness'),
                                             ('lam_displacement_thickness', 'upper_lam_displacement_thickness'),
                                             ('turb_displacement_thickness', 'upper_turb_displacement_thickness'),
                                             ('lam_Cf', 'upper_lam_Cf'),
                                             ('turb_Cf', 'upper_turb_Cf'),
                                             ('transition_distance', 'upper_transition_length')])

        self.add_subsystem('LowerBoundaryLayer', transitionalboundarylayer.TransitionalBoundaryLayerGroup(num_lam_nodes=num_lower_lam_nodes, num_turb_nodes=num_lower_turb_nodes),
                           promotes_inputs=[('total_length', 'lower_length'),
                                            ('lam_node_external_tangential_velocities', 'neg_lower_laminar_external_tangential_velocities'),
                                            ('turb_node_external_tangential_velocities', 'neg_lower_turbulent_external_tangential_velocities'),
                                            'kinematic_viscosity'],
                           promotes_outputs=[('lam_momentum_thickness', 'lower_lam_momentum_thickness'),
                                             ('turb_momentum_thickness', 'lower_turb_momentum_thickness'),
                                             ('lam_displacement_thickness', 'lower_lam_displacement_thickness'),
                                             ('turb_displacement_thickness', 'lower_turb_displacement_thickness'),
                                             ('lam_Cf', 'lower_lam_Cf'),
                                             ('turb_Cf', 'lower_turb_Cf'),
                                             ('transition_distance', 'lower_transition_length')])

        self.linear_solver = om.LinearRunOnce()
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, maxiter=30, iprint=2)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    aerofoil_path = r"C:\Users\euana\Documents\Important Documents\Optimisation Project\TASWiG\aerofoils\NACA_64-012"
    soln_path = r"C:\Users\euana\Documents\Important Documents\Optimisation Project\TASWiG\aerofoils\NACA_64-012_JF"
    aerofoil = np.loadtxt(aerofoil_path).T
    num_points = aerofoil.shape[1]
    num_ul = 20
    num_ut = 75
    num_ll = 50
    num_lt = 50

    p = om.Problem(model=BoundaryLayerCoupling(num_base_points=num_points,
                                               num_upper_lam_nodes=num_ul, num_upper_turb_nodes=num_ut,
                                               num_lower_lam_nodes=num_ll, num_lower_turb_nodes=num_lt))

    p.model.set_input_defaults('alpha', 3, units='deg')
    p.model.set_input_defaults('aerofoil_ccw_coordinates', aerofoil)
    p.model.set_input_defaults('kinematic_viscosity', 1.81e-5/10)

    p.setup()
    p.run_model()

    node_positions = np.concatenate((p.get_val('upper_laminar_node_positions'), p.get_val('upper_turbulent_node_positions')))
    Cf = np.concatenate((p.get_val('upper_lam_Cf'), p.get_val('upper_turb_Cf')))
    d1 = np.concatenate((p.get_val('upper_lam_displacement_thickness'), p.get_val('upper_turb_displacement_thickness')))
    d2 = np.concatenate((p.get_val('upper_lam_momentum_thickness'), p.get_val('upper_turb_momentum_thickness')))

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
