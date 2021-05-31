import openmdao.api as om
import numpy as np

import Turbulent as turbulent
import Laminar as laminar
import Transition as transition

# Create a group
# Add the three subgroups
# Connect transition distance to region length on laminar component
# Write component to output final values of theta, H at transition
# Write component to calculate turbulent b.l length as total length - transition distance


class TransitionalBoundaryLayerGroup(om.Group):

    def initialize(self):
        self.options.declare('num_lam_nodes')
        self.options.declare('num_turb_nodes')

    def setup(self):
        num_lam_nodes = self.options['num_lam_nodes']
        num_turb_nodes = self.options['num_turb_nodes']

        self.add_subsystem('LaminarGroup', laminar.LaminarGroup(num_nodes=num_lam_nodes),
                           promotes_inputs=[('region_length', 'transition_distance'),
                                            'kinematic_viscosity',
                                            ('node_external_tangential_velocities', 'lam_node_external_tangential_velocities')],
                           promotes_outputs=[('momentum_thickness', 'lam_momentum_thickness'),
                                             ('displacement_thickness', 'lam_displacement_thickness'),
                                             ('H', 'lam_H'),
                                             ('Cf', 'lam_Cf')])

        # self.add_subsystem('TransitionGroup', transition.TransitionGroup(num_nodes=num_lam_nodes, transition_log_max_amplification_ratio=2.0),
        #                    promotes_inputs=[('H', 'lam_H'),
        #                                     ('momentum_thickness', 'lam_momentum_thickness')],
        #                    promotes_outputs=['log_max_amplification_ratio',
        #                                      'transition_distance'])
        self.add_subsystem('FakeTransitionLength', om.ExecComp('transition_distance = 0.1'),
                           promotes_outputs=['transition_distance'])

        self.add_subsystem('ExtractTransitionMomentumThickness', om.ExecComp('turb_initial_momentum_thickness = lam_momentum_thickness[-1]', turb_initial_momentum_thickness=0.1, lam_momentum_thickness=0.1*np.ones(num_lam_nodes)),
                           promotes_inputs=['lam_momentum_thickness'],
                           promotes_outputs=['turb_initial_momentum_thickness'])
        self.add_subsystem('ExtractTransitionH', om.ExecComp('turb_initial_H = lam_H[-1]', turb_initial_H=1.5, lam_H=1.5*np.ones(num_lam_nodes)),  # Convert this to use src_indices?
                           promotes_inputs=['lam_H'],
                           promotes_outputs=['turb_initial_H'])
        self.add_subsystem('CalculateTurbulentLength', om.ExecComp('turb_region_length = total_length-transition_distance'),
                           promotes_inputs=['transition_distance',
                                            'total_length'],
                           promotes_outputs=['turb_region_length'])

        self.add_subsystem('TurbulentGroup', turbulent.TurbulentGroup(num_nodes=num_turb_nodes),
                           promotes_inputs=[('region_length', 'turb_region_length'),
                                            'kinematic_viscosity',
                                            ('node_external_tangential_velocities', 'turb_node_external_tangential_velocities'),
                                            ('initial_momentum_thickness', 'turb_initial_momentum_thickness'),
                                            ('initial_H', 'turb_initial_H')],
                           promotes_outputs=[('momentum_thickness', 'turb_momentum_thickness'),
                                             ('displacement_thickness', 'turb_displacement_thickness'),
                                             ('H', 'turb_H'),
                                             ('Cfs', 'turb_Cf')])

        self.linear_solver = om.DirectSolver()
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, maxiter=30, iprint=2)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    numero_nodeso = 200

    p = om.Problem(model=om.Group())

    p.model.add_subsystem('TransitionalBoundaryLayerGroup',
                          TransitionalBoundaryLayerGroup(num_lam_nodes=numero_nodeso, num_turb_nodes=numero_nodeso),
                          promotes_inputs=['total_length',
                                           'lam_node_external_tangential_velocities', # Temporary, this will be replaced by the interpolating function at some point
                                           'turb_node_external_tangential_velocities', # Temporary, this will be replaced by the interpolating function at some point
                                           'kinematic_viscosity'],
                          promotes_outputs=['lam_Cf',
                                            'turb_Cf',
                                            'turb_displacement_thickness',
                                            'lam_displacement_thickness',
                                            'turb_momentum_thickness',
                                            'lam_momentum_thickness',
                                            'transition_distance'])

    total_length = 2
    test_vels = np.linspace(0.1, 60.4141875, 2*numero_nodeso)#np.repeat(6.4141875, numero_nodeso)
    p.model.set_input_defaults('kinematic_viscosity', 1.81e-5)
    p.model.set_input_defaults('total_length', total_length)
    p.model.set_input_defaults('lam_node_external_tangential_velocities', test_vels[:numero_nodeso])
    p.model.set_input_defaults('turb_node_external_tangential_velocities', test_vels[numero_nodeso:])

    p.setup()
    p.run_model()

    xtr = p.get_val('transition_distance')
    lam_nodes = np.linspace(0, xtr, numero_nodeso)
    turb_nodes = np.linspace(xtr, total_length, numero_nodeso)
    node_positions = np.concatenate((lam_nodes, turb_nodes))
    Cf = np.concatenate((p.get_val('lam_Cf'), p.get_val('turb_Cf')))
    d1 = np.concatenate((p.get_val('lam_displacement_thickness'), p.get_val('turb_displacement_thickness')))
    d2 = np.concatenate((p.get_val('lam_momentum_thickness'), p.get_val('turb_momentum_thickness')))

    fig, axs = plt.subplots(3, 1)
    axs[0].plot(node_positions, d1, label='Displacement Thickness')
    axs[0].set_title('Displacement Thickness')
    axs[0].axvline(x=xtr, color='k', linestyle='--', label='Transition Onset')
    axs[0].legend()
    axs[1].plot(node_positions, d2, label='Momentum Thickness')
    axs[1].set_title('Momentum Thickness')
    axs[1].axvline(x=xtr, color='k', linestyle='--', label='Transition Onset')
    axs[1].legend()
    axs[2].plot(node_positions, Cf, label='Cf')
    axs[2].set_title('Skin Friction Coefficient')
    axs[2].axvline(x=xtr, color='k', linestyle='--', label='Transition Onset')
    axs[2].set_ylim(-0.0025, max(p.get_val('turb_Cf'))*2)
    axs[2].legend()
    plt.show()

    # with np.printoptions(linewidth=1024, edgeitems=20, formatter={'float_kind': '{:5.5f}'.format}):
    #     # formatter={'float_kind': '{:5.2f}'.format})
    #     p.check_partials(show_only_incorrect=True, compact_print=True)