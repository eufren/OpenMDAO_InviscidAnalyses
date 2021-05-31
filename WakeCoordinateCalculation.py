import openmdao.api as om
import numpy as np
import matplotlib.pyplot as plt

import TurbulentWake as turbulentwake
import DisplacementBody as displacementbody


class WakeShapeCalculation(om.Group):

    def initialize(self):
        self.options.declare('num_wake_nodes')
        self.options.declare('num_base_points')

    def setup(self):
        num_wake_nodes = self.options['num_wake_nodes']
        num_base_points = self.options['num_base_points']

        self.add_subsystem('TurbulentWakeGroup', turbulentwake.TurbulentWakeGroup(num_nodes=num_wake_nodes),
                           promotes_inputs=['region_start',
                                            'region_length',
                                            'initial_momentum_thickness',
                                            'initial_H',
                                            'node_external_tangential_velocities'],
                           promotes_outputs=[('displacement_thickness', 'wake_thickness')])

        self.add_subsystem('DisplacementBodyGroup', displacementbody.DisplacementBodyGroup(num_wake_points_per_side=num_wake_nodes, num_base_points=num_base_points),
                           promotes_inputs=['aerofoil_cw_coordinates',
                                            'displacement_thickness',  # This is the boundary layer one
                                            'wake_camber_lengths',
                                            'wake_camber_angles',
                                            'wake_thickness'],
                           promotes_outputs=['displacement_body_coordinates'])


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    aerofoil_path = r"C:\Users\euana\Documents\Important Documents\Optimisation Project\TASWiG\aerofoils\NACA_63-515"
    aerofoil = np.loadtxt(aerofoil_path).T
    aerofoil = np.flip(aerofoil, 1)
    n_body_pts = aerofoil.shape[1]
    n_w_pts = 75
    L = 0.5
    ue_TE = 3
    ue_W = 4

    p = om.Problem(model=om.Group())

    p.model.add_subsystem('WakeShapeCalculation', WakeShapeCalculation(num_wake_nodes=n_w_pts, num_base_points=n_body_pts),
                          promotes_inputs=['region_start',
                                           'region_length',
                                           'initial_momentum_thickness',
                                           'initial_H',
                                           'node_external_tangential_velocities',
                                           'aerofoil_cw_coordinates',
                                           'displacement_thickness',  # This is the boundary layer one
                                           'wake_camber_lengths',
                                           'wake_camber_angles',
                                           'wake_thickness'],
                          promotes_outputs=['displacement_body_coordinates'])

    p.model.set_input_defaults('aerofoil_cw_coordinates', aerofoil)
    p.model.set_input_defaults('displacement_thickness', np.repeat(0.02, n_body_pts))

    p.model.set_input_defaults('region_start', 0)
    p.model.set_input_defaults('region_length', L)
    p.model.set_input_defaults('wake_camber_lengths', np.repeat(L/(n_w_pts-1), n_w_pts))
    p.model.set_input_defaults('wake_camber_angles', np.linspace(-0.1, np.deg2rad(3), n_w_pts))
    p.model.set_input_defaults('initial_momentum_thickness', 2*0.01)
    p.model.set_input_defaults('initial_H', 2)
    p.model.set_input_defaults('node_external_tangential_velocities', np.linspace(ue_TE, ue_W, n_w_pts))

    p.setup()
    p.run_model()

    with np.printoptions(linewidth=1024, edgeitems=20, formatter={'float_kind': '{:5.5f}'.format}):
        # formatter={'float_kind': '{:5.2f}'.format})
        p.check_partials(show_only_incorrect=True, compact_print=False)

    base = p.get_val('aerofoil_cw_coordinates')
    displacement_body = p.get_val('displacement_body_coordinates')
    plt.plot(base[0, :], base[1, :], label="Base aerofoil")
    plt.plot(displacement_body[0, :], displacement_body[1, :], label="Displacement body")
    plt.legend()
    plt.gca().set_aspect('equal')
    plt.show()