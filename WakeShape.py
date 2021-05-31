import openmdao.api as om
import numpy as np

# Optimise shape of wake panels to make pressure constant across upper/lower wake

# Calculate d*(s) given ue(s), due/ds from previous component
# Calculate wake length implicitly by where d*(s) = 0


class AverageWakeVelocity(om.ExplicitComponent):

    def setup(self):
        self.add_input('lower_trailing_edge_velocity')
        self.add_input('upper_trailing_edge_velocity')
        self.add_input('wake_trailing_edge_velocity')
        self.add_input('wake_length')

        self.add_output('average_trailing_edge_velocity')
        self.add_output('average_edge_velocity_gradient')

        self.declare_partials('average_trailing_edge_velocity', 'lower_trailing_edge_velocity', val=0.5)
        self.declare_partials('average_trailing_edge_velocity', 'upper_trailing_edge_velocity', val=0.5)
        self.declare_partials('average_edge_velocity_gradient', 'lower_trailing_edge_velocity')
        self.declare_partials('average_edge_velocity_gradient', 'upper_trailing_edge_velocity')
        self.declare_partials('average_edge_velocity_gradient', 'wake_trailing_edge_velocity')
        self.declare_partials('average_edge_velocity_gradient', 'wake_length')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ueTEl = inputs['lower_trailing_edge_velocity']
        ueTEu = inputs['upper_trailing_edge_velocity']
        ueWTE = inputs['wake_trailing_edge_velocity']
        Lw = inputs['wake_length']

        ueTE = (ueTEl+ueTEu)/2
        due_ds = (ueWTE - ueTE)/Lw

        outputs['average_trailing_edge_velocity'] = ueTE
        outputs['average_edge_velocity_gradient'] = due_ds

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        Lw = inputs['wake_length']

        partials['average_edge_velocity_gradient', 'lower_trailing_edge_velocity'] = -1/(2*Lw)
        partials['average_edge_velocity_gradient', 'upper_trailing_edge_velocity'] = -1/(2*Lw)
        partials['average_edge_velocity_gradient', 'wake_trailing_edge_velocity'] = 1/Lw
        partials['average_edge_velocity_gradient', 'wake_length'] = -1/Lw**2


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    aerofoil_path = r"C:\Users\euana\Documents\Important Documents\Optimisation Project\TASWiG\aerofoils\NACA_63-515"
    aerofoil = np.loadtxt(aerofoil_path).T
    aerofoil = np.flip(aerofoil, 1)
    n_pts = aerofoil.shape[1]
    n_wk = 4

    p = om.Problem(model=om.Group())

    p.model.add_subsystem('DisplacementBodyGroup', AverageWakeVelocity(),
                          promotes_inputs=['lower_trailing_edge_velocity',
                                           'upper_trailing_edge_velocity',
                                           'wake_trailing_edge_velocity',
                                           'wake_length'],
                          promotes_outputs=['average_trailing_edge_velocity',
                                            'average_edge_velocity_gradient'])

    p.model.set_input_defaults('lower_trailing_edge_velocity', 26)
    p.model.set_input_defaults('upper_trailing_edge_velocity', 30)
    p.model.set_input_defaults('wake_trailing_edge_velocity', 29)
    p.model.set_input_defaults('wake_length', 2)

    p.setup()

    p.run_model()

    with np.printoptions(linewidth=1024, edgeitems=20, formatter={'float_kind': '{:5.5f}'.format}):
        # formatter={'float_kind': '{:5.2f}'.format})
        p.check_partials(show_only_incorrect=True, compact_print=False)
