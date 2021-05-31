import openmdao.api as om
import numpy as np
from CoupledLiftingLineQuadraticDoublet import CoupledLLTQD
from LiftingLine import LiftingLineTestGroup

g = 9.81

num_secs = 4
num_nodes_per_seg = 10
num_nodes = num_nodes_per_seg*(num_secs-1)

aerofoil_path = r"C:\Users\euana\Documents\Important Documents\Optimisation Project\TASWiG\aerofoils\naca0015_10pans"
aerofoil = np.loadtxt(aerofoil_path).T
num_aerofoil_points = aerofoil.shape[1]


class FlatPlateDrag(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_sections', types=int)

    def setup(self):
        num_sections = self.options['num_sections']

        self.add_input('section_chords', shape=(num_sections,))
        self.add_input('section_spans', shape=(num_sections,))
        self.add_input('freestream_velocity')
        self.add_input('kinematic_viscosity')
        self.add_input('density')

        self.add_output('flat_plate_drag')

        self.declare_partials('flat_plate_drag', 'section_chords', method='fd')
        self.declare_partials('flat_plate_drag', 'section_spans', method='fd')
        self.declare_partials('flat_plate_drag', 'freestream_velocity', method='fd')
        self.declare_partials('flat_plate_drag', 'density', method='fd')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        section_chords = inputs['section_chords']
        section_spans = inputs['section_spans']
        v_inf = inputs['freestream_velocity']
        rho = inputs['density']
        nu = inputs['kinematic_viscosity']

        chord_drags = 1.328/np.sqrt(section_chords*v_inf/nu)
        flat_plate_drag = np.sum(np.trapz(chord_drags, section_spans))
        outputs['flat_plate_drag'] = 0.5*rho*v_inf**2*flat_plate_drag


class GenerateSpans(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_sections', types=int)
        self.options.declare('num_nodes_per_segment', types=int)

    def setup(self):
        num_sections = self.options['num_sections']

        self.add_input('design_span_spacing')

        self.add_output('section_spans', shape=(num_sections,))

        self.declare_partials('section_spans', 'design_span_spacing',
                              rows=np.arange(1, num_sections),
                              cols=np.repeat(0, num_sections-1),
                              val=1)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        num_sections = self.options['num_sections']

        design_spans = inputs['design_span_spacing']
        outputs['section_spans'] = np.concatenate([[0], np.repeat(design_spans, num_sections-1)])


p = om.Problem(model=om.Group())

p.model.add_subsystem('GenerateSpans', GenerateSpans(num_sections=num_secs, num_nodes_per_segment=num_nodes_per_seg),
                      promotes_inputs=['design_span_spacing'],
                      promotes_outputs=['section_spans'])

p.model.add_subsystem('LiftingLineTestGroup',
                      LiftingLineTestGroup(num_sections=num_secs, num_nodes_per_segment=num_nodes_per_seg),
                      promotes_inputs=['*'],
                      promotes_outputs=['*'])

p.model.add_subsystem('FlatPlateDrag',
                      FlatPlateDrag(num_sections=num_secs),
                      promotes_inputs=['section_chords',
                                       'section_spans',
                                       'freestream_velocity',
                                       'kinematic_viscosity',
                                       'density'],
                      promotes_outputs=['flat_plate_drag'])

p.model.add_subsystem('WingArea', om.ExecComp('wing_area = sum(surface_areas)', surface_areas=np.ones(num_nodes)),
                      promotes_inputs=['surface_areas'],
                      promotes_outputs=['wing_area'])

p.model.add_subsystem('WingMass', om.ExecComp('wing_mass = 2*wing_area*(foam_area_density + panel_area_density)'),
                      promotes_inputs=['foam_area_density',
                                       'panel_area_density',
                                       'wing_area'],
                      promotes_outputs=['wing_mass'])

p.model.add_subsystem('DroneWeight', om.ExecComp('drone_weight = g*(fuselage_mass + wing_mass)'),
                      promotes_inputs=['fuselage_mass',
                                       'wing_mass',
                                       'g'],
                      promotes_outputs=['drone_weight'])

p.model.add_subsystem('SolarPower', om.ExecComp('solar_power = 2*wing_area*panel_efficiency'),
                      promotes_inputs=['wing_area',
                                       'panel_efficiency'],
                      promotes_outputs=['solar_power'])

p.model.add_subsystem('FlightPower', om.ExecComp('flight_power = (drag+flat_plate_drag)*freestream_velocity'),
                      promotes_inputs=['freestream_velocity',
                                       'drag',
                                       'flat_plate_drag'],
                      promotes_outputs=['flight_power'])

p.model.add_subsystem('PowerConstraint', om.ExecComp('power_constraint = (drivetrain_efficiency*solar_power)-flight_power'),
                      promotes_inputs=['solar_power',
                                       'flight_power',
                                       'drivetrain_efficiency'],
                      promotes_outputs=['power_constraint'])

p.model.add_subsystem('LiftConstraint', om.ExecComp('lift_constraint = lift - drone_weight'),
                      promotes_inputs=['lift',
                                       'drone_weight'],
                      promotes_outputs=['lift_constraint'])


p.model.linear_solver = om.DirectSolver()
p.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, maxiter=10, iprint=2, atol=1e-6, rtol=1e-6)
#solve_subsystems=True,
# setup the optimization

p.driver = om.ScipyOptimizeDriver()
p.driver.options['optimizer'] = 'SLSQP'
p.driver.options['tol'] = 1e-6
p.driver.options['disp'] = True
p.driver.options['debug_print'] = ['desvars', 'ln_cons', 'nl_cons', 'objs']
#p.driver.options['debug_print'] = ['totals']

p.model.add_design_var('freestream_velocity', lower=10, upper=50)
p.model.add_design_var('alpha', lower=0, upper=3, units='deg')
p.model.add_design_var('design_span_spacing', lower=0.1, upper=0.5)
p.model.add_design_var('section_chords', lower=0.01, upper=2)
p.model.add_design_var('section_twists', lower=-3, upper=3, units='deg')

p.model.add_constraint('lift_constraint', equals=0)
p.model.add_constraint('power_constraint', lower=0)

p.model.add_objective('drone_weight')

section_chords = np.array([0.5, 0.4, 0.3, 0.2])
design_span_spacing = np.array([0.25])


p.model.set_input_defaults('density', 1.225)
p.model.set_input_defaults('kinematic_viscosity', 1.4207E-5)
p.model.set_input_defaults('g', 9.81)
p.model.set_input_defaults('section_offsets', ([0, 0, 0, 0]))
p.model.set_input_defaults('section_dihedrals', ([0, 0, 0, 0]), units='deg')
# for control_section in np.arange(num_secs):
#     p.model.set_input_defaults(f'source_aerofoil_{control_section}', aerofoil)
p.model.set_input_defaults('panel_efficiency', 150)
p.model.set_input_defaults('foam_area_density', 0.625)
p.model.set_input_defaults('panel_area_density', 0.32)
p.model.set_input_defaults('drivetrain_efficiency', 0.9*0.8*0.5)
p.model.set_input_defaults('fuselage_mass', 1)

p.setup()

p.set_val('freestream_velocity', 10)
p.set_val('alpha', 1, units='deg')
p.set_val('design_span_spacing', design_span_spacing)
p.set_val('section_chords', [0.5, 0.4, 0.3, 0.2])
p.set_val('section_twists', [0.0, 0.0, 0.0, 0.0], units='deg')
p.set_val('2d_lift_coefficients', np.linspace(1, 0, (num_secs-1)*num_nodes_per_seg))
p.set_val('total_angles', np.repeat(1.5, (num_secs-1)*num_nodes_per_seg), units='deg')

p.run_driver()

#p.run_model()

# with np.printoptions(linewidth=2024, edgeitems=20, formatter={'float_kind': '{:5.5f}'.format}):
#     #formatter={'float_kind': '{:5.2f}'.format})
#       p.check_partials(show_only_incorrect=True, compact_print=True, method='fd')

# Base, Cl = 1.1863445

# Optimised for Cl = 0.4 by Newton's
# optimised_upper_weights = np.array([0.2865416, 0.18585428, 0.12139648])
# optimised_lower_weights = np.array([-0.26501766, -0.04916124, -0.001])

# Optimised for Cl = 0.1 by NLBGS
# optimised_upper_weights = np.array([0.03947886, 0.03725292, 0.06710897])
# optimised_lower_weights = np.array([-0.04871469, -0.02735157, -0.03220049])
