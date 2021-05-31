import openmdao.api as om
import numpy as np
from LiftingLine import LiftingLineGroup
from Inviscid import InviscidGroup


class InterpolateAerofoil(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_points')

    def setup(self):
        num_points = self.options['num_points']

        self.add_input('aerofoil_coordinates_A', shape=(2, num_points))
        self.add_input('aerofoil_coordinates_B', shape=(2, num_points))
        self.add_input('B_span')  # Span of aerofoil A to B
        self.add_input('interpolated_span')  # Span of aerofoil A to interpolating point

        self.add_output('interpolated_coordinates', shape=(2, num_points))

        self.declare_partials('interpolated_coordinates', 'aerofoil_coordinates_A',
                              rows=np.arange(2*num_points),
                              cols=np.arange(2*num_points))
        self.declare_partials('interpolated_coordinates', 'aerofoil_coordinates_B',
                              rows=np.arange(2*num_points),
                              cols=np.arange(2*num_points))
        self.declare_partials('interpolated_coordinates', 'B_span')
        self.declare_partials('interpolated_coordinates', 'interpolated_span')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        A = inputs['aerofoil_coordinates_A']
        B = inputs['aerofoil_coordinates_B']
        s = inputs['B_span']
        z = inputs['interpolated_span']

        outputs['interpolated_coordinates'] = A + ((B-A)/s)*z

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        num_points = self.options['num_points']

        A = inputs['aerofoil_coordinates_A']
        B = inputs['aerofoil_coordinates_B']
        s = inputs['B_span']
        z = inputs['interpolated_span']

        partials['interpolated_coordinates', 'aerofoil_coordinates_A'] = np.repeat(1 - z/s, 2*num_points)
        partials['interpolated_coordinates', 'aerofoil_coordinates_B'] = np.repeat(z/s, 2*num_points)
        partials['interpolated_coordinates', 'B_span'] = (-((B-A)/s**2)*z).reshape(-1)
        partials['interpolated_coordinates', 'interpolated_span'] = ((B-A)/s).reshape(-1)


class DemuxInterpolatedSpans(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_sections')
        self.options.declare('num_nodes_per_segment')

    def setup(self):
        num_sections = self.options['num_sections']
        num_nodes_per_segment = self.options['num_nodes_per_segment']
        num_segments = num_sections - 1
        num_nodes = num_segments * num_nodes_per_segment
        num_interpolated_sections = num_nodes + 1

        self.add_input('section_spans', shape=(num_sections,))

        for source_pair in np.arange(num_segments):
            for interpolated_aerofoil in np.arange(1, num_nodes_per_segment+2):
                if interpolated_aerofoil == num_nodes_per_segment+1:
                    self.add_output(f'source_span_{source_pair}')
                    self.declare_partials(f'source_span_{source_pair}', 'section_spans',
                                          rows=np.array([0, 0]),
                                          cols=np.array([source_pair, source_pair+1]),
                                          val=np.array([-1, 1]))
                else:
                    self.add_output(f'interpolated_span_{source_pair}_{interpolated_aerofoil-1}')
                    self.declare_partials(f'interpolated_span_{source_pair}_{interpolated_aerofoil-1}', 'section_spans',
                                          rows=np.array([0, 0]),
                                          cols=np.array([source_pair, source_pair+1]))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        num_sections = self.options['num_sections']
        num_nodes_per_segment = self.options['num_nodes_per_segment']
        num_segments = num_sections - 1
        num_nodes = num_segments * num_nodes_per_segment

        section_spans = inputs['section_spans']

        for source_pair in np.arange(num_segments):
            interpolated_spans = np.linspace(0, section_spans[source_pair+1]-section_spans[source_pair], num_nodes_per_segment + 2).T.reshape(-1)
            for interpolated_aerofoil in np.arange(1, num_nodes_per_segment+2):
                if interpolated_aerofoil == num_nodes_per_segment+1:
                    outputs[f'source_span_{source_pair}'] = section_spans[source_pair+1]
                else:
                    outputs[f'interpolated_span_{source_pair}_{interpolated_aerofoil-1}'] = interpolated_spans[interpolated_aerofoil]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        num_sections = self.options['num_sections']
        num_nodes_per_segment = self.options['num_nodes_per_segment']
        num_segments = num_sections - 1
        num_nodes = num_segments * num_nodes_per_segment

        section_spans = inputs['section_spans']

        for source_pair in np.arange(num_segments):
            interpolated_spans = np.linspace(0, section_spans[source_pair+1]-section_spans[source_pair], num_nodes_per_segment + 2).T.reshape(-1)
            for interpolated_aerofoil in np.arange(1, num_nodes_per_segment+1):
                    deriv = -interpolated_spans[interpolated_aerofoil]/(section_spans[source_pair+1])
                    partials[f'interpolated_span_{source_pair}_{interpolated_aerofoil-1}', 'section_spans'] = np.array([deriv, -deriv])


class DemuxTotalAngles(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_sections')
        self.options.declare('num_nodes_per_segment')

    def setup(self):
        num_sections = self.options['num_sections']
        num_nodes_per_segment = self.options['num_nodes_per_segment']
        num_segments = num_sections - 1
        num_nodes = num_segments * num_nodes_per_segment

        self.add_input('total_angles', shape=(num_nodes,), units='rad')

        for node in np.arange(num_nodes):
            self.add_output(f'total_angle_{node}', units='rad')
            self.declare_partials(f'total_angle_{node}', 'total_angles',
                                  rows=np.array([0]),
                                  cols=np.array([node]),
                                  val=1)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        num_sections = self.options['num_sections']
        num_nodes_per_segment = self.options['num_nodes_per_segment']
        num_segments = num_sections - 1
        num_nodes = num_segments * num_nodes_per_segment

        total_angles = inputs['total_angles']

        for node in np.arange(num_nodes):
            outputs[f'total_angle_{node}'] = total_angles[node]


class MuxLiftCoefficients(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_sections')
        self.options.declare('num_nodes_per_segment')

    def setup(self):
        num_sections = self.options['num_sections']
        num_nodes_per_segment = self.options['num_nodes_per_segment']
        num_segments = num_sections - 1
        num_nodes = num_segments * num_nodes_per_segment

        self.add_output('2d_lift_coefficients', shape=(num_nodes,))

        for node in np.arange(num_nodes):
            self.add_input(f'2d_lift_coefficient_{node}')
            self.declare_partials('2d_lift_coefficients', f'2d_lift_coefficient_{node}',
                                  rows=np.array([node]),
                                  cols=np.array([0]),
                                  val=1)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        num_sections = self.options['num_sections']
        num_nodes_per_segment = self.options['num_nodes_per_segment']
        num_segments = num_sections - 1
        num_nodes = num_segments * num_nodes_per_segment

        cls = np.array([inputs[f'2d_lift_coefficient_{node}'] for node in np.arange(num_nodes)])
        outputs['2d_lift_coefficients'] = cls.reshape(-1)


class InterpolateMultipleAerofoils(om.Group):

    def initialize(self):
        self.options.declare('num_sections')
        self.options.declare('num_nodes_per_segment')
        self.options.declare('num_points')

    def setup(self):
        num_sections = self.options['num_sections']
        num_nodes_per_segment = self.options['num_nodes_per_segment']
        num_points = self.options['num_points']
        num_segments = num_sections - 1

        # For each pair of original sections
            # For each interpolated foil between them
                # Add an InterpolateAerofoil component with sequentially numbered inputs/outputs

        self.add_subsystem('PassthroughSourceAerofoil_0',
                           om.ExecComp('interpolated_coordinates_0 = source_aerofoil_0', shape=(2, num_points), has_diag_partials=True),
                           promotes_inputs=['source_aerofoil_0'],
                           promotes_outputs=['interpolated_coordinates_0'])

        for source_pair in np.arange(num_segments):
            for interpolated_aerofoil in np.arange(1, num_nodes_per_segment+1):
                if interpolated_aerofoil == num_nodes_per_segment:
                    self.add_subsystem(f'PassthroughSourceAerofoil_{source_pair+1}',
                                       om.ExecComp(f'interpolated_coordinates_{(source_pair+1)*num_nodes_per_segment} = source_aerofoil_{source_pair+1}',
                                                   shape=(2, num_points), has_diag_partials=True),
                                       promotes_inputs=[f'source_aerofoil_{source_pair+1}'],
                                       promotes_outputs=[f'interpolated_coordinates_{(source_pair+1)*num_nodes_per_segment}'])
                else:
                    self.add_subsystem(f'InterpolateAerofoil_{source_pair}{interpolated_aerofoil}',
                                       InterpolateAerofoil(num_points=num_points),
                                       promotes_inputs=[('aerofoil_coordinates_A', f'source_aerofoil_{source_pair}'),
                                                        ('aerofoil_coordinates_B', f'source_aerofoil_{source_pair+1}'),
                                                        ('B_span', f'source_span_{source_pair}'),
                                                        ('interpolated_span', f'interpolated_span_{source_pair}_{interpolated_aerofoil}')],
                                       promotes_outputs=[('interpolated_coordinates', f'interpolated_coordinates_{source_pair*num_nodes_per_segment + interpolated_aerofoil}')])


class MultipleInviscid(om.Group):

    def initialize(self):
        self.options.declare('num_sections')
        self.options.declare('num_nodes_per_segment')
        self.options.declare('num_points')

    def setup(self):
        num_sections = self.options['num_sections']
        num_nodes_per_segment = self.options['num_nodes_per_segment']
        num_points = self.options['num_points']
        num_segments = num_sections-1
        num_nodes = num_segments*num_nodes_per_segment
        num_interpolated_sections = num_nodes + 1

        for node in np.arange(num_nodes):
            self.add_subsystem(f'InviscidGroup_{node}',
                               InviscidGroup(num_points=num_points),
                               promotes_inputs=[('alpha', f'total_angle_{node}'),
                                                ('aerofoil_ccw_coordinates', f'interpolated_coordinates_{node}')],
                               promotes_outputs=[('lift_coefficient', f'2d_lift_coefficient_{node}')])


class CoupledLLTQD(om.Group):

    def initialize(self):
        self.options.declare('num_sections')
        self.options.declare('num_nodes_per_segment')
        self.options.declare('num_points')

    def setup(self):
        num_sections = self.options['num_sections']
        num_nodes_per_segment = self.options['num_nodes_per_segment']
        num_points = self.options['num_points']

        interpolate_inputs = self.add_subsystem('InterpolateInputs', om.Group(), promotes=['*'])

        interpolate_inputs.add_subsystem('DemuxInterpolatedSpans',
                                         DemuxInterpolatedSpans(num_sections=num_sections, num_nodes_per_segment=num_nodes_per_segment),
                                         promotes_inputs=['section_spans'],
                                         promotes_outputs=['*'])

        interpolate_inputs.add_subsystem('InterpolateMultipleAerofoils',
                                         InterpolateMultipleAerofoils(num_points=num_points, num_sections=num_sections,
                                                                      num_nodes_per_segment=num_nodes_per_segment),
                                         promotes_inputs=['*'],
                                         promotes_outputs=['*'])

        analysis = self.add_subsystem('Analysis', om.Group(), promotes=['*'])

        analysis.add_subsystem('LiftingLineGroup',
                                       LiftingLineGroup(num_sections=num_sections, num_nodes_per_segment=num_nodes_per_segment),
                                       promotes_inputs=['alpha',
                                                        'density',
                                                        'freestream_velocity',
                                                        'section_chords',
                                                        'section_offsets',
                                                        'section_twists',
                                                        'section_spans',
                                                        'section_dihedrals',
                                                        '2d_lift_coefficients'],
                                       promotes_outputs=['spans',
                                                         'surface_areas',
                                                         'quarter_chords',
                                                         'total_angles',
                                                         'induced_velocities',
                                                         'vortex_strengths',
                                                         '3d_forces',
                                                         '2d_forces',
                                                         'lift',
                                                         'drag'])

        analysis.add_subsystem('DemuxTotalAngles',
                                       DemuxTotalAngles(num_sections=num_sections, num_nodes_per_segment=num_nodes_per_segment),
                                       promotes_inputs=['total_angles'],
                                       promotes_outputs=['*'])

        analysis.add_subsystem('MultipleInviscid',
                                       MultipleInviscid(num_points=num_points, num_sections=num_sections, num_nodes_per_segment=num_nodes_per_segment),
                                       promotes_inputs=['*'],
                                       promotes_outputs=['*'])

        analysis.add_subsystem('MuxLiftCoefficients',
                                        MuxLiftCoefficients(num_sections=num_sections, num_nodes_per_segment=num_nodes_per_segment),
                                        promotes_inputs=['*'],
                                        promotes_outputs=['2d_lift_coefficients'])

        analysis.linear_solver = om.DirectSolver()
        analysis.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, maxiter=10, iprint=2)