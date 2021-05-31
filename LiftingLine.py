import openmdao.api as om
import numpy as np


class FreestreamUnitVector(om.ExplicitComponent):

    def setup(self):
        self.add_input('alpha', units='rad')

        self.add_output('freestream_unit_vector', shape=(3,))

        self.declare_partials('freestream_unit_vector', 'alpha')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        alpha = inputs['alpha']

        outputs['freestream_unit_vector'] = np.array([np.cos(alpha), 0, np.sin(alpha)])

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        alpha = inputs['alpha']

        partials['freestream_unit_vector', 'alpha'] = np.array([-np.sin(alpha), 0, np.cos(alpha)])


class InterpolateGeometry(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_sections', types=int)
        self.options.declare('num_nodes_per_segment', types=int)
        self.options.declare('num_interpolated_sections', types=int)

    def setup(self):
        num_sections = self.options['num_sections']
        num_nodes_per_segment = self.options['num_nodes_per_segment']
        num_interpolated_sections = self.options['num_interpolated_sections']

        self.add_input('section_chords', shape=(num_sections,))
        self.add_input('section_offsets', shape=(num_sections,))
        self.add_input('section_twists', shape=(num_sections,), units='rad')
        self.add_input('section_spans', shape=(num_sections,))
        self.add_input('section_dihedrals', shape=(num_sections,), units='rad')

        self.add_output('chords', shape=(num_interpolated_sections,))
        self.add_output('offsets', shape=(num_interpolated_sections,))
        self.add_output('twists', shape=(num_interpolated_sections,))
        self.add_output('spans', shape=(num_interpolated_sections,))
        self.add_output('dihedrals', shape=(num_interpolated_sections,))

        rows = np.concatenate((np.arange(num_nodes_per_segment), (np.arange(1, 2*num_nodes_per_segment) + num_nodes_per_segment*np.arange(num_sections-2)[:, None]).reshape(-1), np.arange((num_sections-2)*num_nodes_per_segment+1, (num_sections-1)*num_nodes_per_segment+1)))
        cols = np.concatenate((np.repeat(0, num_nodes_per_segment), np.repeat(np.arange(1, num_sections-1), 2*num_nodes_per_segment-1), np.repeat(num_sections-1, num_nodes_per_segment)))
        a = np.linspace(0, 1, num_nodes_per_segment+1)[1:-1]
        b = np.linspace(1, 0, num_nodes_per_segment+1)[:-1]
        c = np.concatenate((a, b))
        val = np.concatenate((np.linspace(1, 0, num_nodes_per_segment+1)[:-1], np.tile(c, num_sections-2), np.linspace(0, 1, num_nodes_per_segment+1)[1:]))

        self.declare_partials('chords', 'section_chords',
                              rows=rows,
                              cols=cols,
                              val=val)
        self.declare_partials('offsets', 'section_offsets',
                              rows=np.arange(num_interpolated_sections),
                              cols=np.repeat(np.arange(num_sections), num_nodes_per_segment)[num_nodes_per_segment-1:],
                              val=1/num_nodes_per_segment)
        self.declare_partials('twists', 'section_twists',
                              rows=rows,
                              cols=cols,
                              val=val)
        self.declare_partials('spans', 'section_spans',
                              rows=np.arange(num_interpolated_sections),
                              cols=np.repeat(np.arange(num_sections), num_nodes_per_segment)[num_nodes_per_segment-1:],
                              val=1/num_nodes_per_segment)
        self.declare_partials('dihedrals', 'section_dihedrals',
                              rows=rows,
                              cols=cols,
                              val=val)

    def linearly_interpolate(self, a, n_per):
        return np.concatenate((a[0:1], np.linspace(a[:-1], a[1:], n_per+1)[1:, :].T.reshape(-1)))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        num_nodes_per_segment = self.options['num_nodes_per_segment']

        section_chords = inputs['section_chords']
        section_offsets = inputs['section_offsets']
        section_twists = inputs['section_twists']
        section_spans = inputs['section_spans']
        section_dihedrals = inputs['section_dihedrals']

        outputs['chords'] = self.linearly_interpolate(section_chords, num_nodes_per_segment)
        outputs['offsets'] = np.repeat(section_offsets/num_nodes_per_segment, num_nodes_per_segment)[num_nodes_per_segment-1:]  # Needs a slightly different treatment
        outputs['twists'] = self.linearly_interpolate(section_twists, num_nodes_per_segment)
        outputs['spans'] = np.repeat(section_spans/num_nodes_per_segment, num_nodes_per_segment)[num_nodes_per_segment-1:]  # Needs a slightly different treatment
        outputs['dihedrals'] = self.linearly_interpolate(section_dihedrals, num_nodes_per_segment)


class QuarterChordPositions(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_interpolated_sections', types=int)

    def setup(self):
        num_interpolated_sections = self.options['num_interpolated_sections']

        self.add_input('offsets', shape=(num_interpolated_sections,))
        self.add_input('spans', shape=(num_interpolated_sections,))
        self.add_input('dihedrals', shape=(num_interpolated_sections,))

        self.add_output('quarter_chords', shape=(num_interpolated_sections, 3))

        rows = np.repeat(3*np.arange(num_interpolated_sections), np.arange(num_interpolated_sections)+1)
        cols = np.tril(np.arange(num_interpolated_sections))[np.tril_indices(num_interpolated_sections)]

        self.declare_partials('quarter_chords', 'offsets',
                              rows=rows,
                              cols=np.tril(np.arange(num_interpolated_sections))[np.tril_indices(num_interpolated_sections)],
                              val=1)
        self.declare_partials('quarter_chords', 'spans',
                              rows=np.concatenate((rows+1, rows+2)),
                              cols=np.concatenate((cols, cols)))
        self.declare_partials('quarter_chords', 'dihedrals',
                              rows=np.concatenate((rows+1, rows+2)),
                              cols=np.concatenate((cols, cols)))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        offsets = inputs['offsets']
        spans = inputs['spans']
        dihedrals = inputs['dihedrals']

        x = np.cumsum(offsets)
        y = np.cumsum(spans * np.cos(dihedrals))
        z = np.cumsum(spans * np.sin(dihedrals))

        outputs['quarter_chords'] = np.dstack([x, y, z])

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        num_interpolated_sections = self.options['num_interpolated_sections']

        offsets = inputs['offsets']
        spans = inputs['spans']
        dihedrals = inputs['dihedrals']

        dqcy_dspans = np.tril(np.cos(dihedrals))[np.tril_indices(num_interpolated_sections)]
        dqcz_dspans = np.tril(np.sin(dihedrals))[np.tril_indices(num_interpolated_sections)]

        dqcy_ddihedrals = np.tril(-spans*np.sin(dihedrals))[np.tril_indices(num_interpolated_sections)]
        dqcz_ddihedrals = np.tril(spans*np.cos(dihedrals))[np.tril_indices(num_interpolated_sections)]

        partials['quarter_chords', 'spans'] = np.concatenate((dqcy_dspans, dqcz_dspans))
        partials['quarter_chords', 'dihedrals'] = np.concatenate((dqcy_ddihedrals, dqcz_ddihedrals))


class LeadingTrailingEdgePositions(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_interpolated_sections', types=int)

    def setup(self):
        num_interpolated_sections = self.options['num_interpolated_sections']

        self.add_input('chords', shape=(num_interpolated_sections,))
        self.add_input('twists', shape=(num_interpolated_sections,))
        self.add_input('dihedrals', shape=(num_interpolated_sections,))
        self.add_input('quarter_chords', shape=(num_interpolated_sections, 3))

        self.add_output('leading_edges', shape=(num_interpolated_sections, 3))
        self.add_output('trailing_edges', shape=(num_interpolated_sections, 3))

        rows = np.arange(3 * num_interpolated_sections)
        cols = np.repeat(np.arange(num_interpolated_sections), 3)

        self.declare_partials('leading_edges', 'chords',
                              rows=rows,
                              cols=cols)
        self.declare_partials('leading_edges', 'twists',
                              rows=rows,
                              cols=cols)
        self.declare_partials('leading_edges', 'dihedrals',
                              rows=rows,
                              cols=cols)
        self.declare_partials('leading_edges', 'quarter_chords',
                              rows=np.arange(3*num_interpolated_sections),
                              cols=np.arange(3*num_interpolated_sections),
                              val=1)
        self.declare_partials('trailing_edges', 'chords',
                              rows=rows,
                              cols=cols)
        self.declare_partials('trailing_edges', 'twists',
                              rows=rows,
                              cols=cols)
        self.declare_partials('trailing_edges', 'dihedrals',
                              rows=rows,
                              cols=cols)
        self.declare_partials('trailing_edges', 'quarter_chords',
                              rows=np.arange(3*num_interpolated_sections),
                              cols=np.arange(3*num_interpolated_sections),
                              val=1)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        num_interpolated_sections = self.options['num_interpolated_sections']

        chords = inputs['chords']
        twists = inputs['twists']
        dihedrals = inputs['dihedrals']
        quarter_chords = inputs['quarter_chords']

        x = quarter_chords[:, 0]
        y = quarter_chords[:, 1]
        z = quarter_chords[:, 2]

        c1 = np.cos(dihedrals)
        s1 = np.sin(dihedrals)
        c2 = np.cos(twists)
        s2 = np.sin(twists)

        o = np.zeros(num_interpolated_sections)  # Just makes it slightly easier to see symbolic 1 and 0 in the upcoming bit
        l = np.ones(num_interpolated_sections)

        T = np.transpose(np.array([[   c2,   o,    s2,  x],
                                   [ s1*s2, c1, -c2*s1, y],
                                   [-c1*s2, s1,  c1*c2, z],
                                   [   o,    o,    o,   l]]),
                         (2, 0, 1))  # Numpy matrix multiplication expects the first dimension to be the stacking dimension, so each T matrix is accessed as [i, :, :]

        T_le = np.transpose(np.array([[-chords/4],
                                      [o],
                                      [o],
                                      [l]]),
                            (2, 0, 1))

        T_te = np.transpose(np.array([[3*chords/4],
                                      [o],
                                      [o],
                                      [l]]),
                            (2, 0, 1))

        leading_edges = np.transpose(((T @ T_le)[:, :3, :]), (2, 0, 1))  # Take it down to just being the coordinates. Access as [i] to get [x, y, z]
        trailing_edges = np.transpose(((T @ T_te)[:, :3, :]), (2, 0, 1))

        outputs['leading_edges'] = np.squeeze(leading_edges)  # Get rid of empty dimensions left by matrix stuff
        outputs['trailing_edges'] = np.squeeze(trailing_edges)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        num_interpolated_sections = self.options['num_interpolated_sections']

        chords = inputs['chords']
        twists = inputs['twists']
        dihedrals = inputs['dihedrals']
        quarter_chords = inputs['quarter_chords']

        x = quarter_chords[:, 0]
        y = quarter_chords[:, 1]
        z = quarter_chords[:, 2]

        c1 = np.cos(dihedrals)
        s1 = np.sin(dihedrals)
        c2 = np.cos(twists)
        s2 = np.sin(twists)

        o = np.zeros(num_interpolated_sections)  # Just makes it slightly easier to see symbolic 1 and 0 in the upcoming bit
        l = np.ones(num_interpolated_sections)

        T = np.transpose(np.array([[   c2,   o,    s2,  x],
                                   [ s1*s2, c1, -c2*s1, y],
                                   [-c1*s2, s1,  c1*c2, z],
                                   [   o,    o,    o,   l]]),
                         (2, 0, 1))  # Numpy matrix multiplication expects the first dimension to be the stacking dimension, so each T matrix is accessed as [i, :, :]

        dT_ddihedrals = np.transpose(np.array([[   o,   o,    o,   o],
                                               [ c1*s2, -s1, -c2*c1, o],
                                               [ s1*s2, c1,  -s1*c2, o],
                                               [   o,    o,    o,    o]]),
                                     (2, 0, 1))

        dT_dtwists = np.transpose(np.array([[   -s2,   o,    c2,  o],
                                            [ s1*c2, o, s2*s1, o],
                                            [-c1*c2, o,  -c1*s2, o],
                                            [   o,    o,    o,   o]]),
                                  (2, 0, 1))

        T_le = np.transpose(np.array([[-chords/4],
                                      [o],
                                      [o],
                                      [l]]),
                            (2, 0, 1))

        T_te = np.transpose(np.array([[3*chords/4],
                                      [o],
                                      [o],
                                      [l]]),
                            (2, 0, 1))

        T_le_dc = np.transpose(np.array([[-l/4],
                                      [o],
                                      [o],
                                      [o]]),
                            (2, 0, 1))

        T_te_dc = np.transpose(np.array([[3*l/4],
                                      [o],
                                      [o],
                                      [o]]),
                            (2, 0, 1))

        dle_dc = np.transpose(((T @ T_le_dc)[:, :3, :]), (2, 0, 1)).reshape(-1)
        dte_dc = np.transpose(((T @ T_te_dc)[:, :3, :]), (2, 0, 1)).reshape(-1)

        dle_ddihedrals = np.transpose(((dT_ddihedrals @ T_le)[:, :3, :]), (2, 0, 1)).reshape(-1)
        dte_ddihedrals = np.transpose(((dT_ddihedrals @ T_te)[:, :3, :]), (2, 0, 1)).reshape(-1)

        dle_dtwists = np.transpose(((dT_dtwists @ T_le)[:, :3, :]), (2, 0, 1)).reshape(-1)
        dte_dtwists = np.transpose(((dT_dtwists @ T_te)[:, :3, :]), (2, 0, 1)).reshape(-1)

        partials['leading_edges', 'chords'] = dle_dc  # Get rid of empty dimensions left by matrix stuff
        partials['trailing_edges', 'chords'] = dte_dc
        partials['leading_edges', 'dihedrals'] = dle_ddihedrals
        partials['trailing_edges', 'dihedrals'] = dte_ddihedrals
        partials['leading_edges', 'twists'] = dle_dtwists
        partials['trailing_edges', 'twists'] = dte_dtwists


class MidpointPositions(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('num_interpolated_sections', types=int)

    def setup(self):
        num_interpolated_sections = self.options['num_interpolated_sections']
        num_nodes = self.options['num_nodes']

        self.add_input('leading_edges', shape=(num_interpolated_sections, 3))
        self.add_input('quarter_chords', shape=(num_interpolated_sections, 3))

        self.add_output('midpoint_leading_edges', shape=(num_nodes, 3))
        self.add_output('midpoint_quarter_chords', shape=(num_nodes, 3))
        self.add_output('midpoint_quarter_chords_sym', shape=(num_nodes, 3))

        self.declare_partials('midpoint_leading_edges', 'leading_edges',
                              rows=np.tile(np.arange(3*num_nodes), 2),
                              cols=np.concatenate((np.arange(3*(num_interpolated_sections-1)), np.arange(3, 3*num_interpolated_sections))),
                              val=0.5)
        self.declare_partials('midpoint_quarter_chords', 'quarter_chords',
                              rows=np.tile(np.arange(3*num_nodes), 2),
                              cols=np.concatenate((np.arange(3*(num_interpolated_sections-1)), np.arange(3, 3*num_interpolated_sections))),
                              val=0.5)
        self.declare_partials('midpoint_quarter_chords_sym', 'quarter_chords',
                              rows=np.tile(np.arange(3*num_nodes), 2),
                              cols=np.concatenate((np.arange(3*(num_interpolated_sections-1)), np.arange(3, 3*num_interpolated_sections))),
                              val=np.tile(np.array((0.5, -0.5, 0.5)), 2*num_nodes))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        leading_edges = inputs['leading_edges']
        quarter_chords = inputs['quarter_chords']

        midpoint_leading_edges = (leading_edges[1:, :] + leading_edges[:-1, :])/2
        midpoint_quarter_chords = (quarter_chords[1:, :] + quarter_chords[:-1, :])/2

        outputs['midpoint_leading_edges'] = midpoint_leading_edges
        outputs['midpoint_quarter_chords'] = midpoint_quarter_chords
        outputs['midpoint_quarter_chords_sym'] = midpoint_quarter_chords*np.array([1, -1, 1])  # Mirrors across xz for later use


class BoundVortexVectors(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('num_interpolated_sections', types=int)

    def setup(self):
        num_interpolated_sections = self.options['num_interpolated_sections']
        num_nodes = self.options['num_nodes']

        self.add_input('quarter_chords', shape=(num_interpolated_sections, 3))

        self.add_output('bound_vortices', shape=(num_nodes, 3))

        self.declare_partials('bound_vortices', 'quarter_chords',
                              rows=np.tile(np.arange(3*num_nodes), 2),
                              cols=np.concatenate((np.arange(3*(num_interpolated_sections-1)), np.arange(3, 3*num_interpolated_sections))),
                              val=np.concatenate((-np.ones(3*num_nodes), np.ones(3*num_nodes))))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        quarter_chords = inputs['quarter_chords']

        outputs['bound_vortices'] = quarter_chords[1:, :] - quarter_chords[:-1, :]


class SurfaceVectors(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']

        self.add_input('bound_vortices', shape=(num_nodes, 3))  # I can be slick and reuse this since control points are on bound vortex
        self.add_input('midpoint_leading_edges', shape=(num_nodes, 3))
        self.add_input('midpoint_quarter_chords', shape=(num_nodes, 3))

        self.add_output('unnormalised_chordwise_vectors', shape=(num_nodes, 3))
        self.add_output('unnormalised_normal_vectors', shape=(num_nodes, 3))

        self.declare_partials('unnormalised_chordwise_vectors', 'midpoint_quarter_chords',
                              rows=np.arange(3*num_nodes),
                              cols=np.arange(3*num_nodes),
                              val=1)
        self.declare_partials('unnormalised_chordwise_vectors', 'midpoint_leading_edges',
                              rows=np.arange(3*num_nodes),
                              cols=np.arange(3*num_nodes),
                              val=-1)
        self.declare_partials('unnormalised_normal_vectors', 'midpoint_quarter_chords',
                              rows=np.repeat(np.arange(0, 3*num_nodes), 3).reshape(-1),
                              cols=np.tile(np.arange(3*num_nodes).reshape(-1, 3), 3).reshape(-1))
        self.declare_partials('unnormalised_normal_vectors', 'midpoint_leading_edges',
                              rows=np.repeat(np.arange(0, 3*num_nodes), 3).reshape(-1),
                              cols=np.tile(np.arange(3*num_nodes).reshape(-1, 3), 3).reshape(-1))
        self.declare_partials('unnormalised_normal_vectors', 'bound_vortices',
                              rows=np.repeat(np.arange(0, 3*num_nodes), 3).reshape(-1),
                              cols=np.tile(np.arange(3*num_nodes).reshape(-1, 3), 3).reshape(-1))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        spanwise_vectors = inputs['bound_vortices']
        midpoint_leading_edges = inputs['midpoint_leading_edges']
        midpoint_quarter_chords = inputs['midpoint_quarter_chords']

        chordwise_vectors = (midpoint_quarter_chords-midpoint_leading_edges)
        normal_vectors = np.cross(chordwise_vectors, spanwise_vectors)

        outputs['unnormalised_chordwise_vectors'] = chordwise_vectors
        outputs['unnormalised_normal_vectors'] = normal_vectors

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        spanwise_vectors = inputs['bound_vortices']
        midpoint_leading_edges = inputs['midpoint_leading_edges']
        midpoint_quarter_chords = inputs['midpoint_quarter_chords']

        chordwise_vectors = midpoint_quarter_chords-midpoint_leading_edges

        cx = chordwise_vectors[:, 0]
        cy = chordwise_vectors[:, 1]
        cz = chordwise_vectors[:, 2]
        sx = spanwise_vectors[:, 0]
        sy = spanwise_vectors[:, 1]
        sz = spanwise_vectors[:, 2]

        o = np.zeros_like(cx)

        c_cross = np.array([[o, -cz, cy],
                            [cz, o, -cx],
                            [-cy, cx, o]])

        s_cross = np.array([[o, -sz, sy],
                            [sz, o, -sx],
                            [-sy, sx, o]])

        dn_dc = np.transpose(-s_cross, (2, 0, 1)).reshape(-1)
        dn_ds = np.transpose(c_cross, (2, 0, 1)).reshape(-1)

        partials['unnormalised_chordwise_vectors', 'midpoint_quarter_chords'] = 1
        partials['unnormalised_chordwise_vectors', 'midpoint_leading_edges'] = -1

        partials['unnormalised_normal_vectors', 'midpoint_quarter_chords'] = dn_dc
        partials['unnormalised_normal_vectors', 'midpoint_leading_edges'] = -dn_dc
        partials['unnormalised_normal_vectors', 'bound_vortices'] = dn_ds


class NormaliseSurfaceVectors(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']

        self.add_input('unnormalised_chordwise_vectors', shape=(num_nodes, 3))
        self.add_input('unnormalised_normal_vectors', shape=(num_nodes, 3))

        self.add_output('chordwise_vectors', shape=(num_nodes, 3))
        self.add_output('normal_vectors', shape=(num_nodes, 3))

        self.declare_partials('chordwise_vectors', 'unnormalised_chordwise_vectors',
                              rows=np.repeat(np.arange(0, 3*num_nodes), 3).reshape(-1),
                              cols=np.tile(np.arange(3*num_nodes).reshape(-1, 3), 3).reshape(-1))
        self.declare_partials('normal_vectors', 'unnormalised_normal_vectors',
                              rows=np.repeat(np.arange(0, 3*num_nodes), 3).reshape(-1),
                              cols=np.tile(np.arange(3*num_nodes).reshape(-1, 3), 3).reshape(-1))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        unnormalised_chordwise_vectors = inputs['unnormalised_chordwise_vectors']
        unnormalised_normal_vectors = inputs['unnormalised_normal_vectors']

        chordwise_vectors = unnormalised_chordwise_vectors / np.linalg.norm(unnormalised_chordwise_vectors, axis=1)[:, None]  # Normalise by length
        normal_vectors = unnormalised_normal_vectors / np.linalg.norm(unnormalised_normal_vectors, axis=1)[:, None]

        outputs['chordwise_vectors'] = chordwise_vectors
        outputs['normal_vectors'] = normal_vectors

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        unnormalised_chordwise_vectors = inputs['unnormalised_chordwise_vectors']
        unnormalised_normal_vectors = inputs['unnormalised_normal_vectors']

        chordwise_vectors = unnormalised_chordwise_vectors / np.linalg.norm(unnormalised_chordwise_vectors, axis=1)[:, None]  # Normalise by length
        normal_vectors = unnormalised_normal_vectors / np.linalg.norm(unnormalised_normal_vectors, axis=1)[:, None]  # Normalise by length

        cx = chordwise_vectors[:, 0]
        cy = chordwise_vectors[:, 1]
        cz = chordwise_vectors[:, 2]
        clen = np.linalg.norm(unnormalised_chordwise_vectors, axis=1)
        dxcn_dxc = (cz**2+cy**2)/clen
        dycn_dyc = (cz**2+cx**2)/clen
        dzcn_dzc = (cx**2+cy**2)/clen
        dxcn_dyc = -cx*cy/clen
        dxcn_dzc = -cx*cz/clen
        dycn_dzc = -cy*cz/clen

        dcnorm_dc = np.dstack([dxcn_dxc, dxcn_dyc, dxcn_dzc, dxcn_dyc, dycn_dyc, dycn_dzc, dxcn_dzc, dycn_dzc, dzcn_dzc]).reshape(-1)

        nx = normal_vectors[:, 0]
        ny = normal_vectors[:, 1]
        nz = normal_vectors[:, 2]
        nlen = np.linalg.norm(unnormalised_normal_vectors, axis=1)
        dxnn_dxn = (nz**2+ny**2)/nlen
        dynn_dyn = (nz**2+nx**2)/nlen
        dznn_dzn = (nx**2+ny**2)/nlen
        dxnn_dyn = -nx*ny/nlen
        dxnn_dzn = -nx*nz/nlen
        dynn_dzn = -ny*nz/nlen

        dnnorm_dn = np.dstack([dxnn_dxn, dxnn_dyn, dxnn_dzn, dxnn_dyn, dynn_dyn, dynn_dzn, dxnn_dzn, dynn_dzn, dznn_dzn]).reshape(-1)

        partials['chordwise_vectors', 'unnormalised_chordwise_vectors'] = dcnorm_dc
        partials['normal_vectors', 'unnormalised_normal_vectors'] = dnnorm_dn


class CreateSAVectors(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('num_interpolated_sections', types=int)

    def setup(self):
        num_interpolated_sections = self.options['num_interpolated_sections']
        num_nodes = self.options['num_nodes']

        self.add_input('leading_edges', shape=(num_interpolated_sections, 3))
        self.add_input('trailing_edges', shape=(num_interpolated_sections, 3))

        self.add_output('v_delta', shape=(num_nodes, 3))
        self.add_output('v2', shape=(num_nodes, 3))
        self.add_output('v3', shape=(num_nodes, 3))

        self.declare_partials('v_delta', 'leading_edges',
                              rows=np.tile(np.arange(3*num_nodes), 2),
                              cols=np.concatenate((np.arange(3*num_nodes), np.arange(3*num_nodes)+3)),
                              val=np.concatenate((np.ones(3*num_nodes), -np.ones(3*num_nodes))))
        self.declare_partials('v_delta', 'trailing_edges',
                              rows=np.tile(np.arange(3*num_nodes), 2),
                              cols=np.concatenate((np.arange(3*num_nodes), np.arange(3*num_nodes)+3)),
                              val=np.concatenate((-np.ones(3*num_nodes), np.ones(3*num_nodes))))
        self.declare_partials('v2', 'leading_edges',
                              rows=np.arange(3*num_nodes),
                              cols=np.arange(3*num_nodes),
                              val=-1)
        self.declare_partials('v2', 'trailing_edges',
                              rows=np.arange(3*num_nodes),
                              cols=np.arange(3*num_nodes),
                              val=1)
        self.declare_partials('v3', 'leading_edges',
                              rows=np.tile(np.arange(3*num_nodes), 2),
                              cols=np.concatenate((np.arange(3*num_nodes), np.arange(3*num_nodes)+3)),
                              val=np.concatenate((-np.ones(3*num_nodes), np.ones(3*num_nodes))))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        leading_edges = inputs['leading_edges']
        trailing_edges = inputs['trailing_edges']

        # Glärbo (https://math.stackexchange.com/users/892839/gl%c3%a4rbo), Calculate surface normal and area for a non-planar quadrilateral, URL (version: 2021-03-03): https://math.stackexchange.com/q/4047694

        a1 = leading_edges[:-1, :]
        a2 = trailing_edges[:-1, :]
        b1 = leading_edges[1:, :]
        b2 = trailing_edges[1:, :]

        outputs['v_delta'] = a1 - a2 - b1 + b2
        outputs['v2'] = a2 - a1
        outputs['v3'] = b1 - a1


class SACrossProducts(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('num_interpolated_sections', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']

        self.add_input('v_delta', shape=(num_nodes, 3))
        self.add_input('v2', shape=(num_nodes, 3))
        self.add_input('v3', shape=(num_nodes, 3))

        self.add_output('vdcv2', shape=(num_nodes, 3))
        self.add_output('vdcv3', shape=(num_nodes, 3))
        self.add_output('v2cvd', shape=(num_nodes, 3))
        self.add_output('v2cv3', shape=(num_nodes, 3))

        rows = np.repeat(np.arange(0, 3 * num_nodes), 3).reshape(-1)
        cols = np.tile(np.arange(3 * num_nodes).reshape(-1, 3), 3).reshape(-1)

        self.declare_partials('vdcv2', 'v_delta',
                              rows=rows,
                              cols=cols)
        self.declare_partials('vdcv2', 'v2',
                              rows=rows,
                              cols=cols)
        self.declare_partials('v2cvd', 'v_delta',
                              rows=rows,
                              cols=cols)
        self.declare_partials('v2cvd', 'v2',
                              rows=rows,
                              cols=cols)
        self.declare_partials('vdcv3', 'v_delta',
                              rows=rows,
                              cols=cols)
        self.declare_partials('vdcv3', 'v3',
                              rows=rows,
                              cols=cols)
        self.declare_partials('v2cv3', 'v2',
                              rows=rows,
                              cols=cols)
        self.declare_partials('v2cv3', 'v3',
                              rows=rows,
                              cols=cols)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        v_delta = inputs['v_delta']
        v2 = inputs['v2']
        v3 = inputs['v3']

        vdcv2 = np.cross(v_delta, v2)
        vdcv3 = np.cross(v_delta, v3)
        v2cvd = -vdcv2
        v2cv3 = np.cross(v2, v3)

        outputs['vdcv2'] = vdcv2
        outputs['vdcv3'] = vdcv3
        outputs['v2cvd'] = v2cvd
        outputs['v2cv3'] = v2cv3
        
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        v_delta = inputs['v_delta']
        v2 = inputs['v2']
        v3 = inputs['v3']
        
        v_deltax = v_delta[:, 0]
        v_deltay = v_delta[:, 1]
        v_deltaz = v_delta[:, 2]
        v2x = v2[:, 0]
        v2y = v2[:, 1]
        v2z = v2[:, 2]
        v3x = v3[:, 0]
        v3y = v3[:, 1]
        v3z = v3[:, 2]
        
        o = np.zeros_like(v2x)

        v_delta_cross = np.array([[o, -v_deltaz, v_deltay],
                                  [v_deltaz, o, -v_deltax],
                                  [-v_deltay, v_deltax, o]])

        v2_cross = np.array([[o, -v2z, v2y],
                             [v2z, o, -v2x],
                             [-v2y, v2x, o]])

        v3_cross = np.array([[o, -v3z, v3y],
                             [v3z, o, -v3x],
                             [-v3y, v3x, o]])

        partials['vdcv2', 'v_delta'] = np.transpose(-v2_cross, (2, 0, 1)).reshape(-1)
        partials['vdcv2', 'v2'] = np.transpose(v_delta_cross, (2, 0, 1)).reshape(-1)
        partials['v2cvd', 'v_delta'] = -np.transpose(-v2_cross, (2, 0, 1)).reshape(-1)
        partials['v2cvd', 'v2'] = -np.transpose(v_delta_cross, (2, 0, 1)).reshape(-1)
        partials['vdcv3', 'v_delta'] = np.transpose(-v3_cross, (2, 0, 1)).reshape(-1)
        partials['vdcv3', 'v3'] = np.transpose(v_delta_cross, (2, 0, 1)).reshape(-1)
        partials['v2cv3', 'v2'] = np.transpose(-v3_cross, (2, 0, 1)).reshape(-1)
        partials['v2cv3', 'v3'] = np.transpose(v2_cross, (2, 0, 1)).reshape(-1)


class SurfaceAreaCoefficients(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('num_interpolated_sections', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']

        self.add_input('vdcv2', shape=(num_nodes, 3))
        self.add_input('vdcv3', shape=(num_nodes, 3))
        self.add_input('v2cvd', shape=(num_nodes, 3))
        self.add_input('v2cv3', shape=(num_nodes, 3))

        self.add_output('Cuu', shape=(1, 1, num_nodes))
        self.add_output('Cvv', shape=(1, 1, num_nodes))
        self.add_output('Cuv', shape=(1, 1, num_nodes))
        self.add_output('Cu', shape=(1, 1, num_nodes))
        self.add_output('Cv', shape=(1, 1, num_nodes))
        self.add_output('C0', shape=(1, 1, num_nodes))

        rows = np.repeat(np.arange(num_nodes), 3)
        cols = np.arange(3 * num_nodes)

        self.declare_partials('Cuu', 'vdcv2',
                              rows=rows,
                              cols=cols)
        self.declare_partials('Cvv', 'vdcv3',
                              rows=rows,
                              cols=cols)
        self.declare_partials('Cuv', 'vdcv3',
                              rows=rows,
                              cols=cols)
        self.declare_partials('Cuv', 'v2cvd',
                              rows=rows,
                              cols=cols)
        self.declare_partials('Cu', 'v2cvd',
                              rows=rows,
                              cols=cols)
        self.declare_partials('Cu', 'v2cv3',
                              rows=rows,
                              cols=cols)
        self.declare_partials('Cv', 'vdcv3',
                              rows=rows,
                              cols=cols)
        self.declare_partials('Cv', 'v2cv3',
                              rows=rows,
                              cols=cols)
        self.declare_partials('C0', 'v2cv3',
                              rows=rows,
                              cols=cols)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        vdcv2 = inputs['vdcv2']
        vdcv3 = inputs['vdcv3']
        v2cvd = inputs['v2cvd']
        v2cv3 = inputs['v2cv3']

        outputs['Cuu'] = np.sum(vdcv2**2, axis=1)[None, None, :]  # Add a third dimension for later
        outputs['Cvv'] = np.sum(vdcv3**2, axis=1)[None, None, :]
        outputs['Cuv'] = np.sum(vdcv3*v2cvd, axis=1)[None, None, :]
        outputs['Cu'] = np.sum(v2cvd*v2cv3, axis=1)[None, None, :]
        outputs['Cv'] = np.sum(vdcv3*v2cv3, axis=1)[None, None, :]
        outputs['C0'] = np.sum(v2cv3**2, axis=1)[None, None, :]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        vdcv2 = inputs['vdcv2'].reshape(-1)
        vdcv3 = inputs['vdcv3'].reshape(-1)
        v2cvd = inputs['v2cvd'].reshape(-1)
        v2cv3 = inputs['v2cv3'].reshape(-1)

        partials['Cuu', 'vdcv2'] = 2*vdcv2
        partials['Cvv', 'vdcv3'] = 2*vdcv3
        partials['Cuv', 'vdcv3'] = v2cvd
        partials['Cuv', 'v2cvd'] = vdcv3
        partials['Cu', 'v2cvd'] = v2cv3
        partials['Cu', 'v2cv3'] = v2cvd
        partials['Cv', 'vdcv3'] = v2cv3
        partials['Cv', 'v2cv3'] = vdcv3
        partials['C0', 'v2cv3'] = 2*v2cv3


class SurfaceAreas(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('num_interpolated_sections', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']

        self.add_input('Cuu', shape=(1, 1, num_nodes))
        self.add_input('Cvv', shape=(1, 1, num_nodes))
        self.add_input('Cuv', shape=(1, 1, num_nodes))
        self.add_input('Cu', shape=(1, 1, num_nodes))
        self.add_input('Cv', shape=(1, 1, num_nodes))
        self.add_input('C0', shape=(1, 1, num_nodes))

        self.add_output('surface_areas', shape=(num_nodes,))

        rows = np.arange(num_nodes)
        cols = np.arange(num_nodes)

        self.declare_partials('surface_areas', 'Cuu',
                              rows=rows,
                              cols=cols)
        self.declare_partials('surface_areas', 'Cvv',
                              rows=rows,
                              cols=cols)
        self.declare_partials('surface_areas', 'Cuv',
                              rows=rows,
                              cols=cols)
        self.declare_partials('surface_areas', 'Cu',
                              rows=rows,
                              cols=cols)
        self.declare_partials('surface_areas', 'Cv',
                              rows=rows,
                              cols=cols)
        self.declare_partials('surface_areas', 'C0',
                              rows=rows,
                              cols=cols)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        Cuu = inputs['Cuu']
        Cvv = inputs['Cvv']
        Cuv = inputs['Cuv']
        Cu = inputs['Cu']
        Cv = inputs['Cv']
        C0 = inputs['C0']

        u_integrand_v0 = (np.sqrt(Cuu + 2*Cu + C0) + np.sqrt(C0))/2
        u_integrand_v1 = (np.sqrt(Cuu + 2*Cuv + Cvv + 2*Cu + 2*Cv + C0) + np.sqrt(Cvv + 2*Cv + C0))/2

        areas = (u_integrand_v0+u_integrand_v1)/2

        outputs['surface_areas'] = areas

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        Cuu = inputs['Cuu']
        Cvv = inputs['Cvv']
        Cuv = inputs['Cuv']
        Cu = inputs['Cu']
        Cv = inputs['Cv']
        C0 = inputs['C0']

        partials['surface_areas', 'Cuu'] = ((1/(2*np.sqrt(Cuu+Cvv+2*Cv+2*Cuv+2*Cu+C0))+1/(2*np.sqrt(Cuu+2*Cu+C0)))/4).reshape(-1)
        partials['surface_areas', 'Cvv'] = ((1/(2*np.sqrt(Cvv+2*Cv+2*Cuv+Cuu+2*Cu+C0))+1/(2*np.sqrt(Cvv+2*Cv+C0)))/4).reshape(-1)
        partials['surface_areas', 'Cuv'] = (1/(4*np.sqrt(2*Cuv+Cvv+2*Cv+Cuu+2*Cu+C0))).reshape(-1)
        partials['surface_areas', 'Cu'] = ((1/np.sqrt(2*Cu+Cvv+2*Cv+2*Cuv+Cuu+C0)+1/np.sqrt(2*Cu+Cuu+C0))/4).reshape(-1)
        partials['surface_areas', 'Cv'] = ((1/np.sqrt(2*Cv+Cvv+2*Cuv+Cuu+2*Cu+C0)+1/np.sqrt(2*Cv+Cvv+C0))/4).reshape(-1)
        partials['surface_areas', 'C0'] = ((1/(2*np.sqrt(C0+Cvv+2*Cv+2*Cuv+Cuu+2*Cu))+1/(2*np.sqrt(C0+Cvv+2*Cv))+1/(2*np.sqrt(C0+Cuu+2*Cu))+1/(2*np.sqrt(C0)))/4).reshape(-1)


class VortexToControlVectors(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('num_interpolated_sections', types=int)

    def setup(self):
        num_interpolated_sections = self.options['num_interpolated_sections']
        num_nodes = self.options['num_nodes']

        self.add_input('quarter_chords', shape=(num_interpolated_sections, 3))
        self.add_input('midpoint_quarter_chords', shape=(num_nodes, 3))

        self.add_output('r1', shape=(num_nodes, num_nodes, 3))
        self.add_output('r2', shape=(num_nodes, num_nodes, 3))

        self.declare_partials('r1', 'quarter_chords',
                              rows=np.arange(3*num_nodes**2),
                              cols=np.tile(np.arange(3*num_interpolated_sections-3), num_nodes),
                              val=-1)
        self.declare_partials('r1', 'midpoint_quarter_chords',
                              rows=np.arange(3*num_nodes**2),
                              cols=np.tile(np.arange(3*num_interpolated_sections-3).reshape(-1, 3), num_nodes).reshape(-1),
                              val=1)
        self.declare_partials('r2', 'quarter_chords',
                              rows=np.arange(3*num_nodes**2),
                              cols=np.tile(np.arange(3, 3*num_interpolated_sections), num_nodes),
                              val=-1)
        self.declare_partials('r2', 'midpoint_quarter_chords',
                              rows=np.arange(3*num_nodes**2),
                              cols=np.tile(np.arange(3*num_interpolated_sections-3).reshape(-1, 3), num_nodes).reshape(-1),
                              val=1)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        quarter_chords = inputs['quarter_chords']
        midpoint_quarter_chords = inputs['midpoint_quarter_chords']

        first_quarter_chord = quarter_chords[:-1, :]
        second_quarter_chord = quarter_chords[1:, :]

        outputs['r1'] = (midpoint_quarter_chords[:, None, :] - first_quarter_chord[None, :, :])
        outputs['r2'] = (midpoint_quarter_chords[:, None, :] - second_quarter_chord[None, :, :])


class InfluenceCoefficientsVectorMaths(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']

        self.add_input('r1', shape=(num_nodes, num_nodes, 3))
        self.add_input('r2', shape=(num_nodes, num_nodes, 3))
        self.add_input('freestream_unit_vector', shape=(3,))

        self.add_output('r1_len', shape=(num_nodes, num_nodes))
        self.add_output('r2_len', shape=(num_nodes, num_nodes))
        self.add_output('vcr1', shape=(num_nodes, num_nodes, 3))
        self.add_output('vcr2', shape=(num_nodes, num_nodes, 3))
        self.add_output('r1cr2', shape=(num_nodes, num_nodes, 3))
        self.add_output('vdr1', shape=(num_nodes, num_nodes))
        self.add_output('vdr2', shape=(num_nodes, num_nodes))
        self.add_output('r1dr2', shape=(num_nodes, num_nodes))

        dot_rows = np.repeat(np.arange(num_nodes**2), 3)
        dot_cols = np.arange(3*num_nodes**2)
        cross_rows = np.repeat(np.arange(0, 3 * num_nodes**2), 3).reshape(-1)
        cross_cols = np.tile(np.arange(3 * num_nodes**2).reshape(-1, 3), 3).reshape(-1)

        self.declare_partials('r1_len', 'r1',
                              rows=dot_rows,
                              cols=dot_cols)
        self.declare_partials('r2_len', 'r2',
                              rows=dot_rows,
                              cols=dot_cols)
        self.declare_partials('vcr1', 'freestream_unit_vector',
                              rows=np.repeat(np.arange(3*num_nodes**2), 3),
                              cols=np.tile(np.arange(3), 3*num_nodes**2))
        self.declare_partials('vcr1', 'r1',
                              rows=cross_rows,
                              cols=cross_cols)
        self.declare_partials('vcr2', 'freestream_unit_vector')
        self.declare_partials('vcr2', 'r2',
                              rows=cross_rows,
                              cols=cross_cols)
        self.declare_partials('r1cr2', 'r1',
                              rows=cross_rows,
                              cols=cross_cols)
        self.declare_partials('r1cr2', 'r2',
                              rows=cross_rows,
                              cols=cross_cols)
        self.declare_partials('vdr1', 'freestream_unit_vector',
                              rows=np.repeat(np.arange(num_nodes**2), 3),
                              cols=np.tile(np.arange(3), num_nodes**2))
        self.declare_partials('vdr1', 'r1',
                              rows=dot_rows,
                              cols=dot_cols)
        self.declare_partials('vdr2', 'freestream_unit_vector',
                              rows=np.repeat(np.arange(num_nodes**2), 3),
                              cols=np.tile(np.arange(3), num_nodes**2))
        self.declare_partials('vdr2', 'r2',
                              rows=dot_rows,
                              cols=dot_cols)
        self.declare_partials('r1dr2', 'r1',
                              rows=dot_rows,
                              cols=dot_cols)
        self.declare_partials('r1dr2', 'r2',
                              rows=dot_rows,
                              cols=dot_cols)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        r1 = inputs['r1']
        r2 = inputs['r2']
        freestream_unit_vector = inputs['freestream_unit_vector']

        r1_len = np.linalg.norm(r1, axis=2)
        r2_len = np.linalg.norm(r2, axis=2)

        vcr1 = np.cross(freestream_unit_vector, r1)
        vcr2 = np.cross(freestream_unit_vector, r2)
        r1cr2 = np.cross(r1, r2)

        vdr1 = np.sum(freestream_unit_vector * r1, axis=2)
        vdr2 = np.sum(freestream_unit_vector * r2, axis=2)
        r1dr2 = np.sum(r1 * r2, axis=2)

        outputs['r1_len'] = r1_len
        outputs['r2_len'] = r2_len
        outputs['vcr1'] = vcr1
        outputs['vcr2'] = vcr2
        outputs['r1cr2'] = r1cr2
        outputs['vdr1'] = vdr1
        outputs['vdr2'] = vdr2
        outputs['r1dr2'] = r1dr2

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        num_nodes = self.options['num_nodes']

        r1 = inputs['r1']
        r2 = inputs['r2']
        freestream_unit_vector = inputs['freestream_unit_vector']

        r1_len = np.linalg.norm(r1, axis=2)
        r2_len = np.linalg.norm(r2, axis=2)

        r1x = r1[:, :, 0]
        r1y = r1[:, :, 1]
        r1z = r1[:, :, 2]
        r2x = r2[:, :, 0]
        r2y = r2[:, :, 1]
        r2z = r2[:, :, 2]
        vx = freestream_unit_vector[0]
        vy = freestream_unit_vector[1]
        vz = freestream_unit_vector[2]

        o = np.zeros_like(r1x)

        r1_cross = np.array([[o, -r1z, r1y],
                             [r1z, o, -r1x],
                             [-r1y, r1x, o]])

        r2_cross = np.array([[o, -r2z, r2y],
                             [r2z, o, -r2x],
                             [-r2y, r2x, o]])

        v_cross = np.array([[0, -vz, vy],
                            [vz, 0, -vx],
                            [-vy, vx, 0]])

        partials['r1_len', 'r1'] = (r1/r1_len[:, :, None]).reshape(-1)
        partials['r2_len', 'r2'] = (r2/r2_len[:, :, None]).reshape(-1)
        partials['vcr1', 'freestream_unit_vector'] = np.transpose(-r1_cross, (2, 3, 0, 1)).reshape(-1)
        partials['vcr1', 'r1'] = np.tile(v_cross.reshape(-1), num_nodes**2)
        partials['vcr2', 'freestream_unit_vector'] = np.transpose(-r2_cross, (2, 3, 0, 1)).reshape(-1)
        partials['vcr2', 'r2'] = np.tile(v_cross.reshape(-1), num_nodes**2)
        partials['r1cr2', 'r1'] = np.transpose(-r2_cross, (2, 3, 0, 1)).reshape(-1)
        partials['r1cr2', 'r2'] = np.transpose(r1_cross, (2, 3, 0, 1)).reshape(-1)
        partials['vdr1', 'freestream_unit_vector'] = r1.reshape(-1)
        partials['vdr1', 'r1'] = np.tile(freestream_unit_vector, num_nodes**2)
        partials['vdr2', 'freestream_unit_vector'] = r2.reshape(-1)
        partials['vdr2', 'r2'] = np.tile(freestream_unit_vector, num_nodes**2)
        partials['r1dr2', 'r1'] = r2.reshape(-1)
        partials['r1dr2', 'r2'] = r1.reshape(-1)


class VortexSegmentInfluence(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']

        self.add_input('r1_len', shape=(num_nodes, num_nodes))
        self.add_input('r2_len', shape=(num_nodes, num_nodes))
        self.add_input('vcr1', shape=(num_nodes, num_nodes, 3))
        self.add_input('vcr2', shape=(num_nodes, num_nodes, 3))
        self.add_input('r1cr2', shape=(num_nodes, num_nodes, 3))
        self.add_input('vdr1', shape=(num_nodes, num_nodes))
        self.add_input('vdr2', shape=(num_nodes, num_nodes))
        self.add_input('r1dr2', shape=(num_nodes, num_nodes))

        self.add_output('trailing_vortex_influences_1', shape=(num_nodes, num_nodes, 3))
        self.add_output('trailing_vortex_influences_2', shape=(num_nodes, num_nodes, 3))
        self.add_output('bound_vortex_influences', shape=(num_nodes, num_nodes, 3))

        self.declare_partials('trailing_vortex_influences_1', 'vcr2',
                              rows=np.arange(3*num_nodes**2),
                              cols=np.arange(3*num_nodes**2))
        self.declare_partials('trailing_vortex_influences_1', 'r2_len',
                              rows=np.arange(3*num_nodes**2),
                              cols=np.repeat(np.arange(num_nodes**2), 3))
        self.declare_partials('trailing_vortex_influences_1', 'vdr2',
                              rows=np.arange(3*num_nodes**2),
                              cols=np.repeat(np.arange(num_nodes**2), 3))
        self.declare_partials('trailing_vortex_influences_2', 'vcr1',
                              rows=np.arange(3*num_nodes**2),
                              cols=np.arange(3*num_nodes**2))
        self.declare_partials('trailing_vortex_influences_2', 'r1_len',
                              rows=np.arange(3*num_nodes**2),
                              cols=np.repeat(np.arange(num_nodes**2), 3))
        self.declare_partials('trailing_vortex_influences_2', 'vdr1',
                              rows=np.arange(3*num_nodes**2),
                              cols=np.repeat(np.arange(num_nodes**2), 3))
        self.declare_partials('bound_vortex_influences', 'r1cr2',
                              rows=np.arange(3*num_nodes**2),
                              cols=np.arange(3*num_nodes**2))
        self.declare_partials('bound_vortex_influences', 'r1_len',
                              rows=np.arange(3*num_nodes**2),
                              cols=np.repeat(np.arange(num_nodes**2), 3))
        self.declare_partials('bound_vortex_influences', 'r2_len',
                              rows=np.arange(3*num_nodes**2),
                              cols=np.repeat(np.arange(num_nodes**2), 3))
        self.declare_partials('bound_vortex_influences', 'r1dr2',
                              rows=np.arange(3*num_nodes**2),
                              cols=np.repeat(np.arange(num_nodes**2), 3))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        num_nodes = self.options['num_nodes']

        r1_len = inputs['r1_len']
        r2_len = inputs['r2_len']
        vcr1 = inputs['vcr1']
        vcr2 = inputs['vcr2']
        r1cr2 = inputs['r1cr2']
        vdr1 = inputs['vdr1']
        vdr2 = inputs['vdr2']
        r1dr2 = inputs['r1dr2']

        trailing_vortex_influences_1 = (vcr2/((r2_len*(r2_len - vdr2))[:, :, None]))
        trailing_vortex_influences_2 = -(vcr1/((r1_len*(r1_len - vdr1))[:, :, None]))
        with np.errstate(divide='ignore', invalid='ignore'):  # Hides away the warnings about dividing by zero which happen because control point is coincident with bound vortex
            bound_vortex_influences = r1cr2*((r1_len + r2_len) / (r1_len*r2_len*(r1_len*r2_len + r1dr2)))[:, :, None]
            bound_vortex_influences[np.arange(num_nodes), np.arange(num_nodes), :] = 0  # Replace the bad values with 0, equivalent to saying no influence from bound vortex on coincident control point

        outputs['trailing_vortex_influences_1'] = trailing_vortex_influences_1
        outputs['trailing_vortex_influences_2'] = trailing_vortex_influences_2
        outputs['bound_vortex_influences'] = bound_vortex_influences

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        num_nodes = self.options['num_nodes']

        r1_len = inputs['r1_len'][:, :, None]  # For broadcasting
        r2_len = inputs['r2_len'][:, :, None]
        vcr1 = inputs['vcr1']
        vcr2 = inputs['vcr2']
        r1cr2 = inputs['r1cr2']
        vdr1 = inputs['vdr1'][:, :, None]
        vdr2 = inputs['vdr2'][:, :, None]
        r1dr2 = inputs['r1dr2'][:, :, None]

        with np.errstate(divide='ignore', invalid='ignore'):  # Hides away the warnings about dividing by zero which happen because control point is coincident with bound vortex
            dtrail1_dvcr2 = 1/(r2_len*(r2_len - vdr2))
            dtrail1_dr2len = -(vcr2*(2*r2_len - vdr2))/(r2_len*(r2_len - vdr2))**2
            dtrail1_dvdr2 = vcr2/(r2_len*(r2_len - vdr2)**2)

            dtrail2_dvcr1 = -1/(r1_len*(r1_len - vdr1))
            dtrail2_dr1len = (vcr1*(2*r1_len - vdr1))/(r1_len*(r1_len - vdr1))**2
            dtrail2_dvdr1 = -vcr1/(r1_len*(r1_len - vdr1)**2)

            dbound_dr1cr2 = ((r1_len + r2_len) / (r1_len*r2_len*(r1_len*r2_len + r1dr2)))
            dbound_dr1cr2[np.arange(num_nodes), np.arange(num_nodes), :] = 0
            dbound_dr1len = -(r1cr2*(r1_len**2 + 2*r2_len*r1_len + r1dr2))/(r1_len*(r2_len*r1_len + r1dr2))**2
            dbound_dr1len[np.arange(num_nodes), np.arange(num_nodes), :] = 0
            dbound_dr2len = -(r1cr2*(r2_len**2 + 2*r2_len*r1_len + r1dr2))/(r2_len*(r2_len*r1_len + r1dr2))**2
            dbound_dr2len[np.arange(num_nodes), np.arange(num_nodes), :] = 0
            dbound_dr1dr2 = -(r1cr2*(r2_len+r1_len))/(r1_len*r2_len*(r1dr2 + r1_len*r2_len)**2)
            dbound_dr1dr2[np.arange(num_nodes), np.arange(num_nodes), :] = 0

        partials['trailing_vortex_influences_1', 'vcr2'] = np.repeat(dtrail1_dvcr2, 3).reshape(-1)
        partials['trailing_vortex_influences_1', 'r2_len'] = dtrail1_dr2len.reshape(-1)
        partials['trailing_vortex_influences_1', 'vdr2'] = dtrail1_dvdr2.reshape(-1)
        partials['trailing_vortex_influences_2', 'vcr1'] = np.repeat(dtrail2_dvcr1, 3).reshape(-1)
        partials['trailing_vortex_influences_2', 'r1_len'] = dtrail2_dr1len.reshape(-1)
        partials['trailing_vortex_influences_2', 'vdr1'] = dtrail2_dvdr1.reshape(-1)
        partials['bound_vortex_influences', 'r1cr2'] = np.repeat(dbound_dr1cr2, 3).reshape(-1)
        partials['bound_vortex_influences', 'r1_len'] = dbound_dr1len.reshape(-1)
        partials['bound_vortex_influences', 'r2_len'] = dbound_dr2len.reshape(-1)
        partials['bound_vortex_influences', 'r1dr2'] = dbound_dr1dr2.reshape(-1)


class CombineVortexSegmentInfluence(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']

        self.add_input('trailing_vortex_influences_1', shape=(num_nodes, num_nodes, 3))
        self.add_input('trailing_vortex_influences_2', shape=(num_nodes, num_nodes, 3))
        self.add_input('bound_vortex_influences', shape=(num_nodes, num_nodes, 3))

        self.add_output('influence_coefficients', shape=(num_nodes, num_nodes, 3))

        self.declare_partials('influence_coefficients', 'trailing_vortex_influences_1',
                              rows=np.arange(3*num_nodes**2),
                              cols=np.arange(3*num_nodes**2),
                              val=1/(4*np.pi))
        self.declare_partials('influence_coefficients', 'trailing_vortex_influences_2',
                              rows=np.arange(3*num_nodes**2),
                              cols=np.arange(3*num_nodes**2),
                              val=1/(4*np.pi))
        self.declare_partials('influence_coefficients', 'bound_vortex_influences',
                              rows=np.arange(3*num_nodes**2),
                              cols=np.arange(3*num_nodes**2),
                              val=1/(4*np.pi))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        trailing_vortex_influences_1 = inputs['trailing_vortex_influences_1']
        trailing_vortex_influences_2 = inputs['trailing_vortex_influences_2']
        bound_vortex_influences = inputs['bound_vortex_influences']

        v = (1/(4*np.pi)) * (trailing_vortex_influences_1 + trailing_vortex_influences_2 + bound_vortex_influences)
        outputs['influence_coefficients'] = v


class InfluenceCoefficients(om.Group):

    def initialize(self):
        self.options.declare('num_interpolated_sections')
        self.options.declare('num_nodes')

    def setup(self):
        num_interpolated_sections = self.options['num_interpolated_sections']
        num_nodes = self.options['num_nodes']

        self.add_subsystem('VortexToControlVectors', VortexToControlVectors(num_nodes=num_nodes, num_interpolated_sections=num_interpolated_sections),
                           promotes_inputs=['quarter_chords',
                                            'midpoint_quarter_chords'],
                           promotes_outputs=['r1',
                                             'r2'])

        self.add_subsystem('InfluenceCoefficientsVectorMaths', InfluenceCoefficientsVectorMaths(num_nodes=num_nodes),
                           promotes_inputs=['r1',
                                            'r2',
                                            'freestream_unit_vector'],
                           promotes_outputs=['r1_len',
                                             'r2_len',
                                             'vcr1',
                                             'vcr2',
                                             'r1cr2',
                                             'vdr1',
                                             'vdr2',
                                             'r1dr2'])
        self.add_subsystem('VortexSegmentInfluence', VortexSegmentInfluence(num_nodes=num_nodes),
                           promotes_inputs=['r1_len',
                                            'r2_len',
                                            'vcr1',
                                            'vcr2',
                                            'r1cr2',
                                            'vdr1',
                                            'vdr2',
                                            'r1dr2'],
                           promotes_outputs=['trailing_vortex_influences_1',
                                             'trailing_vortex_influences_2',
                                             'bound_vortex_influences'])

        self.add_subsystem('CombineVortexSegmentInfluence', CombineVortexSegmentInfluence(num_nodes=num_nodes),
                           promotes_inputs=['trailing_vortex_influences_1',
                                            'trailing_vortex_influences_2',
                                            'bound_vortex_influences'],
                           promotes_outputs=['influence_coefficients'])


class CombineInfluenceCoefficients(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']

        self.add_input('influence_coefficients', shape=(num_nodes, num_nodes, 3))
        self.add_input('influence_coefficients_sym', shape=(num_nodes, num_nodes, 3))

        self.add_output('influence_coefficients_total', shape=(num_nodes, num_nodes, 3))

        self.declare_partials('influence_coefficients_total', 'influence_coefficients',
                              rows=np.arange(3*num_nodes**2),
                              cols=np.arange(3*num_nodes**2),
                              val=1)
        self.declare_partials('influence_coefficients_total', 'influence_coefficients_sym',
                              rows=np.arange(3*num_nodes**2),
                              cols=np.arange(3*num_nodes**2),
                              val=np.tile(np.array([1, -1, 1]), num_nodes**2))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        influence_coefficients = inputs['influence_coefficients']
        influence_coefficients_sym = inputs['influence_coefficients_sym']

        outputs['influence_coefficients_total'] = influence_coefficients + (influence_coefficients_sym*np.array([1, -1, 1]))  # (u, v, w) = (u_reg + u_sym, u_reg - u_sym, v_reg + v_sym)


class ApplySymmetryCondition(om.Group):

    def initialize(self):
        self.options.declare('num_interpolated_sections')
        self.options.declare('num_nodes')

    def setup(self):
        num_interpolated_sections = self.options['num_interpolated_sections']
        num_nodes = self.options['num_nodes']

        self.add_subsystem('InfluenceCoefficients', InfluenceCoefficients(num_nodes=num_nodes, num_interpolated_sections=num_interpolated_sections),
                           promotes_inputs=['quarter_chords',
                                            'midpoint_quarter_chords',
                                            'freestream_unit_vector'],
                           promotes_outputs=['influence_coefficients'])
        self.add_subsystem('SymInfluenceCoefficients', InfluenceCoefficients(num_nodes=num_nodes, num_interpolated_sections=num_interpolated_sections),
                           promotes_inputs=['quarter_chords',
                                            ('midpoint_quarter_chords', 'midpoint_quarter_chords_sym'),
                                            'freestream_unit_vector'],
                           promotes_outputs=[('influence_coefficients', 'influence_coefficients_sym')])

        self.add_subsystem('CombineInfluenceCoefficients', CombineInfluenceCoefficients(num_nodes=num_nodes),
                           promotes_inputs=['influence_coefficients',
                                            'influence_coefficients_sym'],
                           promotes_outputs=['influence_coefficients_total'])


class InducedVelocities(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']

        self.add_input('vortex_strengths', shape=(num_nodes,))
        self.add_input('influence_coefficients_total', shape=(num_nodes, num_nodes, 3))
        self.add_input('freestream_unit_vector', shape=(3,))
        self.add_input('freestream_velocity')

        self.add_output('induced_velocities', shape=(num_nodes, 3))

        self.declare_partials('induced_velocities', 'vortex_strengths',
                              rows=np.tile(np.arange(3*num_nodes).reshape(-1, 3), num_nodes).reshape(-1),
                              cols=np.tile(np.repeat(np.arange(num_nodes), 3), num_nodes))
        self.declare_partials('induced_velocities', 'influence_coefficients_total',
                              rows=np.tile(np.arange(3*num_nodes).reshape(-1, 3), num_nodes).reshape(-1),
                              cols=np.arange(3*num_nodes**2))
        self.declare_partials('induced_velocities', 'freestream_unit_vector',
                              rows=np.arange(3*num_nodes),
                              cols=np.tile(np.arange(3), num_nodes))
        self.declare_partials('induced_velocities', 'freestream_velocity')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        vortex_strengths = inputs['vortex_strengths']
        influence_coefficients_total = inputs['influence_coefficients_total']
        freestream_unit_vector = inputs['freestream_unit_vector']
        freestream_velocity = inputs['freestream_velocity']

        outputs['induced_velocities'] = freestream_velocity*freestream_unit_vector + np.sum(vortex_strengths[:, None]*influence_coefficients_total, axis=1)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        num_nodes = self.options['num_nodes']

        vortex_strengths = inputs['vortex_strengths']
        influence_coefficients_total = inputs['influence_coefficients_total']
        freestream_unit_vector = inputs['freestream_unit_vector']
        freestream_velocity = inputs['freestream_velocity']

        partials['induced_velocities', 'vortex_strengths'] = influence_coefficients_total.reshape(-1)
        partials['induced_velocities', 'influence_coefficients_total'] = np.tile(np.repeat(vortex_strengths, 3), num_nodes)
        partials['induced_velocities', 'freestream_unit_vector'] = np.repeat(freestream_velocity, 3*num_nodes)
        partials['induced_velocities', 'freestream_velocity'] = np.tile(freestream_unit_vector, num_nodes)


class TotalAngles(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']

        self.add_input('induced_velocities', shape=(num_nodes, 3))
        self.add_input('chordwise_vectors', shape=(num_nodes, 3))
        self.add_input('normal_vectors', shape=(num_nodes, 3))

        self.add_output('total_angles', shape=(num_nodes,), units='rad')

        rows = np.repeat(np.arange(num_nodes), 3)
        cols = np.arange(3*num_nodes)

        self.declare_partials('total_angles', 'induced_velocities',
                              rows=rows,
                              cols=cols)
        self.declare_partials('total_angles', 'chordwise_vectors',
                              rows=rows,
                              cols=cols)
        self.declare_partials('total_angles', 'normal_vectors',
                              rows=rows,
                              cols=cols)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        induced_velocities = inputs['induced_velocities']
        chordwise_vectors = inputs['chordwise_vectors']
        normal_vectors = inputs['normal_vectors']

        vdotn = np.sum(induced_velocities*normal_vectors, axis=1)
        vdotc = np.sum(induced_velocities*chordwise_vectors, axis=1)

        outputs['total_angles'] = np.arctan(vdotn/vdotc)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        induced_velocities = inputs['induced_velocities']
        chordwise_vectors = inputs['chordwise_vectors']
        normal_vectors = inputs['normal_vectors']

        vdotn = np.sum(induced_velocities*normal_vectors, axis=1)[:, None]  # For broadcasting
        vdotc = np.sum(induced_velocities*chordwise_vectors, axis=1)[:, None]

        denom = ((vdotn**2/vdotc**2) + 1)

        partials['total_angles', 'induced_velocities'] = ((normal_vectors/vdotc - chordwise_vectors*(vdotn/vdotc**2))/denom).reshape(-1)
        partials['total_angles', 'chordwise_vectors'] = (-(induced_velocities*(vdotn/vdotc**2))/denom).reshape(-1)
        partials['total_angles', 'normal_vectors'] = ((induced_velocities/vdotc)/denom).reshape(-1)


class TwoDForces(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']

        self.add_input('density')
        self.add_input('induced_velocities', shape=(num_nodes, 3))
        self.add_input('surface_areas', shape=(num_nodes,))
        self.add_input('2d_lift_coefficients', shape=(num_nodes,))

        self.add_output('2d_forces', shape=(num_nodes,))

        rows = np.arange(num_nodes)
        cols = np.arange(num_nodes)

        self.declare_partials('2d_forces', 'density')
        self.declare_partials('2d_forces', 'induced_velocities',
                              rows=np.repeat(rows, 3),
                              cols=np.arange(3*num_nodes))
        self.declare_partials('2d_forces', 'surface_areas',
                              rows=rows,
                              cols=cols)
        self.declare_partials('2d_forces', '2d_lift_coefficients',
                              rows=rows,
                              cols=cols)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        density = inputs['density']
        induced_velocities = inputs['induced_velocities']
        surface_areas = inputs['surface_areas']
        cls = inputs['2d_lift_coefficients']

        mag_v = np.linalg.norm(induced_velocities, axis=1)
        outputs['2d_forces'] = 0.5*density*(mag_v**2)*surface_areas*cls

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        density = inputs['density']
        induced_velocities = inputs['induced_velocities']
        surface_areas = inputs['surface_areas']
        cls = inputs['2d_lift_coefficients']

        mag_v = np.linalg.norm(induced_velocities, axis=1)
        partials['2d_forces', 'density'] = (0.5*(mag_v**2)*surface_areas*cls).reshape(-1)
        partials['2d_forces', 'induced_velocities'] = density*(induced_velocities*(surface_areas*cls)[:, None]).reshape(-1)
        partials['2d_forces', 'surface_areas'] = (0.5*density*(mag_v**2)*cls).reshape(-1)
        partials['2d_forces', '2d_lift_coefficients'] = (0.5*density*(mag_v**2)*surface_areas).reshape(-1)


class ThreeDForces(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']

        self.add_input('density')
        self.add_input('induced_velocities', shape=(num_nodes, 3))
        self.add_input('vortex_strengths', shape=(num_nodes,))
        self.add_input('bound_vortices', shape=(num_nodes, 3))

        self.add_output('3d_forces', shape=(num_nodes, 3))

        self.declare_partials('3d_forces', 'density')
        self.declare_partials('3d_forces', 'induced_velocities',
                              rows=np.repeat(np.arange(0, 3*num_nodes), 3).reshape(-1),
                              cols=np.tile(np.arange(3 * num_nodes).reshape(-1, 3), 3).reshape(-1))
        self.declare_partials('3d_forces', 'vortex_strengths',
                              rows=np.arange(3*num_nodes),
                              cols=np.repeat(np.arange(num_nodes), 3))
        self.declare_partials('3d_forces', 'bound_vortices',
                              rows=np.repeat(np.arange(0, 3*num_nodes), 3).reshape(-1),
                              cols=np.tile(np.arange(3 * num_nodes).reshape(-1, 3), 3).reshape(-1))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        density = inputs['density']
        induced_velocities = inputs['induced_velocities']
        vortex_strengths = inputs['vortex_strengths']
        bound_vortices = inputs['bound_vortices']

        outputs['3d_forces'] = density*np.cross(vortex_strengths[:, None]*induced_velocities, bound_vortices)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        density = inputs['density']
        induced_velocities = inputs['induced_velocities']
        vortex_strengths = inputs['vortex_strengths']
        bound_vortices = inputs['bound_vortices']

        V = vortex_strengths[:, None]*induced_velocities
        Vx = V[:, 0]
        Vy = V[:, 1]
        Vz = V[:, 2]
        o = np.zeros_like(Vx)

        V_cross = np.array([[o, -Vz, Vy],
                            [Vz, o, -Vx],
                            [-Vy, Vx, o]])

        bx = bound_vortices[:, 0]
        by = bound_vortices[:, 1]
        bz = bound_vortices[:, 2]

        b_cross = np.array([[o, -bz, by],
                            [bz, o, -bx],
                            [-by, bx, o]])

        partials['3d_forces', 'density'] = np.cross(vortex_strengths[:, None]*induced_velocities, bound_vortices).reshape(-1)
        partials['3d_forces', 'induced_velocities'] = -(density*vortex_strengths[:, None, None]*np.transpose(b_cross, (2, 0, 1))).reshape(-1)
        partials['3d_forces', 'vortex_strengths'] = (density*np.cross(induced_velocities, bound_vortices)).reshape(-1)
        partials['3d_forces', 'bound_vortices'] = np.transpose((density*V_cross), (2, 0, 1)).reshape(-1)


class ImplicitLiftingLine(om.ImplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']

        self.add_input('2d_forces', shape=(num_nodes,))
        self.add_input('3d_forces', shape=(num_nodes, 3))
        self.add_input('normal_vectors', shape=(num_nodes, 3))
        self.add_input('total_angles', shape=(num_nodes,), units='rad')

        self.add_output('vortex_strengths', np.ones(num_nodes), shape=(num_nodes,))

    def setup_partials(self):
        num_nodes = self.options['num_nodes']

        self.declare_partials('vortex_strengths', '2d_forces',
                              rows=np.arange(num_nodes),
                              cols=np.arange(num_nodes))
        self.declare_partials('vortex_strengths', '3d_forces',
                              rows=np.repeat(np.arange(num_nodes), 3),
                              cols=np.arange(3*num_nodes))
        self.declare_partials('vortex_strengths', 'normal_vectors',
                              rows=np.repeat(np.arange(num_nodes), 3),
                              cols=np.arange(3*num_nodes))
        self.declare_partials('vortex_strengths', 'total_angles',
                              rows=np.arange(num_nodes),
                              cols=np.arange(num_nodes))

    def apply_nonlinear(self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None):
        F2D = inputs['2d_forces']
        F3D = inputs['3d_forces']
        normal_vectors = inputs['normal_vectors']
        total_angle = inputs['total_angles']

        residuals['vortex_strengths'] = np.sum(F3D*normal_vectors, axis=1) - F2D*np.cos(total_angle)

    def linearize(self, inputs, outputs, jacobian, discrete_inputs=None, discrete_outputs=None):
        F3D = inputs['3d_forces']
        F2D = inputs['2d_forces']
        normal_vectors = inputs['normal_vectors']
        total_angle = inputs['total_angles']

        jacobian['vortex_strengths', '3d_forces'] = normal_vectors.reshape(-1)
        jacobian['vortex_strengths', 'normal_vectors'] = F3D.reshape(-1)
        jacobian['vortex_strengths', 'total_angles'] = F2D*np.sin(total_angle)
        jacobian['vortex_strengths', '2d_forces'] = -np.cos(total_angle)


class LiftDrag(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']

        self.add_input('3d_forces', shape=(num_nodes, 3))
        self.add_input('alpha', units='rad')

        self.add_output('lift')
        self.add_output('drag')

        self.declare_partials('lift', '3d_forces')
        self.declare_partials('lift', 'alpha')
        self.declare_partials('drag', '3d_forces')
        self.declare_partials('drag', 'alpha')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        F3D = inputs['3d_forces']
        alpha = inputs['alpha']

        force_sum = np.sum(F3D, axis=0)
        outputs['lift'] = -2*np.sin(alpha)*force_sum[0] + 2*np.cos(alpha)*force_sum[2]
        outputs['drag'] = 2*np.cos(alpha)*force_sum[0] + 2*np.sin(alpha)*force_sum[2]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        num_nodes = self.options['num_nodes']

        F3D = inputs['3d_forces']
        alpha = inputs['alpha']

        force_sum = np.sum(F3D, axis=0)
        partials['lift', '3d_forces'] = np.tile(np.dstack((-2*np.sin(alpha), 0, 2*np.cos(alpha))).reshape(-1), num_nodes)
        partials['lift', 'alpha'] = -2*np.cos(alpha)*force_sum[0] - 2*np.sin(alpha)*force_sum[2]
        partials['drag', '3d_forces'] = np.tile(np.dstack((2*np.cos(alpha), 0, 2*np.sin(alpha))).reshape(-1), num_nodes)
        partials['drag', 'alpha'] = -2*np.sin(alpha)*force_sum[0] + 2*np.cos(alpha)*force_sum[2]


class LiftingLineGroup(om.Group):

    def initialize(self):
        self.options.declare('num_sections')
        self.options.declare('num_nodes_per_segment')

    def setup(self):
        num_sections = self.options['num_sections']
        num_nodes_per_segment = self.options['num_nodes_per_segment']
        num_segments = num_sections-1
        num_nodes = num_segments*num_nodes_per_segment
        num_interpolated_sections = num_nodes + 1

        self.add_subsystem('FreestreamUnitVector', FreestreamUnitVector(),
                           promotes_inputs=['alpha'],
                           promotes_outputs=['freestream_unit_vector'])
        self.add_subsystem('InterpolateGeometry', InterpolateGeometry(num_sections=num_sections, num_interpolated_sections=num_interpolated_sections, num_nodes_per_segment=num_nodes_per_segment),
                           promotes_inputs=['section_chords', 
                                            'section_offsets', 
                                            'section_twists', 
                                            'section_spans', 
                                            'section_dihedrals'],
                           promotes_outputs=['chords',
                                             'offsets',
                                             'twists',
                                             'spans',
                                             'dihedrals'])
        self.add_subsystem('QuarterChordPositions', QuarterChordPositions(num_interpolated_sections=num_interpolated_sections),
                           promotes_inputs=['offsets',
                                            'spans',
                                            'dihedrals'],
                           promotes_outputs=['quarter_chords'])
        self.add_subsystem('LeadingTrailingEdgePositions', LeadingTrailingEdgePositions(num_interpolated_sections=num_interpolated_sections),
                           promotes_inputs=['chords',
                                            'twists',
                                            'dihedrals',
                                            'quarter_chords'],
                           promotes_outputs=['leading_edges',
                                             'trailing_edges'])
        self.add_subsystem('MidpointPositions', MidpointPositions(num_nodes=num_nodes, num_interpolated_sections=num_interpolated_sections),
                           promotes_inputs=['leading_edges',
                                            'quarter_chords'],
                           promotes_outputs=['midpoint_leading_edges',
                                             'midpoint_quarter_chords',
                                             'midpoint_quarter_chords_sym'])
        self.add_subsystem('BoundVortexVectors', BoundVortexVectors(num_nodes=num_nodes, num_interpolated_sections=num_interpolated_sections),
                           promotes_inputs=['quarter_chords'],
                           promotes_outputs=['bound_vortices'])
        self.add_subsystem('SurfaceVectors', SurfaceVectors(num_nodes=num_nodes),
                           promotes_inputs=['bound_vortices',
                                            'midpoint_leading_edges',
                                            'midpoint_quarter_chords'],
                           promotes_outputs=['unnormalised_chordwise_vectors',
                                             'unnormalised_normal_vectors'])
        self.add_subsystem('NormaliseSurfaceVectors', NormaliseSurfaceVectors(num_nodes=num_nodes),
                           promotes_inputs=['unnormalised_chordwise_vectors',
                                            'unnormalised_normal_vectors'],
                           promotes_outputs=['chordwise_vectors',
                                             'normal_vectors'])

        self.add_subsystem('CreateSAVectors', CreateSAVectors(num_nodes=num_nodes, num_interpolated_sections=num_interpolated_sections),
                           promotes_inputs=['leading_edges',
                                            'trailing_edges'],
                           promotes_outputs=['v_delta',
                                             'v2',
                                             'v3'])
        self.add_subsystem('SACrossProducts', SACrossProducts(num_nodes=num_nodes, num_interpolated_sections=num_interpolated_sections),
                           promotes_inputs=['v_delta',
                                            'v2',
                                            'v3'],
                           promotes_outputs=['vdcv2',
                                             'vdcv3',
                                             'v2cvd',
                                             'v2cv3'])
        self.add_subsystem('SurfaceAreaCoefficients', SurfaceAreaCoefficients(num_nodes=num_nodes, num_interpolated_sections=num_interpolated_sections),
                           promotes_inputs=['vdcv2',
                                            'vdcv3',
                                            'v2cvd',
                                            'v2cv3'],
                           promotes_outputs=['Cuu',
                                             'Cvv',
                                             'Cuv',
                                             'Cu',
                                             'Cv',
                                             'C0'])
        self.add_subsystem('SurfaceAreas', SurfaceAreas(num_nodes=num_nodes, num_interpolated_sections=num_interpolated_sections),
                           promotes_inputs=['Cuu',
                                            'Cvv',
                                            'Cuv',
                                            'Cu',
                                            'Cv',
                                            'C0'],
                           promotes_outputs=['surface_areas'])

        self.add_subsystem('ApplySymmetryCondition', ApplySymmetryCondition(num_nodes=num_nodes, num_interpolated_sections=num_interpolated_sections),
                           promotes_inputs=['quarter_chords',
                                            'midpoint_quarter_chords',
                                            'midpoint_quarter_chords_sym',
                                            'freestream_unit_vector'],
                           promotes_outputs=['influence_coefficients_total'])

        self.add_subsystem('InducedVelocities', InducedVelocities(num_nodes=num_nodes),
                           promotes_inputs=['vortex_strengths',
                                            'influence_coefficients_total',
                                            'freestream_unit_vector',
                                            'freestream_velocity'],
                           promotes_outputs=['induced_velocities'])
        self.add_subsystem('TotalAngles', TotalAngles(num_nodes=num_nodes),
                           promotes_inputs=['induced_velocities',
                                            'chordwise_vectors',
                                            'normal_vectors'],
                           promotes_outputs=['total_angles'])

        self.add_subsystem('TwoDForces', TwoDForces(num_nodes=num_nodes),
                           promotes_inputs=['density',
                                            'induced_velocities',
                                            'surface_areas',
                                            '2d_lift_coefficients'],
                           promotes_outputs=['2d_forces'])
        self.add_subsystem('ThreeDForces', ThreeDForces(num_nodes=num_nodes),
                           promotes_inputs=['density',
                                            'induced_velocities',
                                            'vortex_strengths',
                                            'bound_vortices'],
                           promotes_outputs=['3d_forces'])

        self.add_subsystem('ImplicitLiftingLine', ImplicitLiftingLine(num_nodes=num_nodes),
                           promotes_inputs=['2d_forces',
                                            '3d_forces',
                                            'normal_vectors',
                                            'total_angles'],
                           promotes_outputs=['vortex_strengths'])

        self.add_subsystem('LiftDrag', LiftDrag(num_nodes=num_nodes),
                           promotes_inputs=['3d_forces',
                                            'alpha'],
                           promotes_outputs=['lift',
                                             'drag'])


# FOR TESTING

class TestCl(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']

        self.add_input('total_angles', np.zeros(num_nodes), shape=(num_nodes,), units='rad')

        self.add_output('2d_lift_coefficients', np.zeros(num_nodes), shape=(num_nodes,))

        self.declare_partials('2d_lift_coefficients', 'total_angles',
                              rows=np.arange(num_nodes),
                              cols=np.arange(num_nodes),
                              val=2 * np.pi)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        total_angles = inputs['total_angles']

        outputs['2d_lift_coefficients'] = 2 * np.pi * total_angles


class LiftingLineTestGroup(om.Group):  # FOR TESTING

    def initialize(self):
        self.options.declare('num_sections')
        self.options.declare('num_nodes_per_segment')

    def setup(self):
        num_sections = self.options['num_sections']
        num_nodes_per_segment = self.options['num_nodes_per_segment']
        num_segments = num_sections - 1
        num_nodes = num_segments * num_nodes_per_segment
        num_interpolated_sections = num_nodes + 1

        self.add_subsystem('TestCl',
                           TestCl(num_nodes=num_nodes),
                           promotes_inputs=['total_angles'],
                           promotes_outputs=['2d_lift_coefficients'])

        self.add_subsystem('LiftingLineGroup',
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
                           promotes_outputs=['total_angles',
                                             'surface_areas',
                                             'induced_velocities',
                                             'vortex_strengths',
                                             '3d_forces',
                                             '2d_forces',
                                             'lift',
                                             'drag'])

        self.options['assembled_jac_type'] = 'csc'
        self.linear_solver = om.DirectSolver(assemble_jac=True)
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, maxiter=10, iprint=2, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # DEFINE WING
    mass = 20
    g = 9.81

    alpha = 10

    num_secs = 2
    num_nodes_per_seg = 20

    p = om.Problem(model=om.Group())

    p.model.add_subsystem('LiftingLineTestGroup',
                          LiftingLineTestGroup(num_sections=num_secs, num_nodes_per_segment=num_nodes_per_seg),
                          promotes_inputs=['alpha',
                                           'density',
                                           'freestream_velocity',
                                           'section_chords',
                                           'section_offsets',
                                           'section_twists',
                                           'section_spans',
                                           'section_dihedrals'],
                          promotes_outputs=['total_angles',
                                            'induced_velocities',
                                            'vortex_strengths',
                                            '3d_forces',
                                            '2d_forces',
                                            '2d_lift_coefficients',
                                            'lift',
                                            'drag'])


    section_chords = np.array([0.18, 0.11])
    section_offsets = np.array([0, 0.7])
    section_twists = np.array([3, -2])
    section_dihedrals = np.array([0, 0])
    section_spans = np.array([0, 1])

    p.model.set_input_defaults('alpha', alpha, units='deg')
    p.model.set_input_defaults('density', 1.225)
    p.model.set_input_defaults('section_chords', section_chords)
    p.model.set_input_defaults('section_offsets', section_offsets)
    p.model.set_input_defaults('section_twists', section_twists, units='deg')
    p.model.set_input_defaults('section_spans', section_spans)
    p.model.set_input_defaults('section_dihedrals', section_dihedrals, units='deg')

    p.model.set_input_defaults('freestream_velocity', 50)

    # p.model.set_input_defaults('alpha', 3, units='deg')
    # p.model.set_input_defaults('density', 1.225)
    # p.model.set_input_defaults('freestream_velocity', 50)
    # p.model.set_input_defaults('section_chords', np.array([1, 0.9, 0.8, 0.7, 0.6]))
    # p.model.set_input_defaults('section_offsets', np.array([0, 0.1, 0.3, 0.45, 0.4]))
    # p.model.set_input_defaults('section_twists', np.array([5, 3, 0, -2, -3]), units='deg')
    # p.model.set_input_defaults('section_spans', np.array([0, 2, 2, 0.5, 0.25]))
    # p.model.set_input_defaults('section_dihedrals', np.array([0, 0, 5, 45, 60]), units='deg')

    p.setup()
    p.run_model()

    # with np.printoptions(linewidth=2024, edgeitems=20, formatter={'float_kind': '{:5.5f}'.format}):
    #     #formatter={'float_kind': '{:5.2f}'.format})
    #       p.check_partials(show_only_incorrect=True, compact_print=False, method='fd')


