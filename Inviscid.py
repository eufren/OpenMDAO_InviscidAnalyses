import openmdao.api as om
import numpy as np
from scipy.sparse import coo_matrix


#  1. Discretise geometry.

class ConvertToClockwise(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_points', types=int)

    def setup(self):
        num_points = self.options['num_points']
        self.add_input('aerofoil_ccw_coordinates', shape=(2, num_points))

        self.add_output('aerofoil_cw_coordinates', shape=(2, num_points))

        self.declare_partials('aerofoil_cw_coordinates', 'aerofoil_ccw_coordinates',
                              rows=np.arange(2*num_points),
                              cols=np.concatenate((np.arange(num_points-1, -1, -1), np.arange((2*num_points)-1, num_points-1, -1))),
                              val=np.ones(2*num_points))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        aerofoil_ccw_coordinates = inputs['aerofoil_ccw_coordinates']

        outputs['aerofoil_cw_coordinates'] = np.flip(aerofoil_ccw_coordinates, 1)


class CreatePanels(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_points', types=int)

    def setup(self):
        num_points = self.options['num_points']

        # Expects a (x1, x2, ..., xn), (y1, y2, ..., yn) array, trailing edge >>> CLOCKWISE <<<.
        self.add_input('aerofoil_cw_coordinates', shape=(2, num_points))

        self.add_output('panel_coordinates', shape=(2, num_points - 1, 2))

        self.declare_partials('panel_coordinates', 'aerofoil_cw_coordinates',
                              rows=np.arange(4 * num_points - 4),
                              cols=np.concatenate([np.repeat(np.arange(num_points), 2)[1:-1],
                                                   np.repeat(np.arange(num_points), 2)[1:-1] + num_points]),
                              val=np.ones(4 * num_points - 4))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        aerofoil_cw_coordinates = inputs['aerofoil_cw_coordinates']

        # Fill an array with ((x1,x2),(y1,y2)) for panel coordinates. These will be trailing edge CLOCKWISE, like the coordinates.
        # Note that aside from the first and last coordinates, each coordinate is used twice as the end of one panel and the start of another.
        # So we'll repeat the aerofoil coordinates array i.e (x1, x2, ..., xn) becomes (x1, x1, x2, x2, ..., xn, xn).
        # Then we'll trim off the extra x1 and xn.
        panel_coordinates = np.repeat(aerofoil_cw_coordinates, repeats=2, axis=1)  # Do the repeat
        panel_coordinates = panel_coordinates[:, 1:-1]  # Slice from the second to second-last element (trim x1, xn)
        panel_coordinates = panel_coordinates.reshape(2, -1,
                                                      2)  # And now reshape the array. See diary 30/05 for notes on array structure.

        outputs['panel_coordinates'] = panel_coordinates


class PanelGeometry(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_points', types=int)

    def setup(self):
        num_points = self.options['num_points']
        n = num_points - 1

        self.add_input('panel_coordinates', shape=(2, n, 2))

        self.add_output('panel_angles', shape=(n,))
        self.add_output('panel_lengths', shape=(n,))
        self.add_output('collocation_point_coordinates', shape=(2, n))

        # We reuse these, so may as well only generate them once.
        rows = np.tile(np.arange(n), 4)  # See OneNote entry for 06/06 for derivation.
        two_times_i = 2 * np.arange(n)
        cols = np.concatenate([two_times_i, two_times_i + 1, two_times_i + 2 * n, two_times_i + 2 * n + 1])

        self.declare_partials('panel_angles', 'panel_coordinates',
                              rows=rows,
                              cols=cols)
        self.declare_partials('panel_lengths', 'panel_coordinates',
                              rows=rows,
                              cols=cols)
        self.declare_partials('collocation_point_coordinates', 'panel_coordinates',
                              rows=np.repeat(np.arange(2 * n), 2),
                              cols=np.arange(4 * n),
                              val=0.5 * np.ones(4 * n))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        panel_coordinates = inputs['panel_coordinates']

        # Fill an array with panel angles. These angles are defined counterclockwise from horizontal, and point in the
        # direction of the panel - i.e, top surface panels pointing left will be obtuse, and bottom surface panels
        # pointing right will be acute.
        panel_angles = np.arctan2(  # We do arctan2(dy, dx)
            np.diff(panel_coordinates)[1, :, :],  # We find dy by subtracting all the y pairs.
            np.diff(panel_coordinates)[0, :, :])  # And similarly for dx.
        panel_angles = np.squeeze(panel_angles)  # This just gets rid of the excess dimension left over by np.diff.

        # Fill an array with panel lengths
        panel_lengths = np.sqrt(np.sum(np.diff(panel_coordinates) ** 2, axis=0))
        panel_lengths = np.squeeze(panel_lengths)  # Have to squeeze out the excess dimension again.

        # Fill an array with collocation points, placed at the midpoints of each panel.
        # This is calculated as (x2-x1)/0.5 + x1.
        # Have to slice :1 instead of accessing 0 in order to maintain dimensions.
        collocation_point_coordinates = np.diff(panel_coordinates) / 2 + panel_coordinates[:, :, :1]
        collocation_point_coordinates = np.squeeze(
            collocation_point_coordinates)  # Have to squeeze out the excess dimension again.

        outputs['panel_angles'] = panel_angles
        outputs['panel_lengths'] = panel_lengths
        outputs['collocation_point_coordinates'] = collocation_point_coordinates

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        panel_coordinates = inputs['panel_coordinates']
        x1 = panel_coordinates[0, :, 1]
        x0 = panel_coordinates[0, :, 0]
        y1 = panel_coordinates[1, :, 1]
        y0 = panel_coordinates[1, :, 0]

        # Angles w.r.t coordinates (See pages 13 to 17 in notes)
        dx_i0 = (y1 - y0) / ((x1 - x0) ** 2 + (y1 - y0) ** 2)
        dy_i1 = (x1 - x0) / ((x1 - x0) ** 2 + (y1 - y0) ** 2)
        dy_i0 = -dy_i1
        dx_i1 = -dx_i0
        partials['panel_angles', 'panel_coordinates'] = np.concatenate([dx_i0, dx_i1, dy_i0, dy_i1], axis=0)

        # Lengths w.r.t coordinates
        # L = ((y1 - y0)^2 + (x1 - x0)^2)^0.5
        dx_i1 = (x1 - x0) / ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        dy_i1 = (y1 - y0) / ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        dx_i0 = -dx_i1
        dy_i0 = -dy_i1
        partials['panel_lengths', 'panel_coordinates'] = np.concatenate([dx_i0, dx_i1, dy_i0, dy_i1], axis=0)

        # Collocation points w.r.t coordinates is defined with val argument in setup, since it's simple and constant.


class SecondPanelCoordinateRelativePositions(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('n', types=int)

    def setup(self):
        n = self.options['n']

        self.add_input('panel_coordinates', shape=(2, n, 2))

        self.add_output('second_panel_coordinate_relative_positions', shape=(2, n))

        self.declare_partials('second_panel_coordinate_relative_positions', 'panel_coordinates',
                              rows=np.repeat(np.arange(2 * n), 2),
                              cols=np.arange(4 * n),
                              val=np.tile(np.array([-1, 1]), 2 * n))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        panel_coordinates = inputs['panel_coordinates']

        # Diff subtracts all the coordinates, squeeze gets rid of the extra dimension.
        outputs['second_panel_coordinate_relative_positions'] = np.squeeze(np.diff(panel_coordinates))


class CollocationPointsRelativePositions(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_col_points', types=int)
        self.options.declare('num_panels', types=int)

    def setup(self):
        n = self.options['num_col_points']
        m = self.options['num_panels']

        self.add_input('panel_coordinates', shape=(2, m, 2))
        self.add_input('collocation_point_coordinates', shape=(2, n))

        self.add_output('collocation_points_relative_positions', shape=(n, m, 2))

        # See diary notes 17/11 for derivations.
        self.declare_partials('collocation_points_relative_positions', 'panel_coordinates',
                              rows=np.concatenate((np.arange(0, 2*n*m, 2), np.arange(1, 2*n*m, 2))),
                              cols=np.concatenate((np.tile(np.arange(0, 2*m, 2), n), np.tile(np.arange(2*m, 4*m, 2), n))),
                              val=-np.ones(2*n*m))

        self.declare_partials('collocation_points_relative_positions', 'collocation_point_coordinates',
                              rows=np.arange(2 * n * m),
                              cols=(np.tile((0, n), m) + np.arange(n).reshape(-1, 1)).reshape(-1),
                              val=np.ones(2 * n * m))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        panel_coordinates = inputs['panel_coordinates']
        collocation_point_coordinates = inputs['collocation_point_coordinates']

        # See diary notes from 08/06
        first_panel_xs = panel_coordinates[0, :, 0]
        first_panel_ys = panel_coordinates[1, :, 0]
        collocation_point_xs = collocation_point_coordinates[np.newaxis, 0].T
        collocation_point_ys = collocation_point_coordinates[np.newaxis, 1].T

        cxr = collocation_point_xs - first_panel_xs  # Collocation, X, relative
        cyr = collocation_point_ys - first_panel_ys

        outputs['collocation_points_relative_positions'] = np.dstack([cxr, cyr])


class SecondPanelCoordinateInPanelReferenceFrame(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('n', types=int)

    def setup(self):
        n = self.options['n']

        self.add_input('panel_angles', shape=(n,))
        self.add_input('second_panel_coordinate_relative_positions', shape=(2, n))

        self.add_output('second_panel_coordinate_transformed_positions', shape=(2, n))

        self.declare_partials('second_panel_coordinate_transformed_positions', 'panel_angles',
                              rows=np.arange(n),
                              cols=np.arange(n))
        self.declare_partials('second_panel_coordinate_transformed_positions',
                              'second_panel_coordinate_relative_positions',
                              rows=np.arange(n),
                              cols=np.arange(n))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        second_panel_coordinate_relative_positions = inputs['second_panel_coordinate_relative_positions']
        panel_angles = inputs['panel_angles']

        x2r = second_panel_coordinate_relative_positions[0, :]  # x2r = x coordinate number 2, relative
        y2r = second_panel_coordinate_relative_positions[1, :]

        cos_angles = np.cos(panel_angles)
        sin_angles = np.sin(panel_angles)

        second_panel_coordinate_transformed_positions = np.zeros_like(second_panel_coordinate_relative_positions)
        second_panel_coordinate_transformed_positions[0, :] = x2r/cos_angles

        outputs['second_panel_coordinate_transformed_positions'] = second_panel_coordinate_transformed_positions

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        second_panel_coordinate_relative_positions = inputs['second_panel_coordinate_relative_positions']
        panel_angles = inputs['panel_angles']

        x2r = second_panel_coordinate_relative_positions[0, :]  # x2r = x coordinate number 2, relative
        y2r = second_panel_coordinate_relative_positions[1, :]

        cos_angles = np.cos(panel_angles)
        sin_angles = np.sin(panel_angles)

        partials['second_panel_coordinate_transformed_positions',
                 'panel_angles'] = x2r*sin_angles/(cos_angles**2)

        partials['second_panel_coordinate_transformed_positions',
                 'second_panel_coordinate_relative_positions'] = 1/cos_angles


class CollocationPointsInPanelReferenceFrames(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_col_points', types=int)
        self.options.declare('num_panels', types=int)

    def setup(self):
        n = self.options['num_col_points']
        m = self.options['num_panels']

        self.add_input('panel_angles', shape=(m,))
        self.add_input('collocation_points_relative_positions', shape=(n, m, 2))

        self.add_output('collocation_points_transformed_positions', shape=(n, m, 2))

        self.declare_partials('collocation_points_transformed_positions', 'panel_angles',
                              rows=np.concatenate((np.arange(0, 2*n*m - 1, 2), np.arange(1, 2*n*m, 2))),
                              cols=np.concatenate((np.tile(np.arange(m), n), np.tile(np.arange(m), n))))
        self.declare_partials('collocation_points_transformed_positions', 'collocation_points_relative_positions',
                              rows=np.concatenate((np.arange(0, (2*n*m)-1, 2), np.arange(0, (2*n*m)-1, 2), np.arange(1, (2*n*m), 2), np.arange(1, (2*n*m), 2))),
                              cols=np.concatenate((np.arange(0, (2*n*m)-1, 2), np.arange(1, (2*n*m), 2), np.arange(0, (2*n*m)-1, 2), np.arange(1, (2*n*m), 2))))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        panel_angles = inputs['panel_angles']
        collocation_points_relative_positions = inputs['collocation_points_relative_positions']

        cxr = collocation_points_relative_positions[:, :, 0]  # cxr = Collocation, X, relative
        cyr = collocation_points_relative_positions[:, :, 1]
        cos_angles = np.cos(panel_angles)
        sin_angles = np.sin(panel_angles)

        cxt = cxr * cos_angles + cyr * sin_angles  # Collocation, X, transformed
        cyt = -cxr * sin_angles + cyr * cos_angles

        outputs['collocation_points_transformed_positions'] = np.dstack([cxt, cyt])

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        n = self.options['num_col_points']
        panel_angles = inputs['panel_angles']
        collocation_points_relative_positions = inputs['collocation_points_relative_positions']

        cxr = collocation_points_relative_positions[:, :, 0]  # cxr = Collocation, X, relative
        cyr = collocation_points_relative_positions[:, :, 1]
        cos_angles = np.cos(panel_angles)
        sin_angles = np.sin(panel_angles)

        x_partials = cyr * cos_angles - cxr * sin_angles
        y_partials = -cyr * sin_angles - cxr * cos_angles
        partials['collocation_points_transformed_positions', 'panel_angles'] = np.concatenate((x_partials.reshape(-1), y_partials.reshape(-1)))

        complete_partials = np.concatenate((np.tile(cos_angles, n), np.tile(sin_angles, n), np.tile(-sin_angles, n), np.tile(cos_angles, n)))
        partials[
            'collocation_points_transformed_positions', 'collocation_points_relative_positions'] = complete_partials


class FindThetaComponents(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_col_points', types=int)
        self.options.declare('num_panels', types=int)

    def setup(self):
        n = self.options['num_col_points']
        m = self.options['num_panels']

        self.add_input('collocation_points_transformed_positions', shape=(n, m, 2))
        self.add_input('second_panel_coordinate_transformed_positions', shape=(2, m))

        self.add_output('collocation_points_angles_from_first_panel_points', shape=(n, m))
        self.add_output('collocation_points_angles_from_second_panel_points', shape=(n, m))

        self.declare_partials('collocation_points_angles_from_first_panel_points',
                              'collocation_points_transformed_positions',
                              rows=np.concatenate((np.arange(n*m), np.arange(n*m))),
                              cols=np.concatenate((np.arange(0, 2*n*m-1, 2), np.arange(1, 2*n*m, 2))))
        self.declare_partials('collocation_points_angles_from_first_panel_points',
                              'second_panel_coordinate_transformed_positions', dependent=False)
        self.declare_partials('collocation_points_angles_from_second_panel_points',
                              'collocation_points_transformed_positions',
                              rows=np.concatenate((np.arange(n*m), np.arange(n*m))),
                              cols=np.concatenate((np.arange(0, 2*n*m-1, 2), np.arange(1, 2*n*m, 2))))
        self.declare_partials('collocation_points_angles_from_second_panel_points',
                              'second_panel_coordinate_transformed_positions',
                              rows=np.arange(n*m),
                              cols=np.tile(np.arange(0, m), n))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        collocation_points_transformed_positions = inputs['collocation_points_transformed_positions']
        second_panel_coordinate_transformed_positions = inputs['second_panel_coordinate_transformed_positions']

        x2t = second_panel_coordinate_transformed_positions[0, :]  # x2r = x coordinate number 2, transformed
        cxt = collocation_points_transformed_positions[:, :, 0]  # cxr = Collocation, X, transformed
        cyt = collocation_points_transformed_positions[:, :, 1]

        collocation_points_angles_from_first_panel_points = np.squeeze(np.arctan2(cyt, cxt))
        collocation_points_angles_from_second_panel_points = np.squeeze(np.arctan2(cyt, cxt - x2t))

        outputs['collocation_points_angles_from_first_panel_points'] = collocation_points_angles_from_first_panel_points
        outputs['collocation_points_angles_from_second_panel_points'] = collocation_points_angles_from_second_panel_points

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        collocation_points_transformed_positions = inputs['collocation_points_transformed_positions']
        second_panel_coordinate_transformed_positions = inputs['second_panel_coordinate_transformed_positions']

        x2t = second_panel_coordinate_transformed_positions[0, :]  # x2r = x coordinate number 2, transformed
        cxt = collocation_points_transformed_positions[:, :, 0]  # cxr = Collocation, X, transformed
        cyt = collocation_points_transformed_positions[:, :, 1]

        dtheta1_dxc = -cyt / (cxt ** 2 + cyt ** 2)  # See page 30 in notes
        dtheta1_dyc = cxt / (cxt ** 2 + cyt ** 2)
        dtheta1_dc = np.concatenate([dtheta1_dxc, dtheta1_dyc])

        a = (cxt - x2t)
        dtheta2_dxc = -cyt / (a ** 2 + cyt ** 2)
        dtheta2_dyc = a / (a ** 2 + cyt ** 2)
        dtheta2_dc = np.concatenate([dtheta2_dxc, dtheta2_dyc])

        dtheta2_ds = -dtheta2_dxc  # Again, see page 30

        partials['collocation_points_angles_from_first_panel_points',
                 'collocation_points_transformed_positions'] = dtheta1_dc.reshape(-1)

        partials['collocation_points_angles_from_second_panel_points',
                 'collocation_points_transformed_positions'] = dtheta2_dc.reshape(-1)

        partials['collocation_points_angles_from_second_panel_points',
                 'second_panel_coordinate_transformed_positions'] = dtheta2_ds.reshape(-1)


class FindRComponents(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_col_points', types=int)
        self.options.declare('num_panels', types=int)

    def setup(self):
        n = self.options['num_col_points']
        m = self.options['num_panels']

        self.add_input('collocation_points_transformed_positions', shape=(n, m, 2))
        self.add_input('second_panel_coordinate_transformed_positions', shape=(2, m))

        self.add_output('collocation_points_distances_from_first_panel_points', shape=(n, m))
        self.add_output('collocation_points_distances_from_second_panel_points', shape=(n, m))

        self.declare_partials('collocation_points_distances_from_first_panel_points', 'collocation_points_transformed_positions',
                              rows=np.concatenate((np.arange(n*m), np.arange(n*m))),
                              cols=np.concatenate((np.arange(0, 2*n*m - 1, 2), np.arange(1, 2*n*m, 2))))
        self.declare_partials('collocation_points_distances_from_first_panel_points', 'second_panel_coordinate_transformed_positions', dependent=False)
        self.declare_partials('collocation_points_distances_from_second_panel_points', 'collocation_points_transformed_positions',
                              rows=np.concatenate((np.arange(n*m), np.arange(n*m))),
                              cols=np.concatenate((np.arange(0, 2*n*m - 1, 2), np.arange(1, 2*n*m, 2))))
        self.declare_partials('collocation_points_distances_from_second_panel_points', 'second_panel_coordinate_transformed_positions',
                              rows=np.arange(n*m),
                              cols=np.tile(np.arange(0, m), n))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        collocation_points_transformed_positions = inputs['collocation_points_transformed_positions']
        second_panel_coordinate_transformed_positions = inputs['second_panel_coordinate_transformed_positions']

        x2t = second_panel_coordinate_transformed_positions[0, :]  # x2r = x coordinate number 2, transformed
        cxt = collocation_points_transformed_positions[:, :, 0]  # cxr = Collocation, X, transformed
        cyt = collocation_points_transformed_positions[:, :, 1]

        collocation_points_distances_from_first_panel_points = np.sqrt(cxt**2 + cyt**2)
        collocation_points_distances_from_second_panel_points = np.sqrt((cxt-x2t)**2 + cyt**2)

        outputs['collocation_points_distances_from_first_panel_points'] = collocation_points_distances_from_first_panel_points
        outputs['collocation_points_distances_from_second_panel_points'] = collocation_points_distances_from_second_panel_points

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        collocation_points_transformed_positions = inputs['collocation_points_transformed_positions']
        second_panel_coordinate_transformed_positions = inputs['second_panel_coordinate_transformed_positions']

        x2t = second_panel_coordinate_transformed_positions[0, :]  # x2r = x coordinate number 2, transformed
        cxt = collocation_points_transformed_positions[:, :, 0]  # cxr = Collocation, X, transformed
        cyt = collocation_points_transformed_positions[:, :, 1]

        dr1_dxc = cxt / np.sqrt(cxt ** 2 + cyt ** 2)  # See diary 12/11/20
        dr1_dyc = cyt / np.sqrt(cxt ** 2 + cyt ** 2)
        dr1_dc = np.concatenate([dr1_dxc, dr1_dyc])

        a = (cxt - x2t)
        dr2_dxc = a / np.sqrt(a ** 2 + cyt ** 2)
        dr2_dyc = cyt / np.sqrt(a ** 2 + cyt ** 2)
        dr2_dc = np.concatenate([dr2_dxc, dr2_dyc])

        dr2_ds = -dr2_dxc  # Again, see page 30

        partials['collocation_points_distances_from_first_panel_points',
                 'collocation_points_transformed_positions'] = dr1_dc.reshape(-1)

        partials['collocation_points_distances_from_second_panel_points',
                 'collocation_points_transformed_positions'] = dr2_dc.reshape(-1)

        partials['collocation_points_distances_from_second_panel_points',
                 'second_panel_coordinate_transformed_positions'] = dr2_ds.reshape(-1)


class GeometryAndDiscretisationGroup(om.Group):

    def initialize(self):
        self.options.declare('num_points')

    def setup(self):
        num_points = self.options['num_points']
        n = num_points - 1

        self.add_subsystem('ConvertToClockwise', ConvertToClockwise(num_points=num_points),
                           promotes_inputs=['aerofoil_ccw_coordinates'],
                           promotes_outputs=['aerofoil_cw_coordinates'])

        self.add_subsystem('CreatePanels', CreatePanels(num_points=num_points),
                           promotes_inputs=['aerofoil_cw_coordinates'],
                           promotes_outputs=['panel_coordinates'])

        self.add_subsystem('PanelGeometry', PanelGeometry(num_points=num_points),
                           promotes_inputs=['panel_coordinates'],
                           promotes_outputs=['panel_angles', 'panel_lengths', 'collocation_point_coordinates'])

        self.add_subsystem('SecondPanelCoordinateRelativePositions', SecondPanelCoordinateRelativePositions(n=n),
                           promotes_inputs=['panel_coordinates'],
                           promotes_outputs=['second_panel_coordinate_relative_positions'])

        self.add_subsystem('CollocationPointsRelativePositions', CollocationPointsRelativePositions(num_col_points=n, num_panels=n),
                           promotes_inputs=['panel_coordinates', 'collocation_point_coordinates'],
                           promotes_outputs=['collocation_points_relative_positions'])

        self.add_subsystem('SecondPanelCoordinateInPanelReferenceFrame',
                           SecondPanelCoordinateInPanelReferenceFrame(n=n),
                           promotes_inputs=['panel_angles', 'second_panel_coordinate_relative_positions'],
                           promotes_outputs=['second_panel_coordinate_transformed_positions'])

        self.add_subsystem('CollocationPointsInPanelReferenceFrames', CollocationPointsInPanelReferenceFrames(num_col_points=n, num_panels=n),
                           promotes_inputs=['panel_angles', 'collocation_points_relative_positions'],
                           promotes_outputs=['collocation_points_transformed_positions'])

        self.add_subsystem('FindThetaComponents', FindThetaComponents(num_col_points=n, num_panels=n),
                           promotes_inputs=['collocation_points_transformed_positions',
                                            'second_panel_coordinate_transformed_positions'],
                           promotes_outputs=['collocation_points_angles_from_first_panel_points',
                                             'collocation_points_angles_from_second_panel_points'])

        self.add_subsystem('FindRComponents', FindRComponents(num_col_points=n, num_panels=n),
                           promotes_inputs=['collocation_points_transformed_positions',
                                            'second_panel_coordinate_transformed_positions'],
                           promotes_outputs=['collocation_points_distances_from_first_panel_points',
                                             'collocation_points_distances_from_second_panel_points'])


#  2. Calculate influence coefficients.


class GenerateExtraCollocationPoint(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_col_points', types=int)
        self.adjustment_factor = 0.98

    def setup(self):
        n = self.options['num_col_points']

        self.add_input('collocation_point_coordinates', shape=(2, n))
        self.add_input('panel_coordinates', shape=(2, n, 2))

        self.add_output('extra_collocation_point_coordinates', shape=(2, 1))

        fac = (1-self.adjustment_factor)*0.5
        self.declare_partials('extra_collocation_point_coordinates', 'collocation_point_coordinates',
                              rows=np.asarray([0, 0, 1, 1]),
                              cols=np.asarray([0, n-1, n, 2*n - 1]),
                              val=np.asarray([fac, fac, fac, fac]))
        self.declare_partials('extra_collocation_point_coordinates', 'panel_coordinates',
                              rows=np.asarray([0, 1]),
                              cols=np.asarray([0, 2*n]),
                              val=self.adjustment_factor)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        collocation_point_coordinates = inputs['collocation_point_coordinates']
        te_point = inputs['panel_coordinates'][:, 0, 0]

        first_point = collocation_point_coordinates[:, 0]
        last_point = collocation_point_coordinates[:, -1]

        mid_point = (first_point+last_point)/2
        new_point = self.adjustment_factor*te_point + (1-self.adjustment_factor)*mid_point
        outputs['extra_collocation_point_coordinates'] = new_point


class PrepareExtraCollocationPoint(om.Group):
    def initialize(self):
        self.options.declare('num_col_points')

    def setup(self):
        n = self.options['num_col_points']

        self.add_subsystem('GenerateExtraCollocationPoint', GenerateExtraCollocationPoint(num_col_points=n),
                           promotes_inputs=['collocation_point_coordinates',
                                            'panel_coordinates'],
                           promotes_outputs=['extra_collocation_point_coordinates'])

        self.add_subsystem('CollocationPointsRelativePositions', CollocationPointsRelativePositions(num_col_points=1, num_panels=n),
                           promotes_inputs=['panel_coordinates', ('collocation_point_coordinates', 'extra_collocation_point_coordinates')],
                           promotes_outputs=[('collocation_points_relative_positions', 'extra_collocation_point_relative_positions')])

        self.add_subsystem('CollocationPointsInPanelReferenceFrames', CollocationPointsInPanelReferenceFrames(num_col_points=1, num_panels=n),
                           promotes_inputs=['panel_angles', ('collocation_points_relative_positions', 'extra_collocation_point_relative_positions')],
                           promotes_outputs=[('collocation_points_transformed_positions', 'extra_collocation_point_transformed_positions')])

        self.add_subsystem('FindThetaComponents', FindThetaComponents(num_col_points=1, num_panels=n),
                           promotes_inputs=[('collocation_points_transformed_positions', 'extra_collocation_point_transformed_positions'),
                                            'second_panel_coordinate_transformed_positions'],
                           promotes_outputs=[('collocation_points_angles_from_first_panel_points', 'extra_collocation_point_angles_from_first_panel_points'),
                                             ('collocation_points_angles_from_second_panel_points', 'extra_collocation_point_angles_from_second_panel_points')])

        self.add_subsystem('FindRComponents', FindRComponents(num_col_points=1, num_panels=n),
                           promotes_inputs=[('collocation_points_transformed_positions', 'extra_collocation_point_transformed_positions'),
                                            'second_panel_coordinate_transformed_positions'],
                           promotes_outputs=[('collocation_points_distances_from_first_panel_points', 'extra_collocation_point_distances_from_first_panel_points'),
                                             ('collocation_points_distances_from_second_panel_points', 'extra_collocation_point_distances_from_second_panel_points')])


class ConstantDoubletInfluence(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_col_points', types=int)
        self.options.declare('num_panels', types=int)

    def setup(self):
        n = self.options['num_col_points']
        m = self.options['num_panels']

        self.add_input('collocation_points_angles_from_first_panel_points', shape=(n, m))
        self.add_input('collocation_points_angles_from_second_panel_points', shape=(n, m))

        self.add_output('constant_influence_coefficients', shape=(n, m))

        diag = np.arange(n*m)
        self.declare_partials('constant_influence_coefficients', 'collocation_points_angles_from_first_panel_points',
                              rows=diag,
                              cols=diag,)
        self.declare_partials('constant_influence_coefficients', 'collocation_points_angles_from_second_panel_points',
                              rows=diag,
                              cols=diag,)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        n = self.options['num_col_points']
        theta1 = inputs['collocation_points_angles_from_first_panel_points']
        theta2 = inputs['collocation_points_angles_from_second_panel_points']

        influence_coefficients = (-1/(2*np.pi)) * (theta2 - theta1)

        if n > 1:
            np.fill_diagonal(influence_coefficients, 0.5)

        outputs['constant_influence_coefficients'] = influence_coefficients

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        n = self.options['num_col_points']
        m = self.options['num_panels']

        theta1 = inputs['collocation_points_angles_from_first_panel_points']
        theta2 = inputs['collocation_points_angles_from_second_panel_points']

        da_dtheta1 = np.repeat((1/(2*np.pi)), n*m)

        if n > 1:
            da_dtheta1[np.arange(0, n*m, m+1)] = 0

        partials['constant_influence_coefficients', 'collocation_points_angles_from_first_panel_points'] = da_dtheta1
        partials['constant_influence_coefficients', 'collocation_points_angles_from_second_panel_points'] = -da_dtheta1


class LinearDoubletInfluence(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_col_points', types=int)
        self.options.declare('num_panels', types=int)

    def setup(self):
        n = self.options['num_col_points']
        m = self.options['num_panels']

        self.add_input('collocation_points_angles_from_first_panel_points', shape=(n, m))
        self.add_input('collocation_points_angles_from_second_panel_points', shape=(n, m))
        self.add_input('collocation_points_distances_from_first_panel_points', shape=(n, m))
        self.add_input('collocation_points_distances_from_second_panel_points', shape=(n, m))
        self.add_input('collocation_points_transformed_positions', shape=(n, m, 2))

        self.add_output('linear_influence_coefficients', shape=(n, m))

        diag = np.arange(n*m)
        self.declare_partials('linear_influence_coefficients', 'collocation_points_angles_from_first_panel_points',
                              rows=diag,
                              cols=diag)
        self.declare_partials('linear_influence_coefficients', 'collocation_points_angles_from_second_panel_points',
                              rows=diag,
                              cols=diag)
        self.declare_partials('linear_influence_coefficients', 'collocation_points_distances_from_first_panel_points',
                              rows=diag,
                              cols=diag)
        self.declare_partials('linear_influence_coefficients', 'collocation_points_distances_from_second_panel_points',
                              rows=diag,
                              cols=diag)
        self.declare_partials('linear_influence_coefficients', 'collocation_points_transformed_positions',
                              rows=np.tile(np.arange(n*m), 2),
                              cols=np.array((np.arange(0, 2*(n*m)-1, 2), np.arange(1, 2*(n*m), 2))).reshape(-1))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        n = self.options['num_col_points']
        th_1 = inputs['collocation_points_angles_from_first_panel_points']
        th_2 = inputs['collocation_points_angles_from_second_panel_points']
        r1 = inputs['collocation_points_distances_from_first_panel_points']
        r2 = inputs['collocation_points_distances_from_second_panel_points']
        collocation_points_transformed_positions = inputs['collocation_points_transformed_positions']
        x = collocation_points_transformed_positions[:, :, 0]
        z = collocation_points_transformed_positions[:, :, 1]

        linear_influence_coefficients = (-1/(2*np.pi))*((x*(th_2-th_1)) + z*np.log(r2/r1))

        if n > 1:
            np.fill_diagonal(linear_influence_coefficients, 0.5*np.diag(x))

        outputs['linear_influence_coefficients'] = linear_influence_coefficients

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        n = self.options['num_col_points']
        m = self.options['num_panels']

        th_1 = inputs['collocation_points_angles_from_first_panel_points']
        th_2 = inputs['collocation_points_angles_from_second_panel_points']
        r1 = inputs['collocation_points_distances_from_first_panel_points']
        r2 = inputs['collocation_points_distances_from_second_panel_points']
        collocation_points_transformed_positions = inputs['collocation_points_transformed_positions']
        x = collocation_points_transformed_positions[:, :, 0]
        z = collocation_points_transformed_positions[:, :, 1]

        db_dx = -1/(2*np.pi)*(th_2-th_1)
        db_dz = -1/(4*np.pi)*np.log((r2**2)/(r1**2))
        db_dc = np.array((db_dx, db_dz)).reshape(-1)

        db_dtheta1 = 1/(2*np.pi)*x.reshape(-1)
        db_dtheta2 = -db_dtheta1
        db_dr1 = 1/(2*np.pi)*(z/r1).reshape(-1)
        db_dr2 = -1/(2*np.pi)*(z/r2).reshape(-1)

        if n > 1:
            db_dc[np.arange(0, n*m, m+1)] = 0.5
            db_dtheta1[np.arange(0, n*m, m+1)] = 0
            db_dtheta2[np.arange(0, n*m, m+1)] = 0
            db_dr1[np.arange(0, n*m, m+1)] = 0
            db_dr2[np.arange(0, n*m, m+1)] = 0

        partials['linear_influence_coefficients', 'collocation_points_angles_from_first_panel_points'] = db_dtheta1
        partials['linear_influence_coefficients', 'collocation_points_angles_from_second_panel_points'] = db_dtheta2
        partials['linear_influence_coefficients', 'collocation_points_distances_from_first_panel_points'] = db_dr1
        partials['linear_influence_coefficients', 'collocation_points_distances_from_second_panel_points'] = db_dr2
        partials['linear_influence_coefficients', 'collocation_points_transformed_positions'] = db_dc


class QuadraticDoubletInfluence(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_col_points', types=int)
        self.options.declare('num_panels', types=int)

    def setup(self):
        n = self.options['num_col_points']
        m = self.options['num_panels']

        self.add_input('collocation_points_angles_from_first_panel_points', shape=(n, m))
        self.add_input('collocation_points_angles_from_second_panel_points', shape=(n, m))
        self.add_input('collocation_points_distances_from_first_panel_points', shape=(n, m))
        self.add_input('collocation_points_distances_from_second_panel_points', shape=(n, m))
        self.add_input('panel_lengths', shape=(m,))
        self.add_input('collocation_points_transformed_positions', shape=(n, m, 2))
        self.add_output('quadratic_influence_coefficients', shape=(n, m))

        diag = np.arange(n*m)
        self.declare_partials('quadratic_influence_coefficients', 'collocation_points_angles_from_first_panel_points',
                              rows=diag,
                              cols=diag)
        self.declare_partials('quadratic_influence_coefficients', 'collocation_points_angles_from_second_panel_points',
                              rows=diag,
                              cols=diag)
        self.declare_partials('quadratic_influence_coefficients', 'collocation_points_distances_from_first_panel_points',
                              rows=diag,
                              cols=diag)
        self.declare_partials('quadratic_influence_coefficients', 'collocation_points_distances_from_second_panel_points',
                              rows=diag,
                              cols=diag)
        self.declare_partials('quadratic_influence_coefficients', 'panel_lengths',
                              rows=diag,
                              cols=np.tile(np.arange(m), n))
        self.declare_partials('quadratic_influence_coefficients', 'collocation_points_transformed_positions',
                              rows=np.tile(np.arange(n*m), 2),
                              cols=np.array((np.arange(0, 2*(n*m)-1, 2), np.arange(1, 2*(n*m), 2))).reshape(-1))  # See diary 17/11

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        n = self.options['num_col_points']
        th_1 = inputs['collocation_points_angles_from_first_panel_points']
        th_2 = inputs['collocation_points_angles_from_second_panel_points']
        r1 = inputs['collocation_points_distances_from_first_panel_points']
        r2 = inputs['collocation_points_distances_from_second_panel_points']
        l = inputs['panel_lengths']
        collocation_points_transformed_positions = inputs['collocation_points_transformed_positions']
        x = collocation_points_transformed_positions[:, :, 0]
        z = collocation_points_transformed_positions[:, :, 1]

        quadratic_influence_coefficients = (1/(2*np.pi))*(((x**2 - z**2)*(th_1-th_2)) - 2*x*z*np.log(r2/r1) - z*l)

        if n > 1:
            np.fill_diagonal(quadratic_influence_coefficients, 0.5*(np.diag(x)**2))

        outputs['quadratic_influence_coefficients'] = quadratic_influence_coefficients

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        n = self.options['num_col_points']
        m = self.options['num_panels']

        th_1 = inputs['collocation_points_angles_from_first_panel_points']
        th_2 = inputs['collocation_points_angles_from_second_panel_points']
        r1 = inputs['collocation_points_distances_from_first_panel_points']
        r2 = inputs['collocation_points_distances_from_second_panel_points']
        l = inputs['panel_lengths']
        collocation_points_transformed_positions = inputs['collocation_points_transformed_positions']
        x = collocation_points_transformed_positions[:, :, 0]
        z = collocation_points_transformed_positions[:, :, 1]

        db_dx = (1/(2*np.pi))*((2*(th_1-th_2)*x) - z*np.log(r2**2/r1**2))
        db_dz = (1/(2*np.pi))*((-2*(th_1-th_2)*z) - x*np.log(r2**2/r1**2) - l)
        db_dc = np.array((db_dx, db_dz)).reshape(-1)

        db_dtheta1 = 1/(2*np.pi)*(x**2 - z**2).reshape(-1)
        db_dtheta2 = -db_dtheta1
        db_dr1 = 1/(np.pi)*((x*z)/r1).reshape(-1)
        db_dr2 = -1/(np.pi)*((x*z)/r2).reshape(-1)
        db_dl = -1/(np.pi)*z.reshape(-1)

        if n > 1:
            db_dc[np.arange(0, n*m, m+1)] = np.diag(x)
            db_dc[np.arange(n*m, 2*n*m, m+1)] = 0
            db_dtheta1[np.arange(0, n*m, m+1)] = 0
            db_dtheta2[np.arange(0, n*m, m+1)] = 0
            db_dr1[np.arange(0, n*m, m+1)] = 0
            db_dr2[np.arange(0, n*m, m+1)] = 0
            db_dl[np.arange(0, n*m, m+1)] = 0

        partials['quadratic_influence_coefficients', 'collocation_points_angles_from_first_panel_points'] = db_dtheta1
        partials['quadratic_influence_coefficients', 'collocation_points_angles_from_second_panel_points'] = db_dtheta2
        partials['quadratic_influence_coefficients', 'collocation_points_distances_from_first_panel_points'] = db_dr1
        partials['quadratic_influence_coefficients', 'collocation_points_distances_from_second_panel_points'] = db_dr2
        partials['quadratic_influence_coefficients', 'panel_lengths'] = db_dl
        partials['quadratic_influence_coefficients', 'collocation_points_transformed_positions'] = db_dc


#  3. Assemble influence matrix.


class InfluenceCoefficientAssembler(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_col_points', types=int)
        self.options.declare('num_panels', types=int)

    def setup(self):
        n = self.options['num_col_points']
        m = self.options['num_panels']

        self.add_input('constant_influence_coefficients', shape=(n, m))
        self.add_input('linear_influence_coefficients', shape=(n, m))
        self.add_input('quadratic_influence_coefficients', shape=(n, m))

        self.add_output('full_influence_coefficients', shape=(n, 3*m))

        self.declare_partials('full_influence_coefficients', 'constant_influence_coefficients',  # See notes 23/11
                              rows=np.arange(0, 3*n*m - 2, 3),
                              cols=np.arange(n*m),
                              val=np.ones(n*m))
        self.declare_partials('full_influence_coefficients', 'linear_influence_coefficients',
                              rows=np.arange(1, 3*n*m - 1, 3),
                              cols=np.arange(n*m),
                              val=np.ones(n*m))
        self.declare_partials('full_influence_coefficients', 'quadratic_influence_coefficients',
                              rows=np.arange(2, 3*n*m, 3),
                              cols=np.arange(n*m),
                              val=np.ones(n*m))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        a = inputs['constant_influence_coefficients']
        b = inputs['linear_influence_coefficients']
        c = inputs['quadratic_influence_coefficients']

        full = np.dstack((a, b, c)).reshape(-1)
        outputs['full_influence_coefficients'] = full


#  4. Establish RHS vector.


class BuildDirichletRHSVector(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('n', types=int)

    def setup(self):
        n = self.options['n']

        self.add_input('collocation_point_coordinates', shape=(2, n))
        self.add_input('alpha', units='rad')

        self.add_output('rhs_vector', shape=(n,))

        self.declare_partials('rhs_vector', 'collocation_point_coordinates',
                              rows=np.tile(np.arange(n), 2),  # See diary 13/10
                              cols=np.arange(2 * n))
        self.declare_partials('rhs_vector', 'alpha',
                              rows=np.arange(n),
                              cols=np.zeros(n))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        n = self.options['n']
        collocation_point_coordinates = inputs['collocation_point_coordinates']
        alpha = inputs['alpha']

        collocation_point_xs = collocation_point_coordinates[0, :].T
        collocation_point_ys = collocation_point_coordinates[1, :].T

        rhs_vector = -(collocation_point_xs * np.cos(alpha) + collocation_point_ys * np.sin(
            alpha))  # This should be negative, according to equation 11.84.

        outputs['rhs_vector'] = rhs_vector

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        n = self.options['n']
        collocation_point_coordinates = inputs['collocation_point_coordinates']
        alpha = inputs['alpha']

        collocation_point_xs = collocation_point_coordinates[np.newaxis, 0].squeeze()
        collocation_point_ys = collocation_point_coordinates[np.newaxis, 1].squeeze()

        dr_dcx = np.cos(alpha) * -np.ones(n)
        dr_dcy = np.sin(alpha) * -np.ones(n)
        dr_dc = np.concatenate((dr_dcx, dr_dcy))

        partials['rhs_vector', 'collocation_point_coordinates'] = dr_dc
        partials['rhs_vector', 'alpha'] = -(collocation_point_ys * np.cos(alpha) - collocation_point_xs * np.sin(alpha))


class QuadraticDoubletDirichletBCGroup(om.Group):

    def initialize(self):
        self.options.declare('num_col_points', types=int)
        self.options.declare('num_panels', types=int)

    def setup(self):
        n = self.options['num_col_points']
        m = self.options['num_panels']

        self.add_subsystem('ConstantDoubletInfluence', ConstantDoubletInfluence(num_col_points=n, num_panels=m),
                           promotes_inputs=['collocation_points_angles_from_first_panel_points',
                                            'collocation_points_angles_from_second_panel_points'],
                           promotes_outputs=['constant_influence_coefficients'])

        self.add_subsystem('LinearDoubletInfluence', LinearDoubletInfluence(num_col_points=n, num_panels=m),
                           promotes_inputs=['collocation_points_angles_from_first_panel_points',
                                            'collocation_points_angles_from_second_panel_points',
                                            'collocation_points_distances_from_first_panel_points',
                                            'collocation_points_distances_from_second_panel_points',
                                            'collocation_points_transformed_positions'],
                           promotes_outputs=['linear_influence_coefficients'])

        self.add_subsystem('QuadraticDoubletInfluence', QuadraticDoubletInfluence(num_col_points=n, num_panels=m),
                           promotes_inputs=['collocation_points_angles_from_first_panel_points',
                                            'collocation_points_angles_from_second_panel_points',
                                            'collocation_points_distances_from_first_panel_points',
                                            'collocation_points_distances_from_second_panel_points',
                                            'collocation_points_transformed_positions',
                                            'panel_lengths'],
                           promotes_outputs=['quadratic_influence_coefficients'])

        self.add_subsystem('BuildDirichletRHSVector', BuildDirichletRHSVector(n=n),
                           promotes_inputs=['collocation_point_coordinates',
                                            'alpha'],
                           promotes_outputs=['rhs_vector'])

        self.add_subsystem('InfluenceCoefficientAssembler', InfluenceCoefficientAssembler(num_col_points=n, num_panels=m),
                           promotes_inputs=['constant_influence_coefficients',
                                            'linear_influence_coefficients',
                                            'quadratic_influence_coefficients'],
                           promotes_outputs=['full_influence_coefficients'])


class WakeInfluenceCondition(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_col_points', types=int)
        self.options.declare('num_panels', types=int)

    def setup(self):
        n = self.options['num_col_points']
        m = self.options['num_panels']

        self.add_input('collocation_point_coordinates', shape=(2, n))
        self.add_input('panel_coordinates', shape=(2, m, 2))

        self.add_output('wake_influence_coefficients', shape=(n, 1))

        self.declare_partials('wake_influence_coefficients', 'collocation_point_coordinates',
                              rows=np.tile(np.arange(n), 2),
                              cols=np.arange(2*n))
        self.declare_partials('wake_influence_coefficients', 'panel_coordinates',
                              rows=np.concatenate((np.arange(n), np.arange(n))),
                              cols=np.concatenate((np.zeros(n), (2 * m) * np.ones(n))))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        collocation_point_coordinates = inputs['collocation_point_coordinates']
        te_point = inputs['panel_coordinates'][:, 0, 0]

        theta_1 = np.arctan2((collocation_point_coordinates[1, :] - te_point[1]),
                                          (collocation_point_coordinates[0, :] - te_point[0]))
        theta_2 = np.arctan2((collocation_point_coordinates[1, :] - te_point[1]),
                                          (collocation_point_coordinates[0, :] - 1E+6))

        wake_influence_coefficients = -(1/(2*np.pi)) * (theta_2 - theta_1)

        outputs['wake_influence_coefficients'] = wake_influence_coefficients

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        collocation_point_coordinates = inputs['collocation_point_coordinates']
        te_point = inputs['panel_coordinates'][:, 0, 0]

        xc_minus_xte = collocation_point_coordinates[0, :] - te_point[0]
        zc_minus_zte = collocation_point_coordinates[1, :] - te_point[1]

        dw_dcx = -1/(2*np.pi) * zc_minus_zte / (xc_minus_xte ** 2 + zc_minus_zte ** 2)
        dw_dcz = 1/(2*np.pi) * xc_minus_xte / (xc_minus_xte ** 2 + zc_minus_zte ** 2)

        partials['wake_influence_coefficients', 'collocation_point_coordinates'] = np.concatenate((dw_dcx.reshape(-1), dw_dcz.reshape(-1)))
        partials['wake_influence_coefficients', 'panel_coordinates'] = np.concatenate((-dw_dcx.reshape(-1), -dw_dcz.reshape(-1)))


class ContinuityCondition(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_panels', types=int)

    def setup(self):
        m = self.options['num_panels']

        self.add_input('panel_lengths', shape=(m,))
        self.add_output('continuity_rows', shape=(m-1, 3*m + 1))

        self.declare_partials('continuity_rows', 'panel_lengths',
                              rows=np.concatenate((1 + (3*m + 4)*np.arange(m-1), 2 + (3*m + 4)*np.arange(m-1))),
                              cols=np.tile(np.arange(m-1), 2))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        m = self.options['num_panels']
        panel_lengths = inputs['panel_lengths'][:-1]  # We slice this to take all but the last panel

        rows = np.tile(np.arange(m-1), 4)
        cols = (np.arange(0, 3*m-3, 3)+np.arange(4).reshape(-1, 1)).reshape(-1)
        data = np.concatenate((np.ones(m-1), panel_lengths, panel_lengths**2, -np.ones(m-1)))
        matrix = coo_matrix((data, (rows, cols)), shape=(m-1, 3*m + 1)).toarray()
        outputs['continuity_rows'] = matrix

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        m = self.options['num_panels']
        panel_lengths = inputs['panel_lengths'][:-1]  # We slice this to take all but the last panel

        partials['continuity_rows', 'panel_lengths'] = np.concatenate((np.ones(m-1), 2*panel_lengths))


class GradientContinuityCondition(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_panels', types=int)

    def setup(self):
        m = self.options['num_panels']

        self.add_input('panel_lengths', shape=(m,))
        self.add_output('gradient_continuity_rows', shape=(m-1, 3*m + 1))

        self.declare_partials('gradient_continuity_rows', 'panel_lengths',
                              rows=2 + (3*m + 4)*np.arange(m-1),
                              cols=np.arange(m-1))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        m = self.options['num_panels']
        panel_lengths = inputs['panel_lengths'][:-1]  # We slice this to take all but the last panel

        rows = np.tile(np.arange(m-1), 3)
        cols = (np.arange(1, 3*m-3, 3)+np.asarray([0, 1, 3]).reshape(-1, 1)).reshape(-1)
        data = np.concatenate((np.ones(m-1), 2*panel_lengths, -np.ones(m-1))).reshape(-1)
        matrix = coo_matrix((data, (rows, cols)), shape=(m-1, 3*m + 1)).toarray()
        outputs['gradient_continuity_rows'] = matrix

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        m = self.options['num_panels']

        partials['gradient_continuity_rows', 'panel_lengths'] = np.array(2*np.ones(m-1))


class KuttaCondition(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_panels', types=int)

    def setup(self):
        m = self.options['num_panels']

        self.add_input('panel_lengths', shape=(m,))
        self.add_input('panel_angles', shape=(m,))
        self.add_input('extra_collocation_point_coordinates', shape=(2, 1))
        self.add_input('panel_coordinates', shape=(2, m, 2))

        self.add_output('kutta_rows', shape=(2, 3*m + 1))

        self.declare_partials('kutta_rows', 'panel_lengths',
                              rows=np.array((3*m - 2, 3*m - 1, 2*(3*m + 1)-2)),
                              cols=np.repeat(m-1, 3))
        self.declare_partials('kutta_rows', 'panel_angles',
                              rows=np.array(((3*m + 1) + 1, 2*(3*m + 1)-3, 2*(3*m + 1)-2)),
                              cols=np.array((0, m-1, m-1)))
        self.declare_partials('kutta_rows', 'extra_collocation_point_coordinates',
                              rows=np.tile(np.array(((3*m + 1) + 1, 2*(3*m + 1)-3, 2*(3*m + 1)-2)), 2),
                              cols=np.repeat((0, 1), 3))
        self.declare_partials('kutta_rows', 'panel_coordinates',
                              rows=np.tile(np.array(((3*m + 1) + 1, 2*(3*m + 1)-3, 2*(3*m + 1)-2)), 2),
                              cols=np.repeat((0, 2*m), 3))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        m = self.options['num_panels']
        panel_angles = inputs['panel_angles']
        panel_length_N = inputs['panel_lengths'][-1]  # We slice this to take only the last panel
        te_point = inputs['panel_coordinates'][:, 0, 0]
        extra_point = inputs['extra_collocation_point_coordinates']

        te_angle = np.arctan2(te_point[1]-extra_point[1], te_point[0]-extra_point[0])
        lower_angle = panel_angles[0] - te_angle
        upper_angle = panel_angles[-1] - te_angle

        kutta = np.zeros((2, 3*m + 1))
        # Continuity
        kutta[0, 0] = -1
        kutta[0, -4] = 1
        kutta[0, -3] = panel_length_N
        kutta[0, -2] = (panel_length_N**2)
        kutta[0, -1] = -1
        # Gradient
        kutta[1, 1] = np.sin(lower_angle)
        kutta[1, -3] = np.sin(upper_angle)
        kutta[1, -2] = 2*panel_length_N*np.sin(upper_angle)
        outputs['kutta_rows'] = kutta

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        m = self.options['num_panels']
        panel_angles = inputs['panel_angles']
        panel_length_N = inputs['panel_lengths'][-1]  # We slice this to take only the last panel
        te_point = inputs['panel_coordinates'][:, 0, 0]
        extra_point = inputs['extra_collocation_point_coordinates']

        delta_x = te_point[0]-extra_point[0]
        delta_y = te_point[1]-extra_point[1]

        te_angle = np.arctan2(te_point[1]-extra_point[1], te_point[0]-extra_point[0])
        lower_angle = panel_angles[0] - te_angle
        upper_angle = panel_angles[-1] - te_angle

        wrt_theta0_factor = np.cos(-lower_angle)/(delta_x**2 + delta_y**2)  # Saves some calculations
        wrt_thetam1_factor = np.cos(-upper_angle)/(delta_x**2 + delta_y**2)

        dkutta_dcoords = np.array((delta_y*wrt_theta0_factor, delta_y*wrt_thetam1_factor, 2*panel_length_N*delta_y*wrt_thetam1_factor,
                                  -delta_x*wrt_theta0_factor, -delta_x*wrt_thetam1_factor, -2*panel_length_N*delta_x*wrt_thetam1_factor)).reshape(-1)

        partials['kutta_rows', 'panel_lengths'] = np.concatenate((np.array([1]), np.array([2*panel_length_N]), 2*np.sin(upper_angle)))
        partials['kutta_rows', 'panel_angles'] = np.concatenate((np.cos(lower_angle), np.cos(upper_angle), 2*panel_length_N*np.cos(upper_angle))).reshape(-1)
        partials['kutta_rows', 'panel_coordinates'] = dkutta_dcoords
        partials['kutta_rows', 'extra_collocation_point_coordinates'] = -dkutta_dcoords


class LinearProblemAssembler(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('n', types=int)

    def setup(self):
        n = self.options['n']

        self.add_input('full_influence_coefficients', shape=(n, 3*n))
        self.add_input('extra_influence_coefficients', shape=(1, 3*n))
        self.add_input('continuity_rows', shape=(n-1, 3*n + 1))
        self.add_input('gradient_continuity_rows', shape=(n-1, 3*n + 1))
        self.add_input('kutta_rows', shape=(2, 3*n + 1))
        self.add_input('wake_influence_coefficients', shape=(n, 1))
        self.add_input('extra_wake_influence_coefficients', shape=(1,))
        self.add_input('rhs_vector', shape=(n,))
        self.add_input('extra_rhs_vector', shape=(1,))

        self.add_output('A_matrix', shape=(3*n + 1, 3*n + 1))
        self.add_output('b_vector', shape=(3*n + 1,))

        self.declare_partials('A_matrix', 'full_influence_coefficients',
                              rows=(np.arange(3*n)+((3*n + 1)*np.arange(n).reshape(-1, 1))).reshape(-1),
                              cols=np.arange(3*n*n),
                              val=np.ones(3*n*n))
        self.declare_partials('A_matrix', 'extra_influence_coefficients',
                              rows=np.arange(3*n)+(n*(3*n + 1)),
                              cols=np.arange(3*n),
                              val=np.ones(3*n))
        self.declare_partials('A_matrix', 'continuity_rows',
                              rows=((n + 1)*(3*n + 1)) + np.arange(((n - 1)*(3*n + 1))),
                              cols=np.arange((n-1)*(3*n + 1)),
                              val=np.ones((n-1)*(3*n + 1)))
        self.declare_partials('A_matrix', 'gradient_continuity_rows',
                              rows=((2*n)*(3*n + 1)) + np.arange(((n - 1)*(3*n + 1))),
                              cols=np.arange((n-1)*(3*n + 1)),
                              val=np.ones((n-1)*(3*n + 1)))
        self.declare_partials('A_matrix', 'kutta_rows',
                              rows=np.arange(9*(n**2)-1, (3*n + 1)**2),
                              cols=np.arange(6*n + 2),
                              val=np.ones(6*n + 2))
        self.declare_partials('A_matrix', 'wake_influence_coefficients',
                              rows=(3*n+1)*(np.arange(n))+3*n,
                              cols=np.arange(n),
                              val=np.ones(n))
        self.declare_partials('A_matrix', 'extra_wake_influence_coefficients',
                              rows=np.ones(1)*((3*n+1)*n+3*n),
                              cols=np.zeros(1),
                              val=np.ones(1))
        self.declare_partials('A_matrix', 'rhs_vector', dependent=False)
        self.declare_partials('A_matrix', 'extra_rhs_vector', dependent=False)

        self.declare_partials('b_vector', 'full_influence_coefficients', dependent=False)
        self.declare_partials('b_vector', 'extra_influence_coefficients', dependent=False)
        self.declare_partials('b_vector', 'continuity_rows', dependent=False)
        self.declare_partials('b_vector', 'gradient_continuity_rows', dependent=False)
        self.declare_partials('b_vector', 'wake_influence_coefficients', dependent=False)
        self.declare_partials('b_vector', 'extra_wake_influence_coefficients', dependent=False)
        self.declare_partials('b_vector', 'rhs_vector',
                              rows=np.arange(n),
                              cols=np.arange(n),
                              val=np.ones(n))
        self.declare_partials('b_vector', 'extra_rhs_vector',
                              rows=np.ones(1)*n,
                              cols=np.zeros(1),
                              val=1)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        n = self.options['n']
        full_influence_coefficients = inputs['full_influence_coefficients']
        extra_influence_coefficients = inputs['extra_influence_coefficients']
        wake_influence_coefficients = inputs['wake_influence_coefficients']
        extra_wake_influence_coefficients = inputs['extra_wake_influence_coefficients']
        continuity_rows = inputs['continuity_rows']
        gradient_continuity_rows = inputs['gradient_continuity_rows']
        kutta_rows = inputs['kutta_rows']
        rhs_vector = inputs['rhs_vector']
        extra_rhs_vector = inputs['extra_rhs_vector']

        A_matrix = np.zeros((3*n + 1, 3*n + 1))
        A_matrix[:n, :3*n] = full_influence_coefficients
        A_matrix[n, :3*n] = extra_influence_coefficients
        A_matrix[:n, 3*n] = wake_influence_coefficients.reshape(-1)
        A_matrix[n, 3*n] = extra_wake_influence_coefficients
        A_matrix[n+1:2*n, :] = continuity_rows
        A_matrix[2*n:3*n - 1, :] = gradient_continuity_rows
        A_matrix[3*n - 1:3*n + 1, :] = kutta_rows

        b_vector = np.zeros((3*n+1))
        b_vector[:n] = rhs_vector
        b_vector[n] = extra_rhs_vector

        outputs['A_matrix'] = A_matrix
        outputs['b_vector'] = b_vector


class LinearProblemGeneratorGroup(om.Group):

    def initialize(self):
        self.options.declare('num_points')

    def setup(self):
        num_points = self.options['num_points']
        n = num_points - 1

        self.add_subsystem('PrepareExtraCollocationPoint', PrepareExtraCollocationPoint(num_col_points=n),
                           promotes_inputs=['collocation_point_coordinates',
                                            'panel_coordinates',
                                            'panel_angles',
                                            'second_panel_coordinate_transformed_positions'],
                           promotes_outputs=['extra_collocation_point_distances_from_first_panel_points',
                                             'extra_collocation_point_distances_from_second_panel_points',
                                             'extra_collocation_point_angles_from_first_panel_points',
                                             'extra_collocation_point_angles_from_second_panel_points',
                                             'extra_collocation_point_coordinates',
                                             'extra_collocation_point_transformed_positions'])

        self.add_subsystem('QuadraticDirichletBCGroup', QuadraticDoubletDirichletBCGroup(num_col_points=n, num_panels=n),
                           promotes_inputs=['collocation_points_angles_from_first_panel_points',
                                            'collocation_points_angles_from_second_panel_points',
                                            'collocation_points_distances_from_first_panel_points',
                                            'collocation_points_distances_from_second_panel_points',
                                            'collocation_point_coordinates',
                                            'collocation_points_transformed_positions',
                                            'panel_lengths',
                                            'alpha'],
                           promotes_outputs=['full_influence_coefficients',
                                             'rhs_vector'])

        self.add_subsystem('ExtraQuadraticDirichletBCGroup', QuadraticDoubletDirichletBCGroup(num_col_points=1, num_panels=n),
                           promotes_inputs=[('collocation_points_angles_from_first_panel_points', 'extra_collocation_point_angles_from_first_panel_points'),
                                            ('collocation_points_angles_from_second_panel_points', 'extra_collocation_point_angles_from_second_panel_points'),
                                            ('collocation_points_distances_from_first_panel_points', 'extra_collocation_point_distances_from_first_panel_points'),
                                            ('collocation_points_distances_from_second_panel_points', 'extra_collocation_point_distances_from_second_panel_points'),
                                            ('collocation_point_coordinates', 'extra_collocation_point_coordinates'),
                                            ('collocation_points_transformed_positions', 'extra_collocation_point_transformed_positions',),
                                            'panel_lengths',
                                            'alpha'],
                           promotes_outputs=[('full_influence_coefficients', 'extra_influence_coefficients'),
                                             ('rhs_vector', 'extra_rhs_vector')])

        self.add_subsystem('WakeInfluenceCondition', WakeInfluenceCondition(num_col_points=n, num_panels=n),
                           promotes_inputs=['collocation_point_coordinates',
                                            'panel_coordinates'],
                           promotes_outputs=['wake_influence_coefficients'])

        self.add_subsystem('ExtraWakeInfluenceCondition', WakeInfluenceCondition(num_col_points=1, num_panels=n),
                           promotes_inputs=[('collocation_point_coordinates', 'extra_collocation_point_coordinates'),
                                            'panel_coordinates'],
                           promotes_outputs=[('wake_influence_coefficients', 'extra_wake_influence_coefficients')])

        self.add_subsystem('ContinuityCondition', ContinuityCondition(num_panels=n),
                           promotes_inputs=['panel_lengths'],
                           promotes_outputs=['continuity_rows'])

        self.add_subsystem('GradientContinuityCondition', GradientContinuityCondition(num_panels=n),
                           promotes_inputs=['panel_lengths'],
                           promotes_outputs=['gradient_continuity_rows'])

        self.add_subsystem('KuttaCondition', KuttaCondition(num_panels=n),
                           promotes_inputs=['panel_lengths',
                                            'panel_angles',
                                            'extra_collocation_point_coordinates',
                                            'panel_coordinates'],
                           promotes_outputs=['kutta_rows'])

        self.add_subsystem('LinearProblemAssembler', LinearProblemAssembler(n=n),
                           promotes_inputs=['full_influence_coefficients',
                                            'extra_influence_coefficients',
                                            'continuity_rows',
                                            'gradient_continuity_rows',
                                            'kutta_rows',
                                            'wake_influence_coefficients',
                                            'extra_wake_influence_coefficients',
                                            'rhs_vector',
                                            'extra_rhs_vector'],
                           promotes_outputs=['A_matrix',
                                             'b_vector'])

#  5. Solve.


class SolveMatrixGroup(om.Group):

    def initialize(self):
        self.options.declare('num_points')

    def setup(self):
        num_points = self.options['num_points']
        n = num_points-1

        self.add_subsystem('lin', om.LinearSystemComp(size=3*n + 1),
                           promotes_inputs=[('A', 'A_matrix'),
                                            ('b', 'b_vector')],
                           promotes_outputs=[('x', 'doublet_strengths')])

#  6. Calculate forces.


class CalculateExternalTangentialVelocities(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('n', types=int)

    def setup(self):
        n = self.options['n']

        self.add_input('panel_lengths', shape=(n,))
        self.add_input('doublet_strengths', shape=(3*n + 1,))

        self.add_output('external_tangential_velocities', shape=(n,))

        self.declare_partials('external_tangential_velocities', 'panel_lengths',
                              rows=np.arange(n),
                              cols=np.arange(n))
        self.declare_partials('external_tangential_velocities', 'doublet_strengths',
                              rows=np.tile(np.arange(n), 2),
                              cols=np.concatenate((np.arange(1, 3*n, 3), np.arange(2, 3*n, 3))))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        panel_lengths = inputs['panel_lengths']
        doublet_strengths = inputs['doublet_strengths'][:-1].reshape(-1, 3)  # Exclude the wake doublet strength, reshape it to mu_0i, mu_1i, mu_2i

        mu_1 = doublet_strengths[:, 1]
        mu_2 = doublet_strengths[:, 2]

        external_tangential_velocities = -mu_2*panel_lengths - mu_1

        outputs['external_tangential_velocities'] = external_tangential_velocities

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        n = self.options['n']
        panel_lengths = inputs['panel_lengths']
        doublet_strengths = inputs['doublet_strengths'][:-1].reshape(-1, 3)  # Exclude the wake doublet strength, reshape it to mu_0i, mu_1i, mu_2i

        mu_2 = doublet_strengths[:, 2]

        partials['external_tangential_velocities', 'panel_lengths'] = -mu_2
        partials['external_tangential_velocities', 'doublet_strengths'] = np.concatenate((-np.ones(n), -panel_lengths))


class CalculatePressureCoefficients(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('n', types=int)

    def setup(self):
        n = self.options['n']

        self.add_input('external_tangential_velocities', shape=(n,))

        self.add_output('pressure_coefficients', shape=(n,))

        self.declare_partials('pressure_coefficients', 'external_tangential_velocities',
                              rows=np.arange(n),
                              cols=np.arange(n))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        external_tangential_velocities = inputs['external_tangential_velocities']

        pressure_coefficients = 1 - external_tangential_velocities ** 2

        outputs['pressure_coefficients'] = pressure_coefficients

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        external_tangential_velocities = inputs['external_tangential_velocities']

        partials['pressure_coefficients', 'external_tangential_velocities'] = -2*external_tangential_velocities


class LiftCoefficient(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('n', types=int)

    def setup(self):
        n = self.options['n']

        self.add_input('doublet_strengths', shape=(3*n + 1,))

        self.add_output('lift_coefficient')

        self.declare_partials('lift_coefficient', 'doublet_strengths',
                              rows=np.array([0]),
                              cols=np.array([3*n]),
                              val=-2)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        mu = inputs['doublet_strengths']

        outputs['lift_coefficient'] = -2*mu[-1]  # Why -2? No idea, but it works and the deadline is fast approaching


class LoadCalculationGroup(om.Group):

    def initialize(self):
        self.options.declare('num_points')

    def setup(self):
        num_points = self.options['num_points']
        n = num_points - 1

        self.add_subsystem('CalculateExternalTangentialVelocities', CalculateExternalTangentialVelocities(n=n),
                           promotes_inputs=['panel_lengths', 'doublet_strengths'],
                           promotes_outputs=['external_tangential_velocities'])

        self.add_subsystem('CalculatePressureCoefficients', CalculatePressureCoefficients(n=n),
                           promotes_inputs=['external_tangential_velocities'],
                           promotes_outputs=['pressure_coefficients'])

        self.add_subsystem('LiftCoefficient', LiftCoefficient(n=n),
                           promotes_inputs=['doublet_strengths'],
                           promotes_outputs=['lift_coefficient'])


class LocateStagnationPoint(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('n', types=int)

    def setup(self):
        n = self.options['n']

        self.add_input('external_tangential_velocities', shape=(n,))
        self.add_input('panel_lengths', shape=(n,))

        self.add_output('stagnation_point_position')

        self.declare_partials('stagnation_point_position', 'external_tangential_velocities')
        self.declare_partials('stagnation_point_position', 'panel_lengths')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        external_tangential_velocities = inputs['external_tangential_velocities']
        panel_lengths = inputs['panel_lengths']

        stagnation_index = np.where(np.diff(np.sign(external_tangential_velocities)))[0][0]

        s_minus = np.sum(panel_lengths[:stagnation_index+1])
        s_plus = s_minus+panel_lengths[stagnation_index+1]

        u_minus = external_tangential_velocities[stagnation_index]
        u_plus = external_tangential_velocities[stagnation_index+1]

        s_stagnation = (u_plus*s_minus - u_minus*s_plus)/(u_plus - u_minus)
        outputs['stagnation_point_position'] = s_stagnation

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        n = self.options['n']

        external_tangential_velocities = inputs['external_tangential_velocities']
        panel_lengths = inputs['panel_lengths']

        stagnation_index = np.where(np.diff(np.sign(external_tangential_velocities)))[0][0]

        s_minus = np.sum(panel_lengths[:stagnation_index+1])
        s_plus = s_minus+panel_lengths[stagnation_index+1]

        u_minus = external_tangential_velocities[stagnation_index]
        u_plus = external_tangential_velocities[stagnation_index+1]

        ds0_du_minus = -((s_plus - s_minus)*u_plus)/(u_plus-u_minus)**2
        ds0_du_plus = ((s_plus - s_minus)*u_minus)/(u_plus-u_minus)**2

        ds0_due = np.zeros(n)
        ds0_due[stagnation_index] = ds0_du_minus
        ds0_due[stagnation_index+1] = ds0_du_plus

        ds0_dl = np.zeros(n)
        ds0_dl[:stagnation_index+1] = 1
        ds0_dl[stagnation_index+1] = -u_minus/(u_plus - u_minus)

        partials['stagnation_point_position', 'external_tangential_velocities'] = ds0_due
        partials['stagnation_point_position', 'panel_lengths'] = ds0_dl


class InviscidGroup(om.Group):

    def initialize(self):
        self.options.declare('num_points')

    def setup(self):
        num_points = self.options['num_points']

        self.add_subsystem('GeometryAndDiscretisationGroup', GeometryAndDiscretisationGroup(num_points=num_points),
                           promotes_inputs=['aerofoil_ccw_coordinates'],
                           promotes_outputs=['panel_angles',
                                             'panel_lengths',
                                             'panel_coordinates',
                                             'second_panel_coordinate_transformed_positions',
                                             'collocation_point_coordinates',
                                             'collocation_points_angles_from_first_panel_points',
                                             'collocation_points_angles_from_second_panel_points',
                                             'collocation_points_distances_from_first_panel_points',
                                             'collocation_points_distances_from_second_panel_points',
                                             'collocation_points_transformed_positions'])
        self.add_subsystem('LinearProblemGeneratorGroup',
                           LinearProblemGeneratorGroup(num_points=num_points),
                           promotes_inputs=['collocation_points_angles_from_first_panel_points',
                                            'collocation_points_angles_from_second_panel_points',
                                            'collocation_points_distances_from_first_panel_points',
                                            'collocation_points_distances_from_second_panel_points',
                                            'collocation_point_coordinates',
                                            'collocation_points_transformed_positions',
                                            'second_panel_coordinate_transformed_positions',
                                            'panel_coordinates',
                                            'panel_lengths',
                                            'panel_angles',
                                            'alpha'],
                           promotes_outputs=['A_matrix',
                                             'b_vector',
                                             'extra_collocation_point_coordinates'])
        self.add_subsystem('SolveMatrixGroup', SolveMatrixGroup(num_points=num_points),
                           promotes_inputs=['A_matrix',
                                            'b_vector'],
                           promotes_outputs=['doublet_strengths'])
        self.add_subsystem('LoadCalculationGroup', LoadCalculationGroup(num_points=num_points),
                           promotes_inputs=['doublet_strengths',
                                            'panel_lengths'],
                           promotes_outputs=['external_tangential_velocities',
                                             'pressure_coefficients',
                                             'lift_coefficient'])
        # self.add_subsystem('LocateStagnationPoint', LocateStagnationPoint(n=num_points-1),
        #                    promotes_inputs=['external_tangential_velocities',
        #                                     'panel_lengths'],
        #                    promotes_outputs=['stagnation_point_position'])


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from CST import CSTAerofoil

    num_aerofoil_points = 5
    num_aerofoil_weights = 3
    alpha=3

    p = om.Problem(model=om.Group())

    p.model.add_subsystem('Aerofoil',
                          CSTAerofoil(num_points=num_aerofoil_points,
                                      num_upper_weights=num_aerofoil_weights,
                                      num_lower_weights=num_aerofoil_weights),
                          promotes_inputs=['upper_weights',
                                           'lower_weights',
                                           'le_weight',
                                           'te_weight'],
                          promotes_outputs=['aerofoil_ccw_coordinates'])

    p.model.add_subsystem('Inviscid',
                          InviscidGroup(num_points=num_aerofoil_points),
                          promotes_inputs=['aerofoil_ccw_coordinates',
                                           'alpha'],
                          promotes_outputs=['lift_coefficient'])

    p.model.set_input_defaults('alpha', alpha, units='deg')
    p.model.set_input_defaults('le_weight', 0)
    p.model.set_input_defaults('te_weight', 0)
    p.model.set_input_defaults('upper_weights', [0.206, 0.2728, 0.2292])
    p.model.set_input_defaults('lower_weights', [-0.1294, -0.0036, -0.0666])

    p.setup()

    p.run_model()

    with np.printoptions(linewidth=400, edgeitems=200, formatter={'float_kind': '{:5.4f}'.format}):
        #formatter={'float_kind': '{:5.2f}'.format})
          p.check_partials(show_only_incorrect=True, compact_print=True, method='fd')