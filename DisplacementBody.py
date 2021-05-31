import openmdao.api as om
import numpy as np

# Combine to single array


class CalculateAverageNormals(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_base_points', types=int)

    def setup(self):
        n = self.options['num_base_points']

        self.add_input('aerofoil_cw_coordinates', shape=(2, n))

        self.add_output('average_normals', shape=(n,))

        self.declare_partials('average_normals', 'aerofoil_cw_coordinates',
                              rows=np.tile(np.concatenate(([0, 0], np.arange(1, n-1), np.arange(1, n-1), [n-1, n-1])), 2),
                              cols=np.concatenate(([0, 1], np.arange(n-2), np.arange(2, n), [n-2, n-1],
                                                   np.concatenate(([0, 1], np.arange(n-2), np.arange(2, n), [n-2, n-1]))+n)))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        num_base_points = self.options['num_base_points']

        aerofoil_cw_coordinates = inputs['aerofoil_cw_coordinates']
        x = aerofoil_cw_coordinates[0, :]
        y = aerofoil_cw_coordinates[1, :]

        # Internal points use average of point before and after
        # TE points just use neighbouring point
        avg_normals = np.zeros(num_base_points)
        avg_normals[0] = np.arctan2(y[1]-y[0], x[1]-x[0]) + np.pi/2
        avg_normals[1:-1] = np.arctan2(y[2:]-y[:-2], x[2:]-x[:-2]) + np.pi/2
        avg_normals[-1] = np.arctan2(y[-1]-y[-2], x[-1]-x[-2]) + np.pi/2

        outputs['average_normals'] = avg_normals

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        aerofoil_cw_coordinates = inputs['aerofoil_cw_coordinates']
        x = aerofoil_cw_coordinates[0, :]
        y = aerofoil_cw_coordinates[1, :]

        dnavg0_dx0 = (y[1] - y[0]) / ((x[1] - x[0]) ** 2 + (y[1] - y[0]) ** 2)
        dnavg0_dx1 = -dnavg0_dx0
        dnavg0_dy0 = -(x[1] - x[0]) / ((x[1] - x[0]) ** 2 + (y[1] - y[0]) ** 2)
        dnavg0_dy1 = -dnavg0_dy0

        dnavgi_dxim1 = (y[2:] - y[:-2]) / ((x[2:] - x[:-2]) ** 2 + (y[2:] - y[:-2]) ** 2)  # wrt x_(i-1)
        dnavgi_dxip1 = -dnavgi_dxim1  # wrt x_(i+1)
        dnavgi_dyim1 = -(x[2:] - x[:-2]) / ((x[2:] - x[:-2]) ** 2 + (y[2:] - y[:-2]) ** 2)  # wrt y_(i-1)
        dnavgi_dyip1 = -dnavgi_dyim1  # wrt y_(i+1)

        dnavgm1_dxm1 = (y[-1] - y[-2]) / ((x[-1] - x[-2]) ** 2 + (y[-1] - y[-2]) ** 2)  # n_avg[-1] wrt x[-1]
        dnavgm1_dxm2 = -dnavgm1_dxm1  # n_avg[-1] wrt x[-2]
        dnavgm1_dym1 = -(x[-1] - x[-2]) / ((x[-1] - x[-2]) ** 2 + (y[-1] - y[-2]) ** 2)  # n_avg[-1] wrt y[-1]
        dnavgm1_dym2 = -dnavgm1_dym1  # n_avg[-1] wrt y[-2]

        partials['average_normals', 'aerofoil_cw_coordinates'] = np.concatenate(([dnavg0_dx0, dnavg0_dx1], dnavgi_dxim1, dnavgi_dxip1,[dnavgm1_dxm1, dnavgm1_dxm2],
                                                                                 [dnavg0_dy0, dnavg0_dy1], dnavgi_dyim1, dnavgi_dyip1, [dnavgm1_dym1, dnavgm1_dym2]))


class ApplyDisplacementThickness(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_base_points', types=int)

    def setup(self):
        n = self.options['num_base_points']

        self.add_input('aerofoil_cw_coordinates', shape=(2, n))
        self.add_input('displacement_thickness', 0.01, shape=(n,))
        self.add_input('average_normals', shape=(n,))

        self.add_output('displaced_cw_coordinates', shape=(2, n))

        self.declare_partials('displaced_cw_coordinates', 'aerofoil_cw_coordinates',
                              rows=np.arange(2*n),
                              cols=np.arange(2*n),
                              val=1)
        self.declare_partials('displaced_cw_coordinates', 'displacement_thickness',
                              rows=np.arange(2*n),
                              cols=np.tile(np.arange(n), 2))
        self.declare_partials('displaced_cw_coordinates', 'average_normals',
                              rows=np.arange(2*n),
                              cols=np.tile(np.arange(n), 2))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        aerofoil_cw_coordinates = inputs['aerofoil_cw_coordinates']
        avg_normals = inputs['average_normals']
        displacement = inputs['displacement_thickness']
        x = aerofoil_cw_coordinates[0, :]
        y = aerofoil_cw_coordinates[1, :]

        x_displaced = x + displacement*np.cos(avg_normals)
        y_displaced = y + displacement*np.sin(avg_normals)

        outputs['displaced_cw_coordinates'] = np.array([x_displaced, y_displaced])

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        avg_normals = inputs['average_normals']
        displacement = inputs['displacement_thickness']

        partials['displaced_cw_coordinates', 'displacement_thickness'] = np.concatenate((np.cos(avg_normals), np.sin(avg_normals)))
        partials['displaced_cw_coordinates', 'average_normals'] = np.concatenate((-displacement*np.sin(avg_normals), displacement*np.cos(avg_normals)))


class ComputeWakePosition(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_wake_points_per_side', types=int)

    def setup(self):
        n = self.options['num_wake_points_per_side']

        self.add_input('wake_camber_lengths', shape=(n,))
        self.add_input('wake_camber_angles', shape=(n,))
        self.add_input('wake_thickness', shape=(n,))
        self.add_input('displaced_cw_coordinates', [[1.07, 1.07], [-0.7, 0.7]], src_indices=[[[0, 0], [0, -1]], [[1, 0], [1, -1]]], shape=(2, 2))  # Get the two trailing edge displaced coords

        self.add_output('upper_wake_coordinates', shape=(2, n))
        self.add_output('lower_wake_coordinates', shape=(2, n))

        stops = np.repeat(n, n - 1)  # Adapted from https://codereview.stackexchange.com/questions/83018/vectorized-numpy-version-of-arange-with-multiple-start-stop
        starts = np.arange(1, n)
        l = stops - starts
        tri_pattern = np.concatenate((np.arange(n), np.repeat(stops - l.cumsum(), l) + np.arange(l.sum())))
        self.declare_partials('upper_wake_coordinates', 'wake_camber_lengths',
                              rows=np.concatenate((tri_pattern, tri_pattern+n)),
                              cols=np.tile(np.repeat(np.arange(n), np.arange(n)[::-1]+1), 2))
        self.declare_partials('upper_wake_coordinates', 'wake_camber_angles',
                              rows=np.concatenate((tri_pattern, tri_pattern+n)),
                              cols=np.tile(np.repeat(np.arange(n), np.arange(n)[::-1]+1), 2))
        self.declare_partials('upper_wake_coordinates', 'wake_thickness',
                              rows=np.arange(2*n),
                              cols=np.tile(np.arange(n), 2))
        self.declare_partials('upper_wake_coordinates', 'displaced_cw_coordinates',
                              rows=np.concatenate((np.tile(np.arange(n), 2), np.tile(np.arange(n, 2*n), 2))),
                              cols=np.repeat(np.arange(4), n),
                              val=0.5)
        self.declare_partials('lower_wake_coordinates', 'wake_camber_lengths',
                              rows=np.concatenate((tri_pattern, tri_pattern+n)),
                              cols=np.tile(np.repeat(np.arange(n), np.arange(n)[::-1]+1), 2))
        self.declare_partials('lower_wake_coordinates', 'wake_camber_angles',
                              rows=np.concatenate((tri_pattern, tri_pattern+n)),
                              cols=np.tile(np.repeat(np.arange(n), np.arange(n)[::-1]+1), 2))
        self.declare_partials('lower_wake_coordinates', 'wake_thickness',
                              rows=np.arange(2*n),
                              cols=np.tile(np.arange(n), 2))
        self.declare_partials('lower_wake_coordinates', 'displaced_cw_coordinates',
                              rows=np.concatenate((np.tile(np.arange(n), 2), np.tile(np.arange(n, 2*n), 2))),
                              cols=np.repeat(np.arange(4), n),
                              val=0.5)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        l = inputs['wake_camber_lengths']
        theta = inputs['wake_camber_angles']
        h = inputs['wake_thickness']/2
        disp_TE_coords = inputs['displaced_cw_coordinates']

        h[-1] = 0 # Make wake come to a point at the end

        xTEl = disp_TE_coords[0, 0]
        yTEl = disp_TE_coords[1, 0]
        xTEu = disp_TE_coords[0, 1]
        yTEu = disp_TE_coords[1, 1]

        x0 = (xTEl+xTEu)/2
        y0 = (yTEl+yTEu)/2

        up_ang = theta + np.pi/4
        down_ang = theta - np.pi/4

        x_camber = x0 + np.cumsum(l*np.cos(theta))
        y_camber = y0 + np.cumsum(l*np.sin(theta))

        xu = x_camber + h*np.cos(up_ang)
        yu = y_camber + h*np.sin(up_ang)
        xl = x_camber + h*np.cos(down_ang)
        yl = y_camber + h*np.sin(down_ang)

        outputs['upper_wake_coordinates'] = np.array([xu, yu])
        outputs['lower_wake_coordinates'] = np.array([xl, yl])

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        n = self.options['num_wake_points_per_side']

        l = inputs['wake_camber_lengths']
        theta = inputs['wake_camber_angles']
        h = inputs['wake_thickness']/2
        disp_TE_coords = inputs['displaced_cw_coordinates']

        h[-1] = 0

        xTEl = disp_TE_coords[0, 0]
        yTEl = disp_TE_coords[1, 0]
        xTEu = disp_TE_coords[0, 1]
        yTEu = disp_TE_coords[1, 1]

        x0 = (xTEl+xTEu)/2
        y0 = (yTEl+yTEu)/2

        up_ang = theta + np.pi/4
        down_ang = theta - np.pi/4

        x_camber = x0 + np.cumsum(l*np.cos(theta))
        y_camber = y0 + np.cumsum(l*np.sin(theta))

        di_in_triag = np.cumsum(np.hstack((0, n - np.arange(n-1))))

        dxl_dl = np.repeat(np.cos(theta), np.arange(n)[::-1]+1)
        dyl_dl = np.repeat(np.sin(theta), np.arange(n)[::-1]+1)
        dxl_dtheta = np.repeat(-l*np.sin(theta), np.arange(n)[::-1]+1)
        dxl_dtheta[di_in_triag] -= h*np.sin(down_ang)
        dyl_dtheta = np.repeat(l*np.cos(theta), np.arange(n)[::-1]+1)
        dyl_dtheta[di_in_triag] += h*np.cos(down_ang)
        dxl_dh = np.cos(down_ang)/2
        dxl_dh[-1] = 0
        dyl_dh = np.sin(down_ang)/2
        dyl_dh[-1] = 0

        dxu_dl = np.repeat(np.cos(theta), np.arange(n)[::-1]+1)
        dyu_dl = np.repeat(np.sin(theta), np.arange(n)[::-1]+1)
        dxu_dtheta = np.repeat(-l*np.sin(theta), np.arange(n)[::-1]+1)
        dxu_dtheta[di_in_triag] -= h*np.sin(up_ang)
        dyu_dtheta = np.repeat(l*np.cos(theta), np.arange(n)[::-1]+1)
        dyu_dtheta[di_in_triag] += h*np.cos(up_ang)
        dxu_dh = np.cos(up_ang)/2
        dxu_dh[-1] = 0
        dyu_dh = np.sin(up_ang)/2
        dyu_dh[-1] = 0

        partials['upper_wake_coordinates', 'wake_camber_lengths'] = np.concatenate((dxu_dl, dyu_dl))
        partials['lower_wake_coordinates', 'wake_camber_lengths'] = np.concatenate((dxl_dl, dyl_dl))
        partials['upper_wake_coordinates', 'wake_camber_angles'] = np.concatenate((dxu_dtheta, dyu_dtheta))
        partials['lower_wake_coordinates', 'wake_camber_angles'] = np.concatenate((dxl_dtheta, dyl_dtheta))
        partials['upper_wake_coordinates', 'wake_thickness'] = np.concatenate((dxu_dh, dyu_dh))
        partials['lower_wake_coordinates', 'wake_thickness'] = np.concatenate((dxl_dh, dyl_dh))


class CombineDisplacementCoordinates(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_wake_points_per_side', types=int)
        self.options.declare('num_base_points', types=int)

    def setup(self):
        n_base = self.options['num_base_points']
        n_wake = self.options['num_wake_points_per_side']

        self.add_input('displaced_cw_coordinates', shape=(2, n_base))
        self.add_input('upper_wake_coordinates', shape=(2, n_wake))
        self.add_input('lower_wake_coordinates', shape=(2, n_wake))

        self.add_output('displacement_body_coordinates', shape=(2, n_base+2*n_wake))

        self.declare_partials('displacement_body_coordinates', 'displaced_cw_coordinates',
                              rows=np.concatenate((np.arange(n_wake, n_base+n_wake), np.arange(n_wake, n_base+n_wake)+n_base+2*n_wake)),
                              cols=np.arange(2*n_base),
                              val=1)
        self.declare_partials('displacement_body_coordinates', 'upper_wake_coordinates',
                              rows=np.concatenate((np.arange(n_base+n_wake, n_base+2*n_wake), np.arange(n_base+n_wake, n_base+2*n_wake)+n_base+2*n_wake)),
                              cols=np.arange(2*n_wake),
                              val=1)
        self.declare_partials('displacement_body_coordinates', 'lower_wake_coordinates',
                              rows=np.concatenate((np.arange(n_wake), np.arange(n_wake)+n_base+2*n_wake)),
                              cols=np.concatenate((np.arange(n_wake-1, -1, -1), np.arange(2*n_wake - 1, n_wake - 1, -1))),
                              val=1)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        n_base = self.options['num_base_points']
        n_wake = self.options['num_wake_points_per_side']

        displaced_cw_coordinates = inputs['displaced_cw_coordinates']
        upper_wake_coordinates = inputs['upper_wake_coordinates']
        lower_wake_coordinates = inputs['lower_wake_coordinates']

        displacement_body = np.zeros((2, n_base+2*n_wake))
        displacement_body[:, :n_wake] = np.flip(lower_wake_coordinates, 1)
        displacement_body[:, n_wake:n_wake+n_base] = displaced_cw_coordinates
        displacement_body[:, n_wake+n_base:] = upper_wake_coordinates

        outputs['displacement_body_coordinates'] = displacement_body


class DisplacementBodyGroup(om.Group):
    
    def initialize(self):
        self.options.declare('num_wake_points_per_side', types=int)
        self.options.declare('num_base_points', types=int)

    def setup(self):
        n_base = self.options['num_base_points']
        n_wake = self.options['num_wake_points_per_side']

        self.add_subsystem('CalculateAverageNormals', CalculateAverageNormals(num_base_points=n_base),
                           promotes_inputs=['aerofoil_cw_coordinates'],
                           promotes_outputs=['average_normals'])
        self.add_subsystem('ApplyDisplacementThickness', ApplyDisplacementThickness(num_base_points=n_base),
                           promotes_inputs=['aerofoil_cw_coordinates',
                                            'displacement_thickness',
                                            'average_normals'],
                           promotes_outputs=['displaced_cw_coordinates'])
        self.add_subsystem('ComputeWakePosition', ComputeWakePosition(num_wake_points_per_side=n_wake),
                           promotes_inputs=['wake_camber_lengths',
                                            'wake_camber_angles',
                                            'wake_thickness',
                                            'displaced_cw_coordinates'],
                           promotes_outputs=['upper_wake_coordinates',
                                             'lower_wake_coordinates'])
        self.add_subsystem('CombineDisplacementCoordinates', CombineDisplacementCoordinates(num_base_points=n_base, num_wake_points_per_side=n_wake),
                           promotes_inputs=['upper_wake_coordinates',
                                            'lower_wake_coordinates',
                                            'displaced_cw_coordinates'],
                           promotes_outputs=['displacement_body_coordinates'])


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    aerofoil_path = r"C:\Users\euana\Documents\Important Documents\Optimisation Project\TASWiG\aerofoils\NACA_63-515"
    aerofoil = np.loadtxt(aerofoil_path).T
    aerofoil = np.flip(aerofoil, 1)
    n_pts = aerofoil.shape[1]
    n_wk = 4

    p = om.Problem(model=om.Group())

    p.model.add_subsystem('DisplacementBodyGroup', DisplacementBodyGroup(num_base_points=n_pts, num_wake_points_per_side=n_wk),
                          promotes_inputs=['aerofoil_cw_coordinates',
                                           'displacement_thickness',
                                           'wake_camber_lengths',
                                           'wake_camber_angles',
                                           'wake_thickness'],
                          promotes_outputs=['displacement_body_coordinates'])

    p.model.set_input_defaults('aerofoil_cw_coordinates', aerofoil)
    p.model.set_input_defaults('wake_camber_lengths', np.repeat(0.04, n_wk))
    p.model.set_input_defaults('wake_camber_angles', np.linspace(0, 0.2, n_wk))
    p.model.set_input_defaults('wake_thickness', np.linspace(0.02, 0, n_wk))

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