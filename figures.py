"""
    General comments about this:
    Using the best-fits obviously gives a nice result, but the best fits
    are tricky to interpolate since the z fluctuates.
    But using a linear z interpolation for the model doesn't give a
    good result, since the particle radius, index, etc vary.

    another option is to fix (n, r, lens_angle), and do a continuously
    varying z, to show a similar plot. I think that might be the
    clearest.

    What I've settled on here is using the mean radius, index, lens angle
    from all the fits. This gives a big bright spot in the center.

    I don't see a big difference in the way it looks from that and
    using an attempt at fitting all the frames at once. Since fitting
    all the frames gives slightly weirder values (to avoid the bright
    spot in the middle...), I am just using mean from the fits.
"""
import numpy as np
from scipy.interpolate import interp1d
import matplotlib as mpl
import matplotlib.pyplot as plt


class ThreeDPlot(object):
    _box_color = '#F0F0F0'
    def __init__(self, axes, azimuth_elevation=(0, 0)):
        self.axes = axes
        self.azimuth_elevation = azimuth_elevation

        self._axes_are_setup = False
        self.matrix = self._setup_matrix()
        self.axes_spines = None
        self.box_walls = None  # meh

        # We default to a (0, 1) xlim etc, and fix it later:
        self.xmin = 0
        self.xmax = 1
        self.ymin = 0
        self.ymax = 1
        self.zmin = 0
        self.zmax = 1
        self._was_rescaled = False
        self._setup_axes()

    def plot(self, x, y, z, *args, rescale=True, **kwargs):
        if rescale:
            self._rescale_to(x, y, z)
        self._plot(x, y, z, *args, **kwargs)
        self._clean_axes()
        self._redraw()  # matplotlib is weird...

    def set_xlim(self, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax
        self._redraw()
        return self.xmin, self.xmax

    def set_ylim(self, ymin, ymax):
        self.ymin = ymin
        self.ymax = ymax
        self._redraw()
        return self.ymin, self.ymax

    def set_zlim(self, zmin, zmax):
        self.zmin = zmin
        self.zmax = zmax
        self._redraw()
        return self.zmin, self.zmax

    def _plot(self, x, y, z, *args, **kwargs):
        xy_plot = self._transform_xyz_to_plotxy([x, y, z])
        return self.axes.plot(xy_plot[0, :], xy_plot[1, :], *args, **kwargs)

    def _setup_axes(self):
        self._update_plot_corners()
        self._make_axes_box()
        self._make_axes_spines()
        self._axes_are_setup = True

    def _update_plot_corners(self):
        # 1. Create the 7 corners of the plot:
        origin = [self.xmin, self.ymin, self.zmin]
        xmax = [self.xmax, self.ymin, self.zmin]
        ymax = [self.xmin, self.ymax, self.zmin]
        zmax = [self.xmin, self.ymin, self.zmax]
        xy_corner = [self.xmax, self.ymax, self.zmin]
        xz_corner = [self.xmax, self.ymin, self.zmax]
        yz_corner = [self.xmin, self.ymax, self.zmax]

        # 2. Transform those points to plotxy coordinates:
        self._origin_plot = self._transform_xyz_to_plotxy(origin)
        self._xmax_plot = self._transform_xyz_to_plotxy(xmax)
        self._ymax_plot = self._transform_xyz_to_plotxy(ymax)
        self._zmax_plot = self._transform_xyz_to_plotxy(zmax)
        self._xy_corner_plot = self._transform_xyz_to_plotxy(xy_corner)
        self._xz_corner_plot = self._transform_xyz_to_plotxy(xz_corner)
        self._yz_corner_plot = self._transform_xyz_to_plotxy(yz_corner)

    def _get_box_corners(self):
        vertices_xy = np.array(
            [self._origin_plot, self._xmax_plot,
             self._xy_corner_plot, self._ymax_plot])
        vertices_xz = np.array(
            [self._origin_plot, self._xmax_plot,
             self._xz_corner_plot, self._zmax_plot])
        vertices_yz = np.array(
            [self._origin_plot, self._ymax_plot,
             self._yz_corner_plot, self._zmax_plot])
        return vertices_xy, vertices_xz, vertices_yz

    def _get_spine_points(self):
        xspine = ([self._origin_plot[0], self._xmax_plot[0]],
                  [self._origin_plot[1], self._xmax_plot[1]])
        yspine = ([self._origin_plot[0], self._ymax_plot[0]],
                  [self._origin_plot[1], self._ymax_plot[1]])
        zspine = ([self._origin_plot[0], self._zmax_plot[0]],
                  [self._origin_plot[1], self._zmax_plot[1]])
        return xspine, yspine, zspine  # must be lists, not numpy arrays

    def _rescale_to(self, x, y, z):
        if self._was_rescaled:
            minx = min(min(x), self.xmin)
            maxx = max(max(x), self.xmax)
            miny = min(min(y), self.ymin)
            maxy = max(max(y), self.ymax)
            minz = min(min(z), self.zmin)
            maxz = max(max(z), self.zmax)
        else:
            minx = min(x)
            maxx = max(x)
            miny = min(y)
            maxy = max(y)
            minz = min(z)
            maxz = max(z)
        self.set_xlim(minx, maxx)
        self.set_ylim(miny, maxy)

        self.set_zlim(minz, maxz)

        self._was_rescaled = True

    def _redraw(self):
        self._update_plot_corners()
        self._redraw_grids()
        self._redraw_spines()
        box_corners = np.vstack(self._get_box_corners())
        min_x, min_y = box_corners.min(axis=0)
        max_x, max_y = box_corners.max(axis=0)
        self.axes.set_xlim(min_x, max_x)
        self.axes.set_ylim(min_y, max_y)

    def _redraw_grids(self):
        vertices_xy, vertices_xz, vertices_yz = self._get_box_corners()
        self._patch_xy.set_xy(vertices_xy)
        self._patch_xz.set_xy(vertices_xz)
        self._patch_yz.set_xy(vertices_yz)

    def _redraw_spines(self):
        xspine, yspine, zspine = self._get_spine_points()
        self._xspine.set_xdata(xspine[0])
        self._xspine.set_ydata(xspine[1])

        self._yspine.set_xdata(yspine[0])
        self._yspine.set_ydata(yspine[1])

        self._zspine.set_xdata(zspine[0])
        self._zspine.set_ydata(zspine[1])

    def _make_axes_box(self):
        vertices_xy, vertices_xz, vertices_yz = self._get_box_corners()

        self._patch_xy = mpl.patches.Polygon(vertices_xy, color=self._box_color)
        self._patch_xz = mpl.patches.Polygon(vertices_xz, color=self._box_color)
        self._patch_yz = mpl.patches.Polygon(vertices_yz, color=self._box_color)
        for patch in [self._patch_xy, self._patch_xz, self._patch_yz]:
            self.axes.add_patch(patch)

    def _make_axes_spines(self):
        xspine, yspine, zspine = self._get_spine_points()
        self._xspine = self.axes.plot(*xspine, 'k-', lw=1)[0]
        self._yspine = self.axes.plot(*yspine, 'k-', lw=1)[0]
        self._zspine = self.axes.plot(*zspine, 'k-', lw=1)[0]

    def _transform_xyz_to_plotxy(self, xyz):
        return self.matrix.dot(xyz)

    def _setup_matrix(self):
        az, el = self.azimuth_elevation
        rotation_az = np.array([  # rotation about the y axis
            [np.cos(az), 0., np.sin(az)],
            [0., 1., 0.],
            [-np.sin(az), 0., np.cos(az)]])
        rotation_el = np.array([  # rotation about the x axis
            [1.0, 0., 0.],
            [0., np.cos(el), np.sin(el)],
            [0., -np.sin(el), np.cos(el)]])
        transform_z_into_y = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0]]).astype('float')
        # full_matrix = transform_z_into_y.dot(rotation_el.dot(rotation_az))
        full_matrix = rotation_el.dot(rotation_az.dot(transform_z_into_y))
        # Then we only want the portion that maps into x, y:
        return full_matrix[:2].copy()

    def _clean_axes(self):
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.axes.spines['right'].set_visible(False)
        self.axes.spines['top'].set_visible(False)
        self.axes.spines['bottom'].set_visible(False)
        self.axes.spines['left'].set_visible(False)
