
from GroupLevel.Analyses import group_oscillation_cluster
from SubjectLevel.Analyses import subject_oscillation_cluster, subject_SME
from GroupLevel import group_brain_viz
from scipy.stats import ttest_ind, pearsonr, ttest_1samp

import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib

import ipyvolume.pylab as p3
import ipyvolume as ipv

import numpy as np
import ram_data_helpers

import RAM_helpers

import time

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets


from ptsa.data.readers.tal import TalReader
from ptsa.data.readers.index import JsonIndexReader


import matplotlib.cm as cmx
import matplotlib.colors as clrs

reader = JsonIndexReader('/protocols/r1.json')


class BrainViewer():
    def __init__(self, subject='R1001P', montage='0', bipolar=False, opacity=.75, use_avg_surf=False, fig=None):
        self.subject = subject
        self.montage = montage
        self.bipolar = bipolar
        self.opacity = opacity
        self.use_avg_surf = use_avg_surf

        self.fig = fig
        self.mpl_ax = None
        self.w_output = None
        self.do_plots = True

        # initialize control widgets
        self.w_subj = widgets.Text(value=self.subject, placeholder='Subject', disabled=False)
        self.w_bipolar = widgets.Checkbox(value=self.bipolar, description='Use Bipolar Elecs', disabled=False)
        self.w_show_elecs = widgets.Checkbox(value=False, description='Show Elecs', disabled=True)
        self.w_box = interactive(self.update_attrs, subject=self.w_subj, bipolar=self.w_bipolar)
        self.button = widgets.Button(description="Load Brain")

        self.split_button = widgets.Button(description="Split View")
        self.is_split = False

        display(self.w_box, self.w_show_elecs, self.button, self.split_button)
        self.button.on_click(self.show_brain)
        self.split_button.on_click(self.rotate_brain_to_lat_view)
        self.w_show_elecs.observe(self.handle_elecs)

        # will hold brain data, electrode info
        self.brain_data = {}
        self.elec_info_mono = None
        self.elec_info_bipol = None

    def update_attrs(self, subject, bipolar):
        self.subject = subject
        self.bipolar = bipolar

    def handle_elecs(self, b):
        # make sure any old scatters are gone
        if isinstance(b['new'], bool):
            if not b['new']:
                self.fig.scatters = []
            else:
                f_path = reader.aggregate_values('pairs' if self.bipolar else 'contacts', subject=self.subject,
                                                 montage=self.montage)
                elec_info = TalReader(filename=list(f_path)[0], struct_type='bi' if self.bipolar else 'mono').read()

                if self.use_avg_surf:
                    xyz = np.array(
                        [[elec['avg.dural']['x'], elec['avg']['y'], elec['avg']['z']] for elec in elec_info.atlases])
                else:
                    xyz = np.array(
                        [[elec['ind.dural']['x'], elec['ind']['y'], elec['ind']['z']] for elec in elec_info.atlases])
                points = p3.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], size=2.5, size_selected=5.0, marker="sphere",
                                    color=[.55, .05, .1])
                points.geo = 'circle_2d'

    def show_brain(self, b):
        if self.fig is None:
            print(self.subject)
            self.make_fig()

        # remove any old meshes and scatter
        self.fig.meshes = []
        self.fig.scatters = []

        # load the current brain
        self.load_brain_data()
        l_curv = r_curv = [.6, .6, .6, self.opacity]

        # plot left hemi
        l_coords = self.brain_data['l_coords']
        l_faces = self.brain_data['l_faces']
        brain_l = p3.plot_trisurf(l_coords[:, 0], l_coords[:, 1], l_coords[:, 2], triangles=l_faces, color=l_curv)
        brain_l.material.transparent = True

        # plot right hemi
        r_coords = self.brain_data['r_coords']
        r_faces = self.brain_data['r_faces']
        brain_r = p3.plot_trisurf(r_coords[:, 0], r_coords[:, 1], r_coords[:, 2], triangles=r_faces, color=r_curv)
        brain_r.material.transparent = True

        ipv.squarelim()

    def load_brain_data(self):
        """
        Loads brain surfaces and electrode information for the specifed patient and montage

        """

        # load brain geometry
        if self.use_avg_surf:
            subj_name = 'average'
        else:
            subj_name = self.subject

        l_coords, l_faces = nib.freesurfer.read_geometry('/data/eeg/freesurfer/subjects/%s/surf/lh.pial' % subj_name)
        r_coords, r_faces = nib.freesurfer.read_geometry('/data/eeg/freesurfer/subjects/%s/surf/rh.pial' % subj_name)

        self.brain_data['l_coords'] = l_coords
        self.brain_data['l_faces'] = l_faces
        self.brain_data['r_coords'] = r_coords
        self.brain_data['r_faces'] = r_faces

        self.w_show_elecs.disabled = False

        # load elec info
        f_path_mono = reader.aggregate_values('contacts', subject=self.subject, montage=self.montage)
        self.elec_info_mono = TalReader(filename=list(f_path)[0], struct_type='bi' if self.bipolar else 'mono').read()

        f_path_bipol = reader.aggregate_values('pairs', subject=self.subject, montage=self.montage)
        self.elec_info_bipol = TalReader(filename=list(f_path)[0], struct_type='bi' if self.bipolar else 'mono').read()

    def make_fig(self):
        self.fig = p3.figure(width=400, height=400, lighting=True)
        # i like to turn off the axes lines

        ipv.style.box_off()
        ipv.style.axes_off()
        ipv.style.background_color('white')
        #         ipv.selector_default()
        self.selector_default()

        self.w_output = widgets.Output()
        with self.w_output:
            mpl_fig, ax = plt.subplots()
            self.mpl_ax = ax
            plt.show()
        h = widgets.HBox([self.fig, self.w_output])
        display(h)

    # modified selector code from ipywidgets
    def selector_default(self, output_widget=None):
        fig = p3.gcf()
        if output_widget is None:
            output_widget = ipywidgets.Output()
            display(output_widget)

        def lasso(data, other=None, fig=fig):
            with output_widget as ow:
                if data['device'] and data['type'] == 'lasso':
                    import shapely.geometry
                    region = shapely.geometry.Polygon(data['device'])

                    @np.vectorize
                    def inside(x, y):
                        return region.contains(shapely.geometry.Point([x, y]))
                if data['device'] and data['type'] == 'circle':
                    x1, y1 = data['device']['begin']
                    x2, y2 = data['device']['end']
                    dx = x2 - x1
                    dy = y2 - y1
                    r = (dx ** 2 + dy ** 2) ** 0.5

                    def inside(x, y):
                        return ((x - x1) ** 2 + (y - y1) ** 2) < r ** 2
                if data['device'] and data['type'] == 'rectangle':
                    x1, y1 = data['device']['begin']
                    x2, y2 = data['device']['end']
                    x = [x1, x2]
                    y = [y1, y2]
                    xmin, xmax = min(x), max(x)
                    ymin, ymax = min(y), max(y)

                    def inside(x, y):
                        return (x > xmin) & (x < xmax) & (y > ymin) & (y < ymax)

                def join(x, y, mode):
                    N = np.max(x) if x is not None else np.max(y)
                    N = max(N, np.max(y))
                    xmask = np.zeros(N + 1, np.bool)
                    ymask = np.zeros(N + 1, np.bool)
                    if x is not None:
                        xmask[x] = True
                    ymask[y] = True
                    if mode == "replace":
                        return np.where(ymask)
                    if mode == "and":
                        mask = xmask & ymask
                        return np.where(ymask if x is None else mask)
                    if mode == "or":
                        mask = xmask | ymask
                        return np.where(ymask if x is None else mask)
                    if mode == "subtract":
                        mask = xmask & ~ymask
                        return np.where(ymask if x is None else mask)

                for scatter in fig.scatters:
                    x, y = fig.project(scatter.x, scatter.y, scatter.z)
                    mask = inside(x, y)
                    if np.any(mask):
                        scatter.selected = join(scatter.selected, np.where(mask), fig.selection_mode)
            if self.do_plots:
                #                 out = widgets.Output():
                self.plot_elec_data(mask)

        fig.on_selection(lasso)

    def plot_elec_data(self, mask):

        # cpature the plot output in the same output manager
        with self.w_output:
            clear_output()
            plt.plot([0, 1], np.random.rand(2) * 4)
            plt.show()

    def rotate_brain_to_lat_view(self, b):

        if not self.is_split:

            # get coords to rotate
            mesh_right = self.fig.meshes[0]
            orig_x_right = mesh_right.x.copy()
            orig_y_right = mesh_right.y.copy()
            orig_z_right = mesh_right.z.copy()

            mesh_left = self.fig.meshes[1]
            orig_x_left = mesh_left.x.copy()
            orig_y_left = mesh_left.y.copy()
            orig_z_left = mesh_left.z.copy()

            # electrodes to shift
            orig_x = self.fig.scatters[0].x.copy()
            orig_y = self.fig.scatters[0].y.copy()
            orig_z = self.fig.scatters[0].z.copy()

            right_inds = self.fig.scatters[0].x < 0
            scatter_x_right = orig_x[right_inds]
            scatter_y_right = orig_y[right_inds]
            scatter_z_right = orig_z[right_inds]

            left_inds = self.fig.scatters[0].x > 0
            scatter_x_left = orig_x[left_inds]
            scatter_y_left = orig_y[left_inds]
            scatter_z_left = orig_z[left_inds]

            # rotate brains to get a lateral view of both hemispheres
            mesh_right.x = orig_y_right + 100
            mesh_right.y = orig_x_right

            mesh_left.x = -orig_y_left - 100
            mesh_left.y = -orig_x_left

            new_x = orig_y.copy()
            new_x[right_inds] += 100
            new_x[left_inds] = -new_x[left_inds] - 100

            new_y = orig_x.copy()
            new_y[left_inds] = -new_y[left_inds]

            self.fig.scatters[0].x = new_x
            self.fig.scatters[0].y = new_y

            # set camera position to look at hemisphers
            self.fig.camera.rotation = (1.57, 0, 0, 'XYZ')
            self.fig.camera.position = (0, -1.35, 0)

            # hack because camera rotation bug
            mesh_right.z = -mesh_right.z + 10
            mesh_left.z = -mesh_left.z + 10
            self.fig.scatters[0].z = -self.fig.scatters[0].z + 10

            orig_right_xyz = [orig_x_right, orig_y_right, orig_z_right]
            orig_left_xyz = [orig_x_left, orig_y_left, orig_z_left]
            orig_scatter_xyz = [orig_x, orig_y, orig_z]
        else:

            mesh_right = self.fig.meshes[0]
            mesh_right.x = orig_right_xyz[0]
            mesh_right.y = orig_right_xyz[1]
            mesh_right.z = orig_right_xyz[2]

            mesh_left = self.fig.meshes[1]
            mesh_left.x = orig_left_xyz[0]
            mesh_left.y = orig_left_xyz[1]
            mesh_left.z = orig_left_xyz[2]

            fig.scatters[0].x = self.orig_scatter_xyz[0]
            fig.scatters[0].y = self.orig_scatter_xyz[1]
            fig.scatters[0].z = self.orig_scatter_xyz[2]
            return orig_right_xyz, orig_left_xyz, orig_scatter_xyz
