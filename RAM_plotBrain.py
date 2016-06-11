from guidata import qthelpers # needed tp fix: ValueError: API 'QString' has already been set to version 1
from mayavi import mlab
from mayavi.mlab import pipeline as mp #?
from surfer import Surface, Brain
from ptsa.data.readers.TalReader import TalReader
import numpy as np
# set path to free surfer data
import os
from os.path import expanduser
os.environ['SUBJECTS_DIR'] = '/data/eeg/freesurfer/subjects/'

# import matplotlib

# matplotlib.use('Agg')

# need to do this?
#%gui qt

#TraitError: The 'colormap' trait of a SurfaceFactory instance must be 'Accent' or 'Blues' or 'BrBG' or 'BuGn' or
# 'BuPu' or 'Dark2' or 'GnBu' or 'Greens' or 'Greys' or 'OrRd' or 'Oranges' or 'PRGn' or 'Paired' or 'Pastel1' or
# 'Pastel2' or 'PiYG' or 'PuBu' or 'PuBuGn' or 'PuOr' or 'PuRd' or 'Purples' or 'RdBu' or 'RdGy' or 'RdPu' or
# 'RdYlBu' or 'RdYlGn' or 'Reds' or 'Set1' or 'Set2' or 'Set3' or 'Spectral' or 'YlGn' or 'YlGnBu' or 'YlOrBr' or
#  'YlOrRd' or 'autumn' or 'binary' or 'black-white' or 'blue-red' or 'bone' or 'cool' or 'copper' or 'file' or
# 'flag' or 'gist_earth' or 'gist_gray' or 'gist_heat' or 'gist_ncar' or 'gist_rainbow' or 'gist_stern' or 'gist_
# yarg' or 'gray' or 'hot' or 'hsv' or 'jet' or 'pink' or 'prism' or 'spectral' or 'spring' or 'summer' or 'winter',
#  but a value of 'test' <type 'str'> was specified

#(subj='R1076D',hemi='lh',cortex=('Greys', -1, 2, True))

class BrainPlot:
    data_path = '/data/eeg'

    def __init__(self, subj=None, hemi='both', surface='pial', views='lateral',
                 cortex='classic', background='dimgray', avg_surf=True, bipolar=True):

        if subj is None:
            print('Subj is required')
            return
        self.subj = subj

        # hemisphere to plot: lh, rh, both (electrode plotting doesn't work for both?)
        self.hemi = hemi

        # type of surface: pial, inflated, smoothwm
        self.surface = surface

        # camera views: lateral, frontal, others?
        self.views = views

        # this is the colormap settings for the brain in the format (colormap, min, max, reverse)
        # ex: ('Greys', -1, 2, True)
        self.cortex = cortex

        # background color
        self.background = background

        # use average brain or individual subject brain
        self.avg_surf = avg_surf

        # use bipolar or original electrode localizations
        self.bipolar = bipolar

        # flag for whether all electrodes are currently shown
        self.all_elecs_on = False

        # create the pysurfer brain object
        subj_str = 'average' if self.avg_surf else subj
        self.brain = Brain(subj_str, self.hemi, self.surface, views=self.views, cortex=self.cortex, background=self.background)

        # get list of underlying mayavi object surfaces, see we can directly change things like opacity
        self.brain_surf = self.brain.brain_matrix[0]

        # load patient electrode information (we need locations and labels)
        self.tal = self.load_tal()

        # create dict of electrodes see we can keep track of which are currently displayed and their color
        self.elec_dict = {k: dict() for k in self.tal['tagName']}
        for key in self.elec_dict.keys():
            self.elec_dict[key]['on'] = False
            # self.elec_dict[key]['color'] = (1, 0, 0)
            self.elec_dict[key]['color'] = (np.random.rand(1)[0],np.random.rand(1)[0],np.random.rand(1)[0])
            self.elec_dict[key]['x'] = self.tal[self.tal['tagName'] == key][self.avg_surf]['x_Dykstra']
            self.elec_dict[key]['y'] = self.tal[self.tal['tagName'] == key][self.avg_surf]['y_Dykstra']
            self.elec_dict[key]['z'] = self.tal[self.tal['tagName'] == key][self.avg_surf]['z_Dykstra']

        # needed to show figure?
        mlab.show()

    def set_brain_opacity(self, opacity=1):
        # maybe there is an approved way to access this property, but I don't know
        for hemi in self.brain_surf:
            hemi._geo_surf.actor.property.opacity = opacity

    def plot_elecs_colors(self, elec_size=1, alpha=1):
        """Allows plotting each electrode with its own color in one call, which is not straightforward. Looping and
        plotting each electrode by itself is slow.
        credit: http://stackoverflow.com/questions/18537172/specify-absolute-colour-for-3d-points-in-mayavi/30266228#30266228
        """

        # basicaly, make a colormap where each entry corresponds to one electrode
        N = len(self.tal)
        scalars = np.arange(N)
        colors = (np.random.random((N, 4)) * 255).astype(np.uint8)
        colors[:, -1] = 255  # No transparency

        # Define coordinates and points
        x = self.tal[self.avg_surf]['x_Dykstra']
        y = self.tal[self.avg_surf]['y_Dykstra']
        z = self.tal[self.avg_surf]['z_Dykstra']
        self.brain.pts = mlab.points3d(x, y, z, scalars, scale_factor=(10. * elec_size), opacity=alpha,
                                       scale_mode='none')
        self.brain.pts.glyph.color_mode = 'color_by_scalar'
        self.brain.pts.module_manager.scalar_lut_manager.lut.table = colors
        self.brain.reset_view()
        
    def plot_all_elecs(self, color='red', elec_size=1, alpha=1):
        if not self.all_elecs_on:
            xyz = self.get_xyz()
            self.brain.add_foci(coords=xyz, color=color, scale_factor=elec_size, alpha=alpha, name='elecs')
            self.all_elecs_on = True

    def remove_all_elecs(self):
        if self.all_elecs_on:
            elecs = self.brain.foci['elecs']
            elecs.remove()
            self.all_elecs_on = False

    def get_xyz(self):
        x = self.tal[self.avg_surf]['x_Dykstra']
        y = self.tal[self.avg_surf]['y_Dykstra']
        z = self.tal[self.avg_surf]['z_Dykstra']
        return zip(x, y, z)

    def load_tal(self):
        tal_ext = '_bipol.mat' if self.bipolar else '_monopol.mat'
        tal_path = os.path.join(BrainPlot.data_path, self.subj, 'tal', self.subj + '_talLocs_database' + tal_ext)
        tal_reader = TalReader(filename=tal_path)
        tal_struct = tal_reader.read()
        return tal_struct


        #brain = Brain(subj, hemi, surface, views=views, background=background)
        # mlab.show()
        # geo = Surface(subj, hemi, surface)
        # geo.load_geometry()

        # f = mlab.figure(bgcolor=(0, 0, 0))
        # mesh = mp.triangular_mesh_source(geo.x, geo.y, geo.z, geo.faces)
        # mesh.data.point_data.normals = geo.nn
        # mesh.data.cell_data.normals = None

        # surf = mlab.pipeline.surface(mesh, figure=f, color=(.4, .4, .4), opacity=.4)
        # mlab.show()
if __name__ == '__main__':
    BrainPlot(subj='R1076D')