from guidata import qthelpers # needed tp fix: ValueError: API 'QString' has already been set to version 1
from mayavi import mlab
from mayavi.mlab import pipeline as mp #?
from surfer import Surface, Brain
from ptsa.data.readers.TalReader import TalReader

# set path to free surfer data
import os
from os.path import expanduser
os.environ['SUBJECTS_DIR'] = '/data/eeg/freesurfer/subjects/'

# import matplotlib

# matplotlib.use('Agg')

# need to do this?
#%gui qt

class BrainPlot:
    data_path = '/data/eeg'

    def __init__(self, subj=None, hemi='both', surface='pial', views='lateral', background='dimgray', bipolar=True):

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

        # background color
        self.background = background

        # use bipolar or original electrode localizations
        self.bipolar = bipolar

        # flag for whether all electrodes are currently shown
        self.all_elecs_on = False

        # create the pysurfer brain object
        self.brain = Brain(self.subj, self.hemi, self.surface, views=self.views, background=self.background)

        # get list of underlying mayavi object surfaces, see we can directly change things like opacity
        self.brain_surf = self.brain.brain_matrix[0]

        # load patient electrode information (we need locations and labels)
        self.tal = self.load_tal()

        # create dict of electrodes see we can keep track of which are currently displayed and their color
        self.elec_dict = {k: dict() for k in self.tal['tagName']}
        for key in self.elec_dict.keys():
            self.elec_dict[key]['on'] = False
            self.elec_dict[key]['color'] = (1, 0, 0)
            # self.elec_dict[key]['color'] = (np.random.rand(1)[0],np.random.rand(1)[0],np.random.rand(1)[0])
            self.elec_dict[key]['x'] = self.tal[self.tal['tagName'] == key]['indivSurf']['x_Dykstra']
            self.elec_dict[key]['y'] = self.tal[self.tal['tagName'] == key]['indivSurf']['y_Dykstra']
            self.elec_dict[key]['z'] = self.tal[self.tal['tagName'] == key]['indivSurf']['z_Dykstra']

        # needed to show figure?
        mlab.show()

    def set_brain_opacity(self, opacity=1):
        # maybe there is an approved way to access this property, but I don't know
        for hemi in self.brain_surf:
            hemi._geo_surf.actor.property.opacity = opacity

    def plot_elecs_colors(self, elec_rgbs):

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
        x = self.tal['indivSurf']['x_Dykstra']
        y = self.tal['indivSurf']['y_Dykstra']
        z = self.tal['indivSurf']['z_Dykstra']
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