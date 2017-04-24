"""
An interactive way to look at the subsequent memory effect at the level of individual electrodes in Treasure Hunt. This
will automatically list all TH subjects, and for a given subject, the brain regions where electrodes were located. You
can select and subject and brain region and view the power spectra and t-statistics for all electrodes in the region.

TO DO: average plots for a region. linked panning would be nice, but I'll have to rescale everything to be in the same
units.

Note to self: to start server on rhino
bokeh serve TH_sme_viz.py --port 8123 --host 127.0.0.1:8157
"""
from os.path import dirname, join
from GroupLevel.Analyses import group_move_vs_still
import numpy as np
from bokeh.plotting import figure
from bokeh.layouts import layout, widgetbox, row, column, gridplot
from bokeh.models import ColumnDataSource, HoverTool, Div, Range1d
from bokeh.models.widgets import Slider, Select, TextInput
from bokeh.io import curdoc, show
from scipy.stats import t, ttest_1samp

# Number of columns in each plot is set here
NUM_COLS = 5

# load data
freqs = np.logspace(np.log10(1), np.log10(200), 50)
data = group_move_vs_still.GroupSME(analysis='sme_enc', subject_settings='default',
                                    open_pool=False, freqs=freqs, load_res_if_file_exists=False)
data.process()

# create dictionary mapping subject name to index into subject_objs
subjs = {x.subj: i for i, x in enumerate(data.subject_objs)}

# create subject drop down menu and populate with subject names
subj_drop = Select(title='Subject', options=sorted(subjs.keys()), value=sorted(subjs.keys())[0])


def create_elec_loc_dropdown_vals():
    """
    The region drop down menu only contains brain regions where the current subject has electrodes. This returns a list
    of strings of brain regions for the current subject, with a blank and 'all' prepended.
    """
    subj_obj = data.subject_objs[subjs[subj_drop.value]]
    if subj_obj.subject_data is None:
        subj_obj.load_data()

    elec_locs = []
    for key in subj_obj.elec_locs:
        if np.any(subj_obj.elec_locs[key]):
            elec_locs.append(key)
    return [' ', 'All'] + sorted(elec_locs)

# create the region drop down for the initial value of subject
elec_locs = create_elec_loc_dropdown_vals()
region_drop = Select(title='Region', options=elec_locs, value=elec_locs[0])


def init_spec_fig(n_panels=1):
    """
    Creates an empty power spectrum figure

    Returns figure, columndatasource
    """
    source = ColumnDataSource(data=dict(xs=[], ys=[], labels=['Good Memory', 'Bad Memory']))
    TOOLS = 'box_zoom,pan,wheel_zoom,undo,zoom_in,zoom_out,crosshair,resize,reset'

    height = int(np.ceil(n_panels/float(NUM_COLS))) * 250
    p = figure(plot_height=height, plot_width=600, title="", tools=TOOLS)
    p.multi_line(xs='xs', ys='ys', line_width=2, source=source, line_color=['#8c564b', '#1f77b4'])
    p.yaxis.axis_label = "log(power)"
    p.yaxis.axis_label_text_font_size = '16pt'
    p.yaxis.major_label_text_font_size = '16pt'
    p.xaxis.axis_label = "Frequency"
    p.xaxis.axis_label_text_font_size = '16pt'
    p.xaxis.major_label_text_font_size = '16pt'
    p.xaxis.major_label_text_font_size = '0pt'
    p.yaxis.major_label_text_font_size = '0pt'
    p.title.text_font_size = '16pt'
    p.title.text_font_style = 'normal'
    p.title.align = 'center'
    return p, source


def init_tstat_fig(n_panels=1):
    """
    Creates an tstat power spectrum figure. Very similar to power spectrum figure, but bokeh is kind of buggy and if I
    don't define the line colors and labels when I create the figure, I can't actually set them later. If that worked as
     expected, I wouldn't have seperate functions.

    Returns figure, columndatasource
    """
    source = ColumnDataSource(data=dict(xs=[], yy=[]))
    TOOLS = 'box_zoom,pan,wheel_zoom,undo,zoom_in,zoom_out,crosshair,resize,reset'

    height = int(np.ceil(n_panels/float(NUM_COLS))) * 250
    p = figure(plot_height=height, plot_width=600, title="", tools=TOOLS)
    p.multi_line(xs='xs', ys='ys', line_width=2, line_color=['#000000', '#000000', '#8c564b', '#1f77b4'], source=source)
    p.yaxis.axis_label = "tstat"
    p.yaxis.axis_label_text_font_size = '16pt'
    p.yaxis.major_label_text_font_size = '16pt'
    p.xaxis.axis_label = "Frequency"
    p.xaxis.axis_label_text_font_size = '16pt'
    p.xaxis.major_label_text_font_size = '0pt'
    p.yaxis.major_label_text_font_size = '0pt'
    p.title.text_font_size = '16pt'
    p.title.text_font_style = 'normal'
    p.title.align = 'center'
    return p, source


def update_spec_fig(elecs):
    """
    Update the figure created by init_spec_fig. Height is determined by the number of electrodes.

    Plots log(power) as a function of frequency for each electrode. Red line = good memory. Blue line = bad memory.
    """

    # make sure data is loaded for this figure
    subj_obj = data.subject_objs[subjs[subj_drop.value]]
    if subj_obj.subject_data is None:
        subj_obj.load_data()
    subj_obj.filter_data_to_task_phases(subj_obj.task_phase_to_use)

    # determine the good memory and bad memory trials so we can plot the seperately
    recalled = subj_obj.recall_filter_func(subj_obj.task, subj_obj.subject_data.events.data, subj_obj.rec_thresh)

    # will hold x and y valyes that go into multi_line. all xs are the same, ys go good memory, bad, good, bad, ...
    ys = []
    xs = []

    # determine how much to shift the y values
    all_rec_mean = subj_obj.subject_data[recalled, :, ].mean('events').data
    all_nrec_mean = subj_obj.subject_data[recalled, :, ].mean('events').data
    max_y_range = np.max([np.ptp(all_rec_mean, axis=0).max(), np.ptp(all_nrec_mean, axis=0).max()])

    # loop over each electrode and append to xs and ys
    for i, elec in enumerate(elecs):

        yscale = (i/NUM_COLS+1) * max_y_range
        ys.append((subj_obj.subject_data[recalled, :, elec].mean('events').data + yscale).tolist())
        ys.append((subj_obj.subject_data[~recalled, :, elec].mean('events').data + yscale).tolist())

        x = (np.log10(subj_obj.freqs) + ((np.ptp(np.log10(subj_obj.freqs))+1) * (i % NUM_COLS))).tolist()
        for line in range(2):
            xs.append(x)

    # init figure and set the data
    p1, source1 = init_spec_fig(len(elecs))
    source1.data = dict(
        xs=xs,
        ys=ys,
        line_width=[2]*len(xs),
        line_color=['#8c564b', '#1f77b4']*(len(xs)/2),
    )
    p1.xaxis.axis_label = 'Frequency (%.2f - %.2f)' % (subj_obj.freqs[0], subj_obj.freqs[-1])
    p1.title.text = 'Power Spectra: %s' % (subj_drop.value)
    return p1


def update_tstat_fig(elecs):
    """
    Update the figure created by init_tstat_fig. Height is determined by the number of electrodes.

    Plots tstat as a function of frequency for each electrode. postive = more power at that frequency for good memory,
    opposite for negative values.
    """

    # make sure data is loaded for this figure
    subj_obj = data.subject_objs[subjs[subj_drop.value]]
    if subj_obj.subject_data is None:
        subj_obj.load_data()
    subj_obj.filter_data_to_task_phases(subj_obj.task_phase_to_use)

    # determine the t-stat for significance at p<.05.
    df = subj_obj.subject_data.shape[0] - 2
    crit_t = t.ppf(0.975, df)

    # will hold x and y valyes that go into multi_line. all xs are the same, ys go tstat, zero, postive critical t,
    # negative critical t, ...
    ys = []
    xs = []

    # determine how much to shift the y values
    max_y_range = np.ptp(subj_obj.res['ts'], axis=0).max()

    # loop over each electrode and append to xs and ys
    for i, elec in enumerate(elecs):

        yscale = (i/NUM_COLS+1) * max_y_range

        # t-stat
        ys.append((subj_obj.res['ts'][:, elec] + yscale).tolist())

        # zero line
        ys.append((np.zeros(subj_obj.res['ts'][:, elec].shape[0]) + yscale).tolist())

        # postitive critical t
        ys.append((np.zeros(subj_obj.res['ts'][:, elec].shape[0]) + crit_t + yscale).tolist())

        # negative critical t
        ys.append((np.zeros(subj_obj.res['ts'][:, elec].shape[0]) - crit_t + yscale).tolist())

        x = (np.log10(subj_obj.freqs) + ((np.ptp(np.log10(subj_obj.freqs))+1) * (i % NUM_COLS))).tolist()
        for line in range(4):
            xs.append(x)

    # init figure and set the data
    p1, source1 = init_tstat_fig(len(elecs))
    source1.data = dict(
        xs=xs,
        ys=ys,
        line_width=[2]*len(xs),
        line_color=['#000000', '#000000', '#8c564b', '#1f77b4']*(len(xs)/4),
    )
    p1.xaxis.axis_label = 'Frequency (%.2f - %.2f)' % (subj_obj.freqs[0], subj_obj.freqs[-1])
    p1.title.text = 't-stat: %s' % (subj_drop.value)
    return p1


def create_tstat_region_average_fig(elecs):
    """

    """

    source = ColumnDataSource(data=dict(xs=[], yy=[]))
    TOOLS = 'box_zoom,pan,wheel_zoom,undo,zoom_in,zoom_out,crosshair,resize,reset'

    p = figure(plot_height=100, plot_width=500, title="", tools=TOOLS, x_axis_type='log')
    # p.multi_line(xs='xs', ys='ys', line_width=2, line_color=['#000000', '#000000', '#8c564b', '#1f77b4'], source=source)
    p.multi_line(xs='xs', ys='ys', line_width=2, line_color=['#000000', '#000000'], source=source)
    p.yaxis.axis_label = "tstat"
    p.yaxis.axis_label_text_font_size = '16pt'
    p.yaxis.major_label_text_font_size = '16pt'
    p.xaxis.axis_label = "Frequency"
    p.xaxis.axis_label_text_font_size = '16pt'
    p.xaxis.major_label_text_font_size = '0pt'
    p.yaxis.major_label_text_font_size = '0pt'
    p.title.text_font_size = '16pt'
    p.title.text_font_style = 'normal'
    p.title.align = 'center'

    # make sure data is loaded for this figure
    subj_obj = data.subject_objs[subjs[subj_drop.value]]
    if subj_obj.subject_data is None:
        subj_obj.load_data()
    subj_obj.filter_data_to_task_phases(subj_obj.task_phase_to_use)

    y = np.mean(subj_obj.res['ts'][:, elecs], axis=1)
    print y
    print y.shape
    t, pval = ttest_1samp(subj_obj.res['ts'][:, elecs], 0, axis=1, nan_policy='omit')
    print t
    print t.shape
    source.data = dict(xs=[subj_obj.freqs.tolist()]*2, ys=[y.tolist(), [0]*len(y)], line_width=[2]*2,
                       line_color=['#000000', '#000000'])
    print [subj_obj.freqs]*2
    print [y, [0]*len(y)]

    return p

def update_region(attr, old, new):
    """
    Updates the region drop down menu.

    This is called whenver the subject dropdown is changed so that the regions match those from the current subject.
    """
    elec_locs = create_elec_loc_dropdown_vals()
    region_drop.options = elec_locs
    region_drop.value = elec_locs[0]


def update(attr, old, new):
    """
    This is called whenever the region is selected. It figures out which electrodes are in the current region and
    passes those electrode indices to the plotting function.

    First the old plots are deleted, and then the new ones are created.
    """

    # get the objects which holds the plots, delete any old plots
    l = doc.get_model_by_name('plots')
    l.children = []

    # l2 = doc.get_model_by_name('avg')
    # l2.children = []

    # figure out which electrodes to plot, the create the new figure
    if region_drop.value != ' ':
        if region_drop.value == 'All':
            n_elecs = data.subject_objs[subjs[subj_drop.value]].subject_data.shape[2]
            elecs = range(n_elecs)
        else:
            elecs = np.where(data.subject_objs[subjs[subj_drop.value]].elec_locs[region_drop.value])[0]

        # create the new figures and append to the now empty list of figures to show
        p = update_spec_fig(elecs)
        p2 = update_tstat_fig(elecs)
        l.children.append(row([p, p2]))

        # p3 = create_tstat_region_average_fig(elecs)
        # l2.children.append(p3)

# setting callbacks for dropdowns
subj_drop.on_change('value', update_region)
region_drop.on_change('value', update)

# place both dropdowns in a single widgetbox
controls = [subj_drop, region_drop]
sizing_mode = 'fixed' # also ok with scale_width
inputs = widgetbox(*controls, sizing_mode=sizing_mode)

# create an empty row to hold the plots
r = row(name='plots')
r.width = 400
r.height = 800


# r2 = row(name='avg')
# r2.width = 800
# r2.height = 400

# make the layout. Top row is dropdowns, below that is the plot
desc = Div(text=open(join(dirname(__file__), "desc.html")).read(), width=500)
l = layout(children=[[desc, inputs], [r]], sizing_mode=sizing_mode, name='mainLayout')

# serve it up
doc = curdoc()
doc.add_root(l)
doc.title = "SME"
