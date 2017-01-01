from os.path import dirname, join
from GroupLevel.Analyses import group_SME
import numpy as np
from bokeh.plotting import figure, show
from bokeh.layouts import layout, widgetbox, row, column, gridplot
from bokeh.models import ColumnDataSource, HoverTool, Div, Range1d
from bokeh.models.widgets import Slider, Select, TextInput
from bokeh.io import curdoc
from bokeh import mpl
from bokeh.sampledata.movies_data import movie_path
from bokeh.models import FuncTickFormatter, FixedTicker
from bokeh.client import push_session, pull_session
from scipy.stats import t

# load data
freqs = np.logspace(np.log10(1), np.log10(200), 50)
data = group_SME.GroupSME(analysis='sme_enc', subject_settings='test',
                          open_pool=False, freqs=freqs, load_res_if_file_exists=False)
data.process()

# create dictionary mapping subject name to index into subject_objs
subjs = {x.subj: i for i, x in enumerate(data.subject_objs)}

# create subject drop down menu and populate with subject names
subj_drop = Select(title='Subject', options=sorted(subjs.keys()), value=sorted(subjs.keys())[0])


def create_elec_loc_dropdown():
    """

    """
    subj_obj = data.subject_objs[subjs[subj_drop.value]]
    if subj_obj.subject_data is None:
        subj_obj.load_data()

    elec_locs = []
    for key in subj_obj.elec_locs:
        if np.any(subj_obj.elec_locs[key]):
            elec_locs.append(key)
    return [' ', 'All'] + sorted(elec_locs)

elec_locs = create_elec_loc_dropdown()
region_drop = Select(title='Region', options=elec_locs, value=elec_locs[0])

# def init_spec_fig():
#     """
#     Creates an empty power spectrum figure
#
#     Returns figure, columndatasource
#     """
#     source = ColumnDataSource(data=dict(xs=[], ys=[], labels=['Good Memory', 'Bad Memory']))
#
#     hover = HoverTool(tooltips=[
#         # ("Frequency", "@xs"),
#         # ("Power", "@ys"),
#         ("(x,y)", "($x, $y)"),
#
#     ])
#
#     p = figure(plot_height=250, plot_width=500, title="", toolbar_location=None, x_axis_type="log", webgl=True)
#     p.multi_line(xs='xs', ys='ys',legend='labels', source=source, line_width=4, line_color=['#8c564b', '#1f77b4'])
#     p.yaxis.axis_label = "log(power)"
#     p.yaxis.axis_label_text_font_size = '16pt'
#     p.yaxis.major_label_text_font_size = '16pt'
#     p.xaxis.axis_label = "Frequency"
#     p.xaxis.axis_label_text_font_size = '16pt'
#     p.xaxis.major_label_text_font_size = '16pt'
#     p.xaxis.ticker=FixedTicker(ticks=np.power(2, range(9)))
#     p.xgrid.ticker=FixedTicker(ticks=np.power(2, range(9)))
#     p.title.text_font_size = '16pt'
#     p.title.text_font_style = 'normal'
#     p.title.align = 'center'
#     return p, source


# def init_tstat_fig():
#     """
#     Creates an empty tstat spectrum figure.
#
#     Returns figure, columndatasource
#     """
#
#     source = ColumnDataSource(data=dict(x=[], y=[]))
#
#     p = figure(plot_height=250, plot_width=500, title="", toolbar_location=None, x_axis_type="log", webgl=True)
#     p.line(x='x', y='y', source=source, line_width=4, line_color='black')
#
#     p.yaxis.axis_label = "t-stat"
#     p.yaxis.axis_label_text_font_size = '16pt'
#     p.yaxis.major_label_text_font_size = '16pt'
#     p.xaxis.axis_label = "Frequency"
#     p.xaxis.axis_label_text_font_size = '16pt'
#     p.xaxis.major_label_text_font_size = '16pt'
#     p.xaxis.ticker=FixedTicker(ticks=np.power(2, range(9)))
#     p.xgrid.ticker=FixedTicker(ticks=np.power(2, range(9)))
#     p.title.text_font_size = '16pt'
#     p.title.text_font_style = 'normal'
#     p.title.align = 'center'
#     return p, source


def init_spec_fig(n_panels=1):
    """
    Creates an empty power spectrum figure

    Returns figure, columndatasource
    """
    source = ColumnDataSource(data=dict(xs=[], ys=[], labels=['Good Memory', 'Bad Memory']))
    TOOLS = 'box_zoom,pan,wheel_zoom,undo,zoom_in,zoom_out,crosshair,resize,reset'

    hover = HoverTool(tooltips=[
        # ("Frequency", "@xs"),
        # ("Power", "@ys"),
        ("(x,y)", "($x, $y)"),

    ])

    height = int(np.ceil(n_panels/5.)) * 250
    p = figure(plot_height=height, plot_width=600, title="", tools=TOOLS)
    # p.multi_line(xs='xs', ys='ys',legend='labels', source=source, line_width=4, line_color=['#8c564b', '#1f77b4'])
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
    Creates an empty power spectrum figure

    Returns figure, columndatasource
    """
    source = ColumnDataSource(data=dict(xs=[], yy=[]))
    TOOLS = 'box_zoom,pan,wheel_zoom,undo,zoom_in,zoom_out,crosshair,resize,reset'

    hover = HoverTool(tooltips=[
        # ("Frequency", "@xs"),
        # ("Power", "@ys"),
        ("(x,y)", "($x, $y)"),

    ])

    height = int(np.ceil(n_panels/5.)) * 250
    p = figure(plot_height=height, plot_width=600, title="", tools=TOOLS)
    p.multi_line(xs='xs', ys='ys', line_width=2, line_color=['#000000', '#000000', '#8c564b', '#1f77b4'], source=source)
    # p.multi_line(xs='xs', ys='ys',legend='labels', source=source, line_width=4, line_color=['#8c564b', '#1f77b4'])
    # p.line(x='x', y='y', source=source, line_width=4, line_color='black')
    # p.multi_line(xs='xs', ys='ys', line_width=3, source=source, line_color=['#8c564b', '#1f77b4'])
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


def create_fig(elecs):

    # make sure data is loaded for this figure
    subj_obj = data.subject_objs[subjs[subj_drop.value]]
    if subj_obj.subject_data is None:
        subj_obj.load_data()
    subj_obj.filter_data_to_task_phases(subj_obj.task_phase_to_use)

    # Data for top panel
    recalled = subj_obj.recall_filter_func(subj_obj.task, subj_obj.subject_data.events.data, subj_obj.rec_thresh)

    ys = []
    xs = []

    # data = subj_obj.subject_data.data
    all_rec_mean = subj_obj.subject_data[recalled, :, ].mean('events').data
    all_nrec_mean = subj_obj.subject_data[recalled, :, ].mean('events').data
    max_y_range = np.max([np.ptp(all_rec_mean, axis=0).max(), np.ptp(all_nrec_mean, axis=0).max()])

    for i, elec in enumerate(elecs):

        yscale = (i/5+1) * max_y_range
        ys.append((subj_obj.subject_data[recalled, :, elec].mean('events').data + yscale).tolist())
        ys.append((subj_obj.subject_data[~recalled, :, elec].mean('events').data + yscale).tolist())

        xs.append((np.log10(subj_obj.freqs) + ((np.ptp(np.log10(subj_obj.freqs))+1) * (i % 5))).tolist())
        xs.append((np.log10(subj_obj.freqs) + ((np.ptp(np.log10(subj_obj.freqs))+1) * (i % 5))).tolist())



    # data needed for bottom panel, tstat
    # t = subj_obj.res['ts'][:, elec_num]

    # shared x-axis data
    # x = subj_obj.freqs

    # init top panel and set the data
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


def create_t_fig(elecs):

    # make sure data is loaded for this figure
    subj_obj = data.subject_objs[subjs[subj_drop.value]]
    if subj_obj.subject_data is None:
        subj_obj.load_data()
    subj_obj.filter_data_to_task_phases(subj_obj.task_phase_to_use)

    # Data for top panel
    recalled = subj_obj.recall_filter_func(subj_obj.task, subj_obj.subject_data.events.data, subj_obj.rec_thresh)
    df = subj_obj.subject_data.shape[0] - 2
    crit_t = t.ppf(0.975, df)

    ys = []
    xs = []

    max_y_range = np.ptp(subj_obj.res['ts'], axis=0).max()


    for i, elec in enumerate(elecs):

        yscale = (i/5+1) * max_y_range

        # t-stat
        ys.append((subj_obj.res['ts'][:, elec] + yscale).tolist())

        # zero line
        ys.append((np.zeros(subj_obj.res['ts'][:, elec].shape[0]) + yscale).tolist())

        # postitive critical t
        ys.append((np.zeros(subj_obj.res['ts'][:, elec].shape[0]) + crit_t + yscale).tolist())

        # negative critical t
        ys.append((np.zeros(subj_obj.res['ts'][:, elec].shape[0]) - crit_t + yscale).tolist())

        xs.append((np.log10(subj_obj.freqs) + ((np.ptp(np.log10(subj_obj.freqs)) + 1) * (i % 5))).tolist())
        xs.append((np.log10(subj_obj.freqs) + ((np.ptp(np.log10(subj_obj.freqs)) + 1) * (i % 5))).tolist())
        xs.append((np.log10(subj_obj.freqs) + ((np.ptp(np.log10(subj_obj.freqs)) + 1) * (i % 5))).tolist())
        xs.append((np.log10(subj_obj.freqs) + ((np.ptp(np.log10(subj_obj.freqs)) + 1) * (i % 5))).tolist())


    # init top panel and set the data
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

def create_two_panel_for_elec(elec_num=0):
    """

    """

    # make sure data is loaded for this figure
    subj_obj = data.subject_objs[subjs[subj_drop.value]]
    if subj_obj.subject_data is None:
        subj_obj.load_data()
    subj_obj.filter_data_to_task_phases(subj_obj.task_phase_to_use)

    # Data for top panel
    recalled = subj_obj.recall_filter_func(subj_obj.task, subj_obj.subject_data.events.data, subj_obj.rec_thresh)
    y1 = subj_obj.subject_data[recalled, :, elec_num].mean('events').data
    y2 = subj_obj.subject_data[~recalled, :, elec_num].mean('events').data

    # data needed for bottom panel, tstat
    t = subj_obj.res['ts'][:, elec_num]

    # shared x-axis data
    x = subj_obj.freqs

    # init top panel and set the data
    p1, source1 = init_spec_fig()
    source1.data = dict(
        xs=[x, x],
        ys=[y1, y2],
        line_width=[4, 4],
        line_color=['#8c564b', '#1f77b4'],
        labels=['Good Memory', 'Bad Memory']
    )
    p1.title.text = 'Power Spectra: %s, elec %d' % (subj_drop.value, elec_num)

    # init bottom panel and set the data
    p2, source2 = init_tstat_fig()
    source2.data = dict(x=x, y=t)
    lim = np.ceil(np.max(np.abs(t)))
    p2.y_range = Range1d(-lim, lim)

    #
    return column([p1, p2])




# def update_spect_fig(this_plot, this_source):
#
#     subj_obj = data.subject_objs[subjs[subj_drop.value]]
#     subj_obj.load_data()
#     subj_obj.filter_data_to_task_phases(subj_obj.task_phase_to_use)
#     recalled = subj_obj.recall_filter_func(subj_obj.task, subj_obj.subject_data.events.data, subj_obj.rec_thresh)
#     y1 = subj_obj.subject_data[recalled, :, 0].mean('events').data
#     y2 = subj_obj.subject_data[~recalled, :, 0].mean('events').data
#     x = subj_obj.freqs
#     subj_obj.subject_data = None
#
#     this_source.data = dict(
#         xs=[x, x],
#         ys=[y1, y2],
#         line_width=[4, 4],
#         line_color=['#8c564b', '#1f77b4'],
#         labels=['Good Memory', 'Bad Memory']
#     )
#     this_plot.title.text = 'Power Spectra: %s' % subj_drop.value
#
#
#     # return x, y1, y2, subj_drop.value
#
#
# def update_tstat_fig(this_plot, this_source):
#
#     subj_obj = data.subject_objs[subjs[subj_drop.value]]
#     subj_obj.load_data()
#     y = subj_obj.res['ts'][:, 0]
#     p = subj_obj.res['ps'][:, 0]
#     x = subj_obj.freqs
#     subj_obj.subject_data = None
#
#     this_source.data = dict(
#         x=x,
#         y=y)
#
#     lim = np.ceil(np.max(np.abs(y)))
#     this_plot.y_range = Range1d(-lim, lim)
#
#     # ax2.plot(x, y, '-k', linewidth=4)
#     # ax2.set_ylim([-np.max(np.abs(ax2.get_ylim())), np.max(np.abs(ax2.get_ylim()))])
#     # ax2.plot(x, np.zeros(x.shape), c=[.5, .5, .5], zorder=-1)
#     # return x, y, subj_drop.value


def update_region(attr, old, new):
    elec_locs = create_elec_loc_dropdown()
    region_drop.options = elec_locs
    region_drop.value = elec_locs[0]

def update(attr, old, new):
    l = doc.get_model_by_name('plots')
    l.children = []

    # l2 = doc.get_model_by_name('plots2')
    # l2.children = []

    figs = []
    if region_drop.value != ' ':
        if region_drop.value == 'All':
            n_elecs = data.subject_objs[subjs[subj_drop.value]].subject_data.shape[2]
            for i in range(n_elecs):
                print i
                print n_elecs
                # l.children.append(create_two_panel_for_elec(i))
                figs.append(create_two_panel_for_elec(i))
        else:
            elecs = np.where(data.subject_objs[subjs[subj_drop.value]].elec_locs[region_drop.value])[0]
            # for i in elecs:
            #     l.children.append(create_two_panel_for_elec(i))
            #     figs.append(create_two_panel_for_elec(i))
            #     print 'pre_grid'
            p = create_fig(elecs)
            p2 = create_t_fig(elecs)
        # r = gridplot(figs, ncols=3)
        # print 'post grid'
        # r = row(figs)
        # print 'pre app'
        l.children.append(row([p, p2]))
        # l2.children.append(p2)
        # print 'post app'

subj_drop.on_change('value', update_region)
region_drop.on_change('value', update)

controls = [subj_drop, region_drop]
# for control in controls:
#     control.on_change('value', lambda attr, old, new: update())

sizing_mode = 'fixed'  # 'scale_width' also looks nice with this example


inputs = widgetbox(*controls, sizing_mode=sizing_mode)
# l = layout([[inputs, column([p, p2])]], sizing_mode=sizing_mode, name='mainLayout')
# l = layout([[inputs]], sizing_mode=sizing_mode, name='mainLayout')
# r = column(name='plots')
# l = row(inputs, name='mainLayout', sizing_mode='fixed')
# l.width=10000
# l = layout([[inputs, column([p, p2], name='plots')]], sizing_mode=sizing_mode, name='mainLayout')
# l = row(inputs, column(), name='mainLayout')
r = row(name='plots')
r.width=800
r.height = 1500

r2 = row(name='plots2')
r2.width=800
r2.height = 1500

l = layout(children=[[inputs], [r]], sizing_mode='scale_width', name='mainLayout')
# l = layout([[row(inputs)]], name='mainLayout')
doc = curdoc()
doc.add_root(l)
# session = push_session(curdoc())
# update()


# plots = row([inputs, column(children=[inputs, p], sizing_mode='stretch_both')

# curdoc().add_root(l)
# curdoc().title = "SME"
# Contact GitHub API Training Shop Blog About
