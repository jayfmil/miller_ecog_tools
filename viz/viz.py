import matplotlib as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
from bokeh.plotting import figure, output_file, show, curdoc
from bokeh.models import ColumnDataSource, HoverTool, TapTool, BasicTicker, Rect, ColorBar, LinearColorMapper, \
    FixedTicker, TickFormatter, Row
from bokeh.client import push_session
from bokeh.layouts import row
from bokeh.document import Document
from scipy.stats import ttest_1samp
from random import choice
from string import ascii_uppercase
import math as Math

document = Document()
session = push_session(document)
output_file("/Users/jmiller/Desktop/heatmap.html", title="heatmap.py example")

def compute_bokeh_palette(cm_name='RdBu_r'):
    colormap = cm.get_cmap(cm_name)
    bokehpalette = [plt.colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]
    return bokehpalette

def compute_fill_colors(bokeh_palette, values, center_val=0.5):
    max_dev = np.abs(np.array([np.nanmin(values), np.nanmax(values)]) - center_val).max()
    color_range = [center_val - max_dev, center_val + max_dev]
    norm_val = np.array((values - color_range[0]) / (color_range[1] - color_range[0])) * 255
    return list(np.array(bokeh_palette)[np.round(norm_val).astype(int)]), color_range




TOOLS = "hover,save,pan,box_zoom,wheel_zoom,box_select,tap"
TOOLS = "hover,tap,save"
opts = dict(tools=TOOLS, plot_width=500, plot_height=500)
bokehpalette = compute_bokeh_palette()



df = pd.read_pickle('/Users/jmiller/Desktop/df5.pkl')
# data = df['loso'].T.groupby(level=[1,2]).mean().unstack().T
d = df['loso'].T.groupby(level=['time', 'window']).mean()
N = df['loso'].T.groupby(level=['subject']).mean().shape[0]
# N = 23
d.reset_index(inplace=True)
time_ax = d['time'].values
window_ax = d['window'].values
auc = d['AUC'].values


source = ColumnDataSource(data=dict(time=time_ax, window=window_ax, auc=auc))
c, c_range = compute_fill_colors(bokehpalette,auc)
plot = figure(name='plot1',**opts)
plot.toolbar_location='above'
renderer = plot.rect(x="time", y="window", width=.1, height=.1,
       source=source, alpha=1,
       fill_color=c, nonselection_fill_color='fill_color', nonselection_fill_alpha=1,
       line_color=None, nonselection_line_color=None)

plot.xaxis.axis_label = 'Time (s)'
plot.xaxis.axis_label_text_font_size = '16pt'
plot.xaxis.axis_label_text_font_style = 'normal'
plot.xaxis.major_label_text_font_size = '12pt'

plot.yaxis.axis_label = 'Window Size (s)'
plot.yaxis.axis_label_text_font_size = '16pt'
plot.yaxis.axis_label_text_font_style = 'normal'
plot.yaxis.major_label_text_font_size = '12pt'

plot.title.text = 'LOSO AUC, N=%d' % N
plot.title.text_font_size = '16pt'
plot.title.text_font_style = 'normal'
plot.title.align = 'center'



plot.select_one(HoverTool).tooltips = [
    ('time', '@time'),
    ('window', '@window'),
    ('auc', '@auc'),
]


color_mapper = LinearColorMapper(palette=bokehpalette, low=c_range[0], high=c_range[1])
color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(),
                     label_standoff=12, border_line_color=None, location=(0,0),
                     major_label_text_font_size='12pt')
plot.add_layout(color_bar, 'right')




mainLayout = Row(plot, name='mainLayout')
curdoc().add_root(mainLayout)
session = push_session(curdoc())
# mainLayout = row(toggle,name='Widgets'),p1,name='mainLayout')
# curdoc().add_root(mainLayout)
# session = push_session(curdoc())





# session.show(row([plot, plot]))
# session.show(row)

# document.validate()
# session.loop_until_closed()



def on_selection_change1(attr, old, new):
    # try:
    if len(new['1d']['indices']) > 0:

        # new_t = time_ax[new['1d']['indices']]
        # new_w = window_ax[new['1d']['indices']]

        new_t = time_ax[new['1d']['indices']]
        new_w = window_ax[new['1d']['indices']]

        # old_name = str(old_t[0]) + str(old_w[0])
        # old_name = str(new_t[0]) +str(new_w[0])

        # new_t = '0.35'
        # new_w = '1.0'
        # print new_t, new_w
        rootLayout = curdoc().get_model_by_name('mainLayout')
        listOfSubLayouts = rootLayout.children
        print listOfSubLayouts, '1'

        # if curdoc().get_model_by_name('plot2') is not None:
        # if curdoc().get_model_by_name(old_name) is not None:
        #     plotToRemove = curdoc().get_model_by_name(old_name)
        #     listOfSubLayouts.remove(plotToRemove)
        #     print listOfSubLayouts, '2'
        #     print 'remove'

        # if len(listOfSubLayouts) > 1:
        #     del(listOfSubLayouts[1])

        # if curdoc().get_model_by_name('plot2') is None:
        print 'add'
        # old_name = ''.join(choice(ascii_uppercase) for i in range(12))
        # df2 = pd.read_pickle('/Users/jmiller/Desktop/df_anat.pkl')
        feat_data = df2.T[str(new_t[0])][str(new_w[0])].T.groupby(level=['subj', 'region']).mean()
        t, p = ttest_1samp(feat_data.unstack(), 0, nan_policy='omit')
        region_ax = np.array(list(feat_data.unstack().columns.levels[1][feat_data.unstack().columns.labels[1]])).astype(
            str)
        freq_ax = np.log10(np.array(list(feat_data.unstack().columns.levels[0][feat_data.unstack().columns.labels[0]])))
        source2 = ColumnDataSource(data=dict(freq=freq_ax, region=region_ax, t=t))

        c2, c2_range = compute_fill_colors(bokehpalette, t, 0)
        plot2 = figure(x_range=np.unique(region_ax).tolist(), y_range=(freq_ax.min() - .5, freq_ax.max() + .5),
                       **opts)
        plot2.toolbar_location = None

        plot2.rect(x="region", y="freq", width=1, height=.33,
                          source=source2, alpha=1,
                          fill_color=c2, nonselection_fill_color='fill_color', nonselection_fill_alpha=1,
                          line_color=None, nonselection_line_color=None)
        # plot2.yaxis[0].ticker=FixedTicker(ticks=np.unique(freq_ax))


        plot2.select_one(HoverTool).tooltips = [
            ('region', '@region'),
            ('freq', '@freq'),
            ('t', '@t'), ]
        #
        color_mapper2 = LinearColorMapper(palette=bokehpalette, low=c2_range[0], high=c2_range[1])
        color_bar2 = ColorBar(color_mapper=color_mapper2, ticker=BasicTicker(),
                              label_standoff=12, border_line_color=None, location=(0, 0))
        plot2.add_layout(color_bar2, 'right')

        #
        # print new_t, new_w
        #

        # plotToAdd = curdoc().get_model_by_name('plot2')
        listOfSubLayouts.append(plot2)

        print listOfSubLayouts
        # except:
        #     pass


source.on_change('selected', on_selection_change1)


old_name = None
df2 = pd.read_pickle('/Users/jmiller/Desktop/df_anat5.pkl')
# feat_data = df2.T['0.75']['1.3'].T.groupby(level=['subj','region']).mean()
# t, p = ttest_1samp(feat_data.unstack(), 0, nan_policy='omit')
# region_ax = np.array(list(feat_data.unstack().columns.levels[1][feat_data.unstack().columns.labels[1]])).astype(str)
# freq_ax = np.log10(np.array(list(feat_data.unstack().columns.levels[0][feat_data.unstack().columns.labels[0]])))
# source2 = ColumnDataSource(data=dict(freq=freq_ax, region=region_ax, t=t))
#
#
# c2, c2_range = compute_fill_colors(bokehpalette,t,0)
# plot2 = figure(x_range=np.unique(region_ax).tolist(), y_range=(freq_ax.min()-.5, freq_ax.max()+.5),**opts)
# plot2.toolbar_location=None
#
# rend = plot2.rect(x="region", y="freq", width=1, height=.33,
#        source=source2, alpha=1,
#        fill_color=c2, nonselection_fill_color='fill_color', nonselection_fill_alpha=1,
#        line_color=None, nonselection_line_color=None)
# # plot2.yaxis[0].ticker=FixedTicker(ticks=np.unique(freq_ax))
#
#
# plot2.select_one(HoverTool).tooltips = [
#     ('region', '@region'),
#     ('freq', '@freq'),
#     ('t', '@t'),]
#
# color_mapper2 = LinearColorMapper(palette=bokehpalette, low=c2_range[0], high=c2_range[1])
# color_bar2 = ColorBar(color_mapper=color_mapper2, ticker=BasicTicker(),
#                      label_standoff=12, border_line_color=None, location=(0,0))
# plot2.add_layout(color_bar2, 'right')

session.show()
session.loop_until_closed()