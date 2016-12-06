"""

"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import optimize_joint_classifier_for_enc
import optimize_joint_classifier_for_rec
import optimize_joint_classifier_for_joint
from GroupLevel import group_classifier


def run(enc_scales, do_plot=False):

    # train on both enc and rec and find the optimal scaling parameter for testing on enc
    aucs_enc, best_scale_enc = optimize_joint_classifier_for_enc.run(enc_scales)

    # train on both enc and rec and find the optimal scaling parameter for testing on rec
    aucs_rec, best_scale_rec = optimize_joint_classifier_for_rec.run(enc_scales)

    # train on both enc and rec and find the optimal scaling parameter for testing on both
    aucs_both, best_scale_both = optimize_joint_classifier_for_joint.run(enc_scales)

    # apply the scaling parameter best optimzed for both to enc and rec individually
    both_to_enc = group_classifier.GroupClassifier(analysis_name='all_events_train_enc_test_enc',
                                                   train_phase=['enc', 'rec'],
                                                   test_phase=['enc'],
                                                   save_class=True,
                                                   start_time=[-1.2, -2.9],
                                                   end_time=[0.5, -0.2],
                                                   scale_enc=best_scale_both)
    both_to_enc.process()
    aucs_both_to_enc = both_to_enc.summary_table[both_to_enc.summary_table['LOSO'] == 1]['AUC']

    both_to_rec = group_classifier.GroupClassifier(analysis_name='all_events_train_enc_test_enc',
                                                   train_phase=['enc', 'rec'],
                                                   test_phase=['rec'],
                                                   save_class=True,
                                                   start_time=[-1.2, -2.9],
                                                   end_time=[0.5, -0.2],
                                                   scale_enc=best_scale_both)
    both_to_rec.process()
    aucs_both_to_rec = both_to_rec.summary_table[both_to_rec.summary_table['LOSO'] == 1]['AUC']

    # also get the enc to enc and rec to rec baselines
    enc_to_enc = group_classifier.GroupClassifier(analysis_name='all_events_train_enc_test_enc',
                                                  train_phase=['enc'],
                                                  test_phase=['enc'],
                                                  save_class=True)
    enc_to_enc.process()
    enc_mean = enc_to_enc.summary_table[enc_to_enc.summary_table['LOSO'] == 1]['AUC'].mean()

    rec_to_rec = group_classifier.GroupClassifier(analysis_name='all_events_train_enc_test_enc',
                                                  train_phase=['rec'],
                                                  test_phase=['rec'],
                                                  save_class=True)
    rec_to_rec.process()
    rec_mean = rec_to_rec.summary_table[rec_to_rec.summary_table['LOSO'] == 1]['AUC'].mean()

    # plotting code is always so ugly
    if do_plot:
        with plt.style.context('/home1/jfm2/python/RAM_classify/myplotstyle.mplstyle'):
            # Plot 1
            # 3 panel plot showing auc as a function of scaling for testing on enc (left), rec (middle), both (right)
            # dashed horizontal line is classifier performance for baseline enc to enc (left) and rec to rec (middle)
            # circle is the auc at the best scaling from the both (right) plot
            plt.figure(1, figsize=(18.5, 11.5))
            f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
            data = [aucs_enc, aucs_rec, aucs_both]
            titles = ['encoding', 'retrieval', 'both']
            means = [enc_mean, rec_mean]

            # create each subplot
            for i, ax in enumerate((ax1, ax2, ax3)):

                # auc as a function of scaling
                ax.plot(np.log10(enc_scales), np.mean(data[i], axis=0), linewidth=4)

                # point to indicate the auc at the optimized for both scaling
                ax.scatter(np.log10(enc_scales)[enc_scales == best_scale_both],
                           np.mean(data[i], axis=0)[enc_scales == best_scale_both],
                           s=200, color='k', label='Optimized for both')

                # ticks, title, legend
                ax.set_xticks(np.log10(enc_scales)[::3])
                ax.set_xticklabels(np.round(enc_scales[::3] * 100) / 100, rotation=-45, fontsize=20)
                ax.set_title('Joint to ' + titles[i], fontsize=20)
                l = ax1.legend(scatterpoints=1)
                l.get_frame().set_facecolor([.99, .99, .99, 1])

                # text overlay to indicate the max auc
                x = np.log10(enc_scales)[np.argmax(np.mean(data[i], axis=0))]
                y = np.mean(data[i], axis=0).max()
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
                ax.text(x, y + .0025, '%.3f' % y, horizontalalignment='center', fontsize=16, bbox=props)

                # only axis labels on first plot
                if i == 0:
                    ax.set_xlabel('Encoding/Retrieval Weighting', fontsize=20)
                    ax.set_ylabel('AUC', fontsize=20)

                # only dashed line on first two plots
                if i != 2:
                    ax.plot([np.log10(enc_scales)[0], np.log10(enc_scales)[-1]],
                            [means[i]] * 2, '--k', linewidth=4, label='Baseline')
            fig = plt.gcf()
            fig.set_size_inches(18.5, 11.5)
            plt.show()
            # plt.savefig('/home1/jfm2/joint_to_enc_and_rec.pdf')

            # Plot 2
            # scatter plot with marginal histograms for the distribution of aucs for:
            #     1) training on both encoding and retrieval testing and optimized for encoding vs training on both
            #        encoding and retrieval testing and optimized for retrieval
            #     2) training on both encoding and retrieval testing and optimized for encoding vs training on both
            #        and optimized for both, testing on encoding.
            #     3) training on both encoding and retrieval testing and optimized for retrieval vs training on both
            #        and optimized for both, testing on retrieval.

            # based on http://matplotlib.org/examples/pylab_examples/scatter_hist.html
            nullfmt = NullFormatter()

            # values and labels to loop over
            xs = [aucs_enc[:, enc_scales == best_scale_enc],
                  aucs_enc[:, enc_scales == best_scale_enc],
                  aucs_rec[:, enc_scales == best_scale_rec]]

            ys = [aucs_rec[:, enc_scales == best_scale_rec],
                  both_to_enc.summary_table[both_to_enc.summary_table['LOSO'] == 1]['AUC'],
                  both_to_rec.summary_table[both_to_rec.summary_table['LOSO'] == 1]['AUC']]

            xlabels = ['Enc, optimized for enc',
                       'Enc, optimized for enc',
                       'Rec, optimized for rec']

            ylabels = ['Rec, optimized for rec',
                       'Enc, optimized for both',
                       'Rec, optimized for both']

            # kind of hacky, create some fake subplots to figure out where the axes should be
            left = []
            bottom_h = []
            plt.figure(1, figsize=(18, 6))
            f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
            for i, ax in enumerate([ax1, ax2, ax3]):
                pos = ax.get_position()
                left.append(pos.x0)
                width = pos.x1 - pos.x0 - .05
                bottom = pos.y0
                height = pos.y1 - pos.y0
                bottom_h.append(left[i] + width + 0.005)
            f.clf()

            # start with a rectangular Figure
            plt.figure(1, figsize=(18, 6))

            # plot each scatter
            for i in range(3):

                # define the axis locations for this subplot
                rect_scatter = [left[i], bottom, width, height]
                rect_histx = [left[i], height + .15, width, 0.06]
                rect_histy = [bottom_h[i], bottom, 0.02, height]

                # make axes
                axScatter = plt.axes(rect_scatter)
                axHistx = plt.axes(rect_histx)
                axHisty = plt.axes(rect_histy)

                # data for this subplot
                x = xs[i]
                y = ys[i]

                # turn off labels for plots 2 and 3
                if i > 0:
                    axScatter.yaxis.set_major_formatter(nullfmt)

                # no labels for any marginal histogramgs
                axHistx.xaxis.set_major_formatter(nullfmt)
                axHisty.yaxis.set_major_formatter(nullfmt)

                # the scatter plot
                axScatter.scatter(x, y, s=100, color='#1f77b4', edgecolor='black', linewidth=1, alpha=.9)
                axScatter.set_ylabel(ylabels[i], fontsize=20)
                axScatter.set_xlabel(xlabels[i], fontsize=20)
                axScatter.plot([.5, .5], [0, 1], '--k', zorder=-1)
                axScatter.plot([0, 1], [.5, .5], '--k', zorder=-1)
                axScatter.plot([0, 1], [0, 1], '-k', linewidth=1, zorder=-1)
                axScatter.set_xlim((.25, 1))
                axScatter.set_xticks(np.linspace(.25, 1, 4))
                axScatter.set_ylim((.25, 1))
                axScatter.set_yticks(np.linspace(.25, 1, 4))

                # histograms
                bins = np.arange(0, 1.025, 0.025)
                axHistx.hist(x, bins=bins)
                axHisty.hist(y, bins=bins, orientation='horizontal')
                axHistx.set_xlim(axScatter.get_xlim())
                axHisty.set_ylim(axScatter.get_ylim())
                axHistx.set_yticks([])
                axHistx.set_xticks(np.linspace(.25, 1, 4))
                axHisty.set_xticks([])
                axHisty.set_yticks(np.linspace(.25, 1, 4))
            plt.show()

if __name__ == '__main__':
    enc_scales = np.logspace(np.log10(10**-1.5), np.log10(10**1.5), 25)
    run(enc_scales)
