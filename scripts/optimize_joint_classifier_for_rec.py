"""
For all subjects in TH1, train on both encoding and retrieval, test on retrieval. Loop over a range of values for how
much to scale the encoding period relative to the retrieval period.
"""
import numpy as np
from GroupLevel import group


def run(enc_scales):
    aucs = []
    for scale in enc_scales:
        res_this_scale = group.Group(analysis_name='all_events_train_enc_test_enc',
                                     train_phase=['enc', 'rec'],
                                     test_phase=['rec'],
                                     save_class=True,
                                     start_time=[-1.2, -2.9],
                                     end_time=[0.5, -0.2],
                                     scale_enc=scale)
        res_this_scale.process()
        aucs.append(res_this_scale.summary_table[res_this_scale.summary_table['LOSO'] == 1]['AUC'])

    aucs = np.stack(aucs).T
    best_ind = np.argmax(np.mean(aucs, axis=0))
    best_auc = np.mean(aucs, axis=0)[best_ind]
    print('Best scaling value %.3f (index %d): AUC %.3f' % (enc_scales[best_ind], best_ind, best_auc))
    return aucs, enc_scales[best_ind]


if __name__ == '__main__':
    enc_scales = np.logspace(np.log10(10**-1.5), np.log10(10**1.5), 25)
    run(enc_scales)
