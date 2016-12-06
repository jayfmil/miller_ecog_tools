
freqs = np.logspace(np.log10(1) ,np.log10(200) ,8)
cs = np.logspace(np.log10(1e-6) ,np.log10(1e4) ,22)
starts = np.arange(-1.5 ,.1 ,.1)
ends = starts+ 1.5
time_bins = np.stack([starts, ends], axis=1)

aucs = []
aucs_lolo = []
aucs_loso = []
for c in cs:
    enc = TH_Classify.ClassifyTH(force_reclass=True,
                                 save_class=True, freqs=freqs, compute_pval=False,
                                 start_time=-2.0, end_time=2.0,
                                 time_bins=time_bins, C=c,
                                 pool=None)
    enc.run_classify_for_all_subjs()
    aucs.append(enc.aucs())
    aucs_lolo.append(enc.aucs(cv_type=['lolo']))
    aucs_loso.append(enc.aucs(cv_type=['loso']))

