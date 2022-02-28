import numpy as np
import csv, sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from scipy.stats import wilcoxon, ttest_rel, ttest_ind

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    for patch in bp['boxes']:
        patch.set(facecolor = color)
    plt.setp(bp['whiskers'], color='cornflowerblue')
    plt.setp(bp['caps'], color='steelblue')
    plt.setp(bp['medians'], color='dodgerblue')

file_dir = 'Results/'
file_name = ['CoTr_ncc_1_diffusion_1', 'PVT_ncc_1_diffusion_1', 'ViTVNet_ncc_1_diffusion_1', 'nnFormer_ncc_1_diffusion_1', 'TransMorphDiff', 'TransMorphBspline_ncc_1_diffusion_1', 'TransMorphBayes_ncc_1_diffusion_1', 'TransMorph_ncc_1_diffusion_1']
substruct = ['Left-Cerebral-White-Matter','Left-Cerebral-Cortex','Left-Lateral-Ventricle','Left-Inf-Lat-Vent','Left-Cerebellum-White-Matter','Left-Cerebellum-Cortex','Left-Thalamus-Proper*',
             'Left-Caudate','Left-Putamen','Left-Pallidum','3rd-Ventricle','4th-Ventricle','Brain-Stem','Left-Hippocampus','Left-Amygdala','CSF','Left-Accumbens-area','Left-VentralDC',
             'Left-vessel','Left-choroid-plexus','Right-Cerebral-White-Matter','Right-Cerebral-Cortex','Right-Lateral-Ventricle','Right-Inf-Lat-Vent','Right-Cerebellum-White-Matter',
             'Right-Cerebellum-Cortex','Right-Thalamus-Proper*','Right-Caudate','Right-Putamen','Right-Pallidum','Right-Hippocampus','Right-Amygdala','Right-Accumbens-area','Right-VentralDC',
             'Right-vessel','Right-choroid-plexus','5th-Ventricle','WM-hypointensities','non-WM-hypointensities','Optic-Chiasm','CC_Posterior','CC_Mid_Posterior','CC_Central','CC_Mid_Anterior,CC_Anterior']

outstruct = ['Brain-Stem', 'Thalamus', 'Cerebellum-Cortex', 'Cerebral-White-Matter', 'Cerebellum-White-Matter', 'Putamen', 'VentralDC', 'Pallidum', 'Caudate', 'Lateral-Ventricle', 'Hippocampus',
             '3rd-Ventricle', '4th-Ventricle', 'Amygdala', 'Cerebral-Cortex', 'CSF', 'choroid-plexus']
all_data = []
all_dsc = []
for exp_name in file_name:
    print(exp_name)
    exp_data = np.zeros((len(outstruct), 115))
    stct_i = 0
    for stct in outstruct:
        tar_idx = []
        with open(file_dir+exp_name+'.csv', "r") as f:
            reader = csv.reader(f, delimiter="\t")
            for i, line in enumerate(reader):
                if i == 1:
                    names = line[0].split(',')
                    idx = 0
                    for item in names:
                        if stct in item:
                            tar_idx.append(idx)
                        idx += 1
                elif i>1:
                    if line[0].split(',')[1]=='':
                        continue
                    val = 0
                    for lr_i in tar_idx:
                        vals = line[0].split(',')
                        val += float(vals[lr_i])
                    val = val/len(tar_idx)
                    exp_data[stct_i, i-2] = val
                    #print(stct_i)
        stct_i+=1
    all_dsc.append(exp_data.mean(axis=0))
    print(exp_data.mean())
    print(exp_data.std())
    all_data.append(exp_data)
    my_list = []
    with open(file_dir + exp_name + '.csv', newline='') as f:
        reader = csv.reader(f)
        my_list = [row[-1] for row in reader]
    my_list = my_list[2:]
    my_list = np.array([float(i) for i in my_list])*100
    print('jec_det: {:.3f} +- {:.3f}'.format(my_list.mean(), my_list.std()))

vec1 = all_dsc[-3]
idx = 0
for i in file_name:
    vec2 = all_dsc[idx]
    rank, pval = ttest_rel(list(vec1), list(vec2))
    print('{}, p-vale: {:.20f}'.format(i, pval))
    idx += 1

flierprops = dict(marker='o', markerfacecolor='cornflowerblue', markersize=2, linestyle='none', markeredgecolor='grey')
meanprops={ "markerfacecolor":"sandybrown", "markeredgecolor":"chocolate"}
fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
spacing_factor = 9
showmeans = False
sep = 0.9
pvt0 = plt.boxplot(all_data[0].T, labels=outstruct, positions=np.array(range(len(outstruct)))*spacing_factor-sep/2-3*sep, widths=0.6, showmeans=showmeans, flierprops=flierprops, meanprops=meanprops, patch_artist=True)
pvt1 = plt.boxplot(all_data[1].T, labels=outstruct, positions=np.array(range(len(outstruct)))*spacing_factor-sep/2-2*sep, widths=0.6, showmeans=showmeans, flierprops=flierprops, meanprops=meanprops, patch_artist=True)
vit = plt.boxplot(all_data[2].T, labels=outstruct, positions=np.array(range(len(outstruct)))*spacing_factor-sep/2-sep, widths=0.6, showmeans=showmeans, flierprops=flierprops, meanprops=meanprops, patch_artist=True)
nnformer = plt.boxplot(all_data[3].T, labels=outstruct, positions=np.array(range(len(outstruct)))*spacing_factor-sep/2, widths=0.6, showmeans=showmeans, flierprops=flierprops, meanprops=meanprops, patch_artist=True)
swdiff = plt.boxplot(all_data[4].T, labels=outstruct, positions=np.array(range(len(outstruct)))*spacing_factor+sep/2, widths=0.6, showmeans=showmeans, flierprops=flierprops, meanprops=meanprops, patch_artist=True)
swbspl = plt.boxplot(all_data[5].T, labels=outstruct, positions=np.array(range(len(outstruct)))*spacing_factor+sep/2+sep, widths=0.6, showmeans=showmeans, flierprops=flierprops, meanprops=meanprops, patch_artist=True)
swmcd = plt.boxplot(all_data[6].T, labels=outstruct, positions=np.array(range(len(outstruct)))*spacing_factor+sep/2+2*sep, widths=0.6, showmeans=showmeans, flierprops=flierprops, meanprops=meanprops, patch_artist=True)
sw = plt.boxplot(all_data[7].T, labels=outstruct, positions=np.array(range(len(outstruct)))*spacing_factor+sep/2+3*sep, widths=0.6, showmeans=showmeans, flierprops=flierprops, meanprops=meanprops,patch_artist=True)
set_box_color(pvt0, 'plum') # colors are from http://colorbrewer2.org/
set_box_color(pvt1, 'slateblue')
set_box_color(vit, 'tan')
set_box_color(nnformer, 'pink')
set_box_color(swdiff, 'wheat')
set_box_color(swbspl, 'olive')
set_box_color(swmcd, 'sandybrown')
set_box_color(sw, 'skyblue')
plt.grid(linestyle='--', linewidth=1)
plt.plot([], c='plum', label='CoTr')
plt.plot([], c='slateblue', label='PVT')
plt.plot([], c='tan', label='ViT-V-Net')
plt.plot([], c='pink', label='nnFormer')
plt.plot([], c='wheat', label='TransMorph-diff')
plt.plot([], c='olive', label='TransMorph-bspl')
plt.plot([], c='sandybrown', label='TransMorph-Bayes')
plt.plot([], c='skyblue', label='TransMorph')
font = font_manager.FontProperties(family='Cambria',
                                   style='normal', size=16)
leg = ax.legend(prop=font)
for line in leg.get_lines():
    line.set_linewidth(4.0)
minor_ticks = np.arange(-10.8, len(outstruct) * spacing_factor, 0.8)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(np.arange(0, 1.05, 0.2))
ax.set_yticks(np.arange(-0.05, 1.05, 0.05), minor=True)
ax.grid(which='major', color='#CCCCCC', linestyle='--')
ax.grid(which='minor', color='#CCCCCC', linestyle=':')
plt.xticks(range(0, len(outstruct) * spacing_factor, spacing_factor), outstruct, fontsize=20)
plt.yticks(fontsize=20)
for tick in ax.get_xticklabels():
    tick.set_fontname("Cambria")
for tick in ax.get_yticklabels():
    tick.set_fontname("Cambria")
plt.xlim(-5, len(outstruct)*spacing_factor-4.2)
plt.ylim(-0.05, 1.05)
plt.tight_layout()
plt.xticks(rotation=90)
plt.gcf().subplots_adjust(bottom=0.4)
plt.show()
