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
file_name = ['affine', 'NiftyReg_IXI', 'ants_IXI', 'deedsBCV_IXI', 'lddmm_IXI', 'Vxm_1_ncc_1_diffusion_1', 'Vxm_2_ncc_1_diffusion_1', 'VxmDiff', 'CycleMorph','MIDIR_ncc_1_diffusion_1', 'TransMorph_ncc_1_diffusion_1']
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

vec1 = all_dsc[-1]
idx = 0
for i in file_name[:-1]:
    vec2 = all_dsc[idx]
    rank, pval = ttest_rel(list(vec1), list(vec2))
    print('{}, p-vale: {:.20f}'.format(i, pval))
    idx += 1

flierprops = dict(marker='o', markerfacecolor='cornflowerblue', markersize=2, linestyle='none', markeredgecolor='grey')
meanprops={ "markerfacecolor":"sandybrown", "markeredgecolor":"chocolate"}
fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
spacing_factor = 14
showmeans = False
sep = 1.0
affine = plt.boxplot(all_data[0].T, labels=outstruct, positions=np.array(range(len(outstruct)))*spacing_factor-sep*5, widths=0.6, showmeans=showmeans, flierprops=flierprops, meanprops=meanprops, patch_artist=True)
nifty = plt.boxplot(all_data[1].T, labels=outstruct, positions=np.array(range(len(outstruct)))*spacing_factor-sep*4, widths=0.6, showmeans=showmeans, flierprops=flierprops, meanprops=meanprops, patch_artist=True)
syn = plt.boxplot(all_data[2].T, labels=outstruct, positions=np.array(range(len(outstruct)))*spacing_factor-sep*3, widths=0.6, showmeans=showmeans, flierprops=flierprops, meanprops=meanprops, patch_artist=True)
deeds = plt.boxplot(all_data[3].T, labels=outstruct, positions=np.array(range(len(outstruct)))*spacing_factor-sep*2, widths=0.6, showmeans=showmeans, flierprops=flierprops, meanprops=meanprops,patch_artist=True)
lddmm = plt.boxplot(all_data[4].T, labels=outstruct, positions=np.array(range(len(outstruct)))*spacing_factor-sep, widths=0.6, showmeans=showmeans, flierprops=flierprops, meanprops=meanprops,patch_artist=True)
vxm1 = plt.boxplot(all_data[5].T, labels=outstruct, positions=np.array(range(len(outstruct)))*spacing_factor, widths=0.6, showmeans=showmeans, flierprops=flierprops, meanprops=meanprops,patch_artist=True)
vxm2 = plt.boxplot(all_data[6].T, labels=outstruct, positions=np.array(range(len(outstruct)))*spacing_factor+sep, widths=0.6, showmeans=showmeans, flierprops=flierprops, meanprops=meanprops,patch_artist=True)
vxmdiff = plt.boxplot(all_data[7].T, labels=outstruct, positions=np.array(range(len(outstruct)))*spacing_factor+sep*2, widths=0.6, showmeans=showmeans, flierprops=flierprops, meanprops=meanprops,patch_artist=True)
cycm = plt.boxplot(all_data[8].T, labels=outstruct, positions=np.array(range(len(outstruct)))*spacing_factor+sep*3, widths=0.6, showmeans=showmeans, flierprops=flierprops, meanprops=meanprops,patch_artist=True)
bsp = plt.boxplot(all_data[9].T, labels=outstruct, positions=np.array(range(len(outstruct)))*spacing_factor+sep*4, widths=0.6, showmeans=showmeans, flierprops=flierprops, meanprops=meanprops,patch_artist=True)
sw = plt.boxplot(all_data[10].T, labels=outstruct, positions=np.array(range(len(outstruct)))*spacing_factor+sep*5, widths=0.6, showmeans=showmeans, flierprops=flierprops, meanprops=meanprops,patch_artist=True)
set_box_color(affine, 'plum') # colors are from http://colorbrewer2.org/
set_box_color(nifty, 'slateblue')
set_box_color(syn, 'tan')
set_box_color(deeds, 'wheat')
set_box_color(lddmm, 'lightgreen')
set_box_color(vxm1, 'peachpuff')
set_box_color(vxm2, 'olive')
set_box_color(vxmdiff, 'sandybrown')
set_box_color(cycm, 'pink')
set_box_color(bsp, 'lightcoral')
set_box_color(sw, 'skyblue')
plt.grid(linestyle='--', linewidth=1)
plt.plot([], c='plum', label='Affine')
plt.plot([], c='slateblue', label='NiftyReg')
plt.plot([], c='tan', label='SyN')
plt.plot([], c='wheat', label='deedsBCV')
plt.plot([], c='lightgreen', label='lddmm')
plt.plot([], c='peachpuff', label='VoxelMorph-1')
plt.plot([], c='olive', label='VoxelMorph-2')
plt.plot([], c='sandybrown', label='VoxelMorph-diff')
plt.plot([], c='pink', label='CycleMorph')
plt.plot([], c='lightcoral', label='MIDIR')
plt.plot([], c='skyblue', label='TransMorph')
font = font_manager.FontProperties(family='Cambria',
                                   style='normal', size=10)
leg = ax.legend(prop=font)
for line in leg.get_lines():
    line.set_linewidth(4.0)
minor_ticks = np.arange(-10.8, len(outstruct) * spacing_factor, 0.8)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(np.arange(0, 1.05, 0.2))
ax.set_yticks(np.arange(-0.05, 1.05, 0.05), minor=True)
ax.grid(which='major', color='#CCCCCC', linestyle='--')
ax.grid(which='minor', color='#CCCCCC', linestyle=':')
plt.xticks(range(0, len(outstruct) * spacing_factor, spacing_factor), outstruct, fontsize=14,)
plt.yticks(fontsize=20)
for tick in ax.get_xticklabels():
    tick.set_fontname("Cambria")
for tick in ax.get_yticklabels():
    tick.set_fontname("Cambria")
plt.xlim(-8, len(outstruct)*spacing_factor-6.2)
plt.ylim(-0.05, 1.05)
plt.tight_layout()
plt.xticks(rotation=90)
plt.gcf().subplots_adjust(bottom=0.4)
plt.show()
