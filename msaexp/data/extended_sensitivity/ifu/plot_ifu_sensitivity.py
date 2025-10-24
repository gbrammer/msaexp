import matplotlib.pyplot as plt
import glob
import numpy as np
import scipy.ndimage as nd

from grizli import utils
import msaexp.utils as msautils
msautils.set_plot_style()

files = glob.glob("msaexp*sens*fits")
files.sort()

if 'ifu' in files[0]:
    keys = ['_'.join(f.split('_')[3:5]) for f in files]
    IS_IFU = True
else:
    keys = ['_'.join(f.split('_')[2:4]) for f in files]
    IS_IFU = False

GRATING_COLORS = {
    'g140h_f070lp': 'powderblue',
    'g140h_f100lp': 'skyblue',
    'g140m_f070lp': 'steelblue',
    'g140m_f100lp': 'cornflowerblue',
    'g235h_f170lp': 'tan',
    'g235m_f170lp': 'goldenrod',
    'g395h_f290lp': 'salmon',
    'g395m_f290lp': 'firebrick',
    'prism_clear': 'darkmagenta'
}

sens_curves = [
    utils.read_catalog(file)
    for file in files
]

for sens, file in zip(sens_curves, files):
    sens["sensitivity"][sens["sensitivity"] < 0] = 0
    
    if 'g140' in file:
        sens["sensitivity"][sens["wavelength"] > 3.302] = 0.

fig, axes = plt.subplots(5, 1, figsize=(6,10), sharex=True)

ax = axes[0]
N = len(files)
for i, sens in enumerate(sens_curves):
    ax.plot(
        sens["wavelength"], sens["sensitivity"] / np.gradient(sens["wavelength"]),
        color=GRATING_COLORS[keys[i]], alpha=0.5,
    )
    
# if IS_IFU:
#     ax.set_ylim(0, 220)
# else:
#     ax.set_ylim(0, 1520)

unk = utils.Unique(keys, verbose=False)

linestyles = ['-','--',':']

for i, band in enumerate(['g140','g235','g395','prism']):
    ax = axes[i+1]
    
    plotted_keys = []
    
    for gr in unk.values:
        if not gr.startswith(band):
            continue
        
        for j in np.where(unk[gr])[0]:
            sens = sens_curves[j]
            if 'x1808' in files[j]:
                continue

            color = GRATING_COLORS[keys[j]]
            
            ax.plot(
                sens["wavelength"], sens["sensitivity"] / np.gradient(sens["wavelength"]),
                color=color,
                alpha=0.5,
                ls=linestyles[plotted_keys.count(keys[j])],
                label=files[j]
            )

            # if "orig_sensitivity" in sens.colnames:
            #     ax.plot(
            #         sens["wavelength"],
            #         sens["orig_sensitivity"] / np.gradient(sens["wavelength"]),
            #         color=color,
            #         linestyle='-.'
            #     )
            
            for c in ["calspec_model", "calspec_model_1"]:
                if (c in sens.colnames) & (0):
                    sdata = sens["data"] / sens[c]
                    sdata[sdata <= 0] = np.nan
                    smed = nd.median_filter(sdata, 11)
                    smed[smed > np.nanmax(sens["sensitivity"])] = np.nan
                    resid = (sdata - smed)/smed
                    sdata[np.abs(resid) > 0.1] = np.nan
                    sdata[~np.isfinite(smed)] = np.nan
                    
                    ax.scatter(
                        sens["wavelength"],
                        sdata / np.gradient(sens["wavelength"]),
                        color=color,
                        #linestyle='-.',
                        alpha=0.05,
                        # marker='.',
                    )

            for order, oscale in zip([2,3], [5, 20]):
                col = f"sensitivity_{order}"
                if col in sens.colnames:
                    trim = sens["wavelength"] * order < sens["wavelength"].max()
                    #trim = sens[col] > 1.e-8
                    # trim = np.isfinite(sens[col])
                    
                    sens_i = np.interp(
                        sens["wavelength"],
                        sens["wavelength"] * order,
                        sens[col],
                        left=0.0,
                        right=0.0,
                    ) # / sens["sensitivity"]
                    trim = sens_i > 0
                    
                    ax.plot(
                        sens["wavelength"][trim],
                        (sens_i / np.gradient(sens["wavelength"]))[trim] * oscale,
                        color=color,
                        alpha=0.3,
                        ls=linestyles[plotted_keys.count(keys[j])],
                        # label=files[j]
                    )

                    ax.fill_between(
                        sens["wavelength"][trim]/order,
                        sens["wavelength"][trim] * 0.,
                        (sens_i / np.gradient(sens["wavelength"]))[trim] * oscale,
                        color=color,
                        alpha=0.05,
                        hatch='///////' if order == 3 else None,
                        fc='None' if order == 3 else color,
                        # label=f"Order {order} x {order**2}" if keys[j] not in plotted_keys else None,
                        # label=files[j]
                    )
                    
                    olabel = r"Order $m=oo~~(\times ss~)$"
                    olabel = olabel.replace("oo", f"{order}")
                    olabel = olabel.replace("ss", f"{oscale}")
                        
                    ax.fill_between(
                        sens["wavelength"][trim],
                        sens["wavelength"][trim] * 0.,
                        (sens_i / np.gradient(sens["wavelength"]))[trim] * oscale,
                        color=color,
                        alpha=0.15,
                        hatch='/////////' if order == 3 else None,
                        fc='None' if order == 3 else color,
                        label=olabel if keys[j] not in plotted_keys else None,
                        # label=files[j]
                    )
                    
            plotted_keys.append(keys[j])
    
    ax.legend(
        # loc='upper right',
        fontsize=5.5
    )

for ax in axes:
    ax.grid()
    ax.set_yticklabels([])

ax.set_xlabel(r"$\lambda$  $\mu$m")
axes[2].set_ylabel(r'DN $\mu$Jy$^{-1}$ $\mu$m$^{-1}$')

fig.tight_layout(pad=0.5)

fig.savefig("msaexp_ifu_sensitivity_001.png")

