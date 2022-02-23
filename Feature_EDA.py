import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from Utils import DataPath, ImgPath, teff_range, logg_range, feh_range

###################
# Load data
###################
csv_filename = os.path.join(DataPath, "dr8_v1.0_stellar_LRS.csv")
df = pd.read_csv(csv_filename, delimiter="|")
print(df.columns)
print("Raw data have %i rows" % len(df))
# Columns of concern (QA, X and Y)
# snru/g/r/i/z: signal to noise ratio, for general data quality
# class: type of target, will focus on stars
# mag1 to mag7: brightness of different colors, input Xs of model
# magtype: meta data for mag1 - mag7, what is the color being measured?
# teff, logg, feh: key parameter of star, will focus on teff, their
#                  errors are *_err, can be used as weights


###################
# Filtering
###################
# Filtering:
# 1. class = star
# 2. valid snru/g/r/i/z (snr* > 0)
# 3. max(snr) > 20
# 4. 3000 < TEFF < 7000, 1 < logg < 5, -2 < FeH < 0.5, sweet point of physical model
# 4. magtype = gribvjh, also change column names, e.g. mag1 --> mag_g
# 5. remove outlier in mag1 - mag7

df = df.loc[df["class"] == "STAR"]

SNR_labels = ["snr" + item for item in ["u", "g", "r", "i", "z"]]
df["SNR"] = df[SNR_labels].max(axis=1)
df = df.loc[df["SNR"] > 20]
for band in SNR_labels:
    df = df.loc[df[band] > 0]


idx = (df["teff"] > teff_range[0]) & (df["teff"] < teff_range[1]) & \
      (df["logg"] > logg_range[0]) & (df["logg"] < logg_range[1]) & \
      (df["feh"] > feh_range[0]) & (df["feh"] < feh_range[1])
df = df.loc[idx]

df = df.loc[df["magtype"] == "gribvjh"]

mag_label_old = ["mag%i" % item for item in np.arange(7) + 1]
mag_label_new = ["mag_" + item for item in "gribvjh"]
rename_dict = {}
for key, value in zip(mag_label_old, mag_label_new):
    rename_dict[key] = value
df.rename(columns=rename_dict, inplace=True)
print(df[mag_label_new].describe().transpose())
df[mag_label_new].boxplot()
plt.show()

# Max of mag_b, j, h is 20.0, looks like some rounding up
# Also mag_v has an outlier at ~17.8, will use it for cutoff of all magx
mag_cutoff = df["mag_v"].max()
for mag in mag_label_new:
    df = df.loc[df[mag] < mag_cutoff]

par_names = ["teff", "logg", "feh"]
par_err_names = [item+"_err" for item in par_names]
df.drop_duplicates(inplace=True)
df = df[SNR_labels + mag_label_new + par_names + par_err_names]
df.reset_index(drop=True, inplace=True)
print("Trimmed table has %i rows" % (len(df)))


###################
# EDA efforts
###################
print(df.dtypes)
print(df.describe().transpose())
# Things that I care about:
# 1. hist for snrx, magx, and the three parameters
# 2. correlations between the input Xs and Y to predict (teff for now)

par_groups = {"SNR": SNR_labels,
              "mag": mag_label_new,
              "pars": par_names}
for key in par_groups:
    n_rows = int(np.ceil(len(par_groups[key]) / 3))
    fig = plt.figure(figsize=[9, n_rows*2 + 1], dpi=200)
    for i, item in enumerate(par_groups[key]):
        ax = fig.add_subplot(n_rows, 3, i+1)
        ax.hist(df[item])
        ax.set_ylabel(item)
    plt.tight_layout()
    plt.savefig(os.path.join(ImgPath, key.join(["Hist_", ".png"])))
    plt.close()

plt.matshow(df[mag_label_new + ["teff"]].corr())
plt.xticks(ticks=np.arange(len(mag_label_new) + 1), labels=mag_label_new + ["teff"])
plt.yticks(ticks=np.arange(len(mag_label_new) + 1), labels=mag_label_new + ["teff"])
plt.colorbar()
plt.savefig(os.path.join(ImgPath, "Corr_Raw.png"))
plt.close()

###################
# Feature Engineering
###################
# There are strong correlations between magx, like bulbs having different wattage but the same color.
# color difference (delta bands) is usually used in physical models to address the overall growth.
# It is also better to normalize the stellar parameters.
# Lastly, I want to know the typical error of Teff, to compare with lgbm method


d_band_namelist = []
for i in range(len(mag_label_new)):
    for j in range(i):
        band1, band2 = mag_label_new[i], mag_label_new[j]
        df["-".join([band1[-1], band2[-1]])] = df[band1] - df[band2]
        d_band_namelist.append("-".join([band1[-1], band2[-1]]))

plt.matshow(df[d_band_namelist + ["teff"]].corr())
plt.xticks(ticks=np.arange(len(d_band_namelist) + 1), labels=d_band_namelist + ["teff"], rotation=-75)
plt.yticks(ticks=np.arange(len(d_band_namelist) + 1), labels=d_band_namelist + ["teff"])
plt.colorbar()
plt.savefig(os.path.join(ImgPath, "Corr_d_Band.png"))
plt.close()

df["teff_norm"] = (df["teff"] - teff_range[0]) / (teff_range[1] - teff_range[0])
df["teff_norm_err"] = df["teff_err"] / (teff_range[1] - teff_range[0])
df["logg_norm"] = (df["logg"] - logg_range[0]) / (logg_range[1] - logg_range[0])
df["logg_norm_err"] = df["logg_err"] / (logg_range[1] - logg_range[0])
df["feh_norm"] = (df["feh"] - feh_range[0]) / (feh_range[1] - feh_range[0])
df["feh_norm_err"] = df["feh_err"] / (feh_range[1] - feh_range[0])


print("Typical error of Teff is %.2f, or %.2f%%" %
      (df["teff_err"].median(), np.nanmedian(df["teff_err"].to_numpy() / df["teff"].to_numpy()) * 100))

# save table
with open(os.path.join(DataPath, "TrimmedData.csv"), "w") as f:
    f.write(df.to_csv(index=False))
