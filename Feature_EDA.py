import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from astropy.io import fits
import os
from Path import DataPath, ImgPath, ModelPath


# Load data from fits file to pandas data frame
data_fits = fits.open('/Users/haoyufan/PNow_Assessment/Data/sage_panstarrs_dered.fits')
print(data_fits[1].data.names)
# Columns of interest:
# RA/DEJ2000: coordinate of target, not related to the project;
# U/V/G/R/IMAG: brightness of UVGRI bands, similar to color;
# TEFF/LOGG/FEH: the major parameters of a star. Focus on TEFF (effective temperature);
# *ERR/E*: The errors of the above parameters, may be used as weights but will be dropped for now;


# Fits file use Big-endian buffer so loading the data into a data frame takes more effort
X_namelist = ['UMAG', 'VMAG', 'GMAG', 'RMAG', 'IMAG']
Y_namelist = ['TEFF', 'LOGG', 'FEH']

data_all = np.asarray([data_fits[1].data.field(n) for n in X_namelist + Y_namelist]).transpose()
data_all.byteswap().newbyteorder()
df = pd.DataFrame(data=data_all, columns=X_namelist + Y_namelist)


# The stellar parameters are from physical models and are more reliable within certain range
# Further trim the table, drop possible duplicate, resort index
idx = (df["TEFF"] > 3000) & (df["TEFF"] < 7000) & \
      (df["LOGG"] > 1) & (df["LOGG"] < 5) & \
      (df["FEH"] > -2) & (df["FEH"] < 0.5)
df = df.loc[idx]

df.drop(columns=["LOGG", "FEH"], inplace=True)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

# Some EDA effort
# Basic print-out
print(df.dtypes)
print(df.describe().transpose())
# Good dtype, no missing values, ~115K rows
# Numbers making sense, no obvious outliers


# Hist of columns
fig = plt.figure(figsize=[10, 5], dpi=200)
for i, col in enumerate(df.columns):
      ax = fig.add_subplot(3, 2, i+1)
      ax.hist(df[col])
      ax.set_ylabel(col)
plt.tight_layout()
plt.savefig(os.path.join(ImgPath, "Hist_RawUVGRI.png"))
plt.close()
# All X features are tightly distributed within a similar range


# Correlation among columns
plt.matshow(df.corr())
plt.xticks(ticks=np.arange(6), labels=df.columns)
plt.yticks(ticks=np.arange(6), labels=df.columns)
plt.colorbar()
plt.savefig(os.path.join(ImgPath, "Corr_RawUVGRI.png"))
plt.close()
# X_features are tightly correlated as expected. If a star is bright, all band will be bright.
# We are more interested in the difference between bands brightness


# Getting Delta brightness
band_all = ["U", "V", "G", "R", "I"]
d_band_namelist = []
for i in range(len(band_all)):
    for j in range(i):
          band1, band2 = band_all[i], band_all[j]
          df["-".join([band1, band2])] = df[band1 + "MAG"] - df[band2 + "MAG"]
          d_band_namelist.append("-".join([band1, band2]))

print(d_band_namelist)

# Redo Corr and Hist plots for d_bands
fig = plt.figure(figsize=[10, 5], dpi=200)
for i, col in enumerate(d_band_namelist):
      ax = fig.add_subplot(3, 4, i+1)
      ax.hist(df[col])
      ax.set_ylabel(col)
plt.tight_layout()
plt.savefig(os.path.join(ImgPath, "Hist_DeltaUVGRI.png"))
plt.close()

# Correlation among columns
plt.matshow(df[d_band_namelist + ["TEFF"]].corr())
plt.xticks(ticks=np.arange(11), labels=d_band_namelist + ["TEFF"])
plt.yticks(ticks=np.arange(11), labels=d_band_namelist + ["TEFF"])
plt.colorbar()
plt.savefig(os.path.join(ImgPath, "Corr_DeltaUVGRI.png"))
plt.close()


# Also, normalize Temperature. This is important for NN but not sure for lightGBM
# Note I trimmed TEFF to between 3000 and 7000
df["TEFF_Norm"] = (df["TEFF"] - 3000) / (7000 - 3000)

# Split data into training and testing set, save to csv file
train_df, test_df = train_test_split(df, test_size=0.2)
with open(os.path.join(DataPath, "TrainingData.csv"), "w") as f:
      f.write(train_df.to_csv(index=False))

with open(os.path.join(DataPath, "TestingData.csv"), "w") as f:
      f.write(test_df.to_csv(index=False))


