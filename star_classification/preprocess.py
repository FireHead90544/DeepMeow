import pandas as pd
import numpy as np
from maps import spectral_classes, star_colors

pd.set_option("future.no_silent_downcasting", True)
df = pd.read_csv("https://github.com/YBIFoundation/Dataset/raw/main/Stars.csv")

df.pop('Star category')
df.replace({'Spectral Class': spectral_classes, 'Star color': star_colors}, inplace=True)
df[['Spectral Class', 'Star color']] = df[['Spectral Class', 'Star color']].astype(np.int64)

df.to_csv("datasets/stars_preprocessed.csv", index=False)