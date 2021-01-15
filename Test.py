import numpy as np
import pandas as pd

df = pd.DataFrame(columns=(range(0,10)),index=(range(0,10)))
print(df)
df.loc[1,2] = 1
print(df)