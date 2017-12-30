import pandas as pd
import math
huff_t1_df=pd.read_csv('./huff_t1.csv')


amp=3

size=int(round(math.log(amp,2)))+1

print(size)
print(huff_t1_df['Code'][size])

