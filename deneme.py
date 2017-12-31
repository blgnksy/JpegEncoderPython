import pandas as pd
import math

huff_t1_df = pd.read_csv('./huff_t1.csv')

amp = -15

size = int(math.log(abs(amp), 2))+1

# print(size)
print(huff_t1_df['Code'][size])
