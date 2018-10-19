import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
params = {'figure.figsize': (18,12)}
plt.rcParams.update(params)
import plotly
import cufflinks as cf


detect_path = 'result.csv'
tag_path = 'tagging.csv'

detect_df = pd.read_csv(detect_path)
tag_df = pd.read_csv(tag_path)

detect_df['Image'] = detect_df['Image'].apply(lambda image_path: image_path.replace('images', 'detect'))
tag_df['Image'] = tag_df['Image'].apply(lambda image_path: image_path.replace('anqi/', '').replace('images', 'detect'))

detect_df.head()
tag_df.head()

combine_df = detect_df.rename(columns = {'People Count': 'People Count (Detect)'}).merge(tag_df.rename(columns = {'People Count': 'People Count (Tag)'}), on = 'Image')
combine_df['Diff'] = combine_df['People Count (Tag)'] - combine_df['People Count (Detect)']
combine_df = combine_df.sort_values('Diff')

from sklearn.metrics import mean_squared_error, mean_absolute_error
mean_squared_error(combine_df['People Count (Detect)'], combine_df['People Count (Tag)'])
mean_absolute_error(combine_df['People Count (Detect)'], combine_df['People Count (Tag)'])

combine_df['Diff'].value_counts().iplot(kind='bar', title = 'Detect vs Tag Diff (All)')
combine_df[combine_df['People Count (Tag)']==0]['Diff'].value_counts().iplot(kind='bar', title = 'Detect vs Tag Diff (Tag = 0)')
combine_df[combine_df['People Count (Tag)']!=0]['Diff'].value_counts().iplot(kind='bar', title = 'Detect vs Tag Diff (Tag != 0)')

index = -1
image = cv2.imread(combine_df.iloc[index]['Image'])
print(combine_df.iloc[index])
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
