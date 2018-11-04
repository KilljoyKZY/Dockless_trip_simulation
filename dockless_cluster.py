
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import os
import sys
import json
from geopy.distance import great_circle
import matplotlib.pyplot as plt
import seaborn as sns

# from GPSTransform import GPSUtil


# In[2]:

# set working directory
os.chdir("C:/Users/kouz/Desktop/research_code/xiamen_dockless")


# In[3]:

df=pd.read_csv('xiamen_university_track_1004.csv', sep='\t')
df.columns = ['TripID','BikeID','BikeStatus','StartTime','StartLon','StartLat','EndDate','EndLon','EndLat','TrajRec','TripTime']


# In[4]:

for i in range(len(df)):
    row = df.loc[i,]
    if (np.isnan([row['StartLat'], row['StartLon'], row['EndLat'], row['EndLon']]).any()) & (row['TripTime'] > 0):
        df['EndLon'][i] = float(row['TrajRec'].split('#')[-1].split(';')[0].split(',')[0])
        df['EndLat'][i] = float(row['TrajRec'].split('#')[-1].split(';')[0].split(',')[1])


# In[5]:

def find_dist(row):
    if np.isnan([row['StartLat'], row['StartLon'], row['EndLat'], row['EndLon']]).any():
        output = np.nan
    else:
        start_loc = row['StartLat'], row['StartLon']
        end_loc = row['EndLat'], row['EndLon']
        output = great_circle(start_loc, end_loc).meters
    return output


# In[12]:

valid_df = df.copy()
valid_df = valid_df[(valid_df['StartLat'] > 24) & (valid_df['StartLat']< 24.75)]
valid_df = valid_df[(valid_df['StartLon'] > 117.9) & (valid_df['StartLon']< 120)]
valid_df = valid_df[(valid_df['EndLat'] > 24) & (valid_df['EndLat']< 24.75)]
valid_df = valid_df[(valid_df['EndLon'] > 117.9) & (valid_df['EndLon']< 120)]


# In[89]:

# sample the data
sample_df = valid_df.copy()


# In[90]:

f1 = sample_df['StartLat'].values
f2 = sample_df['StartLon'].values
X = np.array(list(zip(f1, f2)))
plt.scatter(f1, f2, c='black', s=7)
# plt.scatter(centroids[:,0], centroids[:,1], marker='*', s=200, c='g')
plt.show()


# In[91]:

from sklearn.cluster import KMeans

# Number of clusters
kmeans = KMeans(n_clusters=200)
# Fitting the input data
kmeans = kmeans.fit(X)
# Getting the cluster labels
labels = kmeans.predict(X)
# Centroid values
centroids = kmeans.cluster_centers_


# In[92]:

plt.scatter(centroids[:,0], centroids[:,1], marker='*', s=100, c='g')
plt.show()


# In[94]:

# columns
column_name = ['lat', 'lon']

# pass in array and columns
cluster_df = pd.DataFrame(X, columns=column_name)


# In[95]:

cluster_df['label'] = labels


# In[85]:

cluster_df['dist_to_center'] = 0.0


# In[87]:

for i in range(len(cluster_df)):
    trip_loc = cluster_df['lat'].loc[i], cluster_df['lon'].loc[i]
    label_num = cluster_df['label'].loc[i]
    cent_lat = centroids[label_num][0]
    cent_lon = centroids[label_num][1]
    cent_loc = cent_lat, cent_lon
    cluster_df['dist_to_center'].loc[i] = great_circle(trip_loc, cent_loc).meters
    


# In[88]:

np.average(cluster_df['dist_to_center'])


# ## simulate the clusters

# In[93]:

sample_df['start_station'] = labels


# In[98]:

e1 = sample_df['EndLat'].values
e2 = sample_df['EndLon'].values
X2 = np.array(list(zip(e1, e2)))
label_ends = kmeans.predict(X2)


# In[99]:

sample_df['end_station'] = label_ends


# In[101]:

sample_df


# In[107]:

station_ids = np.arange(200)
sta_status = pd.DataFrame(station_ids, columns=['station_id'])
sta_status['now_status'] = 0
sta_status['max_status'] = 0
sta_status['min_status'] = 0


# In[109]:

sta_status.head()


# In[110]:

sort_df = sample_df.sort_values(by='StartTime', ascending=True)


# In[111]:

sort_df.head()


# In[123]:

for i in sort_df.index:
    from_sta = sort_df['start_station'].loc[i]
    to_sta = sort_df['end_station'].loc[i]
    k_from = np.where(sta_status['station_id'] == from_sta)[0][0]
    sta_status['now_status'].loc[k_from] = sta_status['now_status'].loc[k_from] - 1
    sta_status['min_status'].loc[k_from] = min(sta_status['now_status'].loc[k_from], sta_status['min_status'].loc[k_from])
    
    k_to = np.where(sta_status['station_id'] == to_sta)[0][0]
    sta_status['now_status'].loc[k_to] = sta_status['now_status'].loc[k_to] + 1
    sta_status['max_status'].loc[k_to] = max(sta_status['now_status'].loc[k_to], sta_status['max_status'].loc[k_to])


# In[124]:

sta_status


# In[127]:

sta_status.to_csv('Xiamen_clustered_station_statues.csv')


# In[128]:

sort_df.to_csv('Xiamen_clustered_to&from_trips.csv')


# In[131]:

center_df = pd.DataFrame(centroids, columns=['station_lat', 'station_lon'])


# In[133]:

center_df.to_csv('clustered_station_200_locations.csv')


# In[136]:

sta_status['bike_needed'] = - sta_status['min_status']


# In[137]:

sta_status['docks_needed'] = sta_status['bike_needed'] + sta_status['max_status']


# In[140]:

sta_status


# In[139]:

sta_status.bike_needed.sum()


# In[141]:

sta_status.docks_needed.sum()


# In[144]:

center_df


# In[145]:

merge_status = pd.merge(sta_status, center_df, left_index=True, right_index=True)


# In[146]:

merge_status


# In[171]:

sta_temp = merge_status.copy()


# In[172]:

pos_sta = sta_temp[sta_temp['now_status']>0]
neg_sta = sta_temp[sta_temp['now_status']<0]
neg_sta['dist_to_pos'] = 0.0


# In[173]:

sta_temp = merge_status.copy()
pos_sta = sta_temp[sta_temp['now_status']>0]
neg_sta = sta_temp[sta_temp['now_status']<0]
neg_sta['dist_to_pos'] = 0.0
total_reb_dist = 0.0
num_of_reb = 0
for i in pos_sta.index:
    pos_loc = pos_sta['station_lat'].loc[i], pos_sta['station_lon'].loc[i]
    for j in neg_sta.index:
        neg_loc = neg_sta['station_lat'].loc[j], neg_sta['station_lon'].loc[j]
        neg_sta['dist_to_pos'].loc[j] = great_circle(pos_loc, neg_loc).meters
    while pos_sta['now_status'].loc[i] >0:
        k_near = neg_sta[neg_sta['dist_to_pos'] == min(neg_sta['dist_to_pos'])].index[0]
        num_to_move = min(pos_sta['now_status'].loc[i], abs(neg_sta['now_status'].loc[k_near]))
        pos_sta['now_status'].loc[i] = pos_sta['now_status'].loc[i] - num_to_move
        neg_sta['now_status'].loc[k_near] = neg_sta['now_status'].loc[k_near] + num_to_move
        total_reb_dist = total_reb_dist + neg_sta['dist_to_pos'].loc[k_near] * num_to_move
        num_of_reb = num_of_reb + 1
        neg_sta = neg_sta[neg_sta['now_status'] != 0]
        print('length of neg_sta')
        print(len(neg_sta))
    print(i)


# In[158]:

abs(neg_sta['now_status'].loc[51])


# In[ ]:



