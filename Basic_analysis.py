
# coding: utf-8

# In[113]:

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
all_files = os.listdir("C:/Users/kouz/Desktop/research_code/xiamen_dockless")
df_list = []


# In[8]:

for i in range(len(all_files)):
    df=pd.read_csv(all_files[i],sep='\t')
    df.columns = ['BikeID','TripID','BikeStatus','StartTime','StartLon','StartLat','EndDate','EndLon','EndLat','TrajRec','TripTime']
    df_list.append(df)


# In[36]:

for i in range(len(df_list)):
    date_list = []
    for j in range(df_list[i].shape[0]):
        date_list.append(df_list[i]['StartTime'][j][:10])
    uniq_date = list(set(date_list))
    print(uniq_date)


# In[21]:

len_list = []
for i in range(len(df_list)):
    len_list.append(df_list[i].shape[0])
print(len_list)


# In[21]:

df=pd.read_csv('xiamen_university_track_1004.csv', sep='\t')
df.columns = ['TripID','BikeID','BikeStatus','StartTime','StartLon','StartLat','EndDate','EndLon','EndLat','TrajRec','TripTime']


# In[29]:

df.head()


# In[125]:

# remove data in Shanghai
df = df[df['StartLat']<30]


# In[126]:

df.nunique()


# In[127]:

df['BikeStatus'].value_counts()


# In[101]:

for i in range(len(df)):
    row = df.loc[i,]
    if (np.isnan([row['StartLat'], row['StartLon'], row['EndLat'], row['EndLon']]).any()) & (row['TripTime'] > 0):
        df['EndLon'][i] = float(row['TrajRec'].split('#')[-1].split(';')[0].split(',')[0])
        df['EndLat'][i] = float(row['TrajRec'].split('#')[-1].split(';')[0].split(',')[1])


# In[102]:

def find_dist(row):
    if np.isnan([row['StartLat'], row['StartLon'], row['EndLat'], row['EndLon']]).any():
        output = np.nan
    else:
        start_loc = row['StartLat'], row['StartLon']
        end_loc = row['EndLat'], row['EndLon']
        output = great_circle(start_loc, end_loc).meters
    return output


# In[98]:

np.isnan([df.StartLat[0], df.StartLon[0]]).any()


# In[103]:

df['gc_distance'] = df.apply(find_dist, axis=1)


# In[128]:

df['gc_distance'][~(df['gc_distance'].isnull())].sum()


# In[130]:

len(df[~(df['gc_distance'].isnull())])


# ## Indentify rebalancing

# In[248]:

valid_df = df[df['gc_distance']>0]
valid_df = valid_df[(valid_df['StartLat'] > 24) & (valid_df['StartLat']< 25)]
valid_df = valid_df[(valid_df['StartLon'] > 117) & (valid_df['StartLon']< 120)]
valid_df = valid_df[(valid_df['EndLat'] > 24) & (valid_df['EndLat']< 25)]
valid_df = valid_df[(valid_df['EndLon'] > 117) & (valid_df['EndLon']< 120)]
bike_ids, trip_count = np.unique(valid_df['BikeID'], return_counts=True)


# In[116]:

# matplotlib histogram
plt.hist(trip_count, color = 'blue', edgecolor = 'black',
         bins = 100)

# Add labels
plt.xlabel('Trips made by each bike on 20171004')
plt.ylabel('Number of bikes')
plt.show()


# In[250]:

print(max(trip_count))
print(np.average(trip_count))


# In[251]:

dict_trip_ct = {
    'BikeID': bike_ids,
    'TripCount': trip_count
}
df_trip_ct = pd.DataFrame(dict_trip_ct)


# In[252]:

df_trip_ct_m1 = df_trip_ct[df_trip_ct['TripCount']>1]


# In[253]:

df_trip_ct_m1.index


# In[279]:

# create dataframe to store rebalance record
reb_df = pd.DataFrame(columns=['BikeID', 'last_lat', 'last_lon', 'moved_lat', 'moved_lon', 'last_time', 'moved_time', 'reb_distance'])
reb_df


# In[280]:

for k in df_trip_ct_m1.index:
    trip_record = valid_df[valid_df['BikeID']==df_trip_ct_m1['BikeID'].loc[k]]
    trip_rec_sort = trip_record.sort_values(by='StartTime', ascending=True)
    idx_arr = trip_rec_sort.index.values
    for i in range(len(idx_arr)-1):
        last_end = trip_rec_sort['EndLat'][idx_arr[i]], trip_rec_sort['EndLon'][idx_arr[i]]
        now_start = trip_rec_sort['StartLat'][idx_arr[i+1]], trip_rec_sort['StartLon'][idx_arr[i+1]]
        diff_dist = great_circle(last_end, now_start).meters
        if diff_dist > 500:
            reb_df.loc[len(reb_df)] = [trip_rec_sort['BikeID'][idx_arr[i]], trip_rec_sort['EndLat'][idx_arr[i]], trip_rec_sort['EndLon'][idx_arr[i]],
                                      trip_rec_sort['StartLat'][idx_arr[i+1]], trip_rec_sort['StartLon'][idx_arr[i+1]],
                                       trip_rec_sort['EndDate'][idx_arr[i]], trip_rec_sort['StartTime'][idx_arr[i+1]], diff_dist]


# In[281]:

reb_df


# In[282]:

reb_df[reb_df['reb_distance']==reb_df.reb_distance.max()]


# In[283]:

np.average(reb_df['reb_distance'])


# In[284]:

reb_less10 = reb_df[reb_df['reb_distance']<10000]


# In[289]:

np.average(reb_less10['reb_distance'])


# In[287]:

len(reb_less10)


# In[288]:

plt.boxplot(reb_less10['reb_distance']/1000,
            notch=False, # box instead of notch shape
            sym='rs',    # red squares for outliers
            vert=True)   # vertical box aligmnent

plt.xlabel('Rebalancing distance (km)')
plt.ylabel('Density')
plt.show()


# In[290]:

reb_bike_ids, reb_count = np.unique(reb_less10['BikeID'], return_counts=True)
len(reb_bike_ids)


# In[291]:

max(reb_count)


# ## Calculate the rebalancing demand given the end of day bike locations and the beginning of day bike locations

# In[294]:

bike_loc = pd.DataFrame(columns=['BikeID', 'first_lat', 'first_lon', 'last_lat', 'last_lon'])
bike_loc


# In[295]:

# find all start location and end location
for k in df_trip_ct.index:
    trip_record = valid_df[valid_df['BikeID']==df_trip_ct['BikeID'].loc[k]]
    trip_rec_sort = trip_record.sort_values(by='StartTime', ascending=True)
    idx_arr = trip_rec_sort.index.values
    bike_loc.loc[len(bike_loc)] = [trip_rec_sort['BikeID'][idx_arr[0]], trip_rec_sort['StartLat'][idx_arr[0]], trip_rec_sort['StartLon'][idx_arr[0]], 
                                   trip_rec_sort['EndLat'][idx_arr[-1]], trip_rec_sort['EndLon'][idx_arr[-1]]]
    


# In[351]:

bike_loc['nearest_last_lat'] = 0.0
bike_loc['nearest_last_lon'] = 0.0
bike_loc['dist_to_reb'] = 0.0


# In[359]:

bike_loc.head()


# In[300]:

end_loc = bike_loc[['BikeID', 'last_lat', 'last_lon']].copy()


# In[358]:

for i in range(len(bike_loc)):
    search_lat = bike_loc['first_lat'][i]
    search_location = bike_loc['first_lat'][i], bike_loc['first_lon'][i]
    search_low = search_lat - 0.0001
    search_up = search_lat + 0.0001
    loc_candidate = end_loc[(end_loc['last_lat']>search_low) & (end_loc['last_lat']<search_up)].copy()
    while len(loc_candidate) == 0:
        search_low = search_low - 0.0001
        search_up = search_up + 0.0001
        loc_candidate = end_loc[(end_loc['last_lat']>search_low) & (end_loc['last_lat']<search_up)].copy()
    loc_candidate['dist_to_start'] = 0.0
    for j in loc_candidate.index:
        cal_lat = loc_candidate['last_lat'].loc[j]
        cal_lon = loc_candidate['last_lon'].loc[j]
        cal_location = cal_lat, cal_lon
        loc_candidate['dist_to_start'].loc[j] = great_circle(search_location, cal_location).meters
    k = loc_candidate[loc_candidate['dist_to_start'] == loc_candidate['dist_to_start'].min()].index[0]
    bike_loc['nearest_last_lat'].loc[i] = loc_candidate['last_lat'].loc[k]
    bike_loc['nearest_last_lon'].loc[i] = loc_candidate['last_lon'].loc[k]
    bike_loc['dist_to_reb'].loc[i] = loc_candidate['dist_to_start'].loc[k]
    end_loc = end_loc[end_loc.index != k]
    print(len(end_loc))

        


# In[360]:

bike_loc['dist_to_reb'].sum()


# In[362]:

bike_loc.to_csv('Xiamen_dockless_rebalanced_to_original.csv')


# In[363]:

valid_df.to_csv('Xiamen_1004_valid_trips_with_distance.csv')


# In[ ]:




# ## Backup

# In[79]:

traj_array = df['TrajRec']
float(traj_array[1].split('#')[1].split(';')[0].split(',')[0])



# In[81]:

traj_array[1].split('#')[-1]


# In[ ]:




# In[5]:

trarecord=df['TrajRec']

tradic={}
for indexs in trarecord.index:
    trjstr=trarecord.loc[indexs]
    trjlist=trjstr.split('#')
    gtools = GPSUtil()
    for str in trjlist :
        str2=str.split(';')
        #print(str2)
        ladlon=str2[0]
        #place=str2[1]
        print(ladlon)


# In[ ]:




# In[32]:

------

import pandas as pd

import os
import sys
import json
sys.path.append(os.path.abspath("../../"))
#f=open('E:/学习相关/Python/数据样例/用户侧数据/账单.csv')
#df=pd.read_csv(f)
from GPSTransform import GPSUtil

#df=pd.read_csv('data\\xiamen_uni\\xiamen_university_track_0911.csv',sep='\t')

# for line in df:
#     print(line(3))
#csv_txt = "date","player1","player2","score1","score2"
# class Point:
#     lat = ''
#     lng = ''
#
#     def __init__(self, lat, lng):
#         self.lat = lat  # 纬度
#         self.lng = lng  # 经度
#
#     def show(self):
#         print
#         self.lat, " ", self.lng

df=pd.read_csv('c:\\Users\\Wei\\PycharmProjects\\xmmobile\\data\\xiamen_uni\\xiamen_university_track_0911.csv',sep='\t')
df.columns = ['BikeID','TripID','BikeStatus','StartTime','StartLon','StartLat','EndDate','EndLon','EndLat','TrajRec','TripTime']
trarecord=df['TrajRec']

tradic={}
for indexs in trarecord.index:
    trjstr=trarecord.loc[indexs]
    trjlist=trjstr.split('#')
    gtools = GPSUtil()
    for str in trjlist :
        str2=str.split(';')
        #print(str2)
        ladlon=str2[0]
        #place=str2[1]
        print(ladlon)
        #print(ladlon)
        # gps=ladlon.split(',')
        # tmplat = float(gps[0])
        # tmplng=float(gps[1])
        # bdgps=gtools.wgs2bd(tmplat,tmplng)
        # print(tmplat,tmplng,bdgps)
    #             if(bdgps is not None):
    #                 OldmanTabGps.objects.filter(latitude=gps[0]).filter(longtitude=gps[1]).update(bdlat=bdgps['lat'],bdlon=bdgps['lon'])





# for
#     gpsls=
#         gtools=GPSUtil()
#         cnt=0
#         for gps in gpsls:
#             tmplat=float(gps[0])
#             tmplng=float(gps[1])
#             bdgps=gtools.wgs2bd(tmplat,tmplng)
#             print tmplat,tmplng,bdgps
#             if(bdgps is not None):
#                 OldmanTabGps.objects.filter(latitude=gps[0]).filter(longtitude=gps[1]).update(bdlat=bdgps['lat'],bdlon=bdgps['lon'])





# df1=df.head(100)
# # print(df1)
#
# dfjson=df1.to_json(orient='records')
# print(dfjson)
# with open("../record.json","w") as f:
#   json.dump(dfjson,f)
#   print("加载入文件完成...")




#print(dfjson)

#print(df.head(10))
#print(df.describe())
#print(df[['TrajRec']])
# trarecord=df['TrajRec']
#print(trarecord[0])
#trjtestp=trarecord[0]

# df1=trarecord.head(1000)
#print(testtrip)
#teststr=testtrip[0]

# python2json = {}
# #构造list
# listData = [1,2,3]
# python2json["listData"] = listData
# python2json["strData"] = "test python obj 2 json"
# dfjson=df1.to_json(orient='index')
# print(dfjson)


# In[ ]:



