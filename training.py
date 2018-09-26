from __future__ import print_function
import pandas as pd
import numpy as np
import math

ap_SSID = pd.Series(['AP1', 'AP2', 'AP3', 'AP4', 'AP5'])
ap_BSSID = pd.Series(['00:1b:2f:a8:e5:21', '00:1b:2f:a8:e5:22', '00:1b:2f:a8:e5:23', '00:1b:2f:a8:e5:24', '00:1b:2f:a8:e5:25'])
ap_isSecure = pd.Series([True, False, True, False, False])
ap_SignalStrength = pd.Series([-30, -30, -30, -30, -30])
loc = pd.Series(['Loc1', 'Loc2', 'Loc3', 'Loc4'])
device_wifi_first = pd.Series(['AP2', 'AP3', 'AP1', 'AP5'])
device_wifi_second = pd.Series(['AP3', 'AP5', 'AP4', 'AP4'])
device_wifi_third = pd.Series(['AP1', 'AP4', 'AP5', 'AP3'])
device_wifi_dist_first = pd.Series([0.9014, 0.5000, 0.5590, 0.5590])
device_wifi_dist_second = pd.Series([1.1180, 0.9014, 0.7071, 0.7071])
device_wifi_dist_third = pd.Series([1.3463, 1.5811, 1.5207, 1.5000])
#device_wifi_signal_first = pd.Series([0.8197, 0.9, 0.8882, 0.8882])
#device_wifi_signal_second = pd.Series([0.7764, 0.8197, 0.8586, 0.8586])
#device_wifi_signal_third = pd.Series([0.7307, 0.6838, 0.6959, 0.7])
def distToLost(freq, lossFactor, dist, floor):
    '''
        freq: Frequency in MHz
        lossFactor: -全開放環境:2.0~2.5 -半開放環境:2.5~3.0 -較封閉環境:3.0~3.5 -隧道環境:1.6~1.8
        dist: Distance in meters
        floor: Floors between AP and device
    '''
    loss = 20*np.log(freq)+10*lossFactor*np.log(dist)+6+3*(floor-1)
    return loss


device_loc = pd.Series(['Loc1', 'Loc2', 'Loc3', 'Loc4'])
location_x = pd.Series([0, 1, 0, 1])
location_y = pd.Series([0, 0, 1, 1])

ap = pd.DataFrame({'SSID':ap_SSID, 'BSSID':ap_BSSID, 'Secure':ap_isSecure})
# SSID        BSSID       Secure
#  AP1  00:1b:2f:a8:e5:21  True
#  AP2  00:1b:2f:a8:e5:22  False
#  AP3  00:1b:2f:a8:e5:23  True
#  AP4  00:1b:2f:a8:e5:24  False
#  AP5  00:1b:2f:a8:e5:25  True
location = pd.DataFrame({'Location':loc, 'Location_X':location_x, 'Location_Y':location_y})
# Location Location_X Location_Y
#   Loc1       0          0
#   Loc2       1          0
#   Loc3       0          1
#   Loc4       1          1
device_wifi_signal_first_loss = distToLost(2400, 3.0, device_wifi_dist_first, 1)
device_wifi_signal_second_loss = distToLost(2400, 3.0, device_wifi_dist_second, 1)
device_wifi_signal_third_loss = distToLost(2400, 3.0, device_wifi_dist_third, 1)

print("device_wifi_signal_first_loss:")
print(device_wifi_signal_first_loss)
print("device_wifi_signal_second_loss:")
print(device_wifi_signal_second_loss)
print("device_wifi_signal_third_loss:")
print(device_wifi_signal_third_loss)

device_wifi_signal_first_loss = 1/pow(10, np.log10((device_wifi_signal_first_loss)/10))
device_wifi_signal_second_loss = 1/pow(10, np.log10((device_wifi_signal_second_loss)/10))
device_wifi_signal_third_loss = 1/pow(10, np.log10((device_wifi_signal_third_loss)/10))

device_wifi_signal_first = 500*device_wifi_signal_first_loss
device_wifi_signal_second = 500*device_wifi_signal_second_loss
device_wifi_signal_third = 500*device_wifi_signal_third_loss


device = pd.DataFrame({'WIFI_SSID_First':device_wifi_first,'WIFI_Signal_First':device_wifi_signal_first,
                        'WIFI_SSID_Second':device_wifi_second, 'WIFI_Signal_Second':device_wifi_signal_second,
                        'WIFI_SSID_Third':device_wifi_third, 'WIFI_Signal_Third':device_wifi_signal_third,
                        'Location':device_loc})
# Location WIFI_SSID_First WIFI_SSID_Second WIFI_Signal_First WIFI_Signal_Second
#   Loc1         AP2             AP3             0.8197             0.7764
#   Loc2         AP3             AP5             0.9000             0.8197
#   Loc3         AP1             AP4             0.8882             0.8586
#   Loc4         AP5             AP4             0.8882             0.8586



device_wifi_bssid_first = []
device_wifi_isSecure_first = []
for item in device['WIFI_SSID_First']:
    wifi_mask = ap['SSID'].isin([item])
    device_wifi_bssid_first.append(ap[wifi_mask].iloc[0]['BSSID'])
    device_wifi_isSecure_first.append(ap[wifi_mask].iloc[0]['Secure'])
device_loc_x = []
device_loc_y = []
for item in device['Location']:
    loc_mask = location['Location'].isin([item])
    device_loc_x.append(location[loc_mask].iloc[0]['Location_X'])
    device_loc_y.append(location[loc_mask].iloc[0]['Location_Y'])

device.insert(loc=7, column='Location_Y', value=device_loc_y)
device.insert(loc=7, column='Location_X', value=device_loc_x)
device.insert(loc=0, column='WIFI_Secure_First', value=device_wifi_isSecure_first)
#device.insert(loc=0, column='WIFI_BSSID_First', value=device_wifi_bssid_first)

# Location  Location_X Location_Y WIFI_SSID_First WIFI_Signal_First     WIFI_BSSID     WIFI_Secure
#   Loc1        0          0            AP2            0.8197        00:1b:2f:a8:e5:22     False
#   Loc2        1          0            AP3            0.9000        00:1b:2f:a8:e5:23     True
#   Loc3        0          1            AP1            0.8882        00:1b:2f:a8:e5:21     True
#   Loc4        1          1            AP5            0.8882        00:1b:2f:a8:e5:25     False

device_wifi_one_hot_first = pd.get_dummies(device['WIFI_SSID_First'])
for item in ap_SSID:
    if item not in device_wifi_one_hot_first:
        device_wifi_one_hot_first[item] = [0, 0, 0, 0]
device_wifi_one_hot_second = pd.get_dummies(device['WIFI_SSID_Second'])
for item in ap_SSID:
    if item not in device_wifi_one_hot_second:
        device_wifi_one_hot_second[item] = [0, 0, 0, 0]
device_wifi_one_hot_third = pd.get_dummies(device['WIFI_SSID_Third'])
for item in ap_SSID:
    if item not in device_wifi_one_hot_third:
        device_wifi_one_hot_third[item] = [0, 0, 0, 0]
device_wifi_one_hot = device_wifi_one_hot_first + device_wifi_one_hot_second + device_wifi_one_hot_third
device = device.join(device_wifi_one_hot)

device_location_one_hot = pd.get_dummies(device['Location'])
device = device.join(device_location_one_hot)

newdevice = device
for i in range(499):
    newdevice = pd.concat([newdevice, device],axis=0, ignore_index=True)

noise_df = pd.DataFrame(np.random.random((2000,3)), columns=['Bias_First', 'Bias_Second', 'Bias_Third'])
noise_df*=10
noise_df_2 = pd.DataFrame(np.random.choice([-1, 1], size=(2000, 3), p=[0.5, 0.5]), columns=['Rate_First', 'Rate_Second', 'Rate_Third'])
noise_df = pd.concat([noise_df, noise_df_2], axis=1)
noise_df['WIFI_Signal_First']=noise_df['Bias_First'].multiply(noise_df['Rate_First'], axis=0)
noise_df['WIFI_Signal_Second']=noise_df['Bias_Second'].multiply(noise_df['Rate_Second'], axis=0)
noise_df['WIFI_Signal_Third']=noise_df['Bias_Third'].multiply(noise_df['Rate_Third'], axis=0)

for item in newdevice:
    if item in noise_df:
        newdevice[item]+=noise_df[item]


for item in newdevice['WIFI_Signal_First']:
    if item < 0:
        newdevice['WIFI_Signal_First'].item = 0.0



print("ap:")
print(ap)
print("\nlocation:")
print(location)
print("\ndevice:")
print(device)
print("\nnoise:")
print(noise_df)
print("\nnewdevice:")
print(newdevice)


#Training
