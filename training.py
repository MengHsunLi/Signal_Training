import pandas as pd

ap_SSID = pd.Series(['AP1', 'AP2', 'AP3', 'AP4', 'AP5'])
ap_BSSID = pd.Series(['00:1b:2f:a8:e5:21', '00:1b:2f:a8:e5:22', '00:1b:2f:a8:e5:23', '00:1b:2f:a8:e5:24', '00:1b:2f:a8:e5:25'])
ap_isSecure = pd.Series([True, False, True, False, False])
loc = pd.Series(['Loc1', 'Loc2', 'Loc3', 'Loc4'])
device_wifi_first = pd.Series(['AP2', 'AP3', 'AP1', 'AP5'])
device_wifi_second = pd.Series(['AP3', 'AP5', 'AP4', 'AP4'])
device_wifi_third = pd.Series(['AP1', 'AP4', 'AP5', 'AP3'])
device_wifi_signal_first = pd.Series(['0.8197', '0.9', '0.8882', '0.8882'])
device_wifi_signal_second = pd.Series(['0.7764', '0.8197', '0.8586', '0.8586'])
device_wifi_signal_third = pd.Series(['0.7307', '0.6838', '0.6959', '0.7'])
device_loc = pd.Series(['Loc1', 'Loc2', 'Loc3', 'Loc4'])
location_x = pd.Series(['0', '1', '0', '1'])
location_y = pd.Series(['0', '0', '1', '1'])

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
device.insert(loc=0, column='WIFI_BSSID_First', value=device_wifi_bssid_first)

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

print("ap:")
print(ap)
print("\nlocation:")
print(location)
print("\ndevice:")
print(device)
