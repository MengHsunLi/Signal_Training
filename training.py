import pandas as pd

ap_SSID = pd.Series(['AP1', 'AP2', 'AP3', 'AP4', 'AP5'])
ap_BSSID = pd.Series(['00:1b:2f:a8:e5:21', '00:1b:2f:a8:e5:22', '00:1b:2f:a8:e5:23', '00:1b:2f:a8:e5:24', '00:1b:2f:a8:e5:25'])
ap_isSecure = pd.Series([True, False, True, False, False])
loc = pd.Series(['Loc1', 'Loc2', 'Loc3', 'Loc4'])
device_wifi = pd.Series(['AP2', 'AP3', 'AP1', 'AP5'])
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
device = pd.DataFrame({'WIFI_SSID':device_wifi,'Location':device_loc})
# WIFI  Location
#  AP2    Loc1
#  AP3    Loc2
#  AP1    Loc3
#  AP5    Loc4

device_wifi_bssid = []
device_wifi_isSecure = []
for item in device['WIFI_SSID']:
    wifi_mask = ap['SSID'].isin([item])
    device_wifi_bssid.append(ap[wifi_mask].iloc[0]['BSSID'])
    device_wifi_isSecure.append(ap[wifi_mask].iloc[0]['Secure'])
device_loc_x = []
device_loc_y = []
for item in device['Location']:
    loc_mask = location['Location'].isin([item])
    device_loc_x.append(location[loc_mask].iloc[0]['Location_X'])
    device_loc_y.append(location[loc_mask].iloc[0]['Location_Y'])

device.insert(loc=2, column='WIFI_Secure', value=device_wifi_isSecure)
device.insert(loc=2, column='WIFI_BSSID', value=device_wifi_bssid)
device.insert(loc=1, column='Location_Y', value=device_loc_y)
device.insert(loc=1, column='Location_X', value=device_loc_x)
# WIFI  Location  Location_X Location_Y     WIFI_BSSID     WIFI_Secure
#  AP2    Loc1        0          0      00:1b:2f:a8:e5:22     False
#  AP3    Loc2        1          0      00:1b:2f:a8:e5:23     True
#  AP1    Loc3        0          1      00:1b:2f:a8:e5:21     True
#  AP5    Loc4        1          1      00:1b:2f:a8:e5:25     False

device_wifi_one_hot = pd.get_dummies(device['WIFI_SSID'])
device = device.join(device_wifi_one_hot)

print("ap:")
print(ap)
print("\nlocation:")
print(location)
print("\ndevice:")
print(device)
