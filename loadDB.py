from firebase import firebase
import pandas as pd

url = "https://test-4fef8.firebaseio.com/"
#url = "https://app01-7a86c.firebaseio.com/"
db = firebase.FirebaseApplication(url, None)
Signals = db.get("/Signals", None)
SSIDs = db.get("/Signals/20181031134650", 'SSIDs')
BSSIDs = db.get("/Signals/20181031134650", 'BSSIDs')
Strengths = db.get("/Signals/20181031134650", 'Strengths')
def printInfo(db):
    print("===============Info===============")
    for key,value in db.items():
        print("Id: {}\nDevice: {}\nLocation: {}\nLocationX={}\tLocationY={}\nStamp: {}"
            .format(key,value["Device"],value["Location"],value["LocationX"],value["LocationY"],
                value["Time"]))
def printSSID(S, num):
    d = pd.Series(["SSID_First", "SSID_Second", "SSID_Third", "SSID_Forth", "SSID_Fifth"])
    print("{}: {}".format(d[num],S[d[num]]))
def printBSSID(B, num):
    d = pd.Series(["BSSID_First", "BSSID_Second", "BSSID_Third", "BSSID_Forth", "BSSID_Fifth"])
    print("{}: {}".format(d[num],B[d[num]]))
def printStrength(St, num):
    d = pd.Series(["Strength_First", "Strength_Second", "Strength_Third", "Strength_Forth", "Strength_Fifth"])
    print("{}: {}".format(d[num],St[d[num]]))

printInfo(Signals)
print("==================================")
for i in range(5):
    printSSID(SSIDs, i)
    printBSSID(BSSIDs, i)
    printStrength(Strengths, i)
    print("----------------------------------")
print("==================================")
