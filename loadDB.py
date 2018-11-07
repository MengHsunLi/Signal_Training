from firebase import firebase
import pandas as pd

url = "https://test-4fef8.firebaseio.com/"
db = firebase.FirebaseApplication(url, None)
Signals = db.get("/Signals", None)
def printInfo(infoDB):
    Keys = pd.Series()
    i = 0
    for key,value in infoDB.items():
        Keys.at[i] = key
        i += 1
    for j in range(i):
        print("===============Info===============")
        dbStr='%s%s'%('/Signals/', Keys[j])
        infoDBDevice = db.get('%s%s'%(dbStr,'/Device'), None)
        infoDBLoc = db.get('%s%s'%(dbStr,'/Location'), None)
        infoDBLocX = db.get('%s%s'%(dbStr,'/LocationX'), None)
        infoDBLocY = db.get('%s%s'%(dbStr,'/LocationX'), None)
        infoDBTime = db.get('%s%s'%(dbStr,'/Time'), None)
        SSIDs = db.get(dbStr, 'SSIDs')
        BSSIDs = db.get(dbStr, 'BSSIDs')
        Strengths = db.get(dbStr, 'Strengths')
        print("Id:\t", Keys[j])
        print("Device:\t", infoDBDevice)
        print("Location:\t", infoDBLoc)
        print("LocationX:\t", infoDBLocX)
        print("LocationX:\t", infoDBLocY)
        print("Time:\t", infoDBTime)
        print("===============APs================")
        for k in range(5):
            printSSID(SSIDs, k)
            printBSSID(BSSIDs, k)
            printStrength(Strengths, k)
            print("----------------------------------")
        print("==================================")
def printInfoOrg(infoDB):
    print("===============Info===============")
    for key,value in infoDB.items():
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
