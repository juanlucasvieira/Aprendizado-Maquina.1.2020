import time
import csv
import os
import gc
import re
import copy
import datetime

path = "31daysOct/"
monthSpecified = 10
yearSpecified = 2001

initial = [1,0,0] #day, hour, minute
final = [31,23,45] #day, hour, minute
timeTuples = {}
#timeTuples = {"day":{"hour":{"minute":{}}}}
#print timeTuples
#timeTuples["30"] = {"bla":{}}
#print timeTuples

ap_list = []
#aps_dict = {"":[]}
aps_dict = {}

for day in range(1,32):
    for hour in range(0,24):
        for minute in range(0,60,1):
            timeTuples[(day,hour,minute)] = {"NONE":[]}
            #timeTuples[(day,hour,minute)] = {"AP":[]}

#print timeTuples



entries = {}
#entries.setdefault("", [])

#files_list = os.listdir("movement-v1.3/movement/2001-2004")
files_list = sorted(os.listdir(path))

processed = []

def getKey(item):
    #print item
    day = item[0]
    hour = item[1]
    minute = item[2]
    #print day,hour,minute,"->", str(day * 1440 + hour * 60 + minute)
    return day * 1440 + hour * 60 + minute

def getKey2(item):
    day,hour,minute = item
    return day * 1440 + hour * 60 + minute


def epoch2date(epoch):
    date = time.strftime('%d-%H-%M', time.localtime(int(epoch)))
    #date = time.strftime('%Y-%m-%d-%H-%M', time.localtime(int(epoch)))
    splittedTime = date.split("-")
    return splittedTime[0],splittedTime[1],splittedTime[2]

def checkValidMAC(text):
    if re.match("[0-9a-f]{12}$", text.lower()):
            return True
    else:
            return False

def process():
    progress = 0
    lastPrint = 0
    lastgc = 0
    count = 0
    global ap_list
    stop = False
    for fileName in files_list:
        # if (progress is not lastPrint):
        #     print(progress)
        #     lastPrint = progress
        #     if(progress - lastgc > 9):
        #         print("Collecting garbage")
        #         gc.collect()
        #         lastgc = progress
        # progress = int(count/len(files_list) * 100)
        #with open("movement-v1.3/movement/2001-2004/"+str(fileName), newline='') as csvfile:
        filePath = path+str(fileName)
        print "Reading file: ", filePath
        with open(filePath) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            seenLine = set()
            for row in spamreader:
            #    print row
                rowTxt = " ".join(row)
                if rowTxt in seenLine: # SKIP DUPLICATES
                    #print "LINE SEEEENN!!", " ".join(row)
                    continue
                action = 0
                epoch = row[0]

                month = row[1]
                day = row[2]
                #print row[3]
                if ":" in row[3]:
                    hour, minute, second = row[3].split(":")
                else:
                    continue #Skip not relevant Syslog messages

                ap_name = row[4]
                client = row[8].replace(",","")
                info1 = row[7]
                info2 = row[9]
                if info1 == "Deauthenticating" or info1 == "Deauthentication" or info1 == "Disassociating" or info2 == "roamed":
                    if info1 == "Deauthentication":
                        client = row[9].replace(",","")
                        #print epoch, ap_name, client
                    action = -1
                elif info2 == "Authenticated" or info2 == "Associated" or info2 == "Reassociated":
                    action = 1
                else:
                    print "DO NOT NO WHAT TO DO WITH", fileName, rowTxt
                #print epoch, ap_name, client, action

                if action is not 0: # CLIENT ADDED OR REMOVED EVENT
                    if ap_name not in ap_list:
                        #print "NEW AP:",ap_name, row
                        ap_list.append(ap_name)
                    if not checkValidMAC(client):
                        print "INVALID CLIENT FOUND!"
                    processed.append(str(epoch)+","+str(ap_name)+","+str(client)+","+str(action))

                    #day, hour, minute = epoch2date(epoch)
                    #print epoch, aps_dict
                    if ap_name in aps_dict:
                        if action > 0:
                            if client not in aps_dict[ap_name]:
                                aps_dict[ap_name].append(client)
                                #print "ADDING", row, "DATE:", day, hour, minute
                        elif action < 0:
                            if client in aps_dict[ap_name]:
                                aps_dict[ap_name].remove(client)
                                #print "REMOVING", row, "DATE:", day, hour, minute
                                #stop = True
                            else:
                                if ("Not Associated" not in rowTxt) and ("Must Authenticate Before Associating" not in rowTxt): #Ignore warnings on broken associations
                                    pass#print "CLIENT NOT IN THE LIST TO REMOVE!! >>>", client, rowTxt
                    else:
                        if action > 0:
                            newList = [client]
                            aps_dict[ap_name] = newList
                            #print "ADDING NEW", row, "DATE:", day, hour, minute
                        elif action < 0:
                            if ("Not Associated" not in rowTxt) and ("Must Authenticate Before Associating" not in rowTxt): #Ignore warnings on broken associations
                                pass#print "LIST NOT EXISTING - TRYING TO REMOVE CLIENT!! ", client, rowTxt
                    
                    timeTuples[(int(day),int(hour),int(minute))] = copy.deepcopy(aps_dict)
                    #timeTuples[(int(day),int(hour),int(minute))] = aps_dict.copy()
                    #print timeTuples
                    if stop:
                        break
                seenLine.add(rowTxt)
            count = count + 1
            #print timeTuples
            #print len(aps_dict.keys())
            #print aps_dict
            #return
        if stop:
            print timeTuples
            break
        #break # Stop at the first file

    #print timeTuples
    #print ap_list
    ap_list = sorted(ap_list)
    #print ap_list

    #print len(ap_list)
    rows = []
    names = ["day", "hour", "minute", "weekday"]
    last_known_state = []
    for key in sorted(timeTuples.keys(),key=getKey2):
        day, hour, minute = key 
        #print "KEY:",key
        weekday = datetime.datetime(yearSpecified, monthSpecified, day).weekday() # Monday - 0, Tuesday - 1, ...
        row = [day, hour, minute, weekday]
        
        #aux_dict = {}
        if "NONE" in timeTuples[key]:
            #continue #Skips rows between events
            if len(last_known_state) > 0:
                for value in last_known_state:
                    row.append(value)
            else:
                for ap in ap_list:
                    row.append(-1)
        else:
            for ap in ap_list:
                if ap not in names:
                    names.append(ap)
                if ap in timeTuples[key]:
                    row.append(len(timeTuples[key][ap]))
                    #aux_dict[ap] = len(timeTuples[key][ap])
                elif "NONE" in timeTuples[key]:
                    row.append(-2) 
                else:
                    row.append(0)
                    #aux_dict[ap] = 0
            last_known_state = row[4:] #Get only APs info, discarding time info
        #timeTuples[key] = aux_dict.copy()
        #print len(row)
        if(len(row) > 301):
            print "Illegal size", len(row)
        rows.append(row)

    #print len(ap_list)

        #print timeTuples[key] 
        #return

    

    # for entry in entries.items():
    #     #print(entry)
    #     key, tupl = entry
    #     epoch, clients = tupl
    #     date, ap = key.split('@')
    #     aps.append([epoch, date, ap, clients])
    #     print([epoch, date, ap, clients])

    # rows = sorted(rows,key=getKey)
        
    with open("dataset-"+str(yearSpecified)+"-"+str(monthSpecified)+".txt", 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_NONE, delimiter=',')
        wr.writerow(names)
        for line in rows:
            wr.writerow(line)
        # for i in range (1,1000):
        #     wr.writerow(rows[i])
            

        #entries.clear()

process()

# Year, Month, Day, Hour, Minute, AP, Number os Stations
#time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(1347517370))
