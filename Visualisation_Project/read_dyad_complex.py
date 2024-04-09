#!/usr/bin/env python3
import numpy as np
import glob
import pandas as pd
from datetime import datetime

data_directory = "./vis_project_data/"

# Default weights; used if not set
default_weights = np.array([-1, 0, 1, 2, 2, 3, 5])

# Types of sessions/files:
session_types = ["cog", "so"]
whatdoors = ["indoor", "outdoor"]
whichs = ["base", "inter"]

# Combine to single iteratable list
combined_scenarios = [
    (ses_type, whatdoor, which)
    for ses_type in session_types
    for whatdoor in whatdoors
    for which in whichs
]
################################################################################

def combined_score(filename):
    """Calculates the 'score' for a single session/file.
    Assumes total session duration is 360s, otherwise returns 'nan'.
    This could be modified simply to also return other details of the session."""
    with open(filename, "r") as file:
        print(filename)
        total_duration = 0.0
        t_end_prev = 0.0

        categories = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        date_track = False
        for count, line in enumerate(file.readlines()):
            print(count, line)
            data = line.split(",", 4)
            if count == 0:
                continue
            if date_track:
                if count == date_count:
                    date = datetime.strptime(line.strip(), "%m-%d-%Y")
                if count == time_count:
                    time = datetime.strptime(line.strip(), "%H:%M:%S").time()
                    combined_datetime = datetime.combine(date.date(), time)
                    categories[8] = combined_datetime
                    break
                else:
                    continue

            print("line", line)
            print("data", data)
            if line[0] == "*":
                date_track = True
                date_count = count + 9
                time_count = date_count + 1
                continue
            t_category = int(data[0])
            t_beg = int(data[1])
            t_end = int(data[2])

            if t_beg != t_end_prev:
                print("Error, missing time stamp?")
            t_end_prev = t_end

            #assert t_end >= t_beg
            if count == 1:
                assert t_beg == 0

            if t_category != 0:
                duration = float(t_end - t_beg)
                total_duration += duration
                #print()
                #for x in categories:
                    #print(x.__class__.__name__)
                #print(t_category)
                #print(categories[t_category])
                #print(duration)
                categories[t_category] += duration

        #4print(t_category, t_beg, t_end)
        #print(categories)
        return categories


################################################################################
def pair_sessions(ca, peer
):
    """Calculates the scores for given ca/peer pair.
    It simply prints the result to screen - to be useful, you will want
    to actually store this data (e.g., return a struct or array etc.).
    """
    outlist = []

    trained = "trained" if "u" <= peer[0] <= "z" else "untrained"
    print(ca, peer, f"({trained})")
    for ses_type, whatdoor, which in combined_scenarios:

        # glob creates the list of filenames that match the given pattern
        # '*' is a wildcard
        files = glob.glob(
            data_directory + f"{ses_type}-*-{which}-*-{ca}-{peer}-{whatdoor}.dtx"
        )

        if len(files) == 0:
            continue

        for file in files:
            newdict = dict()
            tmp_score = combined_score(file)
            for i in range(9):
                newdict[i] = tmp_score[i]

            newdict["cog"] = ses_type
            newdict["ca"] = ca
            newdict["peer"] = peer
            newdict["trained"] = trained
            newdict["whatdoor"] = whatdoor
            newdict["which"] = which
            outlist.append(newdict)

    return outlist



################################################################################
def unique_pairs():
    """Returns list of unique ca/peer pairs"""
    all_files = glob.glob(data_directory + "/*.dtx")
    list = []
    for file in all_files:
        t = file.split("-")
        list.append([t[4], t[5]])

    return np.unique(list, axis=0)


################################################################################

if __name__ == "__main__":

    # Example usage:
    ca_peer_list = unique_pairs()

    data = {
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        "cog",
        "ca",
        "peer",
        "trained",
        "whatdoor",
        "what",
        "date"
    }

    df = pd.DataFrame(data)

    dict_list = []
    #print(ca_peer_list)

    print()

    # From the thesis:
    soc_weights = np.array([-1, 0, 1, 2, 2, 3, 5])

    # Match table 5.1 of thesis
    # cog_weights = np.array([-1, 0, 1, 2, 2, 3, 5])

    # Match matlab example:
    cog_weights = np.array([-1, 0, 1, 2, 2, 3, 5])


    #print(pair_sessions("Albert", "Lydia"))
    # Or, for all pairs:
    for ca, peer in ca_peer_list:
        test = pair_sessions(ca, peer)
        dict_list = dict_list + test

    print(dict_list)
    print(len(dict_list))
    df = pd.DataFrame(dict_list)
    df.to_csv('out.csv')