##########################################################
# Move the pickings to the peaks
# ---
# Author: Hongtao Wang
# Email: colin315wht@gmail.com
##########################################################

import numpy as np


"""
Correct the FB result to the peak of amplitude
"""

def mv2peak(curve, gth, SearchRange=2):
    curve_corr = []
    for row_i, t in curve:
        trace = gth[:, row_i]
        # mode correct or no-picking
        if CheckPeak(trace, t):
            t_new = t
        # need to correct
        else:
            SRange = [t-SearchRange, t+SearchRange]
            try:
                PeakList = FindPeak(trace, SRange)
            except ValueError:
                PeakList = []
            if len(PeakList) > 0:
                t_new = PeakList[np.argmin(np.abs(PeakList-t))]
            else:
                t_new = t
        curve_corr.append([row_i, t_new])

    return np.array(curve_corr, dtype=np.int32)


"""
Check t whether is peak or trough
"""
def CheckPeak(trace, t):
    if trace[t] < np.max(trace[(t-1):(t+2)]):
        return False
    else:
        return True


"""
Find the peaks in a range
"""
def FindPeak(trace, Range):
    PeakList = []
    # check range valid
    for ind in range(Range[0], Range[1]+1):
        try:
            if trace[ind] == np.max(trace[(ind-1):(ind+2)]):
                PeakList.append(ind)
        except IndexError:
            pass
    return np.array(PeakList)