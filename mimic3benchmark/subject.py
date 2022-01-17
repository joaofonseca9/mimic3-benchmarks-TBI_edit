from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import os
import pandas as pd

from mimic3benchmark.util import dataframe_from_csv


def read_stays(subject_path):
    stays = dataframe_from_csv(os.path.join(subject_path, 'stays.csv'), index_col=None)
    stays.INTIME = pd.to_datetime(stays.INTIME)
    stays.OUTTIME = pd.to_datetime(stays.OUTTIME)
    stays.DOB = pd.to_datetime(stays.DOB)
    stays.DOD = pd.to_datetime(stays.DOD)
    stays.DEATHTIME = pd.to_datetime(stays.DEATHTIME)
    stays.sort_values(by=['INTIME', 'OUTTIME'], inplace=True)
    return stays


def read_diagnoses(subject_path):
    return dataframe_from_csv(os.path.join(subject_path, 'diagnoses.csv'), index_col=None)

def get_TBI_ICD9():
    skull_fract=[80000, 80001, 80002, 80003, 80004, 80005, 80006, 80009, 80010, 80011,
             80012, 80013, 80014, 80015, 80016, 80019, 80020, 80021, 80022, 80023, 
             80024, 80025, 80026, 80029, 80030, 80031, 80032, 80033, 80034, 80035, 
             80036, 80039, 80040, 80041, 80042, 80043, 80044, 80045, 80046, 80049, 
             80050, 80051, 80052, 80053, 80054, 80055, 80059, 80060, 80061, 80062, 
             80063, 80064, 80065, 80066, 80069, 80070, 80071, 80072, 80073, 80074, 
             80075, 80076, 80079, 80080, 80081, 80082, 80083, 80084, 80085, 80086, 
             80089, 80090, 80091, 80092, 80093, 80094, 80095, 80096, 80099]

    concussion=np.array(skull_fract)+100
    concussion=list(concussion)

    cerebral_contusion=[8502,8503,8504,8505,8509]
    cerebral_contusion.extend(list(np.array(skull_fract)+5000))

    #traumatic subarachnoid hemorrahges, subdurals and extradurals
    hemorrahges=list(np.array(skull_fract)+5200)

    #other hemorrahges
    hemorrahges2=list(np.array(skull_fract)+5300)

    #intracranial injury of unspecified nature
    cran_injury=list(np.array(skull_fract)+5400)

    #subdurals + extra/epidurals
    subdurals=[4320,4321]

    TBI_ICD9=[skull_fract, concussion, cerebral_contusion, hemorrahges, hemorrahges2,
            cran_injury, subdurals]

    return TBI_ICD9


def read_events(subject_path, remove_null=True):
    events = dataframe_from_csv(os.path.join(subject_path, 'events.csv'), index_col=None)
    if remove_null:
        events = events[events.VALUE.notnull()]
    events.CHARTTIME = pd.to_datetime(events.CHARTTIME)
    events.HADM_ID = events.HADM_ID.fillna(value=-1).astype(int)
    events.ICUSTAY_ID = events.ICUSTAY_ID.fillna(value=-1).astype(int)
    events.VALUEUOM = events.VALUEUOM.fillna('').astype(str)
    # events.sort_values(by=['CHARTTIME', 'ITEMID', 'ICUSTAY_ID'], inplace=True)
    return events


def get_events_for_stay(events, icustayid, intime=None, outtime=None):
    idx = (events.ICUSTAY_ID == icustayid)
    if intime is not None and outtime is not None:
        idx = idx | ((events.CHARTTIME >= intime) & (events.CHARTTIME <= outtime))
    events = events[idx]
    del events['ICUSTAY_ID']
    return events


def add_hours_elpased_to_events(events, dt, remove_charttime=True):
    events = events.copy()
    events['HOURS'] = (events.CHARTTIME - dt).apply(lambda s: s / np.timedelta64(1, 's')) / 60./60
    if remove_charttime:
        del events['CHARTTIME']
    return events


def convert_events_to_timeseries(events, variable_column='VARIABLE', variables=[]):
    metadata = events[['CHARTTIME', 'ICUSTAY_ID']].sort_values(by=['CHARTTIME', 'ICUSTAY_ID'])\
                    .drop_duplicates(keep='first').set_index('CHARTTIME')
    timeseries = events[['CHARTTIME', variable_column, 'VALUE']]\
                    .sort_values(by=['CHARTTIME', variable_column, 'VALUE'], axis=0)\
                    .drop_duplicates(subset=['CHARTTIME', variable_column], keep='last')
    timeseries = timeseries.pivot(index='CHARTTIME', columns=variable_column, values='VALUE')\
                    .merge(metadata, left_index=True, right_index=True)\
                    .sort_index(axis=0).reset_index()
    for v in variables:
        if v not in timeseries:
            timeseries[v] = np.nan
    return timeseries


def get_first_valid_from_timeseries(timeseries, variable):
    if variable in timeseries:
        idx = timeseries[variable].notnull()
        if idx.any():
            loc = np.where(idx)[0][0]
            return timeseries[variable].iloc[loc]
    return np.nan
