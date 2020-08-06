import os
import pandas as pd
import datetime as dtm
import pickle
import numpy as np
import subprocess
import logging
logging.basicConfig(
    format=
    "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
    level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR_cwd = os.getcwd()
BASE_DIR_abs = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
BASE_DIR = BASE_DIR_cwd


def reduce_mem(df, use_float16=False):
    start_mem = df.memory_usage().sum() / 1024**2
    tm_cols = df.select_dtypes('datetime').columns
    for col in df.columns:
        if col in tm_cols:
            continue
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(
                        np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(
                        np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(
                        np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(
                        np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if use_float16 and c_min > np.finfo(
                        np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(
                        np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print('{:.2f} Mb, {:.2f} Mb ({:.2f} %)'.format(
        start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def mkdirs(dir2make):
    if isinstance(dir2make, list):
        for i_dir in dir2make:
            if not os.path.exists(i_dir):
                os.makedirs(i_dir)
    elif isinstance(dir2make, str):
        if not os.path.exists(dir2make):
            os.makedirs(dir2make)
    else:
        raise ValueError("dir2make should be string or list type.")


def bash2py(shell_command, split=True):
    """
    Args:
        shell_command: str, shell_command
        No capture_output arg for < py3.7 !
    example:
        bash2py('du -sh')    
    """
    res = subprocess.run(shell_command,
                         shell=True,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    if res.returncode == 0:
        logger.info(f"Execute <{shell_command}> successfully!")
    else:
        raise Exception(f"ERROR: {res.stderr.decode('utf-8')}")
    res = res.stdout.decode('utf-8').strip()  #.split('\n')
    if split:
        return res.split('\n')
    else:
        return res


class Cache():
    @staticmethod
    def cache_data(data, nm_marker=None, dt_format='%Y%m%d_%Hh'):
        mkdirs(os.path.join(BASE_DIR, 'cached_data'))
        name_ = dtm.datetime.now().strftime(dt_format)
        if nm_marker is not None:
            name_ = nm_marker
        path_ = os.path.join(BASE_DIR, f'cached_data/CACHE_{name_}.pkl')
        with open(path_, 'wb') as file:
            pickle.dump(data, file, protocol=4)
        logger.info(f'Cache Successfully! File name: {path_}')

    @staticmethod
    def reload_cache(file_nm,
                     pure_nm=False,
                     base_dir=None,
                     prefix='CACHE_',
                     postfix='.pkl'):
        if pure_nm:
            file_nm = prefix + file_nm + postfix
        if base_dir is None:
            base_dir = os.path.join(BASE_DIR, 'cached_data')
        file_path = os.path.join(base_dir, file_nm)
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        logger.info(f'Successfully Reload: {file_path}')
        return data

    @staticmethod
    def clear_cache(AreYouSure):
        if AreYouSure == 'clear_cache':
            bash2py(f"cd {BASE_DIR} && rm -rf cached_data/")
        else:
            logger.info(
                "If you truely wanna clear all cached data, set AreYouSure as 'clear_cache'"
            )


def get_cur_dt_str():
    cur_dt_str = dtm.datetime.now().strftime("%Y%m%d_%H%M%S")
    return cur_dt_str


def show_all_feas(df, unit='GB'):
    print("=" * 100)
    if isinstance(df, pd.DataFrame):
        print(len(df.columns), '||',
              "'" + "','".join(df.columns.tolist()) + "'")
        if unit.upper() in ['G', 'GB']:
            aa = 3
            bb = 'GB'
        elif unit.upper() in ['M', 'MB']:
            aa = 2
            bb = 'MB'
        elif unit.upper() in ['K', 'KB']:
            aa = 1
            bb = 'KB'
        else:
            raise ValueError(f"Unknown {unit} !")
        print("Memory Usage: ", df.memory_usage().sum() / 1024**aa, bb)
    else:
        # if df is a list like object:
        print(len(df), '||', "'" + "','".join(df) + "'")
    print("=" * 100)


def rm_feas(raw_feas, feas2del):
    if isinstance(feas2del, str):
        feas2del = [feas2del]
    return [col for col in raw_feas if col not in feas2del]


def add_feas(raw_feas, feas2add):
    if isinstance(feas2add, str):
        feas2add = [feas2add]
    cols2add = [col for col in feas2add if col not in raw_feas]
    return raw_feas + cols2add
