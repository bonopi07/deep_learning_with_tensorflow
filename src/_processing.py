import csv
from data import *
import hashlib
import multiprocessing as mp
import os
import shutil
import simplejson
import time
import _pickle


# function: 입력 경로에 대한 모든 파일 경로를 리스트로 반환하는 함수
def walk_dir(input_path):
    result = list()
    for path, _, files in os.walk(input_path):
        for file in files:
            file_path = os.path.join(path, file)  # store "file path"
            result.append(file_path)
    return result

