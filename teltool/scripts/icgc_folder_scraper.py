import sys
import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
import time

d = pd.read_csv(sys.argv[1], sep="\t")
fids = d["File ID"]
fids = list(fids)

base_url = "https://dcc.icgc.org/repositories/files/"

browser = webdriver.Firefox()

folder_names = []
total_files = len(fids)

for x, i in enumerate(fids):
    print("On: ", x+1, "/", total_files, end="\r")
    url = base_url+i
    browser.get(url)
    time.sleep(2)
    el = browser.find_element(By.XPATH, '//*[@id="dataBundleID"]')
    folder_names.append(el.text)

out_file=open(sys.argv[2],'w')
for items in folder_names:
    out_file.writelines([items])
    out_file.writelines("\n")

out_file.close()
