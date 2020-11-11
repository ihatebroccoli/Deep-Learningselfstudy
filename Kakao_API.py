# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 15:33:07 2020

@author: my
"""


# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


from bs4 import BeautifulSoup as bs
import urllib.request
import urllib.parse
from selenium import webdriver
import time
import pandas as pd
from selenium.webdriver.support.ui import Select
import random
url = "https://speech-api.kakao.com/"


driver = webdriver.Chrome()
driver.get(url)
text_box = driver.find_element_by_id('txtDemo')
button = driver.find_element_by_class_name('btn_listen')

select_click = driver.find_element_by_xpath('//*[@id="speechSynthesis"]/div/div[2]/ul/li[5]/div/div[1]') #click select voice box
select_click.click()
select_click_after = driver.find_element_by_xpath('//*[@id="speechSynthesis"]/div/div[2]/ul/li[5]/div/div[2]/div[1]') #click voice // 1 spring 2 ryan 3 naomi
select_click_after.click()

select_click = driver.find_element_by_id('txtDemo_ssml_btn') # Use SSML
select_click.click()

for_speed= ['0.8', '0.9', '1.1', '1.2', '1.3']


def spell_out(txt):
    # spell out
    begin = '<say-as interpret-as=\"spell-out\">'
    end = "</say-as>"
    if 'IPC' in txt:
        parts = txt.split('IPC')
        txt = parts[0] + begin + 'IPC'
        txt = txt + end + parts[1]
    elif 'AMM' in txt:
        parts = txt.split('AMM')
        txt = parts[0] + begin + 'AMM'
        txt = txt + end + parts[1]        
    return txt

def speed(txt, speed_opt): #adjust speed
    
    begin = "<prosody rate=\"" + for_speed[speed_opt] + "\">"
    end= "</prosody>"
    txt = begin + txt + end
    return txt

def volume(txt): #adjust volume
    for_begin= ['\"soft\"', '\"loud\"']
    begin = "<prosody volume=" + for_begin[random.randrange(0,2)] + ">"
    end= "</prosody>"
    txt = begin + txt + end
    return txt

def friendly(txt):#voice _ friendly
    return '<kakao:effect tone = \"friendly\">' + txt + '</kakao:effect>'

def add_ssml(txt, speed_opt):
    
    txt = speed(txt, speed_opt)
    if ('IPC' in txt or 'AMM' in txt):
        txt = spell_out(txt)
    
    txt = friendly(txt)
    txt = "<speak>" + txt + "</speak>"
    return txt

inst_file = pd.read_csv("instr.csv", encoding = 'CP949')
instruction_list = inst_file['instruction']
url_list = []

for i in range(len(instruction_list)):
    text_box.click()
    erase = driver.find_element_by_class_name('close_btn') 
    erase.click() # erase default, old data
        
    text_box.send_keys(add_ssml(instruction_list[i], j)) # type the data
    button.click()
        
    time.sleep(1)
        
    audio = driver.find_element_by_id('audioPlayer')
    audio_name = audio.get_attribute('src')
        
    urllib.request.urlretrieve(audio_name,"./kakao_Plain_Spring_S(" + for_speed[j] + ")_Spell/"+"kakao_Plain_Spring_S(" + for_speed[j] + ")_Spell" + instruction_list[i] + '.mp3')
    time.sleep(4)

