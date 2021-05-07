from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from selenium.webdriver.chrome.options import Options
import re

usr = '******'
pwd = '******'

path_to_extension = r'C:\Users\User\Desktop\3.10.2_0'
option = Options()

option.add_experimental_option("prefs", {
    "profile.default_content_setting_values.notifications": 1
})

chrome_options = Options()
chrome_options.add_argument('load-extension=' + path_to_extension)

driver = webdriver.Chrome()
driver.get("https://www.netacad.com/")
time.sleep(1)
driver.implicitly_wait(10)

element = driver.find_element_by_xpath('/html/body/header/div/div[2]/div/button')
element.click()
time.sleep(1)
driver.implicitly_wait(10)

login = driver.find_element_by_xpath('/html/body/main/nav/div/section[2]/div/div/ul/li[8]')
login.click()
time.sleep(1)
driver.implicitly_wait(10)

login = driver.find_element_by_xpath('/html/body/main/nav/div/section[2]/div/div/ul/li[8]/ul/li[2]/a')
login.click()
time.sleep(1)
driver.implicitly_wait(10)
DONTSHOWME = driver.find_element_by_xpath('/html/body/div[5]/div[1]/div[2]/div[2]/div[1]/div/div/div[3]/a')
DONTSHOWME.click()


id = driver.find_element_by_xpath('/html/body/div/div[2]/div/div[2]/div/div/form/div[1]/div[2]/div/div[2]/span/input')
id.send_keys(usr)
time.sleep(3)
driver.implicitly_wait(10)

time.sleep(1)
driver.implicitly_wait(10)

password = driver.find_element_by_xpath('/html/body/div[1]/div[2]/div/div[2]/div/div/form/div[1]/div[2]/div[2]/div[2]/span/input')
password.send_keys(pwd)
password.send_keys(Keys.RETURN)
time.sleep(1)
driver.implicitly_wait(10)

chapter = driver.find_element_by_xpath('/html/body/main/div/div/div[3]/section/div/div[2]/div/div[1]/div[2]/div/div[1]/div/div[2]/div[2]/a')
chapter.click()

module = driver.find_element_by_xpath('/html/body/div[5]/main/div[2]/div/div[2]/div/section[1]/div/div/ul/li[4]/div[3]/ul/li[1]/div/div/div[2]/div/a')








