from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import NoSuchElementException

from selenium.webdriver.support import expected_conditions as EC
import time
import os
   
service = Service()
options = webdriver.ChromeOptions()

options.add_argument('--ignore-certificate-errors')
options.add_argument('--ignore-ssl-errors')
driver = webdriver.Chrome(service=service, options=options)

URL = 'https://www.google.com'
driver.get(URL)
driver.implicitly_wait(5)  

file_path = './jobs.txt'

with open(file_path, 'r', encoding='utf-8') as file:
    file_contents = file.read()

lines_array = file_contents.splitlines()

for line in lines_array:
    driver.switch_to.window(driver.window_handles[0])
    

    # focus search bar
    search_bar = driver.find_element(By.XPATH, '//*[@id="APjFqb"]')
    # Clear any existing text
    search_bar.clear()

    search_string = line + ' careers'
    search_bar.send_keys(search_string)
    search_bar.send_keys(Keys.ENTER)

    time.sleep(1)

    # get first result
    try:
        first_link = driver.find_element(By.CSS_SELECTOR, '#rso > div.hlcw0c > div > div > div > div > div > div > div > div.yuRUbf > div > span > a').get_attribute('href')
    except NoSuchElementException: 
        print('try another selector')
        try:
            first_link = driver.find_element(By.CSS_SELECTOR, '#rso > div.hlcw0c > div > div > div > div.kb0PBd.cvP2Ce.A9Y9g.jGGQ5e > div > div > span > a').get_attribute('href')
        except NoSuchElementException:
            print('try third selector')

    driver.execute_script("window.open('');")
    driver.switch_to.window(driver.window_handles[-1])
    driver.get(first_link)

driver.quit()
