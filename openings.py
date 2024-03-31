from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os
   
service = Service()
options = webdriver.ChromeOptions()

driver = webdriver.Chrome(service=service, options=options)

# URL = 'https://www.tesla.com/careers/search/?site=US'
# driver.get(URL)

# driver.implicitly_wait(5)  # Adjust the time as needed

# search_bar = driver.find_element(By.XPATH, '/html/body/div[1]/div/form/div/aside/div/div/div[1]/div/input')
# print(search_bar)
# search_string = 'OpenAI'
# search_bar.send_keys(search_string)
# time.sleep(100)
# driver.quit()



URL = 'https://www.google.com'
driver.get(URL)
driver.implicitly_wait(5)  # Adjust the time as needed

search_bar = driver.find_element(By.XPATH, '//*[@id="APjFqb"]')


# Specify the path to your file
file_path = './jobs.txt'

with open(file_path, 'r', encoding='utf-8') as file:
    file_contents = file.read()

lines_array = file_contents.splitlines()

for line in lines_array:
    search_string = line + ' careers'
    search_bar.send_keys(search_string)
    search_bar.send_keys(Keys.ENTER)

    time.sleep(10)
    first_link = driver.find_element(By.XPATH, '//*[@id="rso"]/div[1]/div/div/div/div/div/div/div/div[1]/div/span/a').get_attribute('href')

    driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.CONTROL + 't')

    driver.switch_to.window(driver.window_handles[-1])
    driver.get(first_link)

driver.quit()


# //*[@id="rso"]/div[1]/div/div/div/div/div/div/div/div[1]/div/span/a
# //*[@id="rso"]/div[1]/div/div/div/div[1]/div/div/span/a

# //*[@id="rso"]/div[1]/div/div/div/div[1]/div/div/span/a
# #rso > div.hlcw0c > div > div > div > div.kb0PBd.cvP2Ce.A9Y9g.jGGQ5e > div > div > span > a
# #rso > div.hlcw0c > div > div > div > div > div > div > div > div > div > div > span > a