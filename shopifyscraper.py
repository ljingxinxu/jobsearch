from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import NoSuchElementException
import requests
import time
import string
import os
import re

#Replace this url with an Etsy website you want to scrape
#Make sure its the home page of a store, eg: https://www.etsy.com/shop/ashjairocreations
#URL = "https://newhaventea.com/shop-online/ols/products?page=2"

ROOTDIR = "./ShopifyData"
try:
    os.mkdir(ROOTDIR)
except:
    print("rootdir exists :DD")

service = Service()
options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--ignore-ssl-errors')
#options.add_argument("--headless")
options.add_argument("--log-level=3")
#options.add_argument('--window-size=120,700')
driver = webdriver.Chrome(service=service, options=options)
printable = set(string.printable)

product_links = []

URL = "https://homeland203.com/collections/all"
driver.get(URL)
product_elements = driver.find_elements(By.CSS_SELECTOR, "h3.card__heading.h5 a")

# for x in range(11):
#     URL = f"https://www.odeonboutique.com/collections/all?page={x+2}"
#     driver.get(URL)
#     # driver.switch_to_alert()
#     time.sleep(5)
#     # if therers popup driver.execute_script( "window.onbeforeunload = function(e){};" )
#     product_elements = driver.find_elements(By.CSS_SELECTOR, "a.product-card")
#         #product_elements = driver.find_elements(By.CSS_SELECTOR, ".product-image__link")
#     print("got elements", URL, len(product_elements))

        # Extract links to each product
for product_element in product_elements:
    product_link = product_element.get_attribute("href")
    product_links.append(product_link)

print(len(product_links))


for x, link in enumerate(product_links):
    #print(x, link)
    if link:
        driver.get(link)
        time.sleep(2)
    
    try:
        #title = driver.find_element(By.CSS_SELECTOR, ".w-product-title").text
        #price = driver.find_element(By.CSS_SELECTOR, ".product__price span").text
        
        # square 
        title = driver.find_element(By.CSS_SELECTOR, ".product__title h1").text
        print(title)
        price = driver.find_element(By.CSS_SELECTOR, "span.price-item.price-item--regular").text
        print(price)
        
        # godaddy
        # title = driver.find_element(By.CSS_SELECTOR, ".x-el.x-el-h1.c2-1.c2-2.c2-2v.c2-2w.c2-66.c2-2k.c2-2l.c2-1l.c2-x.c2-w.c2-3.c2-35.c2-7i.c2-36.c2-y.c2-7j.c2-7k.c2-7l.c2-7m").text
        #price = driver.find_element(By.CSS_SELECTOR, ".x-el.x-el-div.c2-1.c2-2.c2-j.c2-k.c2-4n.c2-31.c2-3y.c2-y.c2-2e.c2-3.c2-o.c2-2t.c2-q.c2-2u.c2-2v.c2-2w.c2-2x").text
        
        
        #shopify title = driver.find_element(By.CSS_SELECTOR, ".x-el.x-el-h1.c2-1.c2-2.c2-j.c2-k.c2-70.c2-3z.c2-40.c2-31.c2-2f.c2-2e.c2-3.c2-o.c2-7k.c2-q.c2-2g.c2-7l.c2-7m.c2-7n.c2-7o").text
        #shopify price = driver.find_element(By.CSS_SELECTOR, ".modal_price").text
        
        desc = ""
        try:
            #desc = driver.find_element(By.CSS_SELECTOR, "div[data-aid=PRODUCT_DESCRIPTION_RENDERED] span").text

            parent = driver.find_element(By.CSS_SELECTOR, ".station-tabs-tabcontent")
            children = parent.find_elements(By.CSS_SELECTOR, "p")
            for p in children:
                desc += p.text + "\n"
            print(desc)
            
        except:
            desc = ""

        image_link = []
        img_elements = driver.find_elements(By.CSS_SELECTOR, '.product__media.media.media--transparent img')
        
        for img in img_elements:
            image_link.append(img.get_attribute("src"))

            # img srcset
            # img_srcset = img.get_attribute("srcset")
            # #pattern = r'(.+?)\s+2048w'
            # #pattern = r'//(.+?)\s+2048w'
            # pattern = r'1728w, (.+?) 2048w'


            # match = re.search(pattern, img_srcset)
            # if match:
            #     url = match.group(1).lstrip(', //').rstrip()
            #     image_link.append('https://'+url)
            #     #print(url)
            # else:
            #     print("URL not found before 2048w")

        # click on thumbnails, for newlinblending
        # try:
        #     #thumbnail = driver.find_elements(By.CSS_SELECTOR, "li[data-index].carousel-slide")
        #     carousel = driver.find_element(By.XPATH,"/html/body/div/div/div/div[2]/div/div/div/span/div/section/div/div/div[2]/div/div/div[1]/div[1]/div/div/div[1]/div/div[2]/div/div/ul")
        #     thumbnail = carousel.find_elements(By.CSS_SELECTOR, "li")
        #     print(thumbnail)
        #     if len(thumbnail) > 5:
        #         try:
        #             while next:
        #                 new_image = driver.find_element(By.CSS_SELECTOR, "[src^='https://img1.wsimg.com/isteam/ip/468b84d1-ff29-496c-a672-9d1df66456c1']")
        #                 image_link.append(new_image.get_attribute("src"))
        #                 next = driver.find_element(By.CSS_SELECTOR, "#chevronRight svg")
        #                 ActionChains(driver).move_to_element(next).click(next).perform()
        #                 print('clicked arrow')
        #                 time.sleep(2)
        #         except NoSuchElementException:
        #             print('no next arrow found')
        #             pass
        #     else:
        #         for t in thumbnail:
        #             ActionChains(driver).move_to_element(t).click(t).perform()
        #             print('clicked thumbnail')
        #             time.sleep(2)
        #             new_image = driver.find_element(By.CSS_SELECTOR, "[src^='https://img1.wsimg.com/isteam/ip/468b84d1-ff29-496c-a672-9d1df66456c1']")
        #             image_link.append(new_image.get_attribute("src"))
        # except NoSuchElementException:
        #     print('no thumbnail found')
        #     pass


        #path_title = re.sub(r'[^\w_. -]', '_', title)
        path_title = 'a'+str(x)
        mydir = ROOTDIR + "/" + path_title
        try: 
            os.mkdir(mydir)
            file1 = open(mydir + "/info.txt","w",encoding="utf-8")
            file1.write(title + "\n")
            file1.write(price + "\n")
            file1.write(desc)
            file1.close()
        except Exception as e:
            print(e)
            pass

        for i in range(len(image_link)):
            data = requests.get(image_link[i]).content
            index = i-1
            if index == -1:
                index = ""

            f = open(mydir + "/image" + str(index) + ".jpg", 'wb')
            f.write(data)
            f.close()
            time.sleep(.5)
    except Exception as e:
        print(e)