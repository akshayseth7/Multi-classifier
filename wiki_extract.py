# -*- coding: utf-8 -*-i
from selenium import webdriver
import csv
import time
import sys

def search(inputFile,outputFile):
    counter1=0
    driver = webdriver.Firefox()
    inFile = inputFile
    outFile = outputFile
    Fileopener= open(outFile, 'w')
    Filewriter = csv.writer(Fileopener)
    data1=[["Company Name","Page Number","Search Name","Webpage","Table Row 1","Table Row 2","Table Row 3","Table Row 4","Table Row 5","Table Row 6","Table Row 7","Table Row 8","Table Row 9","Table Row 10","Table Row 11","Table Row 12","Table Row 13","Table Row 14","Table Row 15","Table Row 16","Table Row 17","Table Row 18","Table Row 19"]]
    Filewriter.writerows(data1)
    file = open(inFile, 'r')
    for line in file:
        counter1 = counter1+1

        print ("******************************************************************")
        print "URL Number: ", counter1
        print "Actual URL: ", line

        (line).strip()
        Company_name=line
        Company_name=Company_name.strip()
        #print "Akshay"
        driver.get("https://www.google.com")
        time.sleep(3)
        element1 = driver.find_element_by_xpath('//*[@id="lst-ib"]')
        #data1=[[Company_name]]
        #Filewriter.writerows(data1)
        element1.send_keys(Company_name+' Wiki')
        #element1.send_keys('115 Solutions P/L')

        element2 = driver.find_element_by_xpath('//*[@id="sblsbb"]/button')
        element2.click()
        counter=0
        time.sleep(10)
        c=[]
        company=""
        wikilinks=""
        wiki_links=""
        content=[]
        a=1
        try:

                # To Fetch all Weblinks
                element4=driver.find_elements_by_xpath('//*[@class="r"]/a')
                try:
                    for element in element4:
                            com2=element.get_attribute('href')
                            print com2
                            if("wikipedia.org" in com2):
                                if (counter==0):
                                    print "WIKI"
                                    wikilinks=com2
                                    counter=counter+1
                except Exception,e:
                    print "E2"
                    continue
                time.sleep(30)
        except Exception,e:
            print "Pages Exception"

        print "page"
        print wikilinks
        driver.get(wikilinks)
        wiki_links=wikilinks
        print "T"
        cont=""
        try:
            element9=driver.find_element_by_xpath('//*[@id="mw-content-text"]/p[1]')
            print element9.text
            print "HH"

            cont=element9.text
        except Exception,e:
            print "Exception 1"
            pass
        try:
            element10=driver.find_element_by_xpath('//*[@id="mw-content-text"]/p[2]')
            print element10.text
            cont=cont+" " +element10.text
        except Exception,e:
            print "Exception 2"
            content.append(cont)
            pass
        try:
            element11=driver.find_element_by_xpath('//*[@id="mw-content-text"]/p[3]')
            print element11.text
            cont=cont+" " +element10.text
        except Exception,e:
            print "Exception 3"
            content.append(cont)
            pass

        content.append(cont)
        if (cont==""):
            content.append("not found")

        try:
            element12=driver.find_element_by_xpath('//*[@class="infobox vcard"]/tbody/tr[2]/th')
            print "TABLE"
            print element12.text

        except Exception,e:
            print "Exception 4"
            pass
        try:
            element13=driver.find_element_by_xpath('//*[@class="infobox vcard"]/tbody/tr[2]/td')
            print "TABLE"
            print element13.text

            table1=str(element12.text)+" : "+ str(element13.text)
            link=table1
            link=link.replace("\n"," . ")
            link=link.replace("\&nbsp"," ")
            #link=link.strip()
            link = link.rstrip().lstrip()
            while "  " in link:
                link = link.replace("  ", " ")
            table1=link

            print table1
        except Exception,e:
            print "Exception 5"
            table1="not found"
            pass
        try:
            element12=driver.find_element_by_xpath('//*[@class="infobox vcard"]/tbody/tr[3]/th')
            print "TABLE"
            print element12.text

        except Exception,e:
            print "Exception 6"
            pass
        try:
            element13=driver.find_element_by_xpath('//*[@class="infobox vcard"]/tbody/tr[3]/td')
            print "TABLE"
            print element13.text
            table2=str(element12.text)+" : "+ str(element13.text)
            table2 = table2.replace('\n'," ")
            print table2
        except Exception,e:
            print "Exception 7"
            table2="not found"
            pass
        try:
            element12=driver.find_element_by_xpath('//*[@class="infobox vcard"]/tbody/tr[4]/th')
            print "TABLE"
            print element12.text

        except Exception,e:
            print "Exception 8"
            pass
        try:
            element13=driver.find_element_by_xpath('//*[@class="infobox vcard"]/tbody/tr[4]/td')
            print "TABLE"
            print element13.text
            table3=str(element12.text)+" : "+ str(element13.text)
            table3 = table3.replace('\n'," ")
            if ('\n' in table3):
                print "AKSHAY"
            print table3
        except Exception,e:
            print "Exception 9"
            table3="not found"
            pass
        try:
            element12=driver.find_element_by_xpath('//*[@class="infobox vcard"]/tbody/tr[5]/th')
            print "TABLE"
            print element12.text

        except Exception,e:
            print "Exception 10"
            pass
        try:
            element13=driver.find_element_by_xpath('//*[@class="infobox vcard"]/tbody/tr[5]/td')
            print "TABLE"
            print element13.text
            table4=str(element12.text)+" : "+ str(element13.text)
            table4 = table4.replace('\n'," ")
            print table4
        except Exception,e:
            print "Exception 11"
            table4="not found"
            pass

        try:
            element12=driver.find_element_by_xpath('//*[@class="infobox vcard"]/tbody/tr[6]/th')
            print "TABLE"
            print element12.text

        except Exception,e:
            print "Exception 12"
            pass
        try:
            element13=driver.find_element_by_xpath('//*[@class="infobox vcard"]/tbody/tr[6]/td')
            print "TABLE"
            print element13.text
            table5=str(element12.text)+" : "+ str(element13.text)
            table5 = table5.replace('\n'," ")
            print table5
        except Exception,e:
            print "Exception 13"
            table5="not found"
            pass

        try:
            element12=driver.find_element_by_xpath('//*[@class="infobox vcard"]/tbody/tr[7]/th')
            print "TABLE"
            print element12.text

        except Exception,e:
            print "Exception 14"
            pass
        try:
            element13=driver.find_element_by_xpath('//*[@class="infobox vcard"]/tbody/tr[7]/td')
            print "TABLE"
            print element13.text
            table6=str(element12.text)+" : "+ str(element13.text)
            table6 = table6.replace('\n'," ")
            print table6
        except Exception,e:
            print "Exception 15"
            table6="not found"
            pass

        try:
            element12=driver.find_element_by_xpath('//*[@class="infobox vcard"]/tbody/tr[8]/th')
            print "TABLE"
            print element12.text

        except Exception,e:
            print "Exception 16"
            pass
        try:
            element13=driver.find_element_by_xpath('//*[@class="infobox vcard"]/tbody/tr[8]/td')
            print "TABLE"
            print element13.text
            table7=str(element12.text)+" : "+ str(element13.text)
            table7 = table7.replace('\n'," ")
            print table7
        except Exception,e:
            print "Exception 17"
            table7="not found"
            pass

        try:
            element12=driver.find_element_by_xpath('//*[@class="infobox vcard"]/tbody/tr[9]/th')
            print "TABLE"
            print element12.text

        except Exception,e:
            print "Exception 18"
            pass
        try:
            element13=driver.find_element_by_xpath('//*[@class="infobox vcard"]/tbody/tr[9]/td')
            print "TABLE"
            print element13.text
            table8=str(element12.text)+" : "+ str(element13.text)
            table8 = table8.replace('\n'," ")
            print table8
        except Exception,e:
            print "Exception 19"
            table8="not found"
            pass

        try:
            element12=driver.find_element_by_xpath('//*[@class="infobox vcard"]/tbody/tr[10]/th')
            print "TABLE"
            print element12.text

        except Exception,e:
            print "Exception 20"
            pass
        try:
            element13=driver.find_element_by_xpath('//*[@class="infobox vcard"]/tbody/tr[10]/td')
            print "TABLE"
            print element13.text
            table9=str(element12.text)+" : "+ str(element13.text)
            table9 = table9.replace('\n'," ")
            print table9
        except Exception,e:
            print "Exception 21"
            table9="not found"
            pass

        try:
            element12=driver.find_element_by_xpath('//*[@class="infobox vcard"]/tbody/tr[11]/th')
            print "TABLE"
            print element12.text

        except Exception,e:
            print "Exception 22"
            pass
        try:
            element13=driver.find_element_by_xpath('//*[@class="infobox vcard"]/tbody/tr[11]/td')
            print "TABLE"
            print element13.text
            table10=str(element12.text)+" : "+ str(element13.text)
            table10 = table10.replace('\n'," ")
            print table10
        except Exception,e:
            print "Exception 23"
            table10="not found"
            pass
        try:
            element12=driver.find_element_by_xpath('//*[@class="infobox vcard"]/tbody/tr[12]/th')
            print "TABLE"
            print element12.text

        except Exception,e:
            print "Exception 24"
            pass
        try:
            element13=driver.find_element_by_xpath('//*[@class="infobox vcard"]/tbody/tr[12]/td')
            print "TABLE"
            print element13.text
            table11=str(element12.text)+" : "+ str(element13.text)
            table11 = table11.replace('\n'," ")
            print table11
        except Exception,e:
            print "Exception 25"
            table11="not found"
            pass
        try:
            element12=driver.find_element_by_xpath('//*[@class="infobox vcard"]/tbody/tr[13]/th')
            print "TABLE"
            print element12.text

        except Exception,e:
            print "Exception 26"
            pass
        try:
            element13=driver.find_element_by_xpath('//*[@class="infobox vcard"]/tbody/tr[13]/td')
            print "TABLE"
            print element13.text
            table12=str(element12.text)+" : "+ str(element13.text)
            table12 = table12.replace('\n'," ")
            print table12
        except Exception,e:
            print "Exception 27"
            table12="not found"
            pass
        try:
            element12=driver.find_element_by_xpath('//*[@class="infobox vcard"]/tbody/tr[14]/th')
            print "TABLE"
            print element12.text

        except Exception,e:
            print "Exception 28"
            pass
        try:
            element13=driver.find_element_by_xpath('//*[@class="infobox vcard"]/tbody/tr[14]/td')
            print "TABLE"
            print element13.text
            table13=str(element12.text)+" : "+ str(element13.text)
            table13 = table13.replace('\n'," ")
            print table13
        except Exception,e:
            print "Exception 29"
            table13="not found"
            pass
        try:
            element12=driver.find_element_by_xpath('//*[@class="infobox vcard"]/tbody/tr[15]/th')
            print "TABLE"
            print element12.text

        except Exception,e:
            print "Exception 30"
            pass
        try:
            element13=driver.find_element_by_xpath('//*[@class="infobox vcard"]/tbody/tr[15]/td')
            print "TABLE"
            print element13.text
            table14=str(element12.text)+" : "+ str(element13.text)
            table14 = table14.replace('\n'," ")
            print table14
        except Exception,e:
            print "Exception 31"
            table14="not found"
            pass

        try:
            element12=driver.find_element_by_xpath('//*[@class="infobox vcard"]/tbody/tr[16]/th')
            print "TABLE"
            print element12.text

        except Exception,e:
            print "Exception 32"
            pass
        try:
            element13=driver.find_element_by_xpath('//*[@class="infobox vcard"]/tbody/tr[16]/td')
            print "TABLE"
            print element13.text
            table15=str(element12.text)+" : "+ str(element13.text)
            table15 = table15.replace('\n'," ")
            print table15
        except Exception,e:
            print "Exception 33"
            table15="not found"
            pass

        try:
            element12=driver.find_element_by_xpath('//*[@class="infobox vcard"]/tbody/tr[17]/th')
            print "TABLE"
            print element12.text

        except Exception,e:
            print "Exception 34"
            pass
        try:
            element13=driver.find_element_by_xpath('//*[@class="infobox vcard"]/tbody/tr[17]/td')
            print "TABLE"
            print element13.text
            table16=str(element12.text)+" : "+ str(element13.text)
            table16 = table16.replace('\n'," ")
            print table16
        except Exception,e:
            print "Exception 35"
            table16="not found"
            pass

        try:
            element12=driver.find_element_by_xpath('//*[@class="infobox vcard"]/tbody/tr[18]/th')
            print "TABLE"
            print element12.text

        except Exception,e:
            print "Exception 36"
            pass
        try:
            element13=driver.find_element_by_xpath('//*[@class="infobox vcard"]/tbody/tr[18]/td')
            print "TABLE"
            print element13.text
            table17=str(element12.text)+" : "+ str(element13.text)
            table17 = table17.replace('\n'," ")
            print table17
        except Exception,e:
            print "Exception 37"
            table17="not found"
            pass

        try:
            element12=driver.find_element_by_xpath('//*[@class="infobox vcard"]/tbody/tr[19]/th')
            print "TABLE"
            print element12.text

        except Exception,e:
            print "Exception 38"
            pass
        try:
            element13=driver.find_element_by_xpath('//*[@class="infobox vcard"]/tbody/tr[19]/td')
            print "TABLE"
            print element13.text
            table18=str(element12.text)+" : "+ str(element13.text)
            table18 = table18.replace('\n'," ")
            print table18
        except Exception,e:
            print "Exception 39"
            table18="not found"
            pass

        try:
            element12=driver.find_element_by_xpath('//*[@class="infobox vcard"]/tbody/tr[20]/th')
            print "TABLE"
            print element12.text

        except Exception,e:
            print "Exception 40"
            pass
        try:
            element13=driver.find_element_by_xpath('//*[@class="infobox vcard"]/tbody/tr[20]/td')
            print "TABLE"
            print element13.text
            table19=str(element12.text)+" : "+ str(element13.text)
            table19 = table19.replace('\n'," ")
            print table19
        except Exception,e:
            print "Exception 41"
            table19="not found"
            pass

        print "DONE"

        data=[[Company_name,wiki_links,content[0].encode('ascii','ignore'),table1,table2,table3,table4,table5,table6,table7,table8,table9,table10,table11,table12,table13,table14,table15,table16,table17,table18,table19]]
        print data
        print "YO"
        Filewriter.writerows(data)
            #i=i+1
            #print i

if __name__ == '__main__':
	search(sys.argv[1],sys.argv[2])