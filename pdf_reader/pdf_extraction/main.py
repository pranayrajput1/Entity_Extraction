# importing the necessary modules
# import tabula
# from tabulate import tabulate
# import pandas as pd

# # reading the pdf
# df = tabula.read_pdf("~/home/knoldus/Desktop/Entity_Extraction/pdf_reader/45CANDIDATES.pdf", pages='1')
# print(df)

# # removing the blank column
# df = tabula.read_pdf("FoodCaloriesList.pdf")
# df = df.dropna(axis='columns')
# print(df)


# df = tabula.read_pdf("FoodCaloriesList.pdf", pages=3)
# print (tabulate(df))

# #reading file in json format
# df = tabula.read_pdf("FoodCaloriesList.pdf", pages=3, output_format="json")
# print(df) 

# #reading file as a single table
# df = tabula.read_pdf("FoodCaloriesList.pdf",  multiple_tables= False)
# print(df)

# #reading pdf via coordinates
# df = tabula.read_pdf("FoodCaloriesList.pdf", encoding = 'ISO-8859-1',
#          stream=True, area = [269.875, 12.75, 790.5, 961], pages = 4, guess = False,  pandas_options={'header':None})
# print(df)

# #coverting files to csv
# tabula.convert_into(loc1, "output1.csv", output_format="csv", pages='all')
# print("done")

#using camelot 
import camelot

#reading pdf via camelot
tables = camelot.read_pdf(loc4)
print( tables) 

tables1 = camelot.read_pdf("FoodCaloriesList.pdf", pages='all', area=[269.875, 120.75, 790.5, 561])
print (tabulate(tables1[0].df))

for i in range(1,5):
    print (i)
    tables = camelot.read_pdf("FoodCaloriesList.pdf", pages='%d' %  i)
    try:
        print (tabulate(tables[0].df))
        print (tabulate(tables[1].df))
    except IndexError:
        print('NOK')

#using pypdf2

import PyPDF2
pdf_file = open("FoodCaloriesList.pdf", 'rb')
read_pdf = PyPDF2.PdfFileReader(pdf_file)
number_of_pages = read_pdf.getNumPages()
page = read_pdf.getPage(1)
page_content = page.extractText()
print (page_content)

import numpy

table_list = page_content.split('\n')
l = numpy.array_split(table_list, len(table_list)/4)
for i in range(0,5):
    print(l[i])