# importing the necessary modules
import tabula
from tabulate import tabulate
import pandas as pd

# reading the pdf
def read_pdf():
    filename=input("enter file name")
    df = tabula.read_pdf(filename, pages='1')
    print(df)

# removing the blank column
def drop_na_pdf():
    filename=input("enter file name")
    df = tabula.read_pdf(filename)
    df = df.dropna(axis='columns')
    print(df)

# extracting the particular page of a pdf
def pdf_page():
    filename=input("enter file name")
    df = tabula.read_pdf(filename, pages=3)
    print (tabulate(df))

#reading file in json format
def pdf_to_json():
    filename=input("enter file name")
    df = tabula.read_pdf(filename, pages=3, output_format="json")
    print(df) 

#reading file as a single table
def read_pdf_table():
    filename=input("enter file name")
    df = tabula.read_pdf(filename,  multiple_tables= False)
    print(df)


#reading pdf via coordinates
def read_pdf_via_coordinates():
    filename=input("enter file name")
    df = tabula.read_pdf(filename, encoding = 'ISO-8859-1',
         stream=True, area = [269.875, 12.75, 790.5, 961], pages = 4, guess = False,  pandas_options={'header':None})
    print(df)

#coverting files to csv
def pdf_to_csv():
    filename=input("enter file name")
    tabula.convert_into(filename, "output1.csv", output_format="csv", pages='all')
    print("done")

if __name__ == '__main__':
    read_pdf()
