
#importing essential library
import camelot

#read the file
def readpdf():
    pdf = camelot.read_pdf("structured.pdf", pages='all')
    print(pdf)
def type_pdf(typdf):
    pdf = camelot.read_pdf("structured.pdf", pages='all')
    print(type(pdf))

def tableshape(tbshape):
    pdf = camelot.read_pdf("structured.pdf", pages='all')
    print(pdf[1])

def tableshape(tbshape):
    pdf = camelot.read_pdf("structured.pdf", pages='all')
    typeofdata = pdf[1]
    print(typeofdata.df)

def pdftocsv():
    pdf = camelot.read_pdf("structured.pdf", pages='all')
    typeofdata = pdf[1]
    typeofdata.to_csv("op.csv")

def report():
    pdf = camelot.read_pdf("structured.pdf", pages='all')
    typeofdata = pdf[1]
    print(pdf[1].parsing_report) #knowing the accuracy

if __name__ == '__main__':
    readpdf()
    report()
    pdftocsv()