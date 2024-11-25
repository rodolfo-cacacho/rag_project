from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
import pandas as pd
import os


def add_metada(document,version,date):
    document.metadata.update({'version':version,'valid_date':date})
    return document

def add_metadata_value(document,field,text):
    document.metadata.update({field:text})
    return document

def modify_metadata(document,field,text):
    document.metadata[field] = text
    return document

"""
Usage:

data_directory = 'data/documents'
target_file = 'Technische FAQ BEG EM'

data = get_pdf_csv(data_directory,target_file)

"""


def get_pdf_csv(docs_directory,target_file):
    clean_directory = os.path.join(docs_directory,'clean')
    original_directory = os.path.join(docs_directory,'original')
    metadata_directory = os.path.join(docs_directory,'metadata')

    version_file = os.path.join(metadata_directory,'files_date_version.csv')
    csv_version = pd.read_csv(version_file)
    c_directory = os.path.join(clean_directory,target_file)
    print(c_directory)
    files = os.listdir(c_directory)
    clean_file_names = [i for i in files if os.path.isdir(os.path.join(c_directory,i))]
    print(clean_file_names)
    datas = []
    for i,file in enumerate(clean_file_names):

        pdf_file = os.path.join(original_directory,target_file,(file+'.pdf'))
        csv_file = os.path.join(clean_directory,target_file,file,(file+'.csv'))
        print(f'File: {file} - pdf: {pdf_file}\ncsv: {csv_file}')
        if os.path.exists(pdf_file):
            temp_datas = []
            print('PDF exists\n')
            version_i = csv_version.loc[csv_version['file'] == (file+'.pdf'), 'version'].iloc[0]
            date_i = csv_version.loc[csv_version['file'] == (file+'.pdf'), 'date'].iloc[0]
            loader_pdf = PyMuPDFLoader(pdf_file,extract_images=False)
            data_pdf = loader_pdf.load()
            temp_datas.append(data_pdf[0])

            if(os.path.exists(csv_file)):
                print('CSV exists\n')
                loader_csv = CSVLoader(file_path=csv_file)
                data_csv = loader_csv.load()
                temp_datas.extend(data_csv)

            temp_datas2 = []
            for i in temp_datas:
                i = add_metada(i,version=version_i,date=date_i)
                i = add_metadata_value(i,'type','text')
                i = add_metadata_value(i,'path','')
                i = modify_metadata(i,'source',(file+'.pdf'))
                i = add_metadata_value(i,'doc_type',target_file)
                print(i)
                temp_datas2.append(i)
            
            datas.append(temp_datas2)

    return datas

    # pdfs = [item for item in files if os.path.isdir(os.path.join(directory, item)) == False and item.endswith('.pdf')]
    # datas = []
    # for i,pdf in enumerate(pdfs):
    #     version_i = csv_version.loc[csv_version['file'] == pdf, 'version'].iloc[0]
    #     date_i = csv_version.loc[csv_version['file'] == pdf, 'date'].iloc[0]
    #     temp_datas = []
    #     csv_file = pdf.replace(" ","_").replace("pdf","csv")
    #     print(f'{i} - {pdf} - {csv_file} - version:{version_i} - date:{date_i}')
    #     csv_file_path = os.path.join(directory,'csv',csv_file)
    #     pdf_file_path = os.path.join(directory,pdf)
    #     # print(f'csv - {os.path.exists(csv_file_path)} pdf - {os.path.exists(pdf_file_path)}')
    #     loader_pdf = PyMuPDFLoader(pdf_file_path,extract_images=False)
    #     data_pdf = loader_pdf.load()
    #     loader_csv = CSVLoader(file_path=csv_file_path)
    #     data_csv = loader_csv.load()
    #     temp_datas.append(data_pdf[0])
    #     temp_datas.extend(data_csv)
    #     temp_datas2 = []
    #     for i in temp_datas:
    #         i = add_metada(i,version=version_i,date=date_i)
    #         temp_datas2.append(i)
    #     datas.append(temp_datas2)
    # return datas
