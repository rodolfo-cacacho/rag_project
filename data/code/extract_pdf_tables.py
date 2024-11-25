"""
 Copyright 2024 Adobe
 All Rights Reserved.

 NOTICE: Adobe permits you to use, modify, and distribute this file in
 accordance with the terms of the Adobe license agreement accompanying it.
"""

import logging
import os
from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
from adobe.pdfservices.operation.exception.exceptions import ServiceApiException, ServiceUsageException, SdkException
from adobe.pdfservices.operation.io.cloud_asset import CloudAsset
from adobe.pdfservices.operation.io.stream_asset import StreamAsset
from adobe.pdfservices.operation.pdf_services import PDFServices
from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
from adobe.pdfservices.operation.pdfjobs.jobs.extract_pdf_job import ExtractPDFJob
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_element_type import ExtractElementType
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_pdf_params import ExtractPDFParams
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_renditions_element_type import \
    ExtractRenditionsElementType
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.table_structure_type import TableStructureType
from adobe.pdfservices.operation.pdfjobs.result.extract_pdf_result import ExtractPDFResult
import zipfile
from datetime import datetime
from adobe.pdfservices.operation.pdfjobs.jobs.split_pdf_job import SplitPDFJob
from adobe.pdfservices.operation.pdfjobs.params.split_pdf.split_pdf_params import SplitPDFParams
from adobe.pdfservices.operation.pdfjobs.result.split_pdf_result import SplitPDFResult
import PyPDF2


# Initialize the logger
logging.basicConfig(level=logging.INFO)

def get_pdf_pages(file):
    with open(file, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        number_of_pages = len(reader.pages)
    return number_of_pages


class ExtractTextTableInfoWithTableStructureFromPDF:
    def __init__(self,file='src/resources/extractPdfInput.pdf',output_dir='output/',max_pages = 50,output_dir_split = 'split/'):
        self.output_dir = output_dir

        # Get the folder where the file is located
        folder_path = os.path.dirname(file)
        # Get the specific folder name from the path
        self.file_type = os.path.basename(folder_path)
        self.name_file = os.path.join(self.output_dir,os.path.splitext(os.path.basename(file))[0]+'.zip')
        self.success = False
        pages = get_pdf_pages(file)
        if pages > max_pages:
            # print('TODO SPLIT')
            self.split = True
            self.expected_pages = pages/max_pages
        else:
            self.split = False
        self.max_pages = max_pages
        self.output_dir_split = output_dir_split
        self.file = file

    def ExtractDocument(self):
        document_list = []
        if self.split:
            ## DO SPLIT
            # print('Doing split')
            split = SplitPDFByNumberOfPages(self.file,self.output_dir_split,page_count=self.max_pages,expected_splits=self.expected_pages)
            splitted = split.split_pdf()
            document_list.extend(splitted)

        else:
            document_list.append(self.file)

        try:
            parts = len(document_list)
            for i,doc in enumerate(document_list):
                file = open(doc, 'rb')
                input_stream = file.read()
                file.close()
                # print(f'Extracting document {file}')
                # Initial setup, create credentials instance
                credentials = ServicePrincipalCredentials(
                    client_id=os.getenv('PDF_SERVICES_CLIENT_ID'),
                    client_secret=os.getenv('PDF_SERVICES_CLIENT_SECRET')
                )

                # # Creates a PDF Services instance
                pdf_services = PDFServices(credentials=credentials)

                # Creates an asset(s) from source file(s) and upload
                input_asset = pdf_services.upload(input_stream=input_stream, mime_type=PDFServicesMediaType.PDF)

                # Create parameters for the job
                extract_pdf_params = ExtractPDFParams(
                    elements_to_extract=[ExtractElementType.TEXT, ExtractElementType.TABLES],
                    elements_to_extract_renditions=[ExtractRenditionsElementType.TABLES],
                    table_structure_type=TableStructureType.CSV,
                )

                # Creates a new job instance
                extract_pdf_job = ExtractPDFJob(input_asset=input_asset, extract_pdf_params=extract_pdf_params)

                # Submit the job and gets the job result
                location = pdf_services.submit(extract_pdf_job)
                pdf_services_response = pdf_services.get_job_result(location, ExtractPDFResult)

                # Get content from the resulting asset(s)
                result_asset: CloudAsset = pdf_services_response.get_result().get_resource()
                stream_asset: StreamAsset = pdf_services.get_content(result_asset)

                # Creates an output stream and copy stream asset's content to it
                output_file_path = self.create_output_file_path(self)
                with open(output_file_path, "wb") as file:
                    file.write(stream_asset.get_input_stream())

                self.extract_and_delete_zip(i,parts)

            self.success = True

            return self.success
        
        except (ServiceApiException, ServiceUsageException, SdkException) as e:
            logging.exception(f'Exception encountered while executing operation: {e}')
            return self.success

    # Generates a string containing a directory structure and file name for the output file
    @staticmethod
    def create_output_file_path(self) -> str:
        # os.makedirs(self.output_dir, exist_ok=True)
        return self.name_file

    def extract_and_delete_zip(self,index,parts):
        """
        Extracts the contents of a zip file into a directory named after the zip file
        (without the .zip extension) in the same location and then deletes the zip file.

        Parameters:
        zip_file_path (str): The path to the zip file.
        """
        try:
            
            # Determine the directory where the zip file is located
            base_directory = os.path.dirname(self.name_file)
            
            # Determine the base name of the zip file (without the .zip extension)
            zip_file_base_name = os.path.splitext(os.path.basename(self.name_file))[0]
            
            # Create a directory with the base name of the zip file
            if parts == 1:
                extract_to_directory = os.path.join(base_directory, zip_file_base_name)
            else:
                extract_to_directory = os.path.join(base_directory, zip_file_base_name,f'part{index}')
            
            os.makedirs(extract_to_directory, exist_ok=True)    

            # Open the ZIP file and extract all contents
            with zipfile.ZipFile(self.name_file, 'r') as zip_ref:
                zip_ref.extractall(extract_to_directory)

            print(f"Extracted all files to {extract_to_directory}")

            # Delete the zip file
            os.remove(self.name_file)
            print(f"Deleted zip file {self.name_file}")
            
        except Exception as e:
            print(f"An error occurred: {e}")

def handle_conflict(file_path):
    base, ext = os.path.splitext(file_path)
    counter = 1
    new_file_path = f"{base}_{counter}{ext}"
    while os.path.exists(new_file_path):
        counter += 1
        new_file_path = f"{base}_{counter}{ext}"
    return new_file_path


class SplitPDFByNumberOfPages:
    def __init__(self,file='src/resources/splitPDFInput.pdf',output_dir='split/',page_count = 50,expected_splits=2):
        # Get the file name
        self.file_name = os.path.splitext(os.path.basename(file))[0]
        file_type_name = os.path.basename(os.path.dirname(file))
        # Get the folder where the file is located
        self.output_dir = output_dir
        self.name_file = os.path.join(self.output_dir,file_type_name,self.file_name)
        # print(f'filename: {self.file_name}\nname file: {self.name_file}')
        self.file = file
        self.page_count = page_count
        self.expected_files = expected_splits

    def split_pdf(self):

        if os.path.exists(self.name_file):
            files = os.listdir(self.name_file)
            files = [os.path.join(self.name_file,i) for i in files]
            if len(files) >= self.expected_files:
                return files
            
        try:

            file_read = open(self.file, 'rb')
            input_stream = file_read.read()
            file_read.close()

            # Initial setup, create credentials instance
            credentials = ServicePrincipalCredentials(
                client_id=os.getenv('PDF_SERVICES_CLIENT_ID'),
                client_secret=os.getenv('PDF_SERVICES_CLIENT_SECRET')
            )
            # Creates a PDF Services instance
            pdf_services = PDFServices(credentials=credentials)

            # # Creates an asset(s) from source file(s) and upload
            input_asset = pdf_services.upload(input_stream=input_stream,
                                              mime_type=PDFServicesMediaType.PDF)

            # # Create parameters for the job
            split_pdf_params = SplitPDFParams(page_count=self.page_count)

            # # Creates a new job instance
            split_pdf_job = SplitPDFJob(input_asset, split_pdf_params)

            # # Submit the job and gets the job result
            location = pdf_services.submit(split_pdf_job)
            pdf_services_response = pdf_services.get_job_result(location, SplitPDFResult)

            # # Get content from the resulting asset(s)
            result_assets = pdf_services_response.get_result().get_assets()

            # Creates an output stream and copy stream asset's content to it
            output_file_path = self.create_output_file_path(self.name_file)
            print(f'File path: {output_file_path}')
            files_splitted = []
            for i, result_asset in enumerate(result_assets):
                stream_asset: StreamAsset = pdf_services.get_content(result_asset)
                file_dest = os.path.join(output_file_path,(self.file_name+f'_part_{i}.pdf'))
                with open(file_dest, "wb") as file:
                    file.write(stream_asset.get_input_stream())
                files_splitted.append(file_dest)
            
            return files_splitted

        except (ServiceApiException, ServiceUsageException, SdkException) as e:
            logging.exception(f'Exception encountered while executing operation: {e}')

            return []

    @staticmethod
    def create_output_file_path(filename) -> str:
        os.makedirs(filename, exist_ok=True)
        return filename


def ExtractTablesDirectory(file_directory,target_directory,target_split,extension = '.pdf',max_pages = 50):
    files = os.listdir(file_directory)
    filtered_files = [file for file in files if file.endswith(extension)]
    base_folder_name = os.path.basename(file_directory.rstrip('/\\'))
    
    # Create the base folder in the target_directory
    target_base_folder = os.path.join(target_directory, base_folder_name)
    os.makedirs(target_base_folder, exist_ok=True)
    count_pr = 0
    count_missing = 0
    count_done = 0
    count_total = 0
    error_files = []

    for i,file in enumerate(filtered_files):
        count_total+=1
        # print(f'Extracting {file} - {i+1}/{len(filtered_files)}...')

        file_path = os.path.join(file_directory,file)

        target_base_folder_name = os.path.join(target_base_folder,os.path.splitext(file)[0])

        # print(f'file extract: {file_path} target base folder: {target_base_folder_name}')

        if os.path.exists(target_base_folder_name):
            # print(f'File: {file} already processed\n')
            count_done+=1
        else:
            # print(f'Processing: {file}\n')
            extract_class = ExtractTextTableInfoWithTableStructureFromPDF(file= file_path,
                                                                          output_dir= target_base_folder,
                                                                          output_dir_split= target_split,
                                                                          max_pages=max_pages)
            extract_class.ExtractDocument()
            if extract_class.success == True:
                count_pr+=1
            else:
                error_files.append(file)
                count_missing+=1

    # print(f'\nTotal:         {count_total}\nProcessed:     {count_pr}\nDone:         {count_done}\nMissing/Error: {count_missing}')
    return [count_total,count_pr,count_done,count_missing],error_files
