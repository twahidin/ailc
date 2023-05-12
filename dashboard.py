
import streamlit as st
import pandas as pd
import pymongo
import certifi
import pypdf
import io
import os
import gridfs
import shutil
import csv
import time
from pypdf import PdfReader
from st_aggrid import AgGrid
import configparser
import ast
from bson import ObjectId
from datetime import datetime
from data_process import generate_resource
from google.cloud import storage

config = configparser.ConfigParser()
config.read('config.ini')
db_host = st.secrets["db_host"]
#db_host = config['constants']['db_host']
db_client = config['constants']['db_client']
client = pymongo.MongoClient(db_host, tlsCAFile=certifi.where())

db = client[db_client]
fs = gridfs.GridFS(db)
data_collection = db[config['constants']['sd']]
user_info_collection = db[config['constants']['ui']]
doc_collection = db[config['constants']['dd']]

subjects_str = config.get('menu_lists', 'subjects')
shared_access_str = config.get('menu_lists', 'shared_access')
subjects = ast.literal_eval(subjects_str)
shared_access = ast.literal_eval(shared_access_str)
bucket_name = config['constants']['bucket_name']
google_application_credentials_json = st.secrets["GOOGLE_APPLICATION_CREDENTIALS_JSON"]
# Save the JSON key to a temporary file
with open("temp_key.json", "w") as f:
	f.write(google_application_credentials_json)

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable to the temporary file path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "temp_key.json"
project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or st.secrets["GOOGLE_CLOUD_PROJECT"]
os.remove("temp_key.json")

def download_csv(): #downloading of conversational data
	user_info = user_info_collection.find_one({"tch_code": st.session_state.vta_code})
	#st.write(user_info)
	if user_info:
		# Step 3: Get all the codes from vta_code1 to vta_code45
		codes = [user_info[f"vta_code{i}"] for i in range(1, 46) if f"vta_code{i}" in user_info]
		#st.write(codes)

		# Step 4: Query the data_collection to get all the documents containing any of the codes
		documents = data_collection.find({"vta_code": {"$in": codes}})
		#st.write(documents)
		# Write the documents to a CSV file
		filename = 'all_records.csv'
		with open(filename, "w", newline="") as csvfile:
			fieldnames = documents[0].keys()
			writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

			writer.writeheader()
			for document in documents:
				writer.writerow(document)

		# Check if the file was written successfully
		try:
			with open(filename, "r") as file:
				data_csv = file.read()
		except:
			st.error("Failed to export records, please try again")
		else:
			st.success("File is ready for downloading")
			st.download_button(
				label="Download class data as CSV",
				data=data_csv,
				file_name=filename,
				mime='text/csv',
			)


def upload_directory(bucket_name, source_directory, destination_blob_prefix):
	storage_client = storage.Client(project=project_id)
	bucket = storage_client.get_bucket(bucket_name)
	
	for root, _, files in os.walk(source_directory):
		for file in files:
			local_file_path = os.path.join(root, file)
			blob_name = os.path.join(destination_blob_prefix, os.path.relpath(local_file_path, source_directory))
			blob = bucket.blob(blob_name)
			blob.upload_from_filename(local_file_path)

def extract_first_n_words(text, n=200):
	words = text.split()
	first_n_words = words[:n]
	return " ".join(first_n_words)

def extract_text_from_pdf(uploaded_file):
	"""
	Extract text from the first page of an uploaded PDF file.
	:param uploaded_file: Streamlit UploadedFile object
	:return: Extracted text as a single string
	"""
	# Store the PDF file in memory
	in_memory_pdf = io.BytesIO(uploaded_file.getvalue())
	
	# Initialize the PDF reader
	pdf_reader = PdfReader(in_memory_pdf)
	
	# Extract text from the first page
	first_page = pdf_reader.pages[0]
	pdf_text = first_page.extract_text()
	
	return pdf_text

# Delete a document by ID along with its corresponding PDF file
def delete_document_by_id_with_file(document_id):
	try:
		# Query the document by its ID
		document = doc_collection.find_one({"_id": ObjectId(document_id)})

		if document is not None:
			# Check if the tch_code of the document matches the st.session_state.vta_code
			if document["tch_code"] == st.session_state.vta_code:
				# Delete the corresponding PDF file from GridFS
				file_id = document.get('file_id')
				if file_id:
					fs.delete(file_id)

				# Delete the document from MongoDB
				result = doc_collection.delete_one({"_id": ObjectId(document_id)})
			   
				return result.deleted_count > 0
			else:
				st.error("You can only delete your own resources.")
				return False
		else:
			st.error("Document not found.")
			return False
	except Exception as e:
		st.write(f"Error: {e}")
		return False


def upload_pdf(file):
	return fs.put(file, filename=file.name, contentType="application/pdf")

def is_document_duplicate(new_content, tch_code, doc_collection):
	for doc in doc_collection.find({"tch_code": tch_code}):
		if doc['content'] == new_content:
			return True
	return False



def upload_resource():

	uploaded_file = st.file_uploader("Choose a file to upload as a resource (Only your first 10 uploaded articles will be used)")
	if uploaded_file is not None:
		txt = extract_text_from_pdf(uploaded_file)

	
		with st.form("Tool Settings"):
			st.write("**:blue[Please fill in the details as accurately as possible so that the AI agent could locate your sources]**")
			#st.warning("""**:red[(Note that you can only upload up to 10 resources)]**""")
			 

			# Create a text input for subject
			subject = st.selectbox("Subject", options=subjects)

			# Create a text input for topic
			topic = st.text_input("Topic: (50 characters)", max_chars=50)
			# Create a text input for source
			source = st.text_input("Document Source", max_chars=50)

			# Create a text input for topic description
			hyperlinks = st.text_input("Enter a hyperlink (YouTube / Webpage ): (Max 250 characters)", max_chars=200)

			# Create a text area input
			#st.write(":blue[Please confirmed that this is the resource you wish to upload: ( First 200 words shown)]")
			st.text_area(":blue[Please confirmed that this is the resource you wish to upload: ( First 200 words shown)]",value=extract_first_n_words(txt, n=200), height=300)
			#stx.scrollableTextbox(txt, height=500, border=True)
			#st.write(txt)
			st.write("#")

			# Add a multiselect input for class access
			class_access = st.selectbox("Select resource access (Note: Shared resource allow sharing across schools):", options=shared_access)


			# Create a submit button
			submit_button = st.form_submit_button("Submit")

			# Handle form submission
			if submit_button:
				if not source or not topic:
					st.warning("Source and Topic cannot be left empty. Please fill in the required fields.")
					return
				content = txt
				tch_code = st.session_state.vta_code

				# Check if the document is a duplicate for the same teacher
				if is_document_duplicate(content, tch_code, doc_collection):
					st.warning("This document has already been uploaded by you. Please upload a different document.")
					return

				#cache_api()
				content = txt
				tch_code = st.session_state.vta_code
				file_id = upload_pdf(uploaded_file)
				#st.success(f"File uploaded with ID: {file_id}")
				
				# Create a dictionary to store the document data
				document_data = {
					"subject": subject,
					"topic": topic,
					"source": source,
					"hyperlinks": hyperlinks,
					"content": content,
					"tch_code": tch_code,
					"class_access": class_access,
					"file_id": file_id
				}

				# Insert the document data into the MongoDB collection
				doc_collection.insert_one(document_data)
				st.success("Document uploaded successfully!")
				st.info(generate_resource(st.session_state.vta_code, st.session_state.admin_key))
				now = datetime.now()
				db_description = "Relevant resources and materials shared by your class teacher to enhance your learning experience and deepen your understanding of the subject."
				formatted_now = now.strftime("%d/%m/%Y %H:%M:%S")
				user_info_collection.update_one(
					{"tch_code": st.session_state.vta_code},
					{"$set": {"db_last_created": formatted_now, "db_description": db_description, "db_subject": subject}}
				)
				st.session_state.generate_key = False
				st.success("Database generated successfully. Please click dashboard to refresh")
				time.sleep(3)
				directory_path = os.path.join(os.getcwd(), st.session_state.vta_code)
				if os.path.exists(directory_path):
					upload_directory(bucket_name, directory_path, st.session_state.vta_code)




def include_document_for_class_resource(tch_code, doc_collection):
	with st.form(key="include_resource_form"):
	# Ask the teacher to input a document ID
		document_id = st.text_input("Please enter the Document ID you would like to include for the class resources: ")
		# Create a text input for subject
		subject = st.selectbox("Enter the subject for your collated database:", options=subjects)
		submit_button = st.form_submit_button("Include Resource")
		

		# Handle form submission
		if submit_button:
		# Check if the document is available and if its class_access says 'Shared resource'
			# Search for the document in the doc_collection
			if not document_id:
				st.warning("Document ID cannot be left empty. Please fill in the required fields.")
				return
			document = doc_collection.find_one({"_id": ObjectId(document_id)})
			if document and document.get("class_access") == "Shared resource":

				# Update or create the tch_sharing_resource field
				if "tch_sharing_resource" not in document:
					document["tch_sharing_resource"] = [tch_code]
				else:
					document["tch_sharing_resource"].append(tch_code)

				# Update the doc_collection with the modified document
				doc_collection.update_one({"_id": document_id}, {"$set": {"tch_sharing_resource": document["tch_sharing_resource"]}})

				# Copy the entire document without the tch_sharing_resource field
				copied_document = document.copy()
				copied_document.pop("tch_sharing_resource")
				copied_document.pop("_id")  # Remove the _id field

				# Update the tch_code and class_access fields
				copied_document["tch_code"] = tch_code
				copied_document["class_access"] = "Copied Resource"

				# Insert the copied document into the doc_collection
				doc_collection.insert_one(copied_document)

				st.success("Document successfully included for the class resources.")
				st.info(generate_resource(st.session_state.vta_code, st.session_state.admin_key))
				now = datetime.now()
				db_description = "Relevant resources and materials shared by your class teacher to enhance your learning experience and deepen your understanding of the subject."
				formatted_now = now.strftime("%d/%m/%Y %H:%M:%S")
				user_info_collection.update_one(
					{"tch_code": st.session_state.vta_code},
					{"$set": {"db_last_created": formatted_now, "db_description": db_description, "db_subject": subject}}
				)
				st.session_state.generate_key = False
				st.success("Database generated successfully. Please click dashboard to refresh")
				time.sleep(3)
				directory_path = os.path.join(os.getcwd(), st.session_state.vta_code)
				if os.path.exists(directory_path):
					upload_directory(bucket_name, directory_path, st.session_state.vta_code)
			else:
				st.error("Document not found or not a shared resource.")



def check_directory_exists():
	if 'admin_key' not in st.session_state:
		st.session_state.admin_key = False 

	directory_path = os.path.join(os.getcwd(), st.session_state.vta_code)

	if os.path.exists(directory_path):
		user_info = user_info_collection.find_one({"tch_code": st.session_state.vta_code})

		if user_info and "db_last_created" in user_info:
			# st.success(f"Document database exists, last created on {user_info['db_last_created']}. Please see the information below:")
			if "db_subject" in user_info and "db_description" in user_info:
			   st.info(f"""Document database exists, last created on {user_info['db_last_created']}\n\nDatabase Subject: {user_info["db_subject"]}\n\nDatabase Description: {user_info["db_description"]}""")
			   

			else:
				st.warning("Database exists, but subject and description not found in the database")

	else:
		st.error("No document database is created. Please upload a document or include a new doucment to generate a new database")
		documents = list(doc_collection.find({"tch_code": st.session_state.vta_code}, {'file_id': 0}))
		if len(documents) > 0:
			with st.form(key="generate_db"):
				st.write("You have existing documents, please generate a database for your students to access your curated resources")
				db_subject = st.selectbox("Select the subject", options=subjects)
				generate_button = st.form_submit_button("Generate Database")
				if generate_button:
					db_description = f"Relevant resources and materials shared by your class teacher to enhance your learning experience and deepen your understanding of the {db_subject}."
					now = datetime.now()
					formatted_now = now.strftime("%d/%m/%Y %H:%M:%S")
					st.info(generate_resource(st.session_state.vta_code, st.session_state.admin_key))
					user_info_collection.update_one(
						{"tch_code": st.session_state.vta_code},
						{"$set": {"db_last_created": formatted_now, "db_description": db_description, "db_subject": db_subject}}
					)
					directory_path = os.path.join(os.getcwd(), st.session_state.vta_code)
					if os.path.exists(directory_path):
						upload_directory(bucket_name, directory_path, st.session_state.vta_code)


def delete_directory(directory):
	try:
		shutil.rmtree(directory)
		st.success(f'Successfully deleted the directory "{directory}".')
	except FileNotFoundError:
		st.error(f'Directory "{directory}" not found.')

def delete_blob(bucket_name, blob_name):
	storage_client = storage.Client()
	bucket = storage_client.get_bucket(bucket_name)
	blob = bucket.blob(blob_name)
	blob.delete()

def dashboard():
	if 'generate_key' not in st.session_state:
		st.session_state.generate_key = False
	if 'zero_docs' not in st.session_state:
		st.session_state.zero_docs = False  

	col1, col2 = st.columns([2,2])
	with col1:

		st.info("Current student and interaction data (Filter the data and right click to download in CSV or Excel format)")
		try:
			codes = st.session_state.codes_key + [st.session_state.vta_code]
			#st.write(codes)
			documents = list(data_collection.find({"vta_code": {"$in": codes }}, {'_id': 0}))
			#documents = list(doc_collection.find({"class_access": "Shared resource"}, {'_id': 0}))
			df = pd.DataFrame(documents)
			#aggrid_interactive_table(df=df)
			AgGrid(df, height='400px', key="data")
		except Exception as e:
			st.write(f"Error: {e}")
			return False
		download_csv()

	with col2:
		# ["Shared resource","School resource", "Class resource"]

		st.info("Resources available that is available for sharing")
		try:
			# Retrieve documents including the _id column
			documents = list(doc_collection.find({"class_access": "Shared resource"}, {'file_id': 0}))
			if documents:
				# Create a DataFrame from the documents
				r_df = pd.DataFrame(documents)

				# Convert the _id column to a string
				r_df['_id'] = r_df['_id'].astype(str)

				# Rename the _id column to 'Document ID'
				r_df = r_df.rename(columns={'_id': 'Document ID'})
				AgGrid(r_df, height='400px', key="global_resources")
			else:
				df = pd.DataFrame(documents)
				#aggrid_interactive_table(df=df)
				AgGrid(df, height='400px', key="no_elements")
		except Exception as e:
			st.write(f"Error: {e}")
			return False

	
	col1, col2 = st.columns([2,2])
	with col1:
		st.warning("Add shared documents to class resource")
		include_document_for_class_resource(st.session_state.vta_code, doc_collection)
		st.warning("Upload PDF resources")
		upload_resource()

	with col2:
		st.warning(f"Resources uploaded or included by teacher : {st.session_state.vta_code}")
		try:
			# Retrieve documents including the _id column
			 # Retrieve documents including the _id column
			documents = list(doc_collection.find({"tch_code": st.session_state.vta_code}, {'file_id': 0}))
			if len(documents) == 0:
				st.info("No documents found.")
				st.session_state.zero_docs = True
			else:
				st.session_state.zero_docs = False
				# Create a DataFrame from the documents
				r_df = pd.DataFrame(documents)

				# Convert the _id column to a string
				r_df['_id'] = r_df['_id'].astype(str)

				# Rename the _id column to 'Document ID'
				r_df = r_df.rename(columns={'_id': 'Document ID'})
				AgGrid(r_df, height='200px',key="class_resources")
		except Exception as e:
			st.write(f"Error: {e}")
			return False
		check_directory_exists()
		#documents = list(doc_collection.find({"tch_code": st.session_state.vta_code}, {'file_id': 0}))
		if st.session_state.zero_docs != True:
			st.warning("Delete a document resource")
			with st.form(key="delete_form"):
				# Create a text input for the document ID
				
				document_id = st.text_input("Enter the Document ID (copy paste)to delete your resource:")
				st.error("You are about to delete the above resource! To proceed please click the delete button")
				# Create a submit button
				submit_button = st.form_submit_button("Delete")

				# Handle form submission
				if submit_button:
					if delete_document_by_id_with_file(document_id):
						st.success(f"Document with ID '{document_id}' deleted successfully.(Click Dashboard to refresh)")
						st.session_state.generate_key = True
					else:
						st.warning(f"Failed to delete document with ID '{document_id}'.")
		if st.session_state.generate_key:
			documents = list(doc_collection.find({"tch_code": st.session_state.vta_code}, {'file_id': 0}))
			if len(documents) > 0:
				with st.form(key="generate_db"):
					st.warning("Changes in your resources, please generate a new database for your students to access your existing resources")
					db_subject = st.selectbox("Select the subject", options=subjects)
					generate_button = st.form_submit_button("Generate Database")
					if generate_button:
						db_description = f"Relevant resources and materials shared by your class teacher to enhance your learning experience and deepen your understanding of the {db_subject}."
						now = datetime.now()
						formatted_now = now.strftime("%d/%m/%Y %H:%M:%S")
						st.info(generate_resource(st.session_state.vta_code, st.session_state.admin_key))
						user_info_collection.update_one(
							{"tch_code": st.session_state.vta_code},
							{"$set": {"db_last_created": formatted_now, "db_description": db_description, "db_subject": db_subject}}
						)
						st.session_state.generate_key = False
						time.sleep(3)
						directory_path = os.path.join(os.getcwd(), st.session_state.vta_code)
						if os.path.exists(directory_path):
							upload_directory(bucket_name, directory_path, st.session_state.vta_code)
			else:
				with st.form(key="delete_db"):
					st.write("You do not have any more documents in your resources, would you like to delete your existing directory")
					delete_confirmation = st.checkbox(f"Yes I want to delete the {st.session_state.vta_code} database")
					submit_button = st.form_submit_button("Submit")

					if submit_button and delete_confirmation:
						delete_directory(st.session_state.vta_code)
						delete_blob(bucket_name, st.session_state.vta_code)
						st.session_state.generate_key = False
						st.session_state.zero_docs = True


	






