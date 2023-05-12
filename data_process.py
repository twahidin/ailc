import streamlit as st
import pymongo
import tempfile
import certifi
import openai
import os
import re
import gridfs
from google.oauth2 import service_account
#from googleapiclient.discovery import build
#from googleapiclient.errors import HttpError
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PagedPDFSplitter
from langchain.document_loaders import GoogleDriveLoader
from pathlib import Path
import configparser
import ast
from bson import ObjectId

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
# google_drive_credentials_json = st.secrets["GOOGLE_DRIVE_CREDENTIALS_JSON"]
# folder_id = st.secrets["FOLDER_ID"]
# # Save the JSON key to a temporary file
# temp_credentials_path = "drive_key.json"
# with open(temp_credentials_path, "w") as f:
# 	f.write(google_drive_credentials_json)

# hylk = "https://drive.google.com/drive/folders/1gC6GQLgcuoHzwDXtEGzTwvzCz0YsYuwg"
# Set the environment variable to use the temporary file
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_credentials_path


# def generate_drive_data():
# 	openai.api_key  = st.session_state.api_key
# 	os.environ["OPENAI_API_KEY"] = st.session_state.api_key
# 	os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 	try:
# 		holy_grail = "holy_grail_drive"
# 		st.write("here")
# 		loader = GoogleDriveLoader(
# 		folder_id=folder_id,
# 		credentials_path=temp_credentials_path,
# 		token_path="token.json",
# 		# Optional: configure whether to recursively fetch files from subfolders. Defaults to False.
# 		recursive=True
# 		)
# 		documents = loader.load()
# 		#st.write(documents)
# 		text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# 		docs = text_splitter.split_documents(documents)
# 		metadata = {"source": "HOLY GRAIL", "topic":"Notes", "url": hylk, "teacher": "Shared Notes"}
# 		#metadata = {"source": source}
# 		#st.write(docs)

# 		for doc in docs:
# 			doc.metadata.update(metadata)

# 		embeddings = OpenAIEmbeddings()
# 		db = FAISS.from_documents(docs, embeddings)
# 		db.save_local(holy_grail)
# 		os.remove(temp_credentials_path)
# 		return f"VectorStore DB Created for {st.session_state.vta_code}"
		
# 	except Exception as e:
# 		st.write(f"Error: {e}")
# 		return False

def generate_resource(vta, flag):
	openai.api_key  = st.session_state.api_key
	vta_code = st.session_state.vta_code
	os.environ["OPENAI_API_KEY"] = st.session_state.api_key
	os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
	
	try:
		documents = find_documents(vta, flag)
		if documents == False:
			return False
		else:
			ans = generate_data(documents)
			return ans
	except Exception as e:
		st.write(f"Error: {e}")
		return False


#@st.cache_resource
def find_documents(vta_code, admin_flag):
	# Search for documents with matching tch_code or class_access
	query = {
		"$or": [
			{"tch_code": vta_code},
			#source finder function to add all the vta codes of teachers who curated the resource
			#need to create another function in dashboard - find all the resources matching your subject and display all the codes selectinbox maximum ten
			#need to have a tick to say my resource is shareable
			#need to mention that there is a limit of 10 uploaded resources per subject - internal cap plus 10 
			#{"class_access": vta_code}, #To add, all the shared resources 
			#{"tch_code": "joe"}, #default resources
		]
	}
	documents = doc_collection.find(query)
	#st.write(documents)

	# Limit number of documents to 10 if admin_flag is false
	if not admin_flag:
		documents = documents.limit(10)

	results = []
	for document in documents:
		result = {
			"subject": document.get("subject"),
			"topic": document.get("topic"),
			"source": document.get("source"),
			"hyperlinks": document.get("hyperlinks"),
			#"content": document.get("content"),
			"tch_code": document.get("tch_code"),
			"file_id": document.get("file_id")
		}
		#st.write(result)
		results.append(result)
	# If no documents are found, return False
	if not results:
		return False
	return results


def download_pdf(file_id):
	#st.write(file_id)
	file_data = fs.get(file_id)
	return file_data.read(), file_data.filename

def format_string(input_string):
	# convert to lowercase
	output_string = input_string.lower()
	
	# remove special characters
	output_string = re.sub(r'[^\w\s]', '', output_string)
	
	# join spaces with underscore
	output_string = re.sub(r'\s+', '_', output_string)
	
	return output_string


def delete_files_from_mongodb(tch_code):
	fs = gridfs.GridFS(db)

	# Find all files with the specified tch_code
	files = fs.find({"tch_code": tch_code})

	# Delete each file
	for file in files:
		fs.delete(file._id)


def save_files_to_mongodb_recursive(temp_dir, tch_code):
	fs = gridfs.GridFS(db)

	def save_files_recursive(directory, tch_code, root):
		# Iterate through all files and directories in the given directory
		for item in os.listdir(directory):
			item_path = os.path.join(directory, item)
			if os.path.isfile(item_path):
				with open(item_path, "rb") as f:
					# Save the file in MongoDB with the tch_code and relative_path as metadata
					relative_path = os.path.relpath(item_path, root)
					fs.put(f, filename=item, tch_code=tch_code, relative_path=relative_path)
			elif os.path.isdir(item_path):
				# Recursively save files in subdirectories
				save_files_recursive(item_path, tch_code, root)

	save_files_recursive(temp_dir, tch_code, temp_dir)


def save_files_to_mongodb(temp_dir, tch_code):
	fs = gridfs.GridFS(db)

	# Iterate through all files in the temp_dir
	for file in os.listdir(temp_dir):
		file_path = os.path.join(temp_dir, file)
		with open(file_path, "rb") as f:
			# Save the file in MongoDB with the tch_code as metadata
			fs.put(f, filename=file, tch_code=tch_code)

def generate_data(documents):
	try:
		full_docs = []
		all_metadatas = []
		source = ""
		topic = ""
		hyperlinks = ""

		for document in documents:
			subject = document.get('subject')
			topic = document.get('topic')
			hyperlinks = document.get('hyperlinks')
			source = document.get('source')
			tch_code = document.get('tch_code')
			file_id = document.get('file_id')
			#topic = format_string(topic)
			pdf_data, filename = download_pdf(file_id)

			with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
				tmp_file.write(pdf_data)
				tmp_file.flush()

			docs = split_meta_docs(tmp_file.name, source, topic, hyperlinks, tch_code)
			full_docs.extend(docs)  # Extend the full_docs list with the new docs

		embeddings = OpenAIEmbeddings()
		db = FAISS.from_documents(full_docs, embeddings)
		db.save_local(st.session_state.vta_code)
		return f"VectorStore DB Created for {st.session_state.vta_code}"
	
	except Exception as e:
		st.write(f"Error: {e}")
		return False



def split_meta_docs(file, source, topic, hylk, tch_code):
#def split_meta_docs(file, source, tch_code):
	loader = PagedPDFSplitter(file)
	documents = loader.load()
	text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
	docs = text_splitter.split_documents(documents)
	metadata = {"source": source, "topic":topic, "url": hylk, "teacher": tch_code}
	#metadata = {"source": source}

	for doc in docs:
		doc.metadata.update(metadata)
	return docs