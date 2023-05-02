
#Check completed - need to upload pdf to mongodb
import streamlit as st
import pandas as pd
import csv
import pymongo
import certifi
from st_aggrid import AgGrid
import configparser
config = configparser.ConfigParser()
config.read('config.ini')
db_host = st.secrets["db_host"]
#db_host = config['constants']['db_host']
db_client = config['constants']['db_client']
client = pymongo.MongoClient(db_host, tlsCAFile=certifi.where())
db = client[db_client]
data_collection = db[config['constants']['sd']]
user_info_collection = db[config['constants']['ui']]

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

def dashboard_da():

	st.info("Current student and interaction data (Filter the data and right click to download in CSV or Excel format)")
	try:
		codes = st.session_state.codes_key + [st.session_state.vta_code]
		#st.write(codes)
		documents = list(data_collection.find({"vta_code": {"$in": codes }}, {'_id': 0}))
		df = pd.DataFrame(documents)
		#aggrid_interactive_table(df=df)
		AgGrid(df, height='400px')
	except Exception as e:
		st.write(f"Error: {e}")
		return False
		
	download_csv()
