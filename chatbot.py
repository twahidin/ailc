import streamlit as st
from PIL import Image
import pymongo
import certifi
import os
import regex
from datetime import datetime
import re
import time
import openai
from dialogic_agent import (
							ailc_agent_serp, 
							ailc_agent_bing, 
							ailc_resources_bot,
							ailc_resource_agent, 
							)
from metacog import (
					metacog_bot, 
					metacog_resources_bot, 
					metacog_agent,
					summarizer
					)

from sourcefinder import sourcefinder_agent, sourcefinder_bot
from chergpt import chergpt_agent, chergpt_bot, chergpt_agent_4
from mapping import generate_mind_map
from langchain.embeddings.openai import OpenAIEmbeddings
from streamlit_extras.stoggle import stoggle
import json
import configparser
import ast
from langchain.vectorstores import FAISS
from google.cloud import storage
config = configparser.ConfigParser()
config.read('config.ini')
db_host = st.secrets["db_host"]
#db_host = config['constants']['db_host']
db_client = config['constants']['db_client']
client = pymongo.MongoClient(db_host, tlsCAFile=certifi.where())
db = client[db_client]
data_collection = db[config['constants']['sd']]
user_info_collection = db[config['constants']['ui']]
#default conversation table 
CB = config['constants']['CB']
CL = config['constants']['CL']
bucket_name = config['constants']['bucket_name']
google_application_credentials_json = st.secrets["GOOGLE_APPLICATION_CREDENTIALS_JSON"]
# Save the JSON key to a temporary file
with open("temp_key.json", "w") as f:
	f.write(google_application_credentials_json)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "temp_key.json"
project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or st.secrets["GOOGLE_CLOUD_PROJECT"]

#chergpt bot and agent
c_agent = "chergpt_agent"
c4_agent = "chergpt_agent_4"
cb_bot = "chergpt_bot"
#ailc bot and agent
ag_agent = "ailc_agent_google"
ab_agent = "ailc_agent_bing"
abm_agent = "ailc_agent_bing_metacog"
ar_agent = "ailc_resource_agent"
arm_agent = "ailc_resource_agent_metacog"
ar_bot = "ailc_resource_bot"
arm_bot = "ailc_resource_bot_metacog"
#sourcefinder bot and agent
s_bot = "sourcefinder_bot"
s_agent = "sourcefinder_agent"
#metacog bot and agent
m_bot = "metacog_bot"
mr_bot = "metacog_resource_bot"
mr_agent = "metacog_resource_agent"


#---------------------session states declaration -------------------------

#---------------functions --------------------------------------------

def main_bot():

	openai.api_key  = st.session_state.api_key
	os.environ["OPENAI_API_KEY"] = st.session_state.api_key
	os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


	if "temp" not in st.session_state:
		st.session_state["temp"] = ""

	if 'document_exist' not in st.session_state:
		st.session_state.document_exist = load_documents()

	if 'data_source' not in st.session_state:
		st.session_state.data_source = None 

	if 'source_bot' not in st.session_state:
		st.session_state.source_bot = False 

	if 's_count' not in st.session_state:
		st.session_state.s_count = 1 

	if 'web_link' not in st.session_state:
		st.session_state.web_link = []

	if 'related_questions' not in st.session_state:
		st.session_state.related_questions = []

	if 'tool_use' not in st.session_state:
		st.session_state.tool_use = False

	if 'doc_tools_names' not in st.session_state:
		st.session_state.doc_tools_names = False

	if 'consolidate' not in st.session_state:
		st.session_state.consolidate = False

	if 'summary_points' not in st.session_state:
		st.session_state.summary_points = ''

	if 'consolidate_learning' not in st.session_state:
		st.session_state.consolidate_learning = "Click on Consolidate my learning"

	if 'metacog_flag' not in st.session_state:
		st.session_state.metacog_flag = False


	c1,c2 = st.columns([5,3])
	with c1:
		try:
			st.info("Dialogic Agent")
			chat_history()
			if st.session_state["temp"] != "":
				result = chat_bot(st.session_state["temp"])
				if result != False:
					question, answer = result
					now = datetime.now()
					if st.session_state.consolidate == False:
						data_collection.insert_one({"vta_code": st.session_state.vta_code, "function":CB,"question": question, "response": answer, "created_at": now.strftime("%d/%m/%Y %H:%M:%S")})
					else:
						data_collection.insert_one({"vta_code": st.session_state.vta_code, "function":CL,"question": question, "response": answer, "created_at": now.strftime("%d/%m/%Y %H:%M:%S")})
						st.session_state.consolidate = False
			if st.session_state.consolidate == True:
				st.session_state.consolidate_learning = summarizer()
			st.session_state["temp"] = st.text_input("Enter your question", key="text", on_change=clear_text)
			c3, c4, c5 = st.columns([2,2,2])
			with c3:
				if st.session_state.metacog_flag == True:
					if st.button("Consolidate my learning", on_click=consolidate):
						pass
			with c4:
				#if st.session_state.metacog_flag == True:
					#if st.button("Map my learning", on_click=map_learning):
				pass
		except Exception as e:
			st.error(e)
			return False
	with c2:
		try:
			st.info("Learning Log")
			if st.session_state.metacog_flag == True:
				st.write(st.session_state.consolidate_learning)
			else:
				st.write("Learning summary not available")
			st.success("Related Questions & Information")
			show_related_links()
			st.warning("Links and Information")
			show_links()
		except Exception as e:
			st.error(e)
			return False


def map_learning():
	st.session_state["temp"] = generate_mind_map()

def read_local_db_date(file_path):
	with open(file_path, "r") as f:
		date_str = f.read()
	return datetime.strptime(date_str, "%d/%m/%Y %H:%M:%S")

def write_local_db_date(file_path, date_str):
	with open(file_path, "w") as f:
		f.write(date_str)

def load_documents():
	try:
		user_info = user_info_collection.find_one({"tch_code": st.session_state.teacher_key})
		if user_info and "db_last_created" in user_info: 
			if "db_subject" in user_info and "db_description" in user_info:
				st.session_state.doc_tools_names = {"subject": user_info["db_subject"], "description": user_info["db_description"]}
				st.success('Teacher documents loaded.')

			directory_path = os.path.join(os.getcwd(), st.session_state.teacher_key)
			local_date_file_path = os.path.join(directory_path, "db_date.txt")
			remote_date_str = user_info.get("db_last_created")

			if os.path.exists(directory_path) and os.path.exists(local_date_file_path):
				local_date = read_local_db_date(local_date_file_path)
				remote_date = datetime.strptime(remote_date_str, "%d/%m/%Y %H:%M:%S")
				
				if remote_date > local_date:
					download_directory(bucket_name, st.session_state.teacher_key, directory_path)
					write_local_db_date(local_date_file_path, remote_date_str)

			else:
				if not os.path.exists(directory_path):
					os.makedirs(directory_path)

				download_directory(bucket_name, st.session_state.teacher_key, directory_path)
				write_local_db_date(local_date_file_path, remote_date_str)

			return True
		else:
			return False

	except Exception as e:
		st.error(e)
		return False


def show_links():
	if not st.session_state.web_link:
		st.write('No links found.')
	else:
		for i in range(0, len(st.session_state.web_link), 2):
			title = st.session_state.web_link[i]
			url = st.session_state.web_link[i+1]
			stoggle(
				f"""<span style="font-weight: normal; color: #187bcd;">{i//2+1}. {title}</span>""",
				f"""<a href="{url}" style="font-weight: normal; color: #187bcd;">{url}</a>""",
			)
			

def show_related_links():
	if not st.session_state.related_questions:
		st.write('No links found.')
		return
	elif st.session_state.source_bot == True:
		for i in range(0, len(st.session_state.related_questions), 2):
			if i+1 >= len(st.session_state.related_questions):
				st.session_state.related_questions = []
				st.write('No links found.')
				return
			title = st.session_state.related_questions[i]
			content = st.session_state.related_questions[i+1]
			stoggle(
				f"""<span style="font-weight: normal; color: #187bcd;">{i//2+1}. {title}</span>""",
				f"""<span style="font-weight: normal; color: black;">{content}</span>""",
			)
	else:
		for i, question_info in enumerate(st.session_state.related_questions):
			if not all(key in question_info for key in ['question', 'snippet', 'url']):
				st.session_state.related_questions = []
				st.write('No links found.')
				return
			question = question_info['question']
			snippet = question_info['snippet']
			url = question_info['url']
			stoggle(
				f"""<span style="font-weight: normal; color: #187bcd;">{i+1}. {question}</span>""",
				f"""<a href="{url}" style="font-weight: normal; color: #187bcd;">{snippet}</a>""",
			)


def download_directory(bucket_name, source_blob_prefix, destination_directory):
	storage_client = storage.Client()
	bucket = storage_client.get_bucket(bucket_name)
	blobs = bucket.list_blobs(prefix=source_blob_prefix)
	
	for blob in blobs:
		file_path = os.path.join(destination_directory, os.path.relpath(blob.name, source_blob_prefix))
		os.makedirs(os.path.dirname(file_path), exist_ok=True)
		blob.download_to_filename(file_path)


def clear_text():
	st.session_state["temp"] = st.session_state["text"]
	st.session_state["text"] = ""


def consolidate():
	st.session_state["temp"] = """Thank you. I would like to end my discussion and move on to a new topic"""
	st.session_state.consolidate_learning = summarizer()
	

def process_dialougic_agent(text):
	if st.session_state.tool_use == True:
		ref = f'Ref No: ({st.session_state.s_count})'
		st.session_state.s_count += 1
		return f'{text} \n\n {ref}', 'blue'
		st.session_state.tool_use = False
	else:
		st.session_state.s_count += 1
		return f'{text}', 'black'


def process_resource_bot(response):
	answer = response.get('answer', '')
	source_documents = response.get('source_documents', [])

	if source_documents:
		first_doc = source_documents[0]
		source = first_doc.metadata['source']
		topic = first_doc.metadata['topic']
		url = first_doc.metadata['url']
		
		st.session_state.web_link.append(f"Ref No: ({st.session_state.s_count})-{source}: {topic}")
		st.session_state.web_link.append(url)

	st.session_state.related_questions = []
	if source_documents:
		for document in source_documents:
			source = document.metadata['source']
			topic = document.metadata['topic']
			page = document.metadata['page']
			page_content = document.page_content
			
			st.session_state.related_questions.append(f"Ref No: ({st.session_state.s_count})-{source}: {topic}, Content page {int(page) + 1}")
			st.session_state.related_questions.append(page_content)

		st.session_state.tool_use = True

	if st.session_state.tool_use:
		ref = f'Ref No: ({st.session_state.s_count})'
		st.session_state.s_count += 1
		st.session_state.tool_use = False
		return f'{answer} \n\n {ref}', 'blue'
	else:
		st.session_state.s_count += 1
		return f'{answer}', 'black'

# def format_lists(text: str) -> str:
#     # Find lists that start with a number followed by a period
#     regex = r"(\d+\.)(\s*[^\d\s][^\n]*)(\n|$)"
#     formatted_text = re.sub(regex, r'<br>\1\2', text)
#     return formatted_text

def format_lists(text: str) -> str:
    pattern = r"(?<=\n|^)(\s*\d+\.)(\s*[^\d\s][^\n]*)(\n|$)"
    formatted_text = regex.sub(pattern, r'<br>\1\2', text)
    return formatted_text

def process_meta_cog(json_data):
	input_text = json_data["input"]
	output = json_data["output"]
	#st.write("Here")
	try:
		data = json.loads(output)
		if 'search_results' in data:
			summaries = [result['summary'] for result in data['search_results']]
			text = ' '.join(summaries)
		else:
			text = output.strip()
	except json.JSONDecodeError:
		text = output.strip()

	if st.session_state.tool_use == True:
		ref = f'Ref No: ({st.session_state.s_count})'
		modified_text = f'(Query: "{input_text}") (Output: {text})'

		#text = follow_up().predict(input=modified_text) #metacog function
		st.session_state.s_count += 1
		return f'{text} \n\n {ref}', 'blue'
	else:
		st.session_state.s_count += 1
		return f'{text}', 'black'

def chat_history():
	if "chat_msg" not in st.session_state:
		st.session_state.chat_msg = []
		"No chat history, you may begin your conversation"

	with st.expander("Click here to see Chat History"):
		messages = st.session_state.chat_msg
		for message in messages:
			bot_msg = message["response"]
			user_msg = message["question"]
			col_msg = message["colour"]
			if isinstance(bot_msg, Image.Image):
				st.image(user_input, caption="Knowledge Map", use_column_width=True)
				st.markdown(f'<div style="text-align: right;"><span style="color: black; font-weight: normal;">{user_msg}</span><span style="color: blue;">:😃 User </span></div>', unsafe_allow_html=True)
				st.write("#")
			else:
				st.markdown(f'<div style="text-align: left; color: black; font-weight: bold;"> <span style="color: red;">Chatbot 🤖:</span> <span style="color: {col_msg}; font-weight: normal;">{bot_msg}</span></div>', unsafe_allow_html=True)
				st.markdown(f'<div style="text-align: right;"><span style="color: black; font-weight: normal;">{user_msg}</span><span style="color: blue;">:😃 User </span></div>', unsafe_allow_html=True)
				st.write("#")


#@st.cache_resource 

def chat_bot(user_input):
	try:
		if isinstance(user_input, Image.Image):
			st.image(user_input, caption="Knowledge Map", use_column_width=True)
			question = "I would like to generate a Knowlege Map please"
			answer = user_input
			colour = "red"
			st.session_state.chat_msg.append({ "question": question, "response": answer, "colour": colour})
			return question, answer
		if user_input == "UML snytax not generated, Knowledge Map cannot be generated, try again later":
			st.error("UML snytax not generated, Knowledge Map cannot be generated, try again later")
			return False
		else:
			if st.session_state.bot_key == cb_bot: #testing with postgres memory ( Google Cloud )
				if st.session_state.document_exist:
					st.session_state.tool_use = False
					st.session_state.source_bot = True
					answer = chergpt_bot(user_input)
					answer, colour = process_resource_bot(answer)
					answer = format_lists(answer)
				else:
					st.session_state.bot_key = c_agent
					pass
			elif st.session_state.bot_key == ar_bot:
				if st.session_state.document_exist:
					st.session_state.tool_use = False
					st.session_state.source_bot = True
					answer = ailc_resources_bot(user_input)
					answer, colour = process_resource_bot(answer)
				else:
					st.session_state.bot_key = ar_agent
					pass
			elif st.session_state.bot_key == m_bot: #no document 
				st.session_state.source_bot = True
				answer = metacog_bot().predict(input=user_input)
				answer = format_lists(answer)
				colour = "black"
			elif st.session_state.bot_key == mr_bot:
				if st.session_state.document_exist:
					st.session_state.tool_use = False
					st.session_state.source_bot = True
					answer = metacog_resources_bot(user_input)
					answer, colour = process_resource_bot(answer)
					answer = format_lists(answer)
				else:
					st.session_state.bot_key = mr_agent
					pass
			elif st.session_state.bot_key == s_bot:
				if st.session_state.document_exist:
					st.session_state.tool_use = False
					st.session_state.source_bot = True
					answer = sourcefinder_bot(user_input)
					answer, colour = process_resource_bot(answer)
				else:
					st.session_state.bot_key = s_agent
					pass
			elif st.session_state.bot_key == c_agent:
				st.session_state.tool_use = False
				ag = chergpt_agent()
				answer = ag.run(input=user_input)
				answer, colour = process_dialougic_agent(answer)
			elif st.session_state.bot_key == c4_agent:
				st.session_state.tool_use = False
				ag = chergpt_agent_4()
				answer = ag.run(input=user_input)
				answer, colour = process_dialougic_agent(answer)
				answer = format_lists(answer)
			elif st.session_state.bot_key == ag_agent:
				st.session_state.tool_use = False
				ag = ailc_agent_serp()
				answer = ag.run(input=user_input)
				answer, colour = process_dialougic_agent(answer)
			elif st.session_state.bot_key == ab_agent:
				st.session_state.tool_use = False
				ag = ailc_agent_bing()
				answer = ag.run(input=user_input)
				answer, colour = process_dialougic_agent(answer)
			elif st.session_state.bot_key == ar_agent:
				st.session_state.tool_use = False
				ag = ailc_resource_agent()
				answer = ag.run(input=user_input)
				answer, colour = process_dialougic_agent(answer)
			elif st.session_state.bot_key == mr_agent: #Testing with redis server ( Worked )
				st.session_state.tool_use = False
				ag = metacog_agent()
				answer = ag.run(input=user_input)
				answer, colour = process_dialougic_agent(answer)
				answer = format_lists(answer)
			elif st.session_state.bot_key == s_agent:	#testing with BabyAGI toolset
				#activate LLM memeory similar to dialougic agent but no metacog answer, uploading of resources is available 
				st.session_state.tool_use = False
				ag = sourcefinder_agent()
				answer = ag.run(input=user_input)
				answer, colour = process_dialougic_agent(answer)
			if user_input:
				question = user_input
				st.markdown(f'<div style="text-align: left; color: black; font-weight: bold;"> <span style="color: red;">Chatbot 🤖:</span> <span style="color: {colour}; font-weight: normal;">{answer}</span></div>', unsafe_allow_html=True)
				st.markdown(f'<div style="text-align: right;"><span style="color: black; font-weight: normal;">{question}</span><span style="color: blue;">:😃 User </span></div>', unsafe_allow_html=True) 
				st.session_state.chat_msg.append({ "question": question, "response": answer, "colour": colour})
				return question, answer

	except openai.APIError as e:
		st.error(e)
		return False


	except Exception as e:
		st.error(e)
		return False


#======================= default Q&A source bot =======================================================





