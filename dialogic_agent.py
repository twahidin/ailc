import streamlit as st
import wikipedia
import openai
import os
import pymongo
import faiss
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryBufferMemory, VectorStoreRetrieverMemory
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.agents import load_tools, initialize_agent, Tool, tool, ZeroShotAgent, AgentExecutor
from langchain.memory.chat_memory import ChatMessageHistory
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from typing import List, Dict
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from agent_tools import wikipedia_to_json_string, document_search, google_search_serp, bing_search_internet, change_or_end_of_topic
from parser import CustomOutputParser

@st.cache_resource
def ailc_agent_serp():

	if 'memory_state' not in st.session_state:
		st.session_state.memory_state = None

	os.environ["OPENAI_API_KEY"] = st.session_state.api_key
	os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

	cb_temperature, cb_max_tokens, cb_n, cb_presence_penalty, cb_frequency_penalty = st.session_state.cb_settings_key.values()  
	cb_engine = st.session_state.engine_key

	llm = ChatOpenAI(
				model_name=cb_engine, 
				temperature=cb_temperature, 
				max_tokens=cb_max_tokens, 
				n=cb_n,
				presence_penalty= cb_presence_penalty,
				frequency_penalty = cb_frequency_penalty
				)
 
	tools = [
		Tool(
		name="Wikipedia_to_JSON_String",
		func=wikipedia_to_json_string,
		description="A tool to search for a query on Wikipedia and return the search results with their URLs and summaries as a JSON formatted string. The input to this tool should be a query.",
		#return_direct=True
		),
		Tool(
		name = "Google Search Results",
		func=google_search_serp,
		description="A tool to search useful about current events and things on the Internet and return the search results with their URLs and summaries as a JSON formatted string.",
		#return_direct=True
		),
	]

	st.session_state.memory_state = ConversationBufferWindowMemory(memory_key="chat_history", k=7, return_messages=True)

	agent_chain = initialize_agent(tools, llm, agent="chat-conversational-react-description", verbose=True, memory=st.session_state.memory_state)
	
	return agent_chain

@st.cache_resource
def ailc_agent_bing():
	
	if 'memory_state' not in st.session_state:
		st.session_state.memory_state = None

	os.environ["OPENAI_API_KEY"] = st.session_state.api_key
	os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

	cb_temperature, cb_max_tokens, cb_n, cb_presence_penalty, cb_frequency_penalty = st.session_state.cb_settings_key.values()  
	cb_engine = st.session_state.engine_key

	llm = ChatOpenAI(
				model_name=cb_engine, 
				temperature=cb_temperature, 
				max_tokens=cb_max_tokens, 
				n=cb_n,
				presence_penalty= cb_presence_penalty,
				frequency_penalty = cb_frequency_penalty
				)

	tools = [
		Tool(
		name="Wikipedia_to_JSON_String",
		func=wikipedia_to_json_string,
		description="A tool to search for a query on Wikipedia and return the search results with their URLs and summaries as a JSON formatted string. The input to this tool should be a query.",
		#return_direct=True
		),
		Tool(
		name = "Bing Search Results",
		func=bing_search_internet,
		description="A tool to search useful about current events and things on the Internet and return the search results with their URLs and summaries as a JSON formatted string.",
		#return_direct=True
		),
	]

	st.session_state.memory_state = ConversationBufferWindowMemory(memory_key="chat_history", k=7, return_messages=True)

	agent_chain = initialize_agent(tools, llm, agent="chat-conversational-react-description", verbose=True, memory=st.session_state.memory_state)
	
	return agent_chain


@st.cache_resource
def ailc_resource_agent():

	if 'memory_state' not in st.session_state:
		st.session_state.memory_state = None

	os.environ["OPENAI_API_KEY"] = st.session_state.api_key
	os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

	cb_temperature, cb_max_tokens, cb_n, cb_presence_penalty, cb_frequency_penalty = st.session_state.cb_settings_key.values()  
	cb_engine = st.session_state.engine_key

	llm = ChatOpenAI(
				model_name=cb_engine, 
				temperature=cb_temperature, 
				max_tokens=cb_max_tokens, 
				n=cb_n,
				presence_penalty= cb_presence_penalty,
				frequency_penalty = cb_frequency_penalty
				)
	if st.session_state.document_exist == True:
		subject, description = st.session_state.doc_tools_names
		#st.write(st.session_state.doc_tools_names)
		tools = [
			Tool(
				name = f"{subject} material search",
				func=document_search,
				description=f"A tool to search for {description}, this tool should be used more than the rest of the tool especially for relating topics on {subject}",
				),
		
			Tool(
				name="Wikipedia_to_JSON_String",
				func=wikipedia_to_json_string,
				description="A tool to search for a query on Wikipedia and return the search results with their URLs and summaries as a JSON formatted string. The input to this tool should be a query.",
				),
		
		]

	else:

		tools = [
				Tool(
				name="Wikipedia_to_JSON_String",
				func=wikipedia_to_json_string,
				description="A tool to search for a query on Wikipedia and return the search results with their URLs and summaries as a JSON formatted string. The input to this tool should be a query.",
				#return_direct=True
				),
			   ]

	st.session_state.memory_state = ConversationBufferWindowMemory(memory_key="chat_history", k=7, return_messages=True)

	agent_chain = initialize_agent(tools, llm, agent="chat-conversational-react-description", verbose=True, memory=st.session_state.memory_state)
	
	return agent_chain


@st.cache_resource
def load_instance_index(_tch_code):
	embeddings = OpenAIEmbeddings()
	vectordb = FAISS.load_local(_tch_code, embeddings)

	return vectordb
	

def ailc_resources_bot(_query): #not in use for now 
	if 'chat_history' not in st.session_state:
		st.session_state.chat_history = []

	if len(st.session_state.chat_history) > 10:
		st.session_state.chat_history.pop(0)

	os.environ["OPENAI_API_KEY"] = st.session_state.api_key
	os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

	cb_temperature, cb_max_tokens, cb_n, cb_presence_penalty, cb_frequency_penalty = st.session_state.cb_settings_key.values()
	cb_engine = st.session_state.engine_key
	
	llm = ChatOpenAI(
				model_name=cb_engine, 
				temperature=cb_temperature, 
				max_tokens=cb_max_tokens, 
				n=cb_n,
				presence_penalty= cb_presence_penalty,
				frequency_penalty = cb_frequency_penalty
				)


	vectorstore = load_instance_index(st.session_state.teacher_key)

	question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
	doc_chain = load_qa_with_sources_chain(llm, chain_type="map_reduce")

	qa = ConversationalRetrievalChain(
	retriever=vectorstore.as_retriever(),
	question_generator=question_generator,
	combine_docs_chain=doc_chain,
	return_source_documents=True
	)

	result = qa({"question": _query, "chat_history": st.session_state.chat_history })
	st.session_state.chat_history = [(_query, result["answer"])]
	return result











