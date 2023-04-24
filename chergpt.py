import streamlit as st
import openai
import os
import pymongo
import time
from plantuml import PlantUML 
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryBufferMemory
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
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from agent_tools import wikipedia_to_json_string, document_search, google_search_serp, bing_search_internet, change_or_end_of_topic
from parser import CustomOutputParser



@st.cache_resource  
def chergpt_agent_4():

	st.session_state.metacog_flag = True

	if 'chat_history' not in st.session_state:
		st.session_state.chat_history = None

	message_history = RedisChatMessageHistory(url=st.secrets['r_password'], ttl=600, session_id=st.session_state.vta_code)
	#memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=message_history)
	

	prefix = f"""
	Imagine that you are a virtual teacher having a lively discussion to help a human teenager learn and clarify about any educational topic. 
	You must prompt the student for a brief description of the new topic that they would like to learn.
	You must provide a short paragraph or key points of not more than 80 words related to your query follow by 3 to 4 questions.
	At all times you must create a list of 3 to 4 related questions to help the human learn and deepen the understanding of the subject. 
	You must number your questions.

	You are given a set of tools to determine the next course of action
	"""

	#prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
	suffix = """Begin!"

	{chat_history}
	Question: {input}
	{agent_scratchpad}"""

	

	os.environ["OPENAI_API_KEY"] = st.session_state.api_key
	os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


	cb_temperature, cb_max_tokens, cb_n, cb_presence_penalty, cb_frequency_penalty = st.session_state.cb_settings_key.values()  
	cb_engine = st.session_state.engine_key
	st.session_state.tool_use = False
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

		tools = [
			Tool(
			name="Wikipedia_to_JSON_String",
			func=wikipedia_to_json_string,
			description="A tool to search for a query on Wikipedia and return the search results with their URLs and summaries as a JSON formatted string. The input to this tool should be a query.",
			),
			Tool(
			name = f"{subject} material search",
			func=document_search,
			description=f"A tool to search for {description}, this tool should be used more than the rest of the tool especially for relating topics on {subject}",
			),
			Tool(
			name = "change_or_end_of_topic_subject_tool",
			func=change_or_end_of_topic,
			description="A tool to use when the human wants to end or change the subject or topic during the discussion to help the student reflect their learning",

			),
		
		]

	else:

		tools = [
			Tool(
			name="Wikipedia_to_JSON_String",
			func=wikipedia_to_json_string,
			description="A tool to search for a query on Wikipedia and return the search results with their URLs and summaries as a JSON formatted string. The input to this tool should be a query.",
			),
			Tool(
			name = "change_or_end_of_topic_subject_tool",
			func=change_or_end_of_topic,
			description="A tool to use when the human wants to end or change the subject or topic during the discussion to help the student reflect their learning",

			),
			
		]

	
	output_parser = CustomOutputParser()

	memory = ConversationSummaryBufferMemory(llm=llm, memory_key="chat_history", max_token_limit=100, chat_memory=message_history)

	st.session_state.chat_history = memory

	#memory = ConversationBufferWindowMemory(memory_key="chat_history", k=7, return_messages=True)
	prompt = ZeroShotAgent.create_prompt(
		tools, 
		prefix=prefix, 
		suffix=suffix, 
		input_variables=["input", "chat_history", "agent_scratchpad"]
	)

	llm_chain = LLMChain(llm=llm, prompt=prompt)
	agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, output_parser=output_parser, max_iterations=3,  verbose=True)
	agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)
	

	#agent_chain = initialize_agent(tools, llm, agent="chat-conversational-react-description", verbose=True, memory=memory)
	
	return agent_chain



@st.cache_resource
def load_instance_index(_tch_code):
	embeddings = OpenAIEmbeddings()
	vectordb = FAISS.load_local(_tch_code, embeddings)

	return vectordb
	
@st.cache_resource
def chergpt_bot(_query): #not in use for now 

	st.session_state.metacog_flag = True

	if 'chat_memory' not in st.session_state:
		st.session_state.chat_memory = []

	if 'chat_history' not in st.session_state:
		st.session_state.chat_history = []

	if len(st.session_state.chat_history) > 14:
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
	
	
	
	question_prompt_template = """
	Imagine that you are a virtual teacher having a discussion to help a human teenager learn and clarify about any educational topic. 
	Use the following portion of a long document to see if any of the text is relevant to the question to help with the discussion.
	{context}
	Question: {question}
	"""
	QUESTION_PROMPT = PromptTemplate(template=question_prompt_template, input_variables=["context", "question"])

	combine_prompt_template = """
	Given the following extracted parts of a long document and a question, you must provide a short paragraph or key points of not more than 80 words related to your query follow by 3 to 4 questions.

	At all times you must create a 3 to 4 questions during the conversation or discussion.

	When the human is discussing or starting a topic, you will create a list of 3 to 4 related questions to help the human learn and deepen the understanding of the subject. 

	If the human decides to change or end the topic or subject in the discussion, you will ask the following questions in context:
	1.What did you previously know about the topic we just discussed? 
	2.What is something new that you have learned about this topic during our conversation? 
	3.What is an area or aspect of the topic that you would like to explore further or find out more about?
	
	Remember you must create 3 to 4 questions at all times and change the context of the questions depending on the discussion
	Each question must have a question number.

	QUESTION: {question}
	=========
	{summaries}
	=========
	"""

	COMBINE_PROMPT = PromptTemplate(template=combine_prompt_template, input_variables=["summaries", "question"])
	doc_chain = load_qa_with_sources_chain(llm, chain_type="map_reduce", question_prompt=QUESTION_PROMPT, combine_prompt=COMBINE_PROMPT)

	vectorstore = load_instance_index(st.session_state.teacher_key)
	question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)

	qa = ConversationalRetrievalChain(
	retriever=vectorstore.as_retriever(),
	question_generator=question_generator,
	combine_docs_chain=doc_chain,
	return_source_documents=True
	)

	result = qa({"question": _query, "chat_history": st.session_state.chat_memory })
	st.session_state.chat_memory = [(_query, result["answer"])]
	st.session_state.chat_history.append(_query)
	st.session_state.chat_history.append(result["answer"])

	return result


@st.cache_resource
def chergpt_agent():

	st.session_state.metacog_flag = True

	if 'chat_history' not in st.session_state:
		st.session_state.chat_history = None

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

	st.session_state.chat_history  = ConversationBufferWindowMemory(memory_key="chat_history", k=7, return_messages=True)

	agent_chain = initialize_agent(tools, llm, agent="chat-conversational-react-description", verbose=True, memory=st.session_state.chat_history )
	
	return agent_chain
