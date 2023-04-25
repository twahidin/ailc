import streamlit as st
import pymongo
import os
import openai
import regex
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.vectorstores import Chroma, FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts import PromptTemplate, BaseChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversation.memory import ConversationBufferMemory, ConversationEntityMemory, ConversationSummaryBufferMemory
from langchain.memory.chat_memory import ChatMessageHistory
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.agents import load_tools, initialize_agent, Tool, tool, ZeroShotAgent, AgentExecutor, LLMSingleActionAgent
from agent_tools import wikipedia_to_json_string, document_search, google_search_serp, bing_search_internet, change_or_end_of_topic
from parser import CustomOutputParser, CustomPromptTemplate

import json
import configparser
config = configparser.ConfigParser()
config.read('config.ini')
openai.api_key  = st.session_state.api_key
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


def summarizer():
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

	if st.session_state.bot_key == mr_bot or st.session_state.bot_key == cb_bot :
		#st.write(st.session_state.chat_history)
		summary_points = ' '.join([msg for msg in st.session_state.chat_history])
		st.session_state.summary_points = summary_points
	elif st.session_state.bot_key == mr_agent or st.session_state.bot_key == c4_agent or st.session_state.bot_key == m_bot or st.session_state.bot_key == c_agent:
		#history = RedisChatMessageHistory(url=st.secrets['r_password'], ttl=600, session_id=st.session_state.vta_code)
		#messages = history.messages
		# Concatenate the content of all messages
		st.session_state.summary_points = st.session_state.chat_history.load_memory_variables({})
	else: #c_agent
		return "Unable to generate learning log"
	template = """ The conversation below is a discussion between a virtual teacher and a human based on a topic or subject
	{conversation}
	You are going to generate two lists:
	The first list are important points based on the discussion called: Summarised Key Points
	The second list are suggested areas of the subject that the human should explore or explore based on the dicussion called: Suggested Learning
	"""
	prompt = PromptTemplate(template=template, input_variables=["conversation"])
	llm_chain = LLMChain(prompt=prompt, llm=OpenAI(temperature=0), verbose=True)
	st.session_state.consolidate = False
	st.session_state.chat_history = []
	return llm_chain.predict(conversation=st.session_state.summary_points)


def metacog_bot():
	st.session_state.metacog_flag = True
	
	if 'chat_history' not in st.session_state:
		st.session_state.chat_history = None

	os.environ["OPENAI_API_KEY"] = st.session_state.api_key
	os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

	cb_temperature, cb_max_tokens, cb_n, cb_presence_penalty, cb_frequency_penalty = st.session_state.cb_settings_key.values()
	cb_engine = st.session_state.engine_key

	template = f"""

	Imagine that you are a virtual teacher having a discussion to help a human teenager learn and clarify about any educational topic. 
	You are given two different teacher roles, the first role is a Question Teacher role and the second role is a Change or End of Topic teacher role. 
	During the start of the discussion or a new topic discussion you must play the Question teacher role and continue the discussion
	When the human decides to change the topic or end the discussion you must play the role of the Change or End of Topic teacher role.


	The description and tasks of the 2 teacher roles are shown below:
	Question Teacher role:
	This question teacher role will engage the student in a lively conversation on the topic. 
	The question teacher role must prompt the student for a brief description of the topic that they would like to learn. 
	The question teacher role must provide a short paragraph or key points of not more than 80 words related to your query. 
	At all times, the question teacher role must create a list of 3 to 4 related questions to help the human learn and deepen the understanding of the subject. 
	Each question must have a question number.


	Change or End of Topic teacher role:
	This Change or End of Topic teacher role will engage the human when the human wants to end the topic or change the topic or attempt to change the topic.
	This Change or End of Topic teacher role must create a list of 3 reflective questions before changing into a new topic discussion. The sample list of reflective questions are 
	1.What did you previously know about the topic we just discussed? 
	2.What is something new that you have learned about this topic during our conversation? 
	3.What is an area or aspect of the topic that you would like to explore further or find out more about?
	Each question must have a question number.
	
	Current conversation:
	{{history}}
	Last line:
	Human: {{input}}
	You:"""

	prompt = PromptTemplate(
	input_variables=["history", "input"], 
	template=template
	)
	

	llm = ChatOpenAI(
				model_name=cb_engine, 
				temperature=cb_temperature, 
				max_tokens=cb_max_tokens, 
				n=cb_n,
				presence_penalty= cb_presence_penalty,
				frequency_penalty = cb_frequency_penalty
				)

	st.session_state.chat_history = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)

	conversation = ConversationChain(
									llm=llm, 
									verbose=True,
									prompt = prompt,
									memory=st.session_state.chat_history
									)
	return conversation


@st.cache_resource
def load_instance_index(_tch_code):
	embeddings = OpenAIEmbeddings()
	vectordb = FAISS.load_local(_tch_code, embeddings)

	return vectordb

def metacog_resources_bot(_query): #not in use for now 
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
def metacog_agent(): #Only can work for GPT4

	template = """Answer the following questions as best you can through a conversation.
	You will always add 3 learning questions with your Final Answer to help with understanding
	Current conversations
	{chat_history}

	You have access to the following tools:
	{tools}

	Use the following format:

	Question: the input question you must answer
	Thought: you should always think about what to do
	Action: the action to take, should be one of [{tool_names}]
	Action Input: the input to the action
	Observation: the result of the action
	... (this Thought/Action/Action Input/Observation can repeat 3 times)
	Thought: I now know the final answer
	Final Answer: the final answer to the original input question


	Question: {input}
	{agent_scratchpad}"""
	

	st.session_state.metacog_flag = True

	if 'chat_history' not in st.session_state:
		st.session_state.chat_history = None

	message_history = RedisChatMessageHistory(url=st.secrets['r_password'], ttl=600, session_id=st.session_state.vta_code)
	
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

	prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps", "chat_history"]
	)

	llm_chain = LLMChain(llm=llm, prompt=prompt)

	tool_names = [tool.name for tool in tools]

	output_parser = CustomOutputParser()

	memory = ConversationSummaryBufferMemory(llm=llm, memory_key="chat_history", max_token_limit=100, chat_memory=message_history)

	st.session_state.chat_history = memory

	agent = LLMSingleActionAgent(
    								llm_chain=llm_chain, 
   									output_parser=output_parser,
								    stop=["\nObservation:"], 
								    allowed_tools=tool_names
								)

	agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)
	
	return agent_chain


