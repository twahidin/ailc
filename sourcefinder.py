import streamlit as st
import pymongo
import os
import openai
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.vectorstores import Chroma, FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversation.memory import ConversationBufferMemory, ConversationEntityMemory, ConversationSummaryBufferMemory
from langchain.memory.chat_memory import ChatMessageHistory
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.agents import load_tools, initialize_agent, Tool, tool, ZeroShotAgent, AgentExecutor
from parser import CustomOutputParser


@st.cache_resource
def load_instance_index(_tch_code):
	embeddings = OpenAIEmbeddings()
	vectordb = FAISS.load_local(_tch_code, embeddings)

	return vectordb

def sourcefinder_bot(_query): #not in use for now 
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

	return result



@st.cache_resource
def sourcefinder_agent():

	st.session_state.metacog_flag = True

	message_history = RedisChatMessageHistory(url=st.secrets['r_password'], ttl=600, session_id=st.session_state.vta_code)
	#memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=message_history)
	

	prefix = f"""
	You are a virtual assistant helping a human teacher to extract information and resources from the various tools that given to you.
	Your role is to support teachers by helping them plan their lessons, creating assignments or test papers, summarising documents and suggesting
	areas for them to explore.
	You can start by asking them how to support them in their teaching.

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
			#return_direct=True
			),
			Tool(
			name = "Google Search Results",
			func=google_search_serp,
			description="A tool to search useful about current events and things on the Internet and return the search results with their URLs and summaries as a JSON formatted string.",
			#return_direct=True
			),
			Tool(
			name = f"{subject} material search",
			func=document_search,
			description=f"A tool to search for {description}, this tool should be used more than the rest of the tools especially for relating topics on {subject}",
			#return_direct=True
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
			Tool(
			name = "Google Search Results",
			func=google_search_serp,
			description="A tool to search useful about current events and things on the Internet and return the search results with their URLs and summaries as a JSON formatted string.",
			#return_direct=True
			),

		]

	output_parser = CustomOutputParser()

	memory = ConversationSummaryBufferMemory(llm=llm, memory_key="chat_history", max_token_limit=100, chat_memory=message_history)

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