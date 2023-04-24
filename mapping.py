import streamlit as st
import time
import os
import openai
from plantuml import PlantUML #not installed 
from langchain.chat_models import ChatOpenAI
import configparser
config = configparser.ConfigParser()
config.read('config.ini')

#KM default values
km_engine = config['constants'].get('km_engine', 'gpt-3.5-turbo')
km_temperature = float(config['constants'].get('km_temperature', 0.5))
km_max_tokens = float(config['constants'].get('km_max_tokens', 500))
km_max_tokens = int(km_max_tokens)
km_n = float(config['constants'].get('km_n', 1))
km_n = int(km_n)
km_presence_penalty = float(config['constants'].get('km_presence_penalty', 0.0))
km_frequency_penalty = float(config['constants'].get('km_frequency_penalty', 0.0))
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


def render_diagram(uml):
        p = PlantUML("http://www.plantuml.com/plantuml/img/")
        image = p.processes(uml)
        return image

def generate_mind_map():

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
    
    st.write(st.session_state.summary_points)

    prompt = f"""
        Study the conversation below and extract the main topic of the conversation
        {st.session_state.summary_points}

        Let's start by creating a simple MindMap based on the extracted topic and conversation dicussion points
        Can you give the mindmap in PlantUML format based on the conversation topic and discussion points? 

        Keep it structured from the core central topic branching out to other domains and sub-domains. 
        You must start from 3 levels up to 6 levels if necessary. Add the start and end mindmap tags and keep it expanding on one side for now. 
        Also, please add color codes to each node based on the complexity of each topic in terms of the time it takes to learn that topic for a beginner. Use the format *[#colour] topic. 
        """

    try:
        response = openai.ChatCompletion.create(
                                    model=km_engine, 
                                    messages=[{"role": "user", "content": prompt}],
                                    temperature=km_temperature, 
                                    max_tokens=km_max_tokens, 
                                    n=km_n,
                                    presence_penalty=km_presence_penalty,
                                    frequency_penalty=km_frequency_penalty)

        if response['choices'][0]['message']['content'] != None:
            msg = response['choices'][0]['message']['content']

            # Extract PlantUML format string from response
            plantuml = re.search(r'@startmindmap.*?@endmindmap', msg, re.DOTALL).group()

            # Render the PlantUML diagram
            image = render_diagram(plantuml)
            return image
        else:
            return "UML snytax not generated, Knowledge Map cannot be generated, try again later"

    except openai.APIError as e:
        return "UML snytax not generated, Knowledge Map cannot be generated, try again later"
    except Exception as e:
        return "UML snytax not generated, Knowledge Map cannot be generated, try again later"
                    
