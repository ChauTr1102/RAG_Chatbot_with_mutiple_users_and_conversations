import os
import base64
import time
import asyncio
import re
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_community.callbacks import get_openai_callback
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents.agent_types import AgentType
from streamlit_float import *
from db_request import  *



st.set_page_config(
    page_title='FDX_GPT',
    layout="wide",
    initial_sidebar_state="auto",
    page_icon="images/for_bot.jpg"
)
float_init()


col1, col2, col3 = st.columns([1, 4, 1])



# Load environment variables
load_dotenv()


# Function to convert image to base64 string
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

def summarize_head_conversation():


    llm = ChatOpenAI(model='gpt-4o-mini')
    prompt = ChatPromptTemplate.from_template("""Bạn hãy phục vụ mục đích rằng bạn sẽ tóm tắt câu hỏi này để tôi có thể cho nó là một tên của một đoạn hội thoại
    và chỉ cần đưa ra tên cuộc hội thoại không cần thiết đưa ra từ ngữ thừa và dấu :
         Câu hỏi hiện tại:
         {question}
         Trợ lý:
    """)

    parser = StrOutputParser()
    qa_chain = RunnableParallel(

        {
         'question': RunnablePassthrough()
        }

    )
    chain = qa_chain | prompt | llm | parser
    return chain

# Function to set the background image
def set_background():
    local_image_path = 'images/back_ground_page.png'
    base64_image = get_base64_image(local_image_path)

    background_image = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
        background-image: url("data:image/jpg;base64,{base64_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    st.markdown(background_image, unsafe_allow_html=True)


# Function to load the FAISS vector store
def get_vector_store():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return FAISS.load_local('faiss-db', embeddings=embeddings, allow_dangerous_deserialization=True)



def clean_text(text):
    # Remove unnecessary line breaks and merge broken words
    text = re.sub(r'\n+', ' ', text)  # Replace multiple newlines with a space
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    # text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters, like accents
    text = re.sub(r'\d+', '', text)  # Remove digits, if irrelevant

    # Standardize common terms (can expand based on need)
    text = text.replace('CÔNG TY CỔ PHẦN', 'Công ty Cổ phần')
    text = text.replace('FPT', 'FPT Corporation')  # Example of standardizing names

    return text.strip()


def clean_retriever(response):
    text = ''
    for doc in response:
        text += clean_text(doc.page_content)
    return text




def repharse_chain():
    model = ChatOpenAI(temperature=0, streaming=True, model='gpt-4o-mini')
    prompt = ChatPromptTemplate.from_template("""
            Bạn là một trợ lý AI thông minh của công ty TNHH FPT Digital. 
            Vui lòng diễn đạt lại câu hỏi sau sao cho nó nắm bắt được ý chính và mục đích ban đầu, 
            đồng thời xem xét bối cảnh đã được cung cấp. Đảm bảo rằng câu mới được viết ngắn gọn, rõ ràng, 
            tối ưu hóa cho việc tìm kiếm trong cơ sở dữ liệu vector, và cung cấp các từ khóa quan trọng có liên quan. 
            Nếu câu hỏi không rõ nghĩa, vui lòng sửa lại dựa trên kiến thức của bạn để làm rõ câu hỏi.
             chỉ cần liệt kê ra thôi không cần phải ghi lại câu hỏi
            Lịch sử cuộc trò chuyện: {chat_history}

            Câu hỏi: {question}

            Trợ lý:
            """)
    parser = StrOutputParser()
    repharese_chain = prompt | model | parser
    return repharese_chain
def Rag_Chain():


    llm = ChatOpenAI(model=st.session_state.model,streaming = True)

    prompt = ChatPromptTemplate.from_template("""Bạn là một trợ lý AI thông minh của công ty TNHH FPT Digital.
        Hãy cung cấp cho tôi thông tin một cách đầy đủ và chi tiết nhất có thể.
        Nếu tôi không yêu cầu rõ ràng về thông tin cụ thể,
        bạn có thể cung cấp thông tin liên quan nhưng đừng quá sa đà vào các chi tiết không cần thiết.
        Nếu bạn không chắc chắn về câu trả lời, bạn có thể nói rằng bạn không biết, nhưng nếu có,
        hãy sử dụng kiến thức và dữ liệu của mình để bổ sung
        và thêm phần tóm gọn nội dung ở cuối mỗi câu trả lời:

        {context}

        Lịch sử cuộc trò chuyện:
        {chat_history}

        Trả lời câu hỏi sau:
        {question}

        Trợ lý:""")


    parser = StrOutputParser()

    chain = prompt | llm | parser
    return chain

def Overall_Chain():

    llm = ChatOpenAI(model=st.session_state.model,streaming = True)
    prompt = ChatPromptTemplate.from_template("""Bạn là trợ lý AI thông minh của công ty TNHH FPT Digital.
    Hãy cung cấp cho tôi thông tin một cách rõ ràng và chi tiết nhất có thể.
    Nếu tôi không yêu cầu rõ ràng về một thông tin cụ thể,
    bạn có thể cung cấp các chi tiết liên quan nhưng không cần lan man vào những điều không cần thiết.
    Nếu bạn không biết chắc câu trả lời, bạn có thể nói rằng bạn không biết, nhưng hãy sử dụng tất cả kiến thức có sẵn của bạn để hỗ trợ tốt nhất.
    Sau mỗi câu trả lời, tóm tắt lại những ý chính một cách ngắn gọn và súc tích nếu cần thiết.
    Lịch sử cuộc trò chuyện (để tham khảo):
    {chat_history}
    Câu hỏi hiện tại:
    {question}

    Trợ lý:""")

    parser = StrOutputParser()

    chain =  prompt | llm | parser
    return chain


def FPT_AI_Products_Chain():

    llm = ChatOpenAI(model=st.session_state.model,streaming = True)
    agent = create_csv_agent(
        llm = llm,
        path= "Tổng hợp thông tin về dịch vụ_sản phẩm của FPT - Chatbot.csv",
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        allow_dangerous_code=True
    )
    return agent



# Main function to handle the user input and generate a response
def handle_userinput(user_question):
    global  chain
    global col2
    with col2:

        st.chat_message("user", avatar="images/user.jpg").markdown(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question, "avatar": "images/user.jpg"})
        add_messages(st.session_state.user_conversation,role='human',msg=user_question)
        with st.chat_message("assistant", avatar="images/for_bot.jpg"):


            response = None
            if st.session_state.chain_type == "FDX Assistant":
                response = st.write_stream(st.session_state.conversation.stream({
        "chat_history": st.session_state.messages,
        "question": user_question
        }))
            elif st.session_state.chain_type == 'FPT AI Products':
                response = st.write_stream(st.session_state.conversation.stream(user_question))
            else:
                repharese_chain = repharse_chain()
                response = st.write_stream(st.session_state.conversation.stream({
                    "chat_history": st.session_state.messages,
                    "question": user_question,
                    "context": clean_retriever(st.session_state.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 25}).invoke(repharese_chain.invoke({
                        "chat_history": st.session_state.messages,
                        "question": user_question}))
                    )
                }))

        st.session_state.messages.append({"role": "assistant", "content": response, "avatar": "images/for_bot.jpg"})
        add_messages(st.session_state.user_conversation,'AI',response)


# Main function to run the Streamlit app
def main():
    set_background()

    # Sidebar: Upload PDFs and Conversation buttons
    # select your model



    # Initialize session state variables if they don't exist
    if "messages" not in st.session_state:
        st.session_state.messages = None

    if "chain_type" not in st.session_state:
        st.session_state.chain_type=None

    if 'vectorstore' not in st.session_state:
        with st.spinner("Loading your database..."):
            st.session_state.vectorstore = get_vector_store()
        st.success('Successfully loaded your database')

    if 'model' not in st.session_state:
        st.session_state.model = 'gpt-4o'

    if "conversation" not in st.session_state:
        st.session_state.conversation = Overall_Chain()

    if 'user_conversation' not in st.session_state:
        st.session_state.user_conversation  = get_latest_conversation(username= st.session_state.username)

    if 'history' not in st.session_state:
        st.session_state.history = st.session_state.history  = get_messages(username=st.session_state.username,conversation_id=st.session_state.user_conversation)

    if 'conversation_name' not in st.session_state:
        st.session_state.conversation_name = get_conversation_name(st.session_state.user_conversation)

    with st.sidebar:

        st.page_link("login.py", label="Logout", icon="↖️")


        if st.button('**New conversation**',type='primary',icon=":material/add:"):
            add_conversation(user_name=st.session_state.username)
            st.session_state.user_conversation = get_latest_conversation(username= st.session_state.username)
            st.session_state.history = get_messages(username=st.session_state.username,
                                                    conversation_id=st.session_state.user_conversation)
            st.session_state.messages = None


        st.header('**History conversation**',divider=True)
        list_conversation = get_conversation_list(st.session_state.username)
        st.markdown("""
                            <style>
                div.stButton > button {
                    text-align: left !important;  /* Force left alignment */
                    justify-content: flex-start !important;  /* Ensure text stays on the left */
                    padding-left: 10px !important;  /* Optional padding for better spacing */
                    width: 100% !important;  /* Ensure buttons take full width */
                    border-radius: 10px !important;  /* Add rounded corners */
                }
                </style>
                            """, unsafe_allow_html=True)
        for con in reversed(list_conversation):

            if st.button(f"{con['conversation_name']}", key=f"Conversation_{con['conversation_id']}",use_container_width=True,type='secondary'  ):


                st.session_state.user_conversation = con['conversation_id']
                st.session_state.history = get_messages(username=st.session_state.username,
                                                        conversation_id=st.session_state.user_conversation)
                st.session_state.messages = None
       




    # Load historical messages  into session state

    if not st.session_state.messages:
        st.session_state.messages = []
        history = st.session_state.history

        for i in range(0, len(history), 2):
            if i + 1 < len(history):

                user_input = history[i]['message_text']
                assistant_response = history[i+1]['message_text']
                st.session_state.messages.append({"role": "user", "content": user_input, "avatar": "images/user.jpg"})
                st.session_state.messages.append(
                    {"role": "assistant", "content": assistant_response, "avatar": "images/for_bot.jpg"})






    with col2:
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar=message["avatar"]):
                st.markdown(message["content"])

    # Handle new user input

    with col1:
        with st.container():
            st.logo('images/LOGO.png')
            option = st.selectbox(
                "",
                ("gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"),
                index=None,
                placeholder="gpt-4o",
                label_visibility='collapsed'
            )
            if option:
                st.session_state.model = option
            with st.popover("Assistant",use_container_width=True):
                chain = st.radio(
                    "Select your assistant",
                    ["FDX Assistant", "FDX QueryPro",'FPT AI products'],
                    captions=[
                        "FDX Assistant is an intuitive assistant designed to handle everyday tasks and complex requests",
                        "FDX QueryPro taps into FDX’s rich data resources to provide comprehensive answers. While slightly slower than real-time assistants, it compensates with accuracy and depth, ensuring you get high-quality insights every time",
                        "Smart agent for FPT AI products"
                    ],
                )
                st.session_state.chain_type = chain
                if chain == "FDX Assistant":
                    st.session_state.conversation = Overall_Chain()
                elif chain == 'FPT AI products':
                    st.session_state.conversation = FPT_AI_Products_Chain()
                else:
                    st.session_state.conversation = Rag_Chain()
                



    col1.float()

    st.session_state.conversation_name = get_conversation_name(st.session_state.user_conversation)
    if prompt := st.chat_input("Tôi có thể giúp gì cho bạn?"):
        handle_userinput(prompt)
        if st.session_state.conversation_name == 'New conversation':

            rephrase = summarize_head_conversation()
            rephrase   = rephrase.invoke({
        "question": prompt
        })

            update_conversation_name(st.session_state.user_conversation,rephrase)
            st.rerun(scope='app')

# Run the app
main()
# st.write(st.session_state.conversation)
