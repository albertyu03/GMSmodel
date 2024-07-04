#chain for prompt, template, gpt
# coding: utf8
import getpass
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain.agents import AgentExecutor
from langsmith import Client
from langchain.smith import RunEvalConfig, run_on_dataset

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"GMS"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "xxx"

#export LANGCHAIN_TRACING_V2="true"
#export LANGCHAIN_API_KEY="..."
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

os.environ["OPENAI_API_KEY"] = "xxx"

llm = ChatOpenAI(model="gpt-4")


# Retrieve and generate using the relevant snippets of the blog.

parser = StrOutputParser()
system_template = "You are an assistant to 极客晨星 Computer Science School, who have a computer science learning tool for children. Given is the automated transcription of a student's question or statement while using the tool, likely about the school itself or using the tool. This transcription may have transcription errors in the form of similar sounding characters and/or missing/extra characters. If needed, correct the question in context, then translate the fixed question to English."
tl_prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)
fixing_template = "The provided text contains some Chinese and its English counterpart. Please return only the English counterpart."
fixing_prompt_template = ChatPromptTemplate.from_messages(
    [("system", fixing_template), ("user", "{text}")]
)
tl_chain = tl_prompt_template | llm | parser
fix_chain = fixing_prompt_template | llm | parser
total_tl_chain = tl_prompt_template | llm | parser | fixing_prompt_template | llm | parser
loader = TextLoader("manual.txt", encoding='utf-8') #TODO
docs = loader.load() #TODO
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=200)
splits = text_splitter.split_documents(docs) #docs TODO
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()


rag_prompt = "You are an assistant for question-answering tasks employed by 极客晨星 Programming School. Use the following pieces of retrieved context to answer the question. If the question is subjective about the school, have a favorable outlook. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. \n Question: {question} \n Context: {context} \n Answer: " 
prompt = ChatPromptTemplate.from_messages(
    [("system", rag_prompt)]
)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

tl_cn_prompt = "You are an instructor for 极客晨星 Children's Programming School. Translate the given English answer to Chinese, and maintain a friendly tone. Soothe any negative emotions. Keep the answers short and concise, and understandable for children."
tl_cn_prompt_template = ChatPromptTemplate.from_messages(
    [("system", tl_cn_prompt), ("user", "{text}")]
)
tl_cn_chain = tl_cn_prompt_template | llm | parser

#need to test tl_chain, rag_chain
client = Client()

example_inputs = [
    ("几颗陈星变成有什么好", "What is good about Geek Morning Star?"),
    ("师我捉什么呀", "What is my task teacher?"),
    ("运行怎么不对啊", "Why isn't my code correct?"),
    ("派森报错了", "There is a Python error."),
    ("我想发给妈妈看", "I want to show my code to mom."),
    ("C加和白粉哪个更好", "Is C++ or Python better?"),
    ("几颗沉醒不好玩", "Geek Morning Star is not fun."),
    ("老死我怎么办", "Teacher, what should I do?"),
    ("为什么不行啊", "Why isn't it working?"),
    ("我程序怎么不对", "Why isn't my code working?"),
    ("我诚实不对啊", "Why isn't my code working?"),
    ("老我要做什么呀", "What should I do teacher?"),
    ("派森和西佳佳哪个更好", "Is C++ or Python better?"),
    ("几颗陈星变成有什么好", "What is good about Geek Morning Star?"),
    ("怎么让妈妈看到我的程序运行呢", "How do I show my code to mom?"),
    ("那个发布是什么意思", "What does the publish button mean?"),
    ("我不知道怎么用", "I don't know how to use the tool."),
    ("我该干什么", "What is my task?"),
    ("我点了运行怎么没有动", "Why isn't my code running?"),
    ("转换是什么意思", "What does convert mean?"),
    ("几颗星吗是啥意思", "What's so good about Geek Morning Star?"),
    ("怎么出错了", "Why is there something wrong with my code?"),
]
'''
dataset_name = "tl dataset"
tl_dataset = client.create_dataset(
    dataset_name=dataset_name,
    description="Fixable Chinese questions",
)

for input_p, output_a in example_inputs:
    client.create_example(
        inputs={"text": input_p},
        outputs={"answer": output_a},
        dataset_id=tl_dataset.id
    )
'''
evaluation_config = RunEvalConfig(
    evaluators=[
        RunEvalConfig.Criteria(
            {"correctness": "How close is the submissed question/statement to the solution in meaning (in the context of using a Computer Science learning tool)?"}
        )
    ]
)

run_on_dataset(
    client=client,
    dataset_name="tl dataset",
    llm_or_chain_factory=total_tl_chain,
    evaluation=evaluation_config
)