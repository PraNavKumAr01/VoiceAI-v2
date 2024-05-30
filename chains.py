from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from prompts import *
from tools import llm

generate_chain = generate_prompt | llm | StrOutputParser()

question_router = router_prompt | llm | JsonOutputParser()

web_query_chain = web_query_prompt | llm | JsonOutputParser()

yt_query_chain = yt_query_prompt | llm | JsonOutputParser()

math_query_chain = math_query_prompt | llm | JsonOutputParser()

wiki_query_chain = wiki_query_prompt | llm | JsonOutputParser()

eval_result_chain = eval_result_prompt | llm | JsonOutputParser()

