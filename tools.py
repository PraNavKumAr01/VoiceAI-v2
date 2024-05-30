from langchain_groq import ChatGroq
from langchain_community.utilities.google_search import GoogleSearchAPIWrapper
from langchain_community.tools.youtube.search import YouTubeSearchTool
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_core.tools import Tool
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY")
os.environ["GOOGLE_CSE_ID"] = os.environ.get("GOOGLE_CSE_ID")
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_SEARCH_API_KEY")
os.environ["WOLFRAM_ALPHA_APPID"] = os.environ.get("WOLFRAM_ALPHA_APPID")

llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        
image_search_tool = genai.GenerativeModel(model_name="models/gemini-1.5-flash-latest")

# GOOGLE SEARCH TOOL
web_search_wrapper = GoogleSearchAPIWrapper(k=10)
web_search_tool = Tool(
    name="google_search",
    description="Search Google for recent results.",
    func=web_search_wrapper.run,
)

# YOUTUBE SEARCH TOOL
youtube_search_tool = YouTubeSearchTool()

# WOLFRAM MATH TOOL
wolfram_math_tool = WolframAlphaAPIWrapper()

# WIKIPEDIA SEARCH TOOL
wiki_search_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(doc_content_chars_max = 2000))
