from typing_extensions import TypedDict

# Graph State
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        search_query: revised question for web search
        youtube_search_query : revised question for youtube search
        context: web_search result
    """
    question : str
    generation : str
    web_search_query : str
    yt_search_query : str
    wiki_query: str
    math_query: str
    math_answer: str
    video_links: list
    emails: list
    videos_found : bool
    mail_search_query: str
    email: dict
    context : str
    chat_history: str
    logs: list
    retries: int
    imageAvailable: bool
    image: str