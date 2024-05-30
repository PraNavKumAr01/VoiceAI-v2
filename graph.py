from langgraph.graph import END, StateGraph
from state import GraphState
from nodes import Nodes

class WorkFlow():
    def __init__(self):
        workflow = StateGraph(GraphState)
        nodes = Nodes()

        workflow.add_node("websearch", nodes.web_search)
        workflow.add_node("imagesearch", nodes.image_search)
        workflow.add_node("youtubesearch", nodes.youtube_search)
        workflow.add_node("wikisearch", nodes.wiki_search)
        workflow.add_node("mathsolve", nodes.math_solve)
        workflow.add_node("transform_web_query", nodes.transform_web_query)
        workflow.add_node("transform_yt_query", nodes.transform_yt_query)
        workflow.add_node("transform_wiki_query",nodes.transform_wiki_query)
        workflow.add_node("transform_math_query",nodes.transform_math_query)ry)
        workflow.add_node("generate", nodes.generate)

        # Build the edges
        workflow.set_conditional_entry_point(
            nodes.route_question,
            {
                "websearch": "transform_web_query",
                "youtubesearch": "transform_yt_query",
                "wikisearch": "transform_wiki_query",
                "mathsolve": "transform_math_query",
                "imagesearch": "imagesearch",
                "generate": "generate",
            },
        )

        # Handle web search path
        workflow.add_edge("transform_web_query", "websearch")
        workflow.add_conditional_edges(
            "websearch",
            nodes.evaluate_search_results,
            {
                "fallback": "transform_wiki_query",
                "generate": "generate",
            },
        )

        # Handle Wikipedia search path
        workflow.add_edge("transform_wiki_query", "wikisearch")
        workflow.add_conditional_edges(
            "wikisearch",
            nodes.evaluate_search_results,
            {
                "fallback": "transform_web_query",
                "generate": "generate",
            },
        )

        # Handle Wikipedia search path
        workflow.add_conditional_edges(
            "imagesearch",
            nodes.evaluate_search_results,
            {
                "fallback": "transform_web_query",
                "generate": "generate",
            },
        )

        # Handle YouTube search path
        workflow.add_edge("transform_yt_query", "youtubesearch")
        workflow.add_edge("youtubesearch", "generate")

        # Handle solving maths problems
        workflow.add_edge("transform_math_query", "mathsolve")
        workflow.add_edge("mathsolve", "generate")

        workflow.add_edge("generate", END)

        # Compile the workflow
        self.local_agent = workflow.compile()
