from chains import *
from tools import *
import base64
from io import BytesIO
from PIL import Image

class Nodes:

    def generate(self, state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("Step: Generating Final Response")
        question = state["question"]
        context = state["context"]
        chat_history = state["chat_history"]
        video_stat = state["videos_found"]
        math_answer = state["math_answer"]
        video_links = state["video_links"]

        generation = generate_chain.invoke({"context": context,"chat_history": chat_history, "question": question, "videos_found": video_stat, "math_answer": math_answer})
        return {"generation": generation, "video_links": video_links}


    def transform_web_query(self, state):
        """
        Transform user question to web search

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Appended search query
        """
        print("Step: Optimizing Query for Web Search")
        question = state['question']
        chat_history = state['chat_history']
        context = state['context']
        gen_web_query = web_query_chain.invoke({"question": question, "chat_history" : chat_history, "context" : context})
        web_search_query = gen_web_query["query"]
        return {"web_search_query": web_search_query}

    def transform_yt_query(self, state):
        """
        Transform user question to yt search

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Appended search query
        """
        
        print("Step: Optimizing Query for Youtube Search")
        question = state['question']
        chat_history = state['chat_history']       
        gen_yt_query = yt_query_chain.invoke({"question": question, "chat_history" : chat_history})
        yt_search_query = gen_yt_query["query"]
        return {"yt_search_query": yt_search_query}

    def transform_math_query(self, state):
        """
        Transform user question to math search

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Appended math query
        """
        
        print("Step: Optimizing Query for Mathematical Solving")
        question = state['question']
        chat_history = state['chat_history']       
        gen_math_query = math_query_chain.invoke({"question": question, "chat_history" : chat_history})
        math_query = gen_math_query["query"]
        return {"math_query": math_query}

    def transform_wiki_query(self, state):
        """
        Transform user question to wiki search

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Appended search query
        """
        
        print("Step: Optimizing Query for Wikipedia Search")
        question = state['question']
        chat_history = state['chat_history']       
        gen_wiki_query = wiki_query_chain.invoke({"question": question, "chat_history" : chat_history})
        wiki_query = gen_wiki_query["query"]
        return {"wiki_query": wiki_query}

    def transform_email_search_query(self, state):
        """
        Transform user question to email search

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Appended search query
        """
        
        print("Step: Optimizing Query for Gmail Search")
        question = state['question']
        gen_mail_search_query = gmail_search_query_chain.invoke({"question": question})
        mail_search_query = gen_mail_search_query["query"]
        return {"mail_search_query": mail_search_query}

    def transform_email_send_query(self, state):
        """
        Transform user question to send email

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Appended send mail query
        """
        
        print("Step: Optimizing Query for sending Gmail")
        question = state['question']
        gen_mail_send_query = gmail_send_query_prompt.invoke({"question": question})
        return {"email": gen_mail_send_query}
    
    def image_search(self, state):
        """
        Image recognition based on the image

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Appended image description to context
        """
        print("Step: Analysing the image")
        image_bytes = state['image']
        image_data = base64.b64decode(image_bytes)
        image = Image.open(BytesIO(image_data))
        response = image_search_tool.generate_content(["You have to act as my eyes and provide a detailed description of the image provided. Include every detail",image])
        print(response.text)
        return {'context' : response.text}

    def web_search(self, state):
        """
        Web search based on the question

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Appended web results to context
        """

        search_query = state['web_search_query']
        print(f'Step: Searching the Web for: "{search_query}"')
        
        search_result = web_search_tool.run(search_query)
        return {
            "context": search_result,
            "video_links" : []
        }

    def youtube_search(self, state):
        """
        Youtube search based on the question

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Appended Youtube results to links
        """

        search_query = state['yt_search_query']
        print(f'Step: Searching Youtube for: "{search_query}"')
        
        search_result = youtube_search_tool.run(search_query+",5")
        if search_result:
            return {
                "video_links": search_result,
                "videos_found": True
            }
        else:
            return {
                "videos_found" : False
            }
        
    def math_solve(self, state):
        """
        Math solve based on the question

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Appended math results to math_answer
        """

        search_query = state['math_query']
        print(f'Step: Solving the math problem : "{search_query}"')
        
        result = wolfram_math_tool.run(search_query)
        return {
            "math_answer": result,
            "video_links" : []
        }

    def wiki_search(self, state):
        """
        Wikipedia search based on the question

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Appended wiki results to context
        """

        search_query = state['wiki_query']
        print(f'Step: Searching Wikipedia for : "{search_query}"')
        
        result = wiki_search_tool.run(search_query)
        return {
            "context": result,
            "video_links" : []
        }

    def email_search(self, state):
        """
        Email search based on the question

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Appended email results to context
        """

        search_query = state['mail_search_query']
        print(f'Step: Searching Inbox for: "{search_query}"')
        
        emails = searchMail._run(search_query)
        emails = [mail["snippet"] for mail in emails]
        return {
            "context": "".join(emails),
            "video_links" : []
        }

    def email_send(self, state):
        """
        Sending email based on the question

        Args:
            state (dict): The current graph state

        Returns:
            Status: Email sent
        """

        email_content = state['email']
        receiver = email_content['receiver']
        body = email_content['body']
        subject = email_content['subject']
        print(f'Step: Sending email to : "{receiver}"')
        
        sendMail._run(body, receiver,subject)
        
        return {
            "generation" : "Email sent succesfully",
            "video_links" : []
        }

    def evaluate_search_results(self, state):
        """
        Evaluate the search results and decide whether to proceed with generation
        or fallback to the other search method.

        Args:
            state (dict): The current graph state

        Returns:
            str: "generate" if the search results are satisfactory, "fallback" otherwise
        """
        print("Step: Evaluating Search Results")
        question = state["question"]
        context = state["context"]
        result_quality = eval_result_chain.invoke({"question": question,"context": context})
        
        if result_quality["score"] >= 3:
            print("Search results are satisfactory, proceeding with generation.")
            return "generate"
        else:
            print("Search results are not satisfactory, falling back to the other search method.")
            return "fallback"

    def route_question(self, state):
        """
        route question to web search or generation.

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """
        print("Step: Routing Query")
        question = state['question']
        image_is = state['imageAvailable']
        output = question_router.invoke({"question": question, "image_available" : image_is})
        if output['choice'] == "web_search":
            print("Step: Routing Query to Web Search")
            return "websearch"
        elif output['choice'] == "youtube_search":
            print("Step: Routing Query to Youtube Search")
            return "youtubesearch"
        elif output['choice'] == "wolfram_alpha":
            print("Step: Routing query to Mathematical Solving")
            return "mathsolve"
        elif output['choice'] == "wiki_search":
            print("Step: Routing Query to Wikipedia Search")
            return "wikisearch"
        elif output['choice'] == "gmail_search":
            print("Step: Routing query to Gmail Search")
            return "gmailsearch"
        elif output['choice'] == "send_gmail":
            print("Step: Routing Query to Gmail sender")
            return "sendgmail"
        elif output['choice'] == "image_search":
            print("Step: Routing Query Image analyser")
            return "imagesearch"
        elif output['choice'] == 'generate':
            print("Step: Routing Query to Generation")
            return "generate"