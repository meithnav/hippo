import os
import json

import openai
from openai import OpenAI

KEY="ENTER KEY HERE"
client = OpenAI(api_key=KEY)


"""
Before submitting the assignment, describe here in a few sentences what you would have built next if you spent 2 more hours on this project:


Current framework generates a coherent, engaging story in a kid-friendly language with a moral at the end. However, its still quite short(~500-800 words). A simple recourse, is to feed each segment with overlapping content in [R3, Grove] format to yield longer story but it takes more API calls. My future directions would be to figure out a way to reduce the API calls and maintain the verbosity. 

GOAL -> 7-8k words (approx reading time 10-15mins)

Direct approach I tried was gpt-3.5-turbo for outline and GPT4o for generation. It is capable of generating longer verbose stories when given our outline. Can see the sample output in the output directory. Alternative, open-sources LLMs also perform quite well (Starling (eloquent and science fiction; Mistral-7B-OpenOrca creative, Llama-3.1-70B-Instruct, Llama 4 Maverick, DeepSeek V3-0324), and Claude 3.7 Sonnet

Exploring RAG-based approaches[using Textbox, Stories] seems infeasible, if we may proceed with __ due to its short context length. 

Prompt logs can be found in `prompt_logs.txt` which contains all the prompts development and problems with additional ablations and improvements. 

FUTURE DEVELOPMENTS: 
- Include the reader as the PROTAGONIST. Makes them feel more connected.  
- For the outline instead of prompting LLM we should first define a function that makes the meta-data by randomly selecting #scenes [1-7], #supporting-actors: [3-5], relations among them, character arc, moral (randomly from moral pool) ...} and then prompt the LLM to develop the outline. Offers more controllability and reliablitly. Then promopt the LLM again for Story using R3 manner(bit updated).
- For categorization I would still suggest adding it to the custom function where we have a pool for genre (adventure, fantasy, animals, satire, friendship...) and morals which we randomly sample(for simplicity) during the outline construction phase.


CLARIFICATIONS REQUIRED: 
Would like to know more about: 
- goal: to determine what does the company intend to achieve through these stories 
- target audience: [patients, education, entertainment...] to better determine the type of the stories


REFERENCE:
R3: https://arxiv.org/pdf/2210.06774
GROVE: https://openreview.net/pdf?id=mRETTyZEJa
TextBox: https://github.com/RUCAIBox/TextBox#dataset
Stories(100 genres with ~1000 words): https://github.com/FareedKhan-dev/NLP-1K-Stories-Dataset-Genres-100?tab=readme-ov-file
ELI5 - can be used for finetuning models to help generate simpler explanations 
Comprehensive Analysis of different models for Story Geenration: https://github.com/lechmazur/writing?tab=readme-ov-file
"""



def saveJson(response, file_path='./output/story.json') -> None:
    '''
        Desc      : stores the story 
        response  : generated story 
        file_path : location 
    '''

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump({'story': response}, f, indent=4)



def makeOutline(user_query="", user=None) -> str:
    '''
        Desc       : Generates prompt for Story Outline 
        user_query : User-fed story requirements 
        user       : User object containing attributes of the user(Personalization)
    '''

    ## Can call our custom functions to generate the plot I mentioned for controllable plot generation. ## 

    prompt = f'''
        ##TASK:## 
        Generate a STORY for a **{user['age'] if user else 7} year old** 

        ##USER QUERY:## 
        {user_query}


        ##PLANNING:##
        - Break down the USER QUERY to understand the mentioned actors, their relations, plot, genre. 
        - Then create dynamic and relatable characters that evolve throughout the story. 
        - Develop character personas, relatinships, and relatable discriptions. 
        - Story ideas need to explore different angles and perspectives  
        - Ensure the story is well-balanced, mixing action, dialogue, and description to maintain interest and momentum.
        - Weave in gentle humor and a light moral

        ##OUTPUT FORMAT:## 
        1. Give a title
        2. Chapter Outlines: Structure the story with a clear introduction(500 words) -> plot setting(800 words) -> problem(1000 words) ->  difficulties/opportunties in overcoming(2000 words) -> climax(1000 words) -> ending(500 words).   
        3. Moral 
        
        
       Be CREATIVE. 
        
    '''

    return prompt 

def makeStoryPrompt(outline, suggestions="") -> str:
    '''
        Desc        : Generates prompt for Story Generation 
        outline     : Story outine and pointers
        suggestions : user feedback to the story 
    '''
    
    return  f''' 
        ##OUTLINE:## 
        {outline}
        
        ##SUGGESTIONS (if any):##
        {suggestions}

        ##CONSTRAINTS:## 
        - Story is for 5-10 years old.
        - Simple and easy to follow.
        - Prevent harmful, foul, horrendous, and lewd language.
        - Positive themes and gentle humor only.
        - Tone: Uplifting, with a clear, gentle lesson.

        ##OUTPUT FORMAT:## 
        1. Give a title
        2. Story  
        3. Moral 
        
        Generate a LONG STORY following the OUTLINE. Be EXTREMELY VERBOSE. Be CREATIVE. 

        '''
        

def call_model(prompt: str, max_tokens=3000, temperature=0.1) -> str:
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        max_tokens=max_tokens,
        temperature=temperature
    )
    return resp.choices[0].message.content  # type: ignore



example_requests = "A story about a girl named Alice and her best friend Bob, who happens to be a cat."
user = {'age':7, "username":"meith12", "name":"Meith", "interests":[]} ## replace with actual user-info we have (if any)


def main():
    
    NUM_RETRY_ATTEMPTS=1
    user_input = input("What kind of story do you want to hear? ")
    # user_input = example_requests

    print("--> Generating Outline")
    outline_prompt = makeOutline(user_input, user)
    outline = call_model(outline_prompt, max_tokens=4096, temperature=0.5)
    
    print("--> Generating Story")
    story_prompt = makeStoryPrompt(outline)
    story = call_model(story_prompt, max_tokens=4096, temperature=0.5)
    
    print("--> Saved Story")
    saveJson(story, file_path='./output/story.json')
    print("\nSTORY:",story)
    
    
    #feedback
    while(NUM_RETRY_ATTEMPTS>0):
        feedback = input("\nDid you like the story? (yes/no) ")
        if feedback.lower() == "no":
            suggestions = input("Any particular suggestions")
            print("\nLet’s try again with a slightly different version!")
            story_prompt = makeStoryPrompt(outline, suggestions)
            story = call_model(story_prompt, max_tokens=4096, temperature=0.5)
        
            print("--> Saved Story")
            saveJson(story, file_path='./output/story.json')
            print("\nSTORY:",story)


if __name__ == "__main__":
    main()