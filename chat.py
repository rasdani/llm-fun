from transformers import AutoTokenizer
import transformers
import torch

#model = "llamaste/Llama-2-70b-chat-hf"
# model = "./Llama-2-7b-chat-hf"
model = "./Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
    # device_map=0,
)
# prompt = """
#   Cook-Bot: I am your friendly Cook-Bot. Ask me anything you want to know about cooking.
#   User:     What do I do with a potato?
#   Cook-Bot: You can make mashed potatoes or a potato salad.
#   User:     What do I need for a potato salad?
#   Cook-Bot: Salt, cream, pepper, and potato.
#   User: How do I make good scrambled eggs?
#   Cook-Bot: 
# """

def format_prompt(user_message, system_prompt):
    template = """
    <s>[INST] <<SYS>>
    {system_prompt}
    <</SYS>>

    {user_message} [/INST]
    """
    
    return template.format(system_prompt=system_prompt, user_message=user_message)

# # Example usage:
# system_prompt = """
# You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

# If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
# """

# user_message = "There's a llama in my garden 😱 What should I do?"
# prompt = format_prompt(user_message, system_prompt)
# prompt = """
#     Only fill in the missing response of Cook-Bot at the end. 
#     DO NOT GO BEYOND Cook-Bot's response.
#     Generate the stop token, when Cook-Bot's last reponse is finished.
#     DO NOT FILL IN FOR User.

#     Cook-Bot: Hello! I am Cook-Bot, your personal culinary assistant. I'm here to guide you through recipes, cooking techniques, and answer any questions you might have about food and cooking. Here are a few examples of how our conversation might go:
#     User: Can you suggest a quick breakfast recipe?
#     Cook-Bot: Sure! How about avocado toast with a poached egg on top? Just mash a ripe avocado on toasted bread, season with salt and pepper, and top with a poached egg. You can also sprinkle some chili flakes for an extra kick!
#     User: That sounds delicious. How do I make the poached egg?
#     Cook-Bot: For dessert, I recommend a classic chocolate mousse. You'll need dark chocolate, egg yolks, sugar, and whipped cream. Melt the chocolate, mix with the egg yolks and sugar, and then fold in the whipped cream. Chill in the fridge for a few hours, and voila! A rich and creamy chocolate mousse.
#     User: Sounds amazing! I'll definitely try that.
#     Cook-Bot: I'm glad you liked the suggestion. If you have any more questions or need recipes, just ask. Happy cooking!
#     User: Can you suggest a vegetarian dish?
#     Cook-Bot: 
# """
prompt = """Only generate Cook-Bot's response based on the latest User's question. 
    DO NOT reference previous interactions. 

    User: What's a quick breakfast recipe?
    Cook-Bot: A quick breakfast idea is a yogurt parfait. Layer Greek yogurt, granola, and fresh berries in a glass. Drizzle with honey for added sweetness. Enjoy!

    User: How do I make a poached egg?
    Cook-Bot: To make a poached egg, bring a pot of water to a gentle simmer and add a splash of vinegar. Crack an egg into a small bowl, and gently slide it into the simmering water. Let it cook for 3-4 minutes for a soft yolk. Use a slotted spoon to remove, drain on a paper towel, season, and serve.

    User: Can you suggest a dessert?
    Cook-Bot: Certainly! How about a chocolate lava cake? It's a rich and decadent dessert with a molten chocolate center. Top with vanilla ice cream for a delightful contrast.

    User: {user_request}
    Cook-Bot: """

user_request = "Can you suggest a vegetarian dish?"
formatted_prompt = prompt.format(user_request=user_request)

sequences = pipeline(
    # 'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
    # prompt,
    formatted_prompt,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    # eos_token_id=tokenizer.eos_token_id,
    max_length=1000,
)
# for seq in sequences:
#     print(f"Result: {seq['generated_text']}")
# print(f"{len(sequences)}")
# text = sequences[0]['generated_text']
# eos_pos = text.find(tokenizer.eos_token)
# if eos_pos != -1:
#     text = text[:eos_pos]
# print(f"Result: {text}")


text = sequences[0]['generated_text']
print(f"{text=}")
print("====================================")
# Extract Cook-Bot's response based on the user_request
try:
    # Find the position of the user_request
    user_pos = text.find(user_request)
    
    # If user_request is found, find the corresponding "Cook-Bot:" response right after it
    if user_pos != -1:
        cookbot_pos = text.find("Cook-Bot:", user_pos)
        
        # Extract the response until the next "User:" or end of the string
        next_user_pos = text.find("User:", cookbot_pos)
        
        if next_user_pos != -1:
            cookbot_response = text[cookbot_pos + 9:next_user_pos].strip()  # +9 to skip "Cook-Bot:"
        else:
            cookbot_response = text[cookbot_pos + 9:].strip()
        
        print(f"Cook-Bot's Response: {cookbot_response}")
    else:
        print("User request not found in the generated text.")

except Exception as e:
    print(f"Error: {e}")