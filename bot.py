from transformers import AutoTokenizer
import transformers
import torch

# model = "./llama-2-7b-chat-hf"
model = "./Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map=0,
)

def format_prompt(user_message, system_prompt):
    template = """
    <s>[INST] <<SYS>>
    {system_prompt}
    <</SYS>>

    {user_message} [/INST]
    """
    
    return template.format(system_prompt=system_prompt, user_message=user_message)

# Example usage:
system_prompt = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
"""

user_message = "There's a llama in my garden 😱 What should I do?"
formatted_prompt = format_prompt(user_message, system_prompt)


# Starting point for the chat
# chat_history = """
# You are a conversational Chat-Bot, that acts as an cooking assistant.

# A SYSTEM prompt is provided, which instructions you have to obey at all times!
# SYSTEM prompt: Your response will be printed to terminal. Use ONLY easy decodable characters! Do NOT use emojis!

# Your conversations should obey the following pattern. Apply it by putting your response right after the last Cook-Bot: .

# Cook-Bot: Hello! I am Cook-Bot, your personal culinary assistant. I'm here to guide you through recipes, cooking techniques, and answer any questions you might have about food and cooking. Here are a few examples of how our conversation might go:
# User: Can you suggest a quick breakfast recipe?
# Cook-Bot: Sure! How about avocado toast with a poached egg on top? Just mash a ripe avocado on toasted bread, season with salt and pepper, and top with a poached egg. You can also sprinkle some chili flakes for an extra kick!
# User: That sounds delicious. How do I make the poached egg?
# Cook-Bot: To make a poached egg, bring a pot of water to a gentle simmer. Add a splash of vinegar. Crack an egg into a small bowl, and then gently slide it into the simmering water. Let it cook for about 3-4 minutes for a soft yolk. Use a slotted spoon to remove the egg and drain on a paper towel. Season with salt and pepper, and it's ready to serve!
# User: Great, thank you! What about a dessert recipe?
# Cook-Bot: For dessert, I recommend a classic chocolate mousse. You'll need dark chocolate, egg yolks, sugar, and whipped cream. Melt the chocolate, mix with the egg yolks and sugar, and then fold in the whipped cream. Chill in the fridge for a few hours, and voila! A rich and creamy chocolate mousse.
# User: Sounds amazing! I'll definitely try that.
# Cook-Bot: I'm glad you liked the suggestion. If you have any more questions or need recipes, just ask. Happy cooking!
# """

chat_history = """
    Only fill in the missing response of Cook-Bot at the end. 
    DO NOT GO BEYOND Cook-Bot's response.
    DO NOT FILL IN FOR User.
    Return the stop token, when Cook-Bot's last reponse is finished.

    Cook-Bot: Hello! I am Cook-Bot, your personal culinary assistant. I'm here to guide you through recipes, cooking techniques, and answer any questions you might have about food and cooking. Here are a few examples of how our conversation might go:
    User: Can you suggest a quick breakfast recipe?
    Cook-Bot: Sure! How about avocado toast with a poached egg on top? Just mash a ripe avocado on toasted bread, season with salt and pepper, and top with a poached egg. You can also sprinkle some chili flakes for an extra kick!
    User: That sounds delicious. How do I make the poached egg?
    Cook-Bot: To make a poached egg, bring a pot of water to a gentle simmer. Add a splash of vinegar. Crack an egg into a small bowl, and then gently slide it into the simmering water. Let it cook for about 3-4 minutes for a soft yolk. Use a slotted spoon to remove the egg and drain on a paper towel. Season with salt and pepper, and it's ready to serve!
    User: Great, thank you! What about a dessert recipe?
    Cook-Bot: For dessert, I recommend a classic chocolate mousse. You'll need dark chocolate, egg yolks, sugar, and whipped cream. Melt the chocolate, mix with the egg yolks and sugar, and then fold in the whipped cream. Chill in the fridge for a few hours, and voila! A rich and creamy chocolate mousse.
    User: Sounds amazing! I'll definitely try that.
    Cook-Bot: I'm glad you liked the suggestion. If you have any more questions or need recipes, just ask. Happy cooking!
    User: Can you suggest a vegetarian dish?
    Cook-Bot: 
"""

# Pre-defined set of questions for the simulation
questions = [
    "What's a good recipe for chicken?",
    # "How long should I bake a cake?",
    # "Can you suggest a vegetarian dish?",
    # "What spices go well with fish?",
    # "How can I make my pasta sauce creamier?"
]

for user_input in questions:
    print(f"User: {user_input}")
    chat_history += f"User: {user_input}\nCook-Bot: "

#     # Generate bot response
    sequences = pipeline(
        chat_history,
        # formatted_prompt
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=2000,  # increased max length to provide more room for conversation
    )

    # Extract and print the generated text
    print(f"{sequences=}")
    bot_response = sequences[0]['generated_text'].split("Cook-Bot: ")[-1]
    print(f"Cook-Bot: {bot_response}")

    # Update chat history
    chat_history += bot_response + "\n"
