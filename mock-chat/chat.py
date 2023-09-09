from transformers import AutoTokenizer
import transformers
import torch

model = "./Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    # device_map="auto",
    device_map=0,
)

prompt = """Only generate Cook-Bot's response based on the latest User's question. 

    User: What's a quick breakfast recipe?
    Cook-Bot: A quick breakfast idea is a yogurt parfait. Layer Greek yogurt, granola, and fresh berries in a glass. Drizzle with honey for added sweetness. Enjoy!
    User: How do I make a poached egg?
    Cook-Bot: To make a poached egg, bring a pot of water to a gentle simmer and add a splash of vinegar. Crack an egg into a small bowl, and gently slide it into the simmering water. Let it cook for 3-4 minutes for a soft yolk. Use a slotted spoon to remove, drain on a paper towel, season, and serve.
    User: Can you suggest a dessert?
    Cook-Bot: Certainly! How about a chocolate lava cake? It's a rich and decadent dessert with a molten chocolate center. Top with vanilla ice cream for a delightful contrast.
    User: {user_request}
    Cook-Bot:"""

# user_request = "Can you suggest a vegetarian dish?"
user_request = "How do I cook Coq au Vin?"
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