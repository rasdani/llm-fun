import argparse
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

def get_cookbot_response(user_request):
    formatted_prompt = prompt.format(user_request=user_request)

    sequences = pipeline(
        formatted_prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        max_length=1000,
    )

    text = sequences[0]['generated_text']

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
            
            return cookbot_response
        else:
            return "Sorry, I couldn't understand the request."

    except Exception as e:
        return f"Error: {e}"
def interactive_chat():
    print("Welcome to Cook-Bot! Ask your cooking questions and I'll try to help.")
    print("Type 'exit' anytime to end the chat.\n")
    
    while True:
        user_request = input("User: ").strip()
        
        if user_request.lower() in ["exit", "quit", "bye"]:
            print("Cook-Bot: Goodbye! Happy cooking!")
            break
        
        response = get_cookbot_response(user_request)
        print(f"Cook-Bot: {response}\n")

def simulate_chat():
    print("Welcome to Cook-Bot! Simulating user questions...\n")
    
    simulated_user_requests = [
        "How do I make spaghetti?",
        "Tell me a recipe for apple pie.",
        "What goes well with roasted chicken?",
        "How long should I bake cookies?",
        "Can you suggest a vegetarian dish?",
    ]
    
    for user_request in simulated_user_requests:
        print(f"User: {user_request}")
        response = get_cookbot_response(user_request)
        print(f"Cook-Bot: {response}\n")
    
    print("Simulation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Cook-Bot in different modes.")
    parser.add_argument("-i", "--interactive", help="Run in interactive chat mode", action="store_true")
    args = parser.parse_args()

    if args.interactive:
        interactive_chat()
    else:
        simulate_chat()
