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
    device_map=0,
)

def update_prompt(prompt, user_request, cookbot_response):
    # Append the latest user request and Cook-Bot's response to the prompt
    updated_prompt = f"{prompt}\n    User: {user_request}\n    Cook-Bot: {cookbot_response}"
    return updated_prompt

def get_cookbot_response(prompt, user_request):
    formatted_prompt = f"{prompt}\n    User: {user_request}\n    Cook-Bot:"
    
    sequences = pipeline(
        formatted_prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        max_length=2000,
    )

    text = sequences[0]['generated_text']

    # Extract Cook-Bot's response after the user_request
    start_pos = text.find(f"User: {user_request}") + len(f"User: {user_request}")
    end_pos = text.find("User:", start_pos)
    cookbot_response = text[start_pos:end_pos].replace("Cook-Bot:", "").strip()

    return cookbot_response

def interactive_chat(prompt):
    print("Welcome to Cook-Bot! Ask your cooking questions and I'll try to help.")
    print("Type 'exit' anytime to end the chat.\n")
    
    while True:
        user_request = input("User: ").strip()
        
        if user_request.lower() in ["exit", "quit", "bye"]:
            print("Cook-Bot: Goodbye! Happy cooking!")
            break
        
        response = get_cookbot_response(prompt, user_request)
        print(f"Cook-Bot: {response}\n")

        # Update the prompt with the new chat history
        prompt = update_prompt(prompt, user_request, response)

    return prompt

def simulate_chat(prompt):
    print("Welcome to Cook-Bot! Simulating user questions...\n")
    
    simulated_user_requests = [
        "What goes well with roasted chicken?",
        "How long should I bake cookies?",
        "Can you suggest a vegetarian dish?"
    ]
    
    for user_request in simulated_user_requests:
        print(f"User: {user_request}")
        response = get_cookbot_response(prompt, user_request)
        print(f"Cook-Bot: {response}\n")

        # Update the prompt with the new chat history
        prompt = update_prompt(prompt, user_request, response)
    
    print("Simulation complete!")
    return prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Cook-Bot in different modes.")
    parser.add_argument("-i", "--interactive", help="Run in interactive chat mode", action="store_true")
    args = parser.parse_args()

    prompt = """Only generate Cook-Bot's response based on the latest User's question. 

    User: What's a quick breakfast recipe?
    Cook-Bot: A quick breakfast idea is a yogurt parfait. Layer Greek yogurt, granola, and fresh berries in a glass. Drizzle with honey for added sweetness. Enjoy!
    User: How do I make a poached egg?
    Cook-Bot: To make a poached egg, bring a pot of water to a gentle simmer and add a splash of vinegar. Crack an egg into a small bowl, and gently slide it into the simmering water. Let it cook for 3-4 minutes for a soft yolk. Use a slotted spoon to remove, drain on a paper towel, season, and serve.
    User: Can you suggest a dessert?
    Cook-Bot: Certainly! How about a chocolate lava cake? It's a rich and decadent dessert with a molten chocolate center. Top with vanilla ice cream for a delightful contrast."""

    if args.interactive:
        prompt = interactive_chat(prompt)
    else:
        prompt = simulate_chat(prompt)
    
    # Save the prompt (chat history) to a file
    with open('chat_history.txt', 'w') as f:
        f.write(prompt)
