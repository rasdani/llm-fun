def preprocess_data(file_path):
    data = []
    entry = {}

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith("------------------------------------------------------------------"):
                if entry:
                    data.append(entry)
                    entry = {}
                continue

            if line.startswith("id:"):
                entry['id'] = line.split("id:")[1].strip()
            elif line.startswith("Titel:"):
                entry['title'] = line.split("Titel:")[1].strip()
            elif line.startswith("Kontext:"):
                entry['context'] = line.split("Kontext:")[1].strip()
            elif line.startswith("Frage:"):
                entry['question'] = line.split("Frage:")[1].strip()
            elif line.startswith("Antwort:"):
                answer_text = line.split("Antwort:")[1].strip()
                answer_start = entry['context'].find(answer_text)
                entry['answers'] = {'text': [answer_text], 'answer_start': [answer_start]}

        if entry and entry not in data:
            data.append(entry)

    return data

processed_data = preprocess_data("kauz_website_data_raw.txt")

# Save to JSON if you wish:
import json
with open("processed_data.json", "w", encoding="utf-8") as f:
    json.dump(processed_data, f, ensure_ascii=False, indent=4)
