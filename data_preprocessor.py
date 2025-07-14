import json
import ollama
from typing import List, Dict, Any
from dataclasses import dataclass
import csv

@dataclass
class InputLog:
    Timestamp:str
    EventID:int
    EventType:str
    Username:str
    IPAddress_Source:str
    IPAddress_Destination:str
    Authentication_Status:str
    Malware_Detection_Status:str
    Firewall_Action:str

@dataclass
class TrainingExample:
    prompt: str
    response: str

class LocalOllamaProcessor:
    def __init__(self, model_name: str = "llama3.2:latest"):
        self.model_name = model_name
    
    
    def _generate(self, prompt: str) -> str:
        """Generate response using local Ollama model"""
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': 0.3,
                    'num_predict': 1000
                }
            )
            return response['response']
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""
    
    def process_diagnostic_log(self, data: InputLog) -> List[TrainingExample]:
        """Process diagnostic log data"""
        prompt = f"""
        You are an expert network systems logs processor. Your task is to analyze the everyday logs of the IPs and
         create high quality training samples for fine-tuning a language model for this Data : {data}

        Guidelines:
        1. Extract all relevant technical information (Event types, nature of events, timestamps, IPs, 
        Malwares)
        2. Create diverse question answer pairs that would help obtain more information on the IPs,
            the event types, what happened on what dates, and the critical events that have occurred.
        3. Format your response as JSON with the following structure:
        
        Create diverse training examples in this format:

        EXAMPLE 1:
        PROMPT: question about the text
        RESPONSE: detailed automotive answer
        
        EXAMPLE 2:
        PROMPT: question about the text
        RESPONSE: detailed automotive answer

        Focus on creating samples that would help a model. For every output, make sure there is a PROMPT and a RESPONSE:
        - Explain technical issues
        - Provide day-to-day functioning details
        - Interpret events.
        """
        
        response = self._generate(prompt)
        return self._parse_examples(response)
    
    def _parse_examples(self, response: str) -> List[TrainingExample]:
        try:
            """Parse the response into training examples"""
            examples = []
            print('Response!! - ',response)
            # print('Response list! - ', response["response"])
            parts = response.split("EXAMPLE")
            for part in parts[1:]: 
                lines = part.strip().split('\n')
                prompt = ""
                response_text = ""
                
                for line in lines:
                    line = line.strip()
                    if line.startswith("PROMPT:"):
                        prompt = line.replace("PROMPT:", "").strip()
                    elif line.startswith("RESPONSE:"):
                        response_text = line.replace("RESPONSE:", "").strip()
                    elif response_text and not line.startswith("EXAMPLE"):
                        response_text += " " + line
                
                if prompt and response_text:
                    examples.append(TrainingExample(
                        prompt=prompt,
                        response=response_text
                    ))
        
            return examples
        except Exception as e:
            print(f'Exception! {e}')
            return []
    
    def process_data(self, data: InputLog) -> List[TrainingExample]:
        """Process any automotive data"""  
        return self.process_diagnostic_log(data)
       
    def process_batch(self, data_list: List[InputLog]) -> List[TrainingExample]:
        """Process multiple data items"""
        all_examples = []
        
        for i, data in enumerate(data_list):
            print(f"Processing item {i+1}/{len(data_list)}")
            examples = self.process_data(data)
            all_examples.extend(examples)
        
        return all_examples
    
    def save_training_data(self, examples: List[TrainingExample], filename: str):
        """Save training examples to file"""
        with open(filename, 'w') as f:
            for example in examples:
                print('Examples!! - ', example)
                training_item = {
                    "prompt": example.prompt,
                    "response": example.response
                }
                f.write(json.dumps(training_item) + '\n')
        
        print(f"Saved {len(examples)} training examples to {filename}")

def main():
    processor = LocalOllamaProcessor()
    sample_data = []

    csv_file_path = 'network_logs_minimal.csv' 
    sample_data= []
    with open(csv_file_path, mode='r', newline='') as file:
        reader = csv.DictReader(file)

        for row in reader:
            to_append = {
                "Timestamp": row['Timestamp'],
                "Event ID": row['Event ID'],
                "Event Type": row['Event Type'],
                "Username": row['Username'],
                "IP Address (Source)": row['IP Address (Source)'],
                "IP Address (Destination)": row['IP Address (Destination)'],
                "Authentication Status": row['Authentication Status'],
                "Malware Detection Status": row['Malware Detection Status'],
                "Firewall Action": row['Firewall Action']
            }
            sample_data.append(to_append)

    # sample_data = [{
	# "Timestamp":"04-01-2025 13:16",
	# "Event ID":6466,
	# "Event Type":"MALWARE ALERT",
	# "Username":"David",
	# "IP Address (Source)":"192.168.102.87",
	# "IP Address (Destination)":"10.0.176.174",
	# "Authentication Status":"FAILURE",
	# "Malware Detection Status":"Clean",
	# "Firewall Action":"BLOCKED"
    # }]
    

    examples = processor.process_batch(sample_data)
    
    processor.save_training_data(examples, "training_data.jsonl")

    print(f"\nGenerated {len(examples)} training examples:")
    for i, example in enumerate(examples[:3]):
        print(f"\nExample {i+1}:")
        print(f"Prompt: {example.prompt}")
        print(f"Response: {example.response[:100]}...")

if __name__ == "__main__":
    main()