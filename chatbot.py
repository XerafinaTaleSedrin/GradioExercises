from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/blenderbot-400M-distill"

# Load model (download on first run and reference local installation for consequent runs)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

conversation_history = []
while True:

    history_string = "\n".join(conversation_history)
    
    #get user input
    input_text = input("> ")

    #toxenize interaction for ease of parsing
    inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")

    # Generate response from model
    outputs = model.generate(**inputs)

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    # Talk back to user
    print(response)

    ## Add interaction to conversation history
    conversation_history.append(input_text)
    conversation_history.append(response)

