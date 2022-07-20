from transformers import AutoTokenizer
#import os 
#import torch
#os.environ["CUDA_VISIBLE_DEVICES"]="6"

#device = torch.device("cuda")

eng = 'This is an english sentence.'
tam = 'Ithu yethu maathiri illama puthu maathiyaala irukku'
tam_script = "ஆண்ட சாதி, ஆண்ட சாதி  னு ஆயிரம் முறை சொல்லி , அதில் இருக்கும் தப்பை மட்டும் வெளிச்சம்போட்டு,எல்லாரையும் நோகடித்து, குற்ற உணர்ச்சி அடையச் செய்தால் அது புரட்சிப் படம். சமூகத்தில் இருக்கும் உண்மையான நாடகக் காதல் விஷயத்தை சொன்னால் அது சாதிப் படம். அவ்ளோதான் சார் போராலீஸ்"

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-xlm-r-multilingual-v1')

test = tam

print(f"original: {test}\n")

encoded_input = tokenizer.encode_plus(
                                test,            
                                add_special_tokens = True,
                                max_length = 512,
                                #padding = 'max_length',
                                #return_attention_mask = True,
                                truncation=True,
                                return_tensors='pt')



print(f"encoded input: {encoded_input['input_ids'][0]}\n")
#decoded_input = tokenizer.decode(encoded_input['input_ids'][0])
decoded_input = tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0])
print(f"decoded tokens: {decoded_input}")