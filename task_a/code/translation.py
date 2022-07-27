import os

EPOCHS = 4
BATCH_SIZE = 24 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

model_name = "facebook/nllb-200-distilled-600M"

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
en_text = "Do not meddle in the affairs of wizards, for they are subtle and quick to anger."
# Does not translate the swedish
fi_text = "Älä sekaannu velhojen asioihin, sillä ne ovat hienovaraisia ja nopeasti vihaisia, kan jag skriva svenska"
# Works, does not change the english
sv_text = "Jag vet inte vad jag ska skriva här, what happens if I write english"

eng = 'This is an english sentence.'
tam = 'Ithu yethu maathiri illama puthu maathiyaala irukku'
tam_script = "ஆண்ட சாதி, ஆண்ட சாதி  னு ஆயிரம் முறை சொல்லி , அதில் இருக்கும் தப்பை மட்டும் வெளிச்சம்போட்டு,எல்லாரையும் நோகடித்து, குற்ற உணர்ச்சி அடையச் செய்தால் அது புரட்சிப் படம். சமூகத்தில் இருக்கும் உண்மையான நாடகக் காதல் விஷயத்தை சொன்னால் அது சாதிப் படம். அவ்ளோதான் சார் போராலீஸ்"

tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="fi_FI")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

encoded_en = tokenizer(fi_text, return_tensors="pt")
generated_tokens = model.generate(encoded_en['input_ids'])
print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))

