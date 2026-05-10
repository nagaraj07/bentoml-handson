from __future__ import annotations
import bentoml

with bentoml.importing():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


EXAMPLE_INPUT = "Breaking News: In an astonishing turn of events, the small \
town of Willow Creek has been taken by storm as local resident Jerry Thompson's cat, \
Whiskers, performed what witnesses are calling a 'miraculous and gravity-defying leap.' \
Eyewitnesses report that Whiskers, an otherwise unremarkable tabby cat, jumped \
a record-breaking 20 feet into the air to catch a fly. The event, which took \
place in Thompson's backyard, is now being investigated by scientists for potential \
breaches in the laws of physics. Local authorities are considering a town festival \
to celebrate what is being hailed as 'The Leap of the Century."


my_image = bentoml.images.Image(python_version="3.11") \
        .python_packages("torch", "transformers")


@bentoml.service(
    image=my_image,
    resources={"cpu": "2"},
    traffic={"timeout": 30},
)
class Summarization:
    # Define the Hugging Face model as a class variable
    model_path = bentoml.models.HuggingFaceModel("sshleifer/distilbart-cnn-12-6")

    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
    
    @bentoml.api
    def summarize(self, text: str = EXAMPLE_INPUT) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        summary_ids = self.model.generate(inputs["input_ids"], max_length=150, min_length=30)
        result = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return f"Hello world! Here's your summary: {result}"
