from dataclasses import dataclass, field
from collections import defaultdict
import os
import json
from dotenv import load_dotenv

load_dotenv()  # before project imports so ONTODISCO_LOG_LEVEL is applied correctly

from pymongo.mongo_client import MongoClient
import numpy as np

from src.ontodisco.utils import dedup_base
from src.ontodisco.utils.openai_utils import LLMTripletExtractor
from src.ontodisco.relation_dedup import deduplicate_relations



def get_mongo_client(mongo_uri):
    client = MongoClient(mongo_uri)
    return client

client = get_mongo_client("mongodb://localhost:27018/?directConnection=true")

db = client.get_database("musique_gpt4_1_mini_onto_triplets")

embedder = dedup_base.ContrieverEmbedder(device="cpu")
embed_batch_size = 100
verifier = LLMTripletExtractor(model="Openai/Gpt-oss-120b", api_key=os.getenv("AIRI_KEY"), base_url=os.getenv("AIRI_BASE_URL"))

all_relation_surface_forms = db.get_collection("initial_triplets").find({}, {"_id": 0, "subject_type": 1, "object_type": 1, "relation": 1})
all_relation_surface_forms = list(all_relation_surface_forms)

relation_dedup_result = deduplicate_relations(triplets=all_relation_surface_forms, llm_extractor=verifier, similarity_threshold=0.85)

dedup_output = []
for item in relation_dedup_result.items.values():
    dedup_output.append((item.canonical_label, item.surface_forms))
    
prompt_tokens, completion_tokens = verifier.calculate_used_tokens()
print(f"Prompt tokens: {prompt_tokens}")
print(f"Completion tokens: {completion_tokens}")
print(f"Total tokens: {prompt_tokens + completion_tokens}")

with open("relation_dedup_result.json", "w") as f:
    json.dump(dedup_output, f)