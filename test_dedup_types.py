from src.ontodisco.utils import dedup_base
from dataclasses import dataclass, field
from collections import defaultdict
import json
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from src.ontodisco.utils.openai_utils import LLMTripletExtractor
import os
import numpy as np

from pymongo.mongo_client import MongoClient
load_dotenv()

def get_mongo_client(mongo_uri):
    client = MongoClient(mongo_uri)
    return client

client = get_mongo_client("mongodb://localhost:27018/?directConnection=true")

db = client.get_database("musique_gpt4_1_mini_onto_triplets")

types = []
for triplet in db.get_collection("initial_triplets").find({}, {"_id": 0, "subject_type": 1, "object_type": 1}):
    types.append(triplet["subject_type"])
    types.append(triplet["object_type"])
types = list(types)

normalizer = dedup_base.normalize_label
all_labels = types
embedder = dedup_base.ContrieverEmbedder()
embed_batch_size = 100
verifier = LLMTripletExtractor(model="Qwen/Qwen3-235B-A22B-Instruct-2507", api_key=os.getenv("AIRI_KEY"), base_url=os.getenv("AIRI_BASE_URL"))

normalized_to_raws: dict[str, set[str]] = defaultdict(set)
surface_form_counts: dict[str, int] = defaultdict(int)
count_per_normalized: dict[str, int] = defaultdict(int)

for label in all_labels:
    norm = normalizer(label)
    normalized_to_raws[norm].add(label)
    surface_form_counts[label] += 1
    count_per_normalized[norm] += 1

unique_labels = sorted(normalized_to_raws.keys())


print(f"After normalisation: {len(unique_labels)} unique labels (from {len(set(all_labels))} raw labels)")

texts = unique_labels
embeddings = embedder.embed(texts, batch_size=embed_batch_size)

cluster_labels = dedup_base.cluster_hac(embeddings, threshold=0.75, linkage='average')

clusters: dict[int, list[str]] = defaultdict(list)
for norm_label, cl in zip(unique_labels, cluster_labels):
    clusters[int(cl)].append(norm_label)

multi_member = sum(1 for m in clusters.values() if len(m) > 1)
print(f"HAC produced {len(clusters)} clusters ({multi_member} with 2+ members, requiring LLM verification)")
print(f"Mean cluster size for clusters with 2+ members: {np.mean([len(v) for v in clusters.values() if len(v) > 1])}")

verified_groups = dedup_base.verify_clusters_with_llm(clusters, verifier, verbose=True, token_log_every=100)

if not os.path.exists("onto_artifacts"):
    os.makedirs("onto_artifacts")
with open("onto_artifacts/verified_groups.json", "w") as f:
    json.dump(verified_groups, f)

prompt_tokens, completion_tokens = verifier.calculate_used_tokens()
print(f"Prompt tokens: {prompt_tokens}")
print(f"Completion tokens: {completion_tokens}")
print(f"Total tokens: {prompt_tokens + completion_tokens}")