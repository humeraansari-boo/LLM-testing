import os

from dotenv import load_dotenv
import time
from typing import Any
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

class ShoeStoreRag:
    def __init__(self, pine_cone_api_key, openai_api_key):
        self.pine_cone_client = Pinecone(api_key=pine_cone_api_key)
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.index_name = "shoe-store-kb"

        self.index = None
        self.setup_pinecone_index()
        self.populate_knowledge_base()

    def setup_pinecone_index(self):
        existing_indices = [index.name for index in self.pine_cone_client.list_indexes()]
        if self.index_name not in existing_indices:
            print("Creating new pinecone index: " + self.index_name)
            self.pine_cone_client.create_index(
                name=self.index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
        self.index = self.pine_cone_client.Index(self.index_name)


    def get_embedding(self, text):
        # text can be a single string or a list of strings
        response = self.openai_client.embeddings.create(model="text-embedding-3-small",
                                                        input=text)
        if isinstance(text, str):
            return response.data[0].embedding
        return [item.embedding for item in response.data]

    def populate_knowledge_base(self):
        documents: list[dict[str, Any]] = [
            {
                "id": "policy_returns",
                "text": "We offer a 30-day full refund policy at no extra cost for all shoes. No questions asked if you're not satisfied with your purchase.",
                "metadata": {"category": "returns", "topic": "refund_policy"}
            },
            {
                "id": "policy_delivery_times",
                "text": "When will I get my shoes? Orders are delivered in 3-5 business days with standard shipping, or 1-2 business days with express shipping.",
                "metadata": {"category": "shipping", "topic": "delivery_times"}
            },
            {
                "id": "policy_shipping_costs",
                "text": "How much does shipping cost? Shipping is free on orders over $75. Express shipping costs an extra $15.",
                "metadata": {"category": "shipping", "topic": "shipping_costs"}
            },
            {
                "id": "inventory_sizes",
                "text": "Our shoe sizes range from US 5 to US 15 in both men's and women's styles. We also carry wide and narrow width options for most models.",
                "metadata": {"category": "inventory", "topic": "sizes_availability"}
            },
            {
                "id": "warranty_athletic",
                "text": "All our athletic shoes come with a 1-year warranty against manufacturing defects. This covers sole separation, stitching issues, and material defects.",
                "metadata": {"category": "warranty", "topic": "product_warranty"}
            },
            {
                "id": "inventory_brands",
                "text": "We carry popular brands including Nike, Adidas, New Balance, Converse, Vans, and our exclusive store brand ComfortWalk.",
                "metadata": {"category": "inventory", "topic": "brands"}
            },
            {
                "id": "store_hours",
                "text": "Our store hours are Monday-Saturday 9 AM to 9 PM, Sunday 11 AM to 7 PM. We're located at 123 Main Street, downtown shopping district.",
                "metadata": {"category": "store_info", "topic": "hours_location"}
            },
            {
                "id": "services_fitting",
                "text": "We offer professional shoe fitting services. Our certified fitters can measure your feet and recommend the best size and width for optimal comfort.",
                "metadata": {"category": "services", "topic": "fitting_service"}
            },
            {
                "id": "discounts_student",
                "text": "Student discounts are available - show your student ID for 15% off your purchase. Military personnel receive 20% discount with valid military ID.",
                "metadata": {"category": "discounts", "topic": "student_military"}
            },
            {
                "id": "loyalty_rewards",
                "text": "We have a loyalty program called SoleRewards. Earn 1 point for every dollar spent, get $5 off for every 100 points earned.",
                "metadata": {"category": "loyalty", "topic": "rewards_program"}
            },
            {
                "id": "services_custom",
                "text": "Custom shoe orders are available for select brands. Custom orders typically take 4-6 weeks to complete and require a 50% deposit upfront.",
                "metadata": {"category": "services", "topic": "custom_orders"}
            }
        ]

        # Skip re-embedding if the knowledge base is already uploaded
        if self.index.describe_index_stats().total_vector_count >= len(documents):
            print("Knowledge base already populated")
            return

        # Embed all documents in a single API call
        texts = [doc["text"] for doc in documents]
        embeddings = self.get_embedding(texts)

        vectors_to_upsert = []
        for doc, embedding in zip(documents, embeddings):
            vectors_to_upsert.append(
                {
                    "id": doc["id"],
                    "values": embedding,
                    "metadata": {
                        **doc["metadata"],
                        "text": doc["text"]
                    }
                })
        if vectors_to_upsert:
            self.index.upsert(vectors=vectors_to_upsert)

        print(f"Waiting for vectors to be available")
        time.sleep(5)

    def retrieve_context(self, query, top_k=3, min_score=0.45, max_gap=0.1):
        query_embedding = self.get_embedding(query)

        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        best_score = results["matches"][0]["score"] if results["matches"] else 0
        context_docs = []
        for doc in results["matches"]:
            # Skip weak matches, and matches much weaker than the best one,
            # so unrelated documents don't reach the answer
            if doc["score"] < min_score or doc["score"] < best_score - max_gap:
                continue
            if 'text' in doc["metadata"]:
                context_docs.append(doc["metadata"]["text"])
        return context_docs if context_docs else ["No relevant document found"]

    def generate_answer(self, query, context):
        context_str = "\n\n".join(context)
        prompt = f"""You're a helpful customer service assistant for a shoe store, respond to every
                  customer's questions respectfully and accurately in a friendly way
                  context = {context_str}
                  Customer question {query}
                  Please provide a helpful and factual answer with respect to the context.
                  If context doesn't have the answer, say so politely
                  """

        response= self.openai_client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                ChatCompletionSystemMessageParam(role="system",
                                                 content="You're a helpful customer service assistant for a shoe store"),
                ChatCompletionUserMessageParam(role="user", content=prompt)
            ],
            max_tokens=200,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()


if __name__ == "__main__":
    load_dotenv()
    pine_cone_key = os.getenv("PINE_CONE_KEY")
    open_ai_key = os.getenv("OPENAI_API_KEY")
    rag = ShoeStoreRag(pine_cone_key, open_ai_key)
