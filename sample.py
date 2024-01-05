import semantic_kernel as sk
import semantic_kernel.connectors.ai.hugging_face as sk_hf


kernel = sk.Kernel()

# Configure LLM service
kernel.add_text_completion_service(
    service_id="gpt2", 
    service=sk_hf.HuggingFaceTextCompletion(ai_model_id="gpt2", task="text-generation")
)
kernel.add_text_embedding_generation_service(
    service_id="sentence-transformers/all-MiniLM-L6-v2",
    service=sk_hf.HuggingFaceTextEmbedding(ai_model_id="sentence-transformers/all-MiniLM-L6-v2"),
)
kernel.register_memory_store(memory_store=sk.memory.VolatileMemoryStore())
kernel.import_skill(sk.core_skills.TextMemorySkill())

await kernel.memory.save_information_async(
    collection="animal-facts", id="info1", text="Sharks are fish."
)
await kernel.memory.save_information_async(
    collection="animal-facts", id="info2", text="Whales are mammals."
)
await kernel.memory.save_information_async(
    collection="animal-facts", id="info3", text="Penguins are birds."
)
await kernel.memory.save_information_async(
    collection="animal-facts", id="info4", text="Dolphins are mammals."
)
await kernel.memory.save_information_async(
    collection="animal-facts", id="info5", text="Flies are insects."
)

# Define semantic function using SK prompt template language
my_prompt = """I know these animal facts: {{recall $query1}} {{recall $query2}} {{recall $query3}} and """

# Create the semantic function
my_function = kernel.create_semantic_function(
    prompt_template=my_prompt, max_tokens=45, temperature=0.5, top_p=0.5
)
Let's now see what the completion 

context = kernel.create_new_context()
context[sk.core_skills.TextMemorySkill.COLLECTION_PARAM] = "animal-facts"
context[sk.core_skills.TextMemorySkill.RELEVANCE_PARAM] = "0.3"

context["query1"] = "animal that swims"
context["query2"] = "animal that flies"
context["query3"] = "penguins are?"
output = await kernel.run_async(my_function, input_vars=context.variables)

output = str(output).strip()

query_result1 = await kernel.memory.search_async(
    collection="animal-facts",
    query=context["query1"],
    limit=1,
    min_relevance_score=0.3
)
query_result2 = await kernel.memory.search_async(
    collection="animal-facts",
    query=context["query2"],
    limit=1,
    min_relevance_score=0.3
)
query_result3 = await kernel.memory.search_async(
    collection="animal-facts",
    query=context["query3"],
    limit=1,
    min_relevance_score=0.3
)

print(f"gpt2 completed prompt with: '{output}'")