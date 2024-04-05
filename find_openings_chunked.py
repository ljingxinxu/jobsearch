# -*- coding: utf-8 -*-

import os

from modal import Image, Volume, Secret, Stub, enter, exit, gpu, method, web_endpoint
from pathlib import Path
MODEL_DIR = "/model"
BASE_MODEL = "google/gemma-7b-it"


stub = Stub(name='job_scraper')

volume = Volume.from_name('jobs-volume', create_if_missing=True)

def download_model_to_folder():
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(MODEL_DIR, exist_ok=True)

    snapshot_download(
        BASE_MODEL,
        local_dir=MODEL_DIR,
        token=os.environ["HF_TOKEN"],
        ignore_patterns=["*.pt", "*.gguf"],
    )
    move_cache()

def retrieve_sources(sources_refs, texts):
    clean_indices = [
        r.replace("-pl","").strip() for r in sources_refs.split(",")
    ]
    numeric_indices = (int(r) if r.isnumeric() else None for r in clean_indices)
    return [
        texts[i] if i is not None else "INVALID SOURCE" for i in numeric_indices
    ]

#@stub.function()
def qanda_langchain(input, query):
    from langchain.chains.qa_with_sources import load_qa_with_sources_chain
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.llms import OpenAI
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.vectorstores.faiss import FAISS

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    print('splitting')

    # does split_text take read_text() or string
    texts = text_splitter.split_text(input)
    print(len(texts))
    global docsearch

    if not docsearch and len(texts) <=3:
        print('generating docsearch indexer')
        embeddings = OpenAIEmbeddings().embed_documents()
        docsearch = FAISS.from_texts(
            texts,
            OpenAIEmbeddings(chunk_size=5),
            metadatas=[{"source": i} for i in range(len(texts))],
        )
    print("selecting text parts by similarity to query")
    docs = docsearch.similarity_search(query)

    chain = load_qa_with_sources_chain(
        OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0),
        chain_type="stuff",
    )
    print("running query against Q&A chain.\n")
    result = chain(
        {"input_documents": docs, "question": query}, return_only_outputs=True
    )
    output: str = result["output_text"]
    parts = output.split("SOURCES: ")
    if len(parts) == 2:
        answer, sources_refs = parts
        sources = retrieve_sources(sources_refs, texts)
    elif len(parts) == 1:
        answer = parts[0]
        sources = []
    else:
        raise RuntimeError(
            f"Expected to receive an answer with a single 'SOURCES' block, got:\n{output}"
        )
    return answer.strip(), sources

image = (
    Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10"
    )
    .pip_install(
        "vllm==0.3.2",
        "huggingface_hub==0.19.4",
        "hf-transfer==0.1.4",
        "torch==2.1.2",
        "playwright==1.30.0",
        #langchain
        "faiss-cpu~=1.7.3",
        "langchain~=0.0.138",
        "openai~=0.27.4",
        "tiktoken==0.3.0",
    ).run_commands(
        "playwright install-deps chromium",
        "playwright install chromium"
    )
    # Use the barebones hf-transfer package for maximum download speeds. Varies from 100MB/s to 1.5 GB/s,
    # so download times can vary from under a minute to tens of minutes.
    # If your download slows down or times out, try interrupting and restarting.
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_folder,
        secrets=[Secret.from_name("my-huggingface-secret")],
        timeout=60 * 20,
    )
)

stub = Stub(f"example-vllm-{BASE_MODEL}", image=image, secrets=[Secret.from_name("my-openai-secret")])

docsearch = None

GPU_CONFIG = gpu.H100(count=1)

@stub.cls(gpu=GPU_CONFIG, secrets=[Secret.from_name("my-huggingface-secret")])
class Model:
    @enter()
    def load(self):
        from vllm import LLM

        if GPU_CONFIG.count > 1:
            # Patch issue from https://github.com/vllm-project/vllm/issues/1116
            import ray

            ray.shutdown()
            ray.init(num_gpus=GPU_CONFIG.count)

        self.template = (
            "start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model"
        )

        # Load the model. Tip: Some models, like MPT, may require `trust_remote_code=true`.
        self.llm = LLM(
            MODEL_DIR,
            enforce_eager=True,  # skip graph capturing for faster cold starts
            tensor_parallel_size=GPU_CONFIG.count,
        )

    @method()
    def generate(self, user_question):
        import time

        from vllm import SamplingParams

        prompt = self.template.format(user=user_question)

        sampling_params = SamplingParams(
            temperature=0.75,
            top_p=0.99,
            max_tokens=256,
            presence_penalty=1.15,
        )
        start = time.monotonic_ns()
        result = self.llm.generate(prompt, sampling_params)
        duration_s = (time.monotonic_ns() - start) / 1e9
        num_tokens = 0

        COLOR = {
            "HEADER": "\033[95m",
            "BLUE": "\033[94m",
            "GREEN": "\033[92m",
            "RED": "\033[91m",
            "ENDC": "\033[0m",
        }

        for output in result:
            num_tokens += len(output.outputs[0].token_ids)
            print(
                f"\n{COLOR['BLUE']}{output.outputs[0].text}",
                "\n\n",
                sep=COLOR["ENDC"],
            )
            time.sleep(0.01)
        print(
            f"{COLOR['HEADER']}{COLOR['GREEN']}Generated {num_tokens} tokens from {BASE_MODEL} in {duration_s:.1f} seconds, throughput = {num_tokens / duration_s:.0f} tokens/second on {GPU_CONFIG}.{COLOR['ENDC']}"
        )

    @exit()
    def stop_engine(self):
        if GPU_CONFIG.count > 1:
            import ray

            ray.shutdown()


@stub.function()
#@web_endpoint()
def get_links(cur_url):
    from playwright.sync_api import sync_playwright
    #url = 'https://jobs.netflix.com/search'
    with sync_playwright() as p:
        browser =  p.chromium.launch()
        ua = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/69.0.3497.100 Safari/537.36"
        )
        page =  browser.new_page(user_agent=ua)
        page.goto(cur_url)

        all_links = []
        disabled = False
        page_index = 1
        while True:
            # find urls with /jobs/ substring href
            substring='/jobs/'
            selector = f'a[href*="{substring}"]'
            page.wait_for_selector(selector, timeout=5000)  # Waits for 5 seconds

            links = page.locator(selector)
            print(links.count())
            links_text = [link.text_content() for link in links.element_handles()]

            #links =  page.get_by_role("link").filter(page.get_by_role("button", name="column 2 button"))
            #print(links_text[0])

            all_links.extend(links_text)
        
            
            next_page_button = page.get_by_label('Next Page')
            disabled = next_page_button.is_disabled()
            page_index += 1

            if not disabled:
                page.goto(cur_url+'?page='+str(page_index))
                print('new page url',page.url)
            else: 
                break
        print('got all links', all_links)        
        
        browser.close()
    return all_links


# @stub.local_entrypoint()
# def main():
    # file_content = ''
    # with open('./source.html', 'r',encoding='utf-8') as file:
    #     for line in file:
    #         file_content += line

    # try:
    #     model = Model()
    #     # GET LINKS, LATER
    #     #url = 'https://jobs.netflix.com/search'
    #     #job_links = get_links(url)
    #     #links_str = ' '.join(str(x) for x in job_links)
        
        
    #     # NO CHUNKING USING GEMINI
    #     # question = "Which element in this html file is most likely a button that leads to more jobs when clicked: " + file_content
    #     # model.generate.remote(question)
    #     # print('model selected links', )

    #     # CHUNKING
    #     query = "Which element in this html file is most likely a button that leads to more jobs when clicked?"
    #     answer, sources = qanda_langchain(file_content,query)

    #     bold, end = "\033[1m", "\033[0m"

    #     print(f"🦜 {bold}ANSWER:{end}")
    #     print(answer)
    #     # if show_sources:
    #     #     print(f"🔗 {bold}SOURCES:{end}")
    #     #     for text in sources:
    #     #         print(text)
    #     #         print("----")
    # except Exception as e:
    #     print('exception:',e)


#@stub.local_entrypoint()
@stub.function(volumes={"/my_vol":volume})
def cli():
    file_content = ''
    with open('/my_vol/source.txt', 'r') as f:
        for line in f:
            file_content += line
    query = "Which element in this html file is most likely a button that leads to more jobs when clicked?"

    answer, sources = qanda_langchain(file_content, query)
    # Terminal codes for pretty-printing.
    bold, end = "\033[1m", "\033[0m"

    print(f"🦜 {bold}ANSWER:{end}")
    print(answer)