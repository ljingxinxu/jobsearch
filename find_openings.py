import os

from modal import Image, Secret, Stub, asgi_app, enter, exit, gpu, method, web_endpoint

from fastapi import FastAPI

web_app = FastAPI()

MODEL_DIR = "/model"
BASE_MODEL = "google/gemma-7b-it"

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
        "gradio~=3.50.2"

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

stub = Stub(f"example-vllm-{BASE_MODEL}", image=image)

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
        res = ''
        print(len(result))
        for output in result:
            num_tokens += len(output.outputs[0].token_ids)
            print(
                f"{COLOR['HEADER']}{COLOR['GREEN']}{output.prompt}",
                f"\n{COLOR['BLUE']}{output.outputs[0].text}",
                "\n\n",
                sep=COLOR["ENDC"],
            )
            print(type(output), output.outputs[0].text)
        
            #res.append(output.outputs[0].text)
        return output.outputs[0].text
        #time.sleep(0.01)
        # print(
        #     f"{COLOR['HEADER']}{COLOR['GREEN']}Generated {num_tokens} tokens from {BASE_MODEL} in {duration_s:.1f} seconds, throughput = {num_tokens / duration_s:.0f} tokens/second on {GPU_CONFIG}.{COLOR['ENDC']}"
        # )

    @exit()
    def stop_engine(self):
        if GPU_CONFIG.count > 1:
            import ray

            ray.shutdown()


@stub.function(keep_warm=1, container_idle_timeout=60 * 20)
@asgi_app()
#@web_endpoint()
def launch_gradio():
    import gradio as gr
    from gradio.routes import mount_gradio_app
    model = Model()
    links_str='manager, software developer, engineer'        
    question = "Here are all the job openings at a company I want to join. Pick out all the 50 exact job titles that match my criteria most closely. criteria #1 no L3, L4, L5 or principal, or senior roles. #2 no roles based outside the U.S. #3 roles that leverage my engineering and product skills"
    input = question + links_str


    iface = gr.Interface(
        model.generate.remote,
        inputs=[gr.Textbox(value=input, label="🎨 Prompt")],
        outputs="text",
        # some extra bits to make it look nicer
        title="LoRAs Galore",
        description="# Try out some of the top custom SDXL models!"
        "\n\nPick a LoRA finetune of SDXL from the dropdown, then prompt it to generate an image."
        "\n\nCheck out [the code on GitHub](https://github.com/modal-labs/modal-examples/blob/main/10_integrations/cloud_bucket_mount_loras.py)"
        " if you want to create your own version or just see how it works."
        "\n\nPowered by [Modal](https://modal.com) 🚀",
        theme="soft",
        allow_flagging="never",
    )
    print('here??')
    return mount_gradio_app(app=web_app, blocks=iface, path="/")
    return

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


@stub.local_entrypoint()
#@stub.function()
#@web_endpoint()
def main():
    try:
        model = Model()
        url = 'https://jobs.netflix.com/search'
        #job_titles = get_links.remote(url)
        #links_str = ', '.join(job_titles)
        links_str=''
        with open('./output.txt','r') as f:
            for line in f:
                links_str+=line
        question = "Here are all the job openings at a company I want to join. Pick out all the 50 exact job titles that match my criteria most closely. criteria #1 no L3, L4, L5 or principal, or senior roles. #2 no roles based outside the U.S. #3 roles that leverage my engineering and product skills"
        
        result= launch_gradio.remote(model, question, links_str)
        print(result)
        return result
        #return model.generate.remote(question + links_str)
    except Exception as e:
        print('exception:',e)