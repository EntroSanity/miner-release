import os
import sys
import time
import logging
import subprocess
from openai import OpenAI

class LLMServerConfig:
    MAX_MODEL_LEN = 4096
    CHAT_TEMPLATE = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    def __init__(self, base_config):
        self.base_config = base_config
        self.num_gpus = len(self.base_config.gpu_to_use.split(','))
        os.environ["CUDA_VISIBLE_DEVICES"] = self.base_config.gpu_to_use

        self.model_id = sys.argv[1]  # HF Model ID from the first argument
        self.model_quantization = None if sys.argv[2] == 'None' else sys.argv[2]  # Model quantization from the second argument
        self.served_model_name = sys.argv[3]  # Served model name from the third argument
        self.gpu_memory_util  = sys.argv[4] # GPU memory utilization ratio for vllm
        self.model_revision = None if len(sys.argv) <= 5 or sys.argv[5] == 'None' else sys.argv[5]  # Model revision from the fourth argument, if present
        self.process = None
    
    def initialize_client(self):
        return OpenAI(base_url=self.base_config.api_base_url, api_key="N/A")

    def start_llm_server(self):
        """
        Start the LLM server with the provided model details.
        """
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_id,
            "--served-model-name", self.served_model_name,
            "--max-model-len", str(self.MAX_MODEL_LEN),
            "--chat-template", self.CHAT_TEMPLATE,
            "--disable-log-requests",
            "--dtype", "half",
            "--port", str(self.base_config.port),
            "--tensor-parallel-size",str(self.num_gpus),
            "--gpu-memory-utilization", self.gpu_memory_util
        ]

        if self.model_revision:
            cmd.extend(["--revision", self.model_revision])
        if self.model_quantization:
            cmd.extend(["--quantization", self.model_quantization])

        self.process = subprocess.Popen(cmd)
        return self.process

    def terminate_llm_server(self):
        if self.process:
            logging.info("Terminating LLM server process...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            logging.info("LLM server process terminated.")

    def wait_for_server_ready(self, timeout=120, interval=10):
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.health_check():
                logging.info("Server is ready.")
                return True
            else:
                logging.info("Server not ready yet, waiting...")
                time.sleep(interval)
        logging.error("Timeout waiting for server to be ready.")
        return False

    def fetch_gpu_temperature(self):
        try:
            temp_output = subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"])
            temp_values = temp_output.decode().strip().split('\n')
            current_temp = int(temp_values[0])  # Take the first temperature value
            return current_temp
        except Exception as e:
            print(f"Error occurred while fetching GPU temperature: {str(e)}")
            return None

    def fetch_ram_usage(self):
        try:
            vram_output = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"])
            vram_values = vram_output.decode().strip().split("\n")[0].split(",")
            used_vram = int(vram_values[0].strip())
            total_vram = int(vram_values[1].strip())
            vram_usage = (used_vram / total_vram) * 100
            return vram_usage
        except Exception as e:
            print(f"Error occurred while fetching VRAM usage: {str(e)}")
            return None

    def fetch_cpu_usage(self):
        try:
            cpu_output = subprocess.check_output(["top", "-bn1"])
            cpu_lines = cpu_output.decode().split("\n")
            cpu_usage = float(cpu_lines[2].split()[1])
            return cpu_usage
        except Exception as e:
            print(f"Error occurred while fetching CPU usage: {str(e)}")
            return None
    
    def perform_inference(self, client, token_count, iteration_count):
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "write a 200-word essay on the topic of the future of Ethereum"}],
            model=self.served_model_name,
            max_tokens=200,
        )
        token_count += response.usage.total_tokens
        iteration_count += 1
        return token_count, iteration_count
    
    def print_load_test_results(self, duration, num_iterations, total_tokens, avg_tokens_per_second, max_temp, max_ram_usage, max_cpu_usage):
        print("=" * 30)
        print(" " * 5 + "Load Test Results")
        print("=" * 30)
        print(f"Duration: {duration} seconds")
        print(f"Iterations: {num_iterations}")
        print(f"Total Tokens: {total_tokens}")
        print(f"Average Tokens/Second: {avg_tokens_per_second:.2f}")
        print(f"Max GPU Temperature: {max_temp}°C")
        print(f"Max RAM Usage: {max_ram_usage:.2f}%")
        print(f"Max CPU Usage: {max_cpu_usage:.2f}%")
        print("=" * 30)
        print()
    
    def health_check(self, duration=30, temp_threshold=80, ram_threshold=80, cpu_threshold=80):
        try:
            client = self.initialize_client()
            print("Initialized client for model:", self.served_model_name)

            start_time = time.time()
            total_tokens = 0
            num_iterations = 0
            max_temp = 0
            max_ram_usage = 0
            max_cpu_usage = 0
    
            while time.time() - start_time < duration:
                total_tokens, num_iterations = self.perform_inference(client, total_tokens, num_iterations)
    
                current_temp = self.fetch_gpu_temperature()
                if current_temp is not None:
                    max_temp = max(max_temp, current_temp)
                    if current_temp > temp_threshold:
                        print(f"Warning: GPU temperature exceeded threshold of {temp_threshold}°C for model {self.served_model_name}.")
                        return False
    
                current_ram_usage = self.fetch_ram_usage()
                if current_ram_usage is not None:
                    max_ram_usage = max(max_ram_usage, current_ram_usage)
                    if current_ram_usage > ram_threshold:
                        print(f"Warning: RAM usage exceeded threshold of {ram_threshold}% for model {self.served_model_name}.")
                        return False
    
                current_cpu_usage = self.fetch_cpu_usage()
                if current_cpu_usage is not None:
                    max_cpu_usage = max(max_cpu_usage, current_cpu_usage)
                    if current_cpu_usage > cpu_threshold:
                        print(f"Warning: CPU usage exceeded threshold of {cpu_threshold}% for model {self.served_model_name}.")
                        return False
    
            end_time = time.time()
            inference_latency = end_time - start_time
            avg_tokens_per_second = total_tokens / inference_latency
    
            self.print_load_test_results(duration, num_iterations, total_tokens, avg_tokens_per_second, max_temp, max_ram_usage, max_cpu_usage)
    
            if avg_tokens_per_second < 5:
                print(f"Warning: Inference speed is too slow for model {self.served_model_name}.")
                return False
    
            return True
        except Exception as e:
            print(f"Model {self.served_model_name} is not ready. Waiting for LLM Server to finish loading the model to start.")
            return False