# coding=utf-8
import app.config_helper as helper

# net
net_host: str = helper.get_str_value('net.host', '0.0.0.0')
net_port: int = helper.get_int_value('net.port', '5321')
net_auth_file_path: str = helper.get_str_value('net.auth_file_path', '')

# api
api_sd_host: str = helper.get_str_value('api.sd_host', 'http://localhost:7860')
api_ollama_host: str = helper.get_str_value('api.ollama_host', 'http://localhost:11434')
api_openai_host: str = helper.get_str_value('api.openai_host', 'https://api.openai.com')
api_openai_key: str = helper.get_str_value('api.openai_key', '')

# sd
sd_model_path: str = helper.get_str_value('sd.model_path', '')
sd_lora_path: str = helper.get_str_value('sd.lora_path', '')
train_output_path: str = helper.get_str_value('sd.train_output_path', '')
sd_styles_file: str = helper.get_str_value('sd.styles_file', '')
sd_default_model: str = helper.get_str_value('sd.default_model', '')
default_llm_type: str = helper.get_str_value('sd.default_llm_type', '').lower()
default_llm_model: str = helper.get_str_value('sd.default_llm_model', '')
