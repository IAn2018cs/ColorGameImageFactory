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
default_llm_type: str = helper.get_str_value('sd.default_llm_type', '').lower()
default_llm_model: str = helper.get_str_value('sd.default_llm_model', '')
default_color_sd_model: str = helper.get_str_value('sd.default_color_sd_model', '')
default_color_sd_lora: str = helper.get_str_value('sd.default_color_sd_lora', '')
default_color_sd_lora_weight: float = helper.get_float_value('sd.default_color_sd_lora_weight', '1')
default_color_sd_prompt: str = helper.get_str_value('sd.default_color_sd_prompt', '')
default_color_sd_negative: str = helper.get_str_value('sd.default_color_sd_negative', '')
default_color_sd_sampling: str = helper.get_str_value('sd.default_color_sd_sampling', '')
default_color_sd_schedule: str = helper.get_str_value('sd.default_color_sd_schedule', '')
default_color_sd_steps: int = helper.get_int_value('sd.default_color_sd_steps', '20')
default_color_sd_cfg: float = helper.get_float_value('sd.default_color_sd_cfg', '7')
default_line_sd_model: str = helper.get_str_value('sd.default_line_sd_model', '')
default_line_sd_lora: str = helper.get_str_value('sd.default_line_sd_lora', '')
default_line_sd_lora_weight: float = helper.get_float_value('sd.default_line_sd_lora_weight', '1')
default_line_sd_prompt: str = helper.get_str_value('sd.default_line_sd_prompt', '')
default_line_sd_negative: str = helper.get_str_value('sd.default_line_sd_negative', '')
default_line_sd_sampling: str = helper.get_str_value('sd.default_line_sd_sampling', '')
default_line_sd_schedule: str = helper.get_str_value('sd.default_line_sd_schedule', '')
default_line_sd_steps: int = helper.get_int_value('sd.default_line_sd_steps', '20')
default_line_sd_cfg: float = helper.get_float_value('sd.default_line_sd_cfg', '7')
