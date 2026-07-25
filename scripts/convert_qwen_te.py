#!/usr/bin/env python3
import os
import sys

import gguf


def convert_qwen_te():
    src_path = ".alpaca-router/companions/Qwen2.5-VL-7B-Instruct.Q4_K_M.gguf"
    dst_path = ".alpaca-router/companions/qwen2.5_vl_te_sd.gguf"

    if not os.path.exists(src_path):
        print(f"Source file {src_path} not found.")
        sys.exit(1)

    print(f"Reading {src_path}...")
    reader = gguf.GGUFReader(src_path)
    writer = gguf.GGUFWriter(dst_path, "qwen-vl")

    mapped = 0
    for tensor in reader.tensors:
        name = tensor.name
        if name == "token_embd.weight":
            new_name = "text_encoders.llm.model.embed_tokens.weight"
        elif name == "output_norm.weight":
            new_name = "text_encoders.llm.model.norm.weight"
        elif name.startswith("blk."):
            parts = name.split(".")
            idx = parts[1]
            sub = ".".join(parts[2:])
            if sub == "attn_k.weight":
                new_name = f"text_encoders.llm.model.layers.{idx}.self_attn.k_proj.weight"
            elif sub == "attn_q.weight":
                new_name = f"text_encoders.llm.model.layers.{idx}.self_attn.q_proj.weight"
            elif sub == "attn_v.weight":
                new_name = f"text_encoders.llm.model.layers.{idx}.self_attn.v_proj.weight"
            elif sub == "attn_output.weight":
                new_name = f"text_encoders.llm.model.layers.{idx}.self_attn.o_proj.weight"
            elif sub == "attn_norm.weight":
                new_name = f"text_encoders.llm.model.layers.{idx}.input_layernorm.weight"
            elif sub == "ffn_norm.weight":
                new_name = f"text_encoders.llm.model.layers.{idx}.post_attention_layernorm.weight"
            elif sub == "ffn_gate.weight":
                new_name = f"text_encoders.llm.model.layers.{idx}.mlp.gate_proj.weight"
            elif sub == "ffn_up.weight":
                new_name = f"text_encoders.llm.model.layers.{idx}.mlp.up_proj.weight"
            elif sub == "ffn_down.weight":
                new_name = f"text_encoders.llm.model.layers.{idx}.mlp.down_proj.weight"
            else:
                new_name = f"text_encoders.llm.model.layers.{idx}.{sub}"
        else:
            new_name = f"text_encoders.llm.model.{name}"

        writer.add_tensor(new_name, tensor.data)
        mapped += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"✅ Successfully converted {mapped} tensors to {dst_path}!")

if __name__ == "__main__":
    convert_qwen_te()
