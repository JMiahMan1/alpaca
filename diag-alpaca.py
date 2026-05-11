import requests
import time
import json

BASE_URL = "http://localhost:11434"

def test_endpoints():
    print("--- Testing Basic Endpoints ---")
    for ep in ["/api/tags", "/api/version", "/api/ps"]:
        try:
            resp = requests.get(f"{BASE_URL}{ep}", timeout=10)
            print(f"GET {ep}: {resp.status_code} - {resp.text[:100]}")
        except Exception as e:
            print(f"GET {ep} failed: {e}")

def test_chat_stream(model_name):
    print(f"\n--- Triggering Switch to Model: {model_name} ---")
    
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Tell me a very short joke about coding."}],
        "stream": True
    }
    
    start_time = time.time()
    dots_count = 0
    full_response = ""
    
    try:
        with requests.post(f"{BASE_URL}/api/chat", json=payload, stream=True, timeout=120) as r:
            r.raise_for_status()
            print("Connection established. Waiting for switch and response...")
            
            buffer = ""
            for chunk in r.iter_content(chunk_size=1):
                char = chunk.decode()
                if char == " " and not full_response:
                    dots_count += 1
                    if dots_count % 5 == 0:
                        print(".", end="", flush=True)
                else:
                    buffer += char
                    if char == "\n":
                        line = buffer.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                content = data.get("message", {}).get("content") or ""
                                print(content, end="", flush=True)
                                full_response += str(content)
                                if data.get("done"):
                                    break
                            except json.JSONDecodeError:
                                pass
                        buffer = ""
            
            elapsed = time.time() - start_time
            print(f"\n\n--- Switch Stats ---")
            print(f"Total time for switch + load: {elapsed:.2f}s")
            print(f"Heartbeat spaces (Client kept alive): {dots_count}")
            print(f"Response received from {model_name}.")
            
    except Exception as e:
        print(f"\nChat switch failed: {e}")

if __name__ == "__main__":
    print("=== TurboQuant Model Switch Diagnostic ===\n")
    
    # 1. Check what's currently active
    current_model = None
    try:
        ps_resp = requests.get(f"{BASE_URL}/api/ps").json()
        active_models = ps_resp.get("models", [])
        if active_models:
            current_model = active_models[0]['name']
            print(f"Current model in memory: {current_model}")
        else:
            print("No model currently reported as active.")
    except Exception as e:
        print(f"Could not check PS: {e}")

    # 2. Get all available models
    try:
        tags_resp = requests.get(f"{BASE_URL}/api/tags").json()
        all_models = [m['name'] for m in tags_resp.get('models', [])]
        print(f"Available tags: {all_models}")
        
        # 3. Find a model to switch TO
        target_model = None
        for m in all_models:
            if m != current_model:
                target_model = m
                break
        
        if not target_model:
            print("No alternative model found to switch to. Trying first available.")
            target_model = all_models[0] if all_models else "qwen3.5:9b"

        # 4. Perform the switch test
        test_chat_stream(target_model)
        
        # 5. Verify the new model is reported as active
        print("\nVerifying updated state...")
        time.sleep(1)
        ps_resp = requests.get(f"{BASE_URL}/api/ps").json()
        new_active = ps_resp.get("models", [])
        if new_active:
            print(f"Active model is now: {new_active[0]['name']}")
            if new_active[0]['name'] == target_model:
                print("SUCCESS: Switch verified.")
            else:
                print(f"FAILURE: Expected {target_model} but found {new_active[0]['name']}")
        
    except Exception as e:
        print(f"Diagnostic failed: {e}")
