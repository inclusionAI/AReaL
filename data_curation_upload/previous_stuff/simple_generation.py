                with open(tok_cfg, "r", encoding="utf-8") as fh:
                    cfg = json.load(fh)
                chat_tmpl = cfg.get("chat_template")
                if chat_tmpl and not getattr(tok, "chat_template", None):
                    # transformers exposes .chat_template on tokenizers after v4.41
                    try:
                        tok.chat_template = chat_tmpl
                    except Exception:
                        pass
            except Exception:
                pass

        return tok

def _render_chat_or_fallback(tokenizer, messages, fallback_text: str) -> str:
    """Try tokenizer.apply_chat_template; if not available, return fallback_text."""
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False)
    except Exception:
        return fallback_text
def wait_for_server(host="127.0.0.1", port=29950, timeout=60):
    """Wait for server to be ready"""
    print("⏳  Waiting for server to start...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://{host}:{port}/health", timeout=5)
            if response.status_code == 200:
                print("✅  Server is ready!")
                return True
        except (requests.ConnectionError, requests.Timeout):
            elapsed = int(time.time() - start_time)
                                                                                                                                             87,1          14%