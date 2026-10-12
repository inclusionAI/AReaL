# Multimodal executor fixture

`sglang_mm_executor_119b5ffe.txt` contains the unmodified
`sglang/srt/multimodal/processors/executor.py` from the validated SGLang rollout build
`0.5.19.dev125+g119b5ffe4`. Its SHA-256 is
`b69d19801ea61a36f6359e3e4f5464d9eaa1de825368a2c1aaa36005098f63b2`.

`test_sglang_mm_executor_patch.py` applies the common compatibility patch to these bytes
and executes the resulting class without importing SGLang or GPU libraries. It tests
successful calls, patch idempotence, rejection of unknown revisions, and conversion of
`StopIteration` before the thread-to-asyncio boundary. The failure case must finish,
release the request lock, and leave the executor usable for the next request.

Keep the fixture bytes unchanged, including whitespace: the patch validates the exact
source checksum. Put provenance and explanations here instead of adding comments to the
fixture. A newly supported upstream revision needs its own validated source, checksum,
and regression coverage.
