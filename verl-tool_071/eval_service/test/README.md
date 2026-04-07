
# LLM Tool Calling Service Tests

This folder contain basic tests for the LLM evaluation service. 

`test_api_mp.py` examine the pipeline by sending multiple requests in parallel while `test_api.py` is a single-threaded program.

When using these two scripts to test whether your deployment of LLM Evaluation Service is successful, Remember to adjust the `model_name` and `base_url` according to your startup settings in deploying the eval_service.