# Verl-Tool

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/imgs/logo.png">
    <img alt="VerlTool" src="assets/imgs/logo.png" width=20%>
  </picture>
</p>

<h3 align="center">
VerlTool: A unified and easy-to-extend tool-agent training framework based on verl.
</h3>

<p align="center">
| 
<a href="https://arxiv.org/abs/2509.01055"><b>Paper</b></a> |
<a href="https://github.com/TIGER-AI-Lab/verl-tool/blob/main/assets/docs/install.md"><b>Quick Start</b></a> |
  <a href="https://github.com/TIGER-AI-Lab/verl-tool/tree/main/examples/train"><b>Training Recipes</b></a> |
  <a href="https://deepwiki.com/TIGER-AI-Lab/verl-tool"><b>DeepWiki</b></a> |
  <a href="assets/imgs/wechat_group.jpg"><b>WeChat Group</b></a> |
  <a href="https://discord.gg/kZggJmaz"><b>Discord</b></a>
|
</p>

---



## News
+ [2026/03/01] 🏆 Our paper has been accepted by [**ICLR 2026 Lifelong Agents Workshop**](https://lifelongagent.github.io/)!
+ [2025/11/10] VerlTool has re-organized its codebase to improve modularity and maintainability, supporting to the latest verl (`0.6.0`) and vllm (`0.11.0`) versions. Please refer to the [verl-tool v0.6.0.dev Upgrade Notes](/assets/docs/updates/verltool_v0.6.0_upgrade.md) for more details.
+ [2025/09/02] VerlTool's tech report is out! See on [Hugging Face Daily Paper](https://huggingface.co/papers/2509.01055)!
+ [2025/06/30] We reproduce Search-R1 with even higher performance on the same benchmarks! See [PR](https://github.com/TIGER-AI-Lab/verl-tool/pull/71) and training [README](examples/train/search_r1/README.md) for more details.
+ [2025/06/28] We support NL2SQL tool RL training. See NL2SQL [README](https://github.com/TIGER-AI-Lab/verl-tool/tree/main/examples/train/skysql) for more details.
+ [2025/06/26] We support DAPO recipe training. See [DAPO.md](./assets/docs/DAPO.md) for more details.
+ [2025/06/18] VerlTool now officially supports Trajectory-Level asynchronous, speeding up the rollout generation with tool calling by at least 2x! see [asyncRL.md](./assets/docs/asyncRL.md) for more details.
+ [2025/06/16] We have updated the verl submodule to the latest version (06/16) and modified some code to adapt to the new version.
+ [2025/06/13] We integrated [DeepWiki](https://deepwiki.com/TIGER-AI-Lab/verl-tool) for Verl-Tool. Feel free to browse the AI-generated docs and chat with Verl-tool codes.
+ [2025/06/06] We have updated a detailed design overview in the README, including how to add new tools, how to use the tool server, and how to train your own models with verl-tool.
+ [2025/05/31] We released the Verl-tool training/evaluation code with ToRL training as an initial example (see [X post](https://x.com/DongfuJiang/status/1929198238017720379)). We are working on the paper and will release it very soon.

## Features

- 🔧 **Complete decoupling of actor rollout and environment interaction** - We use verl as a submodule to benefit from ongoing verl repository updates. All tool calling is integrated via a unified API, allowing you to easily add new tools by simply adding a Python file and testing independently.
- 🌍 **Tool-as-environment paradigm** - Each tool interaction can modify the environment state. We store and reload environment states for each trajectory.
- ⚡ **Native RL framework for tool-calling agents** - verl-tool natively supports multi-turn interactive loops between agents and their tool environments.
- 📊 **User-friendly evaluation suite** - Launch your trained model with OpenAI API alongside the tool server. Simply send questions and get final outputs with all interactions handled internally. See [benchmarks](benchmarks).

![Verl-Tool Architecture](assets/imgs/verl_tool_architecture.png)

## 📚 Contents Link
- 📖 [Installation Guide](./assets/docs/install.md)
- ⚡ [Synchronous Rollout Design](./assets/docs/sync_design.md)
- 🔄 [Asynchronous Rollout Design](./assets/docs/asyncRL.md)
- 🛠️ [Tool Server Design](./assets/docs/tool_server.md)
- 🎯 [Training Guide](./assets/docs/training_guide.md)
- 📊 [Evaluation Guide](./assets/docs/evaluation.md)
- 🔧 [Update Verl Submodule Version](./assets/docs/update_verl.md)
- 📈 [Existing Training Results](./assets/docs/training_results.md)
- 🤝 [Contributing Guide](./assets/docs/contributing.md)

## Core Contributors

<table>
<tr>
    <td align="center">
        <a href="https://github.com/jdf-prog">
            <img src="https://github.com/jdf-prog.png" width="75px;" alt="Dongfu Jiang"/>
            <br />
            <sub><b>Dongfu Jiang</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/Zhuofeng-Li">
            <img src="https://github.com/Zhuofeng-Li.png" width="75px;" alt="Zhuofeng Li"/>
            <br />
            <sub><b>Zhuofeng Li</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/EigenTom">
            <img src="https://github.com/EigenTom.png" width="75px;" alt="Yi Lu"/>
            <br />
            <sub><b>Yi Lu</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/cogito233">
            <img src="https://github.com/cogito233.png" width="75px;" alt="Zhiheng Lvu"/>
            <br />
            <sub><b>Zhiheng Lvu</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/erenup">
            <img src="https://github.com/erenup.png" width="75px;" alt="Ping Nie"/>
            <br />
            <sub><b>Ping Nie</b></sub>
        </a>
    </td>
</tr>
</table>

## Advisors

<table>
<tr>
    <td align="center">
        <a href="https://github.com/wenhuchen">
            <img src="https://github.com/wenhuchen.png" width="75px;" alt="Wenhu Chen"/>
            <br />
            <sub><b>Wenhu Chen</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/P2333">
            <img src="https://github.com/P2333.png" width="75px;" alt="Tianyu Pang"/>
            <br />
            <sub><b>Tianyu Pang</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/duchao0726">
            <img src="https://github.com/duchao0726.png" width="75px;" alt="Chao Du"/>
            <br />
            <sub><b>Chao Du</b></sub>
        </a>
    </td>
</tr>
</table>

## Acknowledgements

We thank the following open-source projects for making verl-tool possible:
- [VLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang) for their fast LLM inference support!
- [verl](https://github.com/volcengine/verl) for the excellent RL framework design.
- [SearchR1](https://github.com/PeterGriffinJin/Search-R1), [RAGEN](https://github.com/RAGEN-AI/RAGEN), and [ToRL](https://github.com/GAIR-NLP/ToRL) for their early-stage exploration of tool-agent RL training.

We thank [Netmind.AI](https://www.netmind.ai/), [SeaAI Lab](https://sail.sea.com/), and [Map](https://huggingface.co/m-a-p) for GPU support!

## Community Projects Inspired by Verl-Tool
- [AgentFlow](https://github.com/lupantech/AgentFlow): In-the-Flow Agentic System Optimization

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=TIGER-AI-Lab/verl-tool&type=Date)](https://www.star-history.com/#TIGER-AI-Lab/verl-tool&Date)


## Badge

[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/tiger-ai-lab-verl-tool-badge.png)](https://mseep.ai/app/tiger-ai-lab-verl-tool)

## Citation
```bibtex
@article{jiang2025verltool,
  title={VerlTool: Towards Holistic Agentic Reinforcement Learning with Tool Use},
  author={Jiang, Dongfu and Lu, Yi and Li, Zhuofeng and Lyu, Zhiheng and Nie, Ping and Wang, Haozhe and Su, Alex and Chen, Hui and Zou, Kai and Du, Chao and others},
  journal={arXiv preprint arXiv:2509.01055},
  year={2025}
}
```
