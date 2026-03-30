## Memory Manager

<p align="center">
    <img src="../../docs/gigpo/framework-comparison.png" alt="framework" width="100%">
</p>

`verl-agent` allows for flexibly choosing what history to include for each step, such as, recent steps, key events, summaries, or external knowledge.

We provide a simplest memory implementation as a starting point. Developers are encouraged to extend this module with custom memory strategies, such as dynamic summarization, selective memory retention, or external knowledge integration, to improve the handling of long-horizon interaction histories.