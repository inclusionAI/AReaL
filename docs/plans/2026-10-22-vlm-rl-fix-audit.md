# Qwen VLM RL repair inventory

## Fixed base and references

The intended base remains `23913f66028ec407198bbdda7dd6d5b88c185c24`
(`chucai.dzq/pr-qwen38-27b-rl-20261021`). Its refactoring and newer framework features
are retained. Reference commit IDs are provenance, not a requirement that the same
commits appear individually in the target history.

References checked on 2026-10-22:

- GitHub `chucai.dzq/fix-qwen38-27b-rl`: `17e1dc46a92cd9fd813822a26aeb47e04ee61e78`.
- Saved Flash Next `chucai.dzq/fix-qwen38-flash-rl`:
  `8d5bbd4379b091c3bd408c5e0be9b952cc6d6c4d`. The original remote branch is no longer
  advertised. Its merged framework implementation is available at
  `448de5235341482465362e8b831097685d8e305d`.
- Target before this audit: `dc47418a9861e2587728479e3c9394b7b95fb634`.

## Repair inventory

| Repair                                          | Status / implementation                                                                                                                         |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| Timeout and context-overflow reward policy      | Already folded into base `a333cc5`; Arena failure attribution and zero-reward rules match reference `febf23a`                                   |
| Episode rewards across compacted segments       | Already present; episode score is propagated across retained segments                                                                           |
| Invalid final response and malformed tool calls | Already present in finalization and parser; validation precedes discounting and export                                                          |
| Prefilled reasoning and reward breakdown        | Already folded into base; reference `17e1dc46` behavior retained                                                                                |
| Consumed-attempt rollout metrics                | Already folded into base; reference `10845b3` behavior retained                                                                                 |
| EOS-only sampled token/logprob preservation     | Already present in ModelResponse                                                                                                                |
| Image preprocessing                             | Keep target `b74dcfda7fff66890cd7889361896a34d37da08c`; no image implementation is replaced by this audit                                       |
| Native Qwen visual freezing controls            | Existing `2abc8df99de4` retained; explicit unfreezing remains supported                                                                         |
| Native Qwen defaults and generated image-pad    | Existing `dc47418a9861` retained; encoder and projection default to frozen, generated image-pad uses the text domain                            |
| Concat sampled prefixes                         | Missing from the target; restore `3be94b3` as an independent commit, retaining complete sampled actions, aligned metadata and prefix validation |
| Frozen-vision feature cache reuse               | Correct unconditional clearing: prove full actor visual freezing, clear once initially, then preserve features across LM-only updates           |
| Generic AWEX metadata dependency                | Move generic publisher out of the Flash binding so native models do not require Flash-only imports                                              |

Sixteen relevant RL/reward/rollout implementation files were compared directly with the
27B reference, rather than by commit names. Nine Flash model, transport, positioning and
residency files matched the merged implementation. Some reference-only diagnostic tests
were deliberately removed during base cleanup; that does not imply their runtime
implementations are missing.

## Frozen vision and AWEX

Native 27B uses `Qwen3_5ForConditionalGeneration`. Actual configured AWEX sources were
checked against the actor image and matched for the converter, training metadata
resolver and Qwen3.5 implementation. `get_mcore_model_parameters()` collects all named
parameters, including frozen vision parameters. The Qwen3.5 converter maps
`vision_model.*`, and inference includes `visual.*` in its transfer metadata. Frozen
native visual weights are therefore resent during weight synchronization after memory
resume.

Flash Next uses the separate Qwen4Exp frozen contract: it explicitly excludes frozen
weights from transfer, verifies their checkpoint source and preserves visual weights
across release/resume. Those model-specific guards already exist in the base. Copying
that exclusion/preservation scheme onto native 27B would change its transfer contract
unnecessarily.

For native feature reuse, inspect a nonempty actor visual owner and require all its
encoder/projection parameters to have `requires_grad=False`. Do not use inference-side
flags or unresolved CLI defaults as proof. Unknown metadata, missing ownership and
partial freezing continue to invalidate embeddings. The first successful flush clears
embeddings even when frozen, establishing the actor checkpoint. Later fully frozen
updates retain features; KV flushing continues normally. No extra environment variable
is required.

## Verification boundary

Run concat, per-token reward, response validation, context-overflow, episode reward,
parser, consumed-attempt filtering, image, AWEX, freezing and token-mask regressions in
the actor container. Check all-files pre-commit and normal commit hooks. Preserve the
intended base and verify all image files, and the image functions in the shared client,
remain unchanged by this audit.

These are implementation and regression checks. They do not establish a single cause of
historical reward collapse, measure full training-step performance, or replace a
same-weight two-engine forward comparison.
