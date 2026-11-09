# Qwen3-32B tokenizer resources

- Hugging Face repository: `Qwen/Qwen3-32B`
- Revision: `f4130b48aa23d4935cda6610b6f82106cdf1f065`
- Source: the five tokenizer/generation files at the repository root; model
  weights were not downloaded or copied.
- Validation environment: `transformers==5.14.1`, `tokenizers==0.22.2`.
- The repository has no separate chat-template file at this revision.
  `AutoTokenizer` loads the `chat_template` embedded in
  `tokenizer_config.json`.
- Embedded chat-template SHA256 (UTF-8 template text):
  `87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5`

## File SHA256

```text
2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2  generation_config.json
8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5  merges.txt
aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4  tokenizer.json
bbc2c089e3bef8753f63348c10595767be8998bc6ca7c7ebcaada346d401fba8  tokenizer_config.json
ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910  vocab.json
```
