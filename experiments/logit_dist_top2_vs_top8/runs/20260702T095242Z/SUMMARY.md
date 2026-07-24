# Top2 vs Top8 First-Token Logprob Experiment

- Date: 2026-07-02T09:54:51.543927+00:00
- Base URL: `http://127.0.0.1:8000`
- Model root: `GLM5.2`
- Prompts: 96
- Paired valid observations: 96
- Request shape: chat completions, `temperature=0`, `max_tokens=1`, `logprobs=true`, `top_logprobs=20`, thinking disabled.

## Main Result

- Mean paired logprob difference, top2 - top8: -0.145191
- 95% bootstrap CI for the mean: [-0.195849, -0.094242]
- Median paired difference: -0.047007
- Paired t normal-approx p-value: 2.158e-08
- Sign test: top2 higher on 12/96 nonzero pairs, p=1.832e-14
- Same first visible token rate: 74.0%
- Median absolute logprob shift: 0.166217
- 90th percentile absolute logprob shift: 0.470519

## Tier Marginals

- Top2 mean logprob: -0.221630
- Top8 mean logprob: -0.076439
- Top2 mean probability of sampled token: 0.824757
- Top8 mean probability of sampled token: 0.935523

## Logprobs Payload Note

`top_logprobs` candidate counts were min/median/max = 1/1.0/1. On this running server the OpenAI-compatible response returns the sampled token logprob reliably, but the requested top-logprobs list is collapsed to one packed candidate. The analysis therefore uses sampled-token logprob distributions, not full vocabulary entropy.

## Example Pairs

| id | prompt | top2 token | top2 logprob | top8 token | top8 logprob |
|---:|---|---:|---:|---:|---:|
| 0 | Answer with only the number: 47 + 10. | `57` | -0.001830 | `57` | -0.000394 |
| 1 | Answer with only the number: 45 * 9. | `40` | -0.015622 | `40` | -0.000884 |
| 2 | Answer with one word: the opposite of up. | `down` | -0.523704 | `Down` | -0.079059 |
| 3 | Answer with one word: the color of snow. | `white` | -0.282050 | `White` | -0.016311 |
| 4 | Complete the phrase with one word only: bread and | `b` | -0.205398 | `b` | -0.039181 |
| 5 | Translate to English with one word only: bonjour. | `Hello` | -0.263956 | `hello` | -0.523260 |
| 6 | Answer yes or no only: fire is cold | `no` | -0.351244 | `No` | -0.020694 |
| 7 | Return only the next item in the sequence: 5, 10, 15, 20, | `25` | -0.000097 | `25` | -0.000008 |
