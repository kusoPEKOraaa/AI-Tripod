This dataset is sampled from `wikitext/wikitext-2-v1/train`.

Codes to generate this dataset:

```python
import datasets
dataset = datasets.load_dataset('wikitext', 'wikitext-2-v1')

selected = []
i = -1
while len(selected) < 24:
    i += 1
    text = dataset['train'][i]['text']
    if 8 < len(text.split(' ')) <= 16 and '=' not in text:
        selected.append(i)        

tiny_dataset = dataset['train'].select(selected)
```