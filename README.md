<img src = './Architecture diagram.png'/>

## mini_transformer
mini_transformer is designed to rapidly construct state-of-the-art new architectures for transformer

## Install
1、Install the GPU version of PyTorch for better performance.
```bash
pip3 install torch --index-url https://download.pytorch.org/whl/cu124
```
2、Install mini_transformer
```bash
pip install mini_transformer
```

## Usage
1、use GQA attention(https://arxiv.org/abs/2305.13245)
```python
import torch
from mini_transformer import Transformer

data = torch.randint(0, 6400, (2, 512))
start_pos = 0
mask = torch.triu(torch.full((data.shape[1], data.shape[1]), float('-inf')), 1)
model = Transformer(attn_name = 'gqa')
out = model(data, start_pos, mask)
print(out.shape)
```

2、use MLA attention(https://arxiv.org/abs/2412.19437)
```python
import torch
from mini_transformer import Transformer

data = torch.randint(0, 6400, (2, 512))
start_pos = 0
mask = torch.triu(torch.full((data.shape[1], data.shape[1]), float('-inf')), 1)
model = Transformer(attn_name = 'mla')
out = model(data, start_pos, mask)
print(out.shape)
```

3、inference(kv_cache)
```python
import torch
from mini_transformer import Transformer

data = torch.randint(0, 6400, (2, 512))
start_pos = 0
mask = torch.triu(torch.full((data.shape[1], data.shape[1]), float('-inf')), 1)
model = Transformer(attn_name = 'mla')
out = model(data, start_pos, mask, use_cache = True)
print(out.shape)

inference_data = torch.randint(0, 6400, (2, 1))
start_pos = data.shape[1]
out = model(inference_data, start_pos, use_cache = True)
print(out.shape)
```

4、inference(no kv_cache)
```python
import torch
from mini_transformer import Transformer

# init
data = torch.randint(0, 6400, (2, 512))
start_pos = 0
mask = torch.triu(torch.full((data.shape[1], data.shape[1]), float('-inf')), 1)
model = Transformer(attn_name = 'mla')
out = model(data, start_pos, mask)
print(out.shape)

# inference
data = torch.cat((data, torch.randint(0, 6400, (2, 1))), dim = 1)
start_pos = 0
mask = torch.triu(torch.full((data.shape[1], data.shape[1]), float('-inf')), 1)
out = model(data, start_pos)
print(out.shape)
```